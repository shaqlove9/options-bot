"""webhook.py — TradingView alert receiver for the iobot engine.

A lightweight stdlib ``http.server`` runs in a daemon thread (started from
``engine.py`` only when ``config.WEBHOOK_ENABLED``). For each incoming alert it:

  1. authenticates (shared secret, constant-time compare; optional IP allowlist),
  2. de-dups repeats within ``WEBHOOK_DEDUP_SEC`` (TradingView can fire twice),
  3. enriches the alert into a ``signals.Signal`` using the live data client
     (momentum / rsi / rel_volume / vwap_dist / atr / spot — the same fields the
     ``evaluate`` signal computes),
  4. pushes that Signal onto a thread-safe ``queue.Queue``.

The engine drains that queue each tick and runs every Signal through the
*identical* ``Engine._handle_signal`` path as scanner signals — so feature
capture, the governor count/kill-switch gate, the meta layer, contract selection,
the per-trade risk cap and buying-power check all still apply, exactly once. **The
server thread never touches the executor / governor / meta / store** (those run
only on the engine thread); it only reads market data and enqueues.

POST /tv-webhook  (Content-Type: application/json)
--------------------------------------------------
    {
      "secret":    "<must equal config.WEBHOOK_SECRET>",   # REQUIRED -> 401 if wrong
      "id":        "<unique alert id>",                     # optional -> dedup key
      "symbol":    "SPY",                                   # REQUIRED
      "direction": "call" | "put",                          # REQUIRED
      "price":     612.34                                   # optional (advisory only)
    }

Only ``symbol`` and ``direction`` are required (iobot has no per-signal strategy;
``structure`` single/spread is decided by config, so a ``strategy`` field — if
sent — is ignored). If live data can't be fetched the Signal is still enqueued but
flagged ``advisory`` so ``_handle_signal`` skips the meta gate (its features would
be unreliable) while every hard risk control still applies.
"""
from __future__ import annotations

import hmac
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from iobot import config, signals
from iobot.clock import now_et
from iobot.signals import Signal

log = logging.getLogger("webhook")

_VALID_DIRECTIONS = {"call", "put"}

# TradingView's published outbound webhook IPs — set config.WEBHOOK_IP_ALLOWLIST
# (IOBOT_WEBHOOK_IP_ALLOWLIST) to these to lock the endpoint down.
TRADINGVIEW_IPS = ("52.89.214.238", "34.212.75.30", "54.218.53.128", "52.32.178.7")


class Deduper:
    """In-memory TTL set of recently seen alert ids. Thread-safe."""

    def __init__(self, ttl_sec: float):
        self.ttl = ttl_sec
        self._seen: dict[str, float] = {}
        self._lock = threading.Lock()

    def seen_recently(self, alert_id: str | None, now: float | None = None) -> bool:
        """True if ``alert_id`` was seen within the TTL (records it when new). A
        missing id never de-dups."""
        if not alert_id:
            return False
        now = time.monotonic() if now is None else now
        with self._lock:
            for k in [k for k, t in self._seen.items() if now - t > self.ttl]:
                del self._seen[k]
            if alert_id in self._seen:
                return True
            self._seen[alert_id] = now
            return False


def validate_payload(body, client_ip: str, *, deduper: Deduper):
    """Authenticate + validate a decoded JSON body. Pure (no I/O).

    Returns ``(payload | None, http_status, message)``; ``payload`` is a
    normalized dict (symbol upper-cased) ready for enrichment, or ``None`` when
    the request must not become a trade."""
    if config.WEBHOOK_IP_ALLOWLIST and client_ip not in config.WEBHOOK_IP_ALLOWLIST:
        return None, 403, f"ip {client_ip} not in allowlist"
    if not isinstance(body, dict):
        return None, 400, "body is not a JSON object"

    secret = body.get("secret", "")
    if not config.WEBHOOK_SECRET or not isinstance(secret, str) or \
            not hmac.compare_digest(secret, config.WEBHOOK_SECRET):
        return None, 401, "bad or missing secret"

    symbol = body.get("symbol")
    if not symbol or not isinstance(symbol, str):
        return None, 400, "missing 'symbol'"
    direction = body.get("direction")
    if direction not in _VALID_DIRECTIONS:
        return None, 400, "missing/invalid 'direction' (call|put)"

    alert_id = body.get("id")
    if deduper.seen_recently(str(alert_id) if alert_id is not None else None):
        return None, 200, f"duplicate id {alert_id} ignored"

    price = body.get("price")
    try:
        price = float(price) if price is not None else None
    except (TypeError, ValueError):
        price = None
    return ({"symbol": symbol.upper(), "direction": direction, "price": price,
             "id": alert_id}, 200, "accepted")


def enrich_signal(payload: dict, signal_source) -> Signal:
    """Build a ``signals.Signal`` from a validated payload, filling
    momentum / rsi / rel_volume / vwap_dist / atr / spot from live bars (reusing
    the live ``StrategySignal`` data helpers + the pure functions in signals.py).
    On ANY data failure the Signal is returned with numeric features 0 and
    ``advisory=True`` so the engine skips the meta gate rather than crashing.

    The returned Signal carries ``source="tradingview"`` and a dynamic
    ``advisory`` attribute (signals.Signal is left unmodified)."""
    symbol = payload["symbol"]
    now = now_et()
    sig = Signal(
        symbol=symbol,
        direction=payload["direction"],
        spot=float(payload["price"]) if payload.get("price") else 0.0,
        momentum_pct=0.0,
        rsi=0.0,
        rel_volume=0.0,
        vwap_dist_pct=0.0,
        atr_pct=0.0,
        time=now,
        source="tradingview",
    )
    sig.advisory = False
    try:
        intraday = signal_source._intraday(symbol)
        if intraday is None or len(intraday) < config.RSI_PERIOD + 2:
            raise ValueError("insufficient intraday bars")
        candle = intraday.iloc[-1]
        if candle["open"] <= 0:
            raise ValueError("bad last candle")
        spot = float(candle["close"])
        vwap = signals._session_vwap(intraday, now.date())
        sig.spot = spot
        sig.momentum_pct = (candle["close"] - candle["open"]) / candle["open"] * 100
        sig.rsi = signals._wilder_rsi(intraday["close"], config.RSI_PERIOD)
        sig.rel_volume = signals._relative_volume(intraday, signal_source._daily(symbol), now)
        sig.vwap_dist_pct = (spot - vwap) / vwap * 100 if vwap else 0.0
        sig.atr_pct = signals._atr_pct(intraday, spot)
    except Exception as exc:
        sig.advisory = True
        log.warning("webhook %s: live enrichment failed (%s) — ADVISORY only "
                    "(meta gate will be skipped)", symbol, exc)
    return sig


def make_handler(signal_queue, signal_source, deduper: Deduper):
    """Build a request handler bound to a queue, signal source and deduper."""

    class _Handler(BaseHTTPRequestHandler):
        server_version = "iobot-webhook/1.0"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            log.debug("http %s - " + fmt, self.client_address[0], *args)

        def _reply(self, status: int, message: str):
            body = json.dumps({"status": "ok" if status < 400 else "error",
                               "message": message}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            self._reply(200, "ok") if self.path in ("/health", "/healthz") \
                else self._reply(404, "not found")

        def do_POST(self):
            if self.path.rstrip("/") != "/tv-webhook":
                self._reply(404, "not found")
                return
            client_ip = self.client_address[0]
            try:
                length = int(self.headers.get("Content-Length", 0) or 0)
            except ValueError:
                length = 0
            raw = self.rfile.read(length) if length > 0 else b""
            try:
                body = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                log.info("webhook REJECTED from %s: invalid JSON", client_ip)
                self._reply(400, "invalid JSON")
                return

            payload, status, message = validate_payload(body, client_ip, deduper=deduper)
            if payload is None:
                if status >= 400:
                    log.info("webhook REJECTED from %s: %s (%d)", client_ip, message, status)
                else:
                    log.info("webhook IGNORED from %s: %s", client_ip, message)
                self._reply(status, message)
                return

            log.info("webhook ACCEPTED from %s: %s %s id=%s", client_ip,
                     payload["symbol"], payload["direction"], payload["id"])
            sig = enrich_signal(payload, signal_source)
            signal_queue.put(sig)
            self._reply(200, "accepted")

    return _Handler


def start_webhook_server(signal_queue, signal_source, deduper: Deduper | None = None):
    """Start the receiver in a daemon thread. Returns the HTTPServer (or None if
    disabled / misconfigured). Refuses to start an unauthenticated public endpoint."""
    if not config.WEBHOOK_ENABLED:
        log.info("Webhook receiver disabled (IOBOT_WEBHOOK_ENABLED=false)")
        return None
    if not config.WEBHOOK_SECRET:
        log.error("WEBHOOK_ENABLED but WEBHOOK_SECRET is empty — refusing to start "
                  "an unauthenticated public endpoint. Set IOBOT_WEBHOOK_SECRET.")
        return None

    deduper = deduper or Deduper(config.WEBHOOK_DEDUP_SEC)
    handler = make_handler(signal_queue, signal_source, deduper)
    httpd = HTTPServer((config.WEBHOOK_HOST, config.WEBHOOK_PORT), handler)
    thread = threading.Thread(target=httpd.serve_forever, name="webhook", daemon=True)
    thread.start()
    log.info("Webhook receiver listening on %s:%d  (POST /tv-webhook)  IP allowlist: %s",
             config.WEBHOOK_HOST, config.WEBHOOK_PORT,
             config.WEBHOOK_IP_ALLOWLIST or "ALL (open)")
    return httpd
