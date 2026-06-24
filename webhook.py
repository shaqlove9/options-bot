"""webhook.py — TradingView alert receiver feeding the EXISTING entry pipeline.

A lightweight stdlib ``http.server`` runs in a daemon thread (started from
``main.py`` only when ``config.WEBHOOK_ENABLED``). For each incoming alert it:

  1. authenticates (shared secret, constant-time compare; optional IP allowlist),
  2. de-dups repeats within ``WEBHOOK_DEDUP_SEC`` (TradingView can fire twice),
  3. enriches the alert into a ``scanner.Signal`` using the live data client
     (rsi / rel_volume / vwap_dist_pct / spot — exactly like ``Scanner._check_symbol``),
  4. pushes that Signal onto a thread-safe ``queue.Queue``.

The MAIN loop drains the queue and runs each Signal through the *identical*
``main.try_enter`` path as scanner signals, so all risk controls, ML gating,
cooldowns and budget logic apply exactly once, in one place. **The server thread
never touches executor / risk / learner** (those are not thread-safe) — it only
reads market data and enqueues.

POST /tv-webhook  (Content-Type: application/json)
--------------------------------------------------
    {
      "secret":    "<must equal config.WEBHOOK_SECRET>",   # REQUIRED -> 401 if wrong
      "id":        "<unique alert id>",                     # optional -> 60s dedup key
      "symbol":    "SPY",                                   # REQUIRED
      "direction": "call" | "put",                          # REQUIRED
      "strategy":  "scalp" | "runner",                      # optional (default "scalp")
      "price":     612.34                                   # optional (advisory only)
    }

Only ``symbol`` and ``direction`` are required. If live data can't be fetched at
enrichment time the Signal is still enqueued but flagged ``advisory`` so the ML
gate is skipped (advisory entry) rather than crashing.
"""
from __future__ import annotations

import hmac
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import alerts
import config
from scanner import Signal, rsi_last
from utils import now_et

log = logging.getLogger("webhook")

_VALID_DIRECTIONS = {"call", "put"}
_VALID_STRATEGIES = {"scalp", "runner"}

# TradingView's published outbound webhook IPs — set config.WEBHOOK_IP_ALLOWLIST
# to these (via the WEBHOOK_IP_ALLOWLIST env var) to lock the endpoint down.
TRADINGVIEW_IPS = ("52.89.214.238", "34.212.75.30", "54.218.53.128", "52.32.178.7")


# ---------------------------------------------------------------------------
# de-duplication
# ---------------------------------------------------------------------------

class Deduper:
    """In-memory TTL set of recently seen alert ids. Thread-safe so it is robust
    even if the server is later switched to a threading variant."""

    def __init__(self, ttl_sec: float):
        self.ttl = ttl_sec
        self._seen: dict[str, float] = {}
        self._lock = threading.Lock()

    def seen_recently(self, alert_id: str | None, now: float | None = None) -> bool:
        """True if ``alert_id`` was already seen within the TTL. Records it as a
        side effect when it is new. A missing id never de-dups (returns False)."""
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


# ---------------------------------------------------------------------------
# validation (pure — no I/O, unit-testable)
# ---------------------------------------------------------------------------

def validate_payload(body, client_ip: str, *, deduper: Deduper):
    """Authenticate + validate a decoded JSON body.

    Returns ``(payload | None, http_status, message)``. ``payload`` is a
    normalized dict (symbol upper-cased, strategy defaulted) ready for
    enrichment, or ``None`` when the request must not become a trade.
    """
    if config.WEBHOOK_IP_ALLOWLIST and client_ip not in config.WEBHOOK_IP_ALLOWLIST:
        return None, 403, f"ip {client_ip} not in allowlist"
    if not isinstance(body, dict):
        return None, 400, "body is not a JSON object"

    # Secret: reject if unset (never run open) or mismatched. Constant-time.
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
    strategy = body.get("strategy", "scalp")
    if strategy not in _VALID_STRATEGIES:
        return None, 400, f"invalid 'strategy' (scalp|runner): {strategy!r}"

    alert_id = body.get("id")
    if deduper.seen_recently(str(alert_id) if alert_id is not None else None):
        return None, 200, f"duplicate id {alert_id} ignored"

    price = body.get("price")
    try:
        price = float(price) if price is not None else None
    except (TypeError, ValueError):
        price = None
    return ({"symbol": symbol.upper(), "direction": direction,
             "strategy": strategy, "price": price, "id": alert_id},
            200, "accepted")


# ---------------------------------------------------------------------------
# enrichment (payload -> scanner.Signal)
# ---------------------------------------------------------------------------

def enrich_signal(payload: dict, scanner) -> Signal:
    """Build a ``scanner.Signal`` from a validated payload, filling
    rsi / rel_volume / vwap_dist_pct / spot from live bars (reusing the Scanner's
    own data client + helpers, exactly like ``_check_symbol``). On ANY data
    failure the Signal is returned with numeric features defaulted to 0 and
    ``advisory=True`` so ``try_enter`` skips ML gating rather than crashing.

    The returned Signal carries two extra attributes (set dynamically so
    ``scanner.py`` is untouched): ``advisory`` (bool) and ``source`` ("tradingview").
    """
    symbol = payload["symbol"]
    now = now_et()
    sig = Signal(
        symbol=symbol,
        direction=payload["direction"],
        strategy=payload["strategy"],
        momentum_pct=0.0,
        day_change_pct=0.0,
        rsi=0.0,
        rel_volume=0.0,
        spot=float(payload["price"]) if payload.get("price") else 0.0,
        vwap_dist_pct=0.0,
        time=now,
    )
    sig.source = "tradingview"
    sig.advisory = False
    try:
        intraday = scanner._intraday_bars(symbol)
        if intraday is None or len(intraday) < config.RSI_PERIOD + 2:
            raise ValueError("insufficient intraday bars")
        candle = intraday.iloc[-1]
        if candle["open"] <= 0:
            raise ValueError("bad last candle")
        spot = float(candle["close"])
        today_bars = intraday[intraday.index.date == now.date()]
        if today_bars.empty or today_bars["open"].iloc[0] <= 0:
            raise ValueError("no session bars yet")
        vwap = scanner._session_vwap(intraday)
        sig.spot = spot
        sig.momentum_pct = (candle["close"] - candle["open"]) / candle["open"] * 100
        sig.day_change_pct = (spot - today_bars["open"].iloc[0]) / today_bars["open"].iloc[0] * 100
        sig.rsi = rsi_last(intraday["close"], config.RSI_PERIOD)
        sig.rel_volume = scanner._relative_volume(symbol, intraday)
        sig.vwap_dist_pct = (spot - vwap) / vwap * 100 if vwap else 0.0
    except Exception as exc:
        sig.advisory = True
        log.warning("webhook %s: live enrichment failed (%s) — ADVISORY only "
                    "(ML gating will be skipped)", symbol, exc)
    return sig


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

def make_handler(signal_queue, scanner, deduper: Deduper):
    """Build a request handler bound to a queue, scanner and deduper."""

    class _Handler(BaseHTTPRequestHandler):
        server_version = "iobot-webhook/1.0"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):       # silence default stderr spam
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
            # Unauthenticated liveness probe only — exposes nothing.
            if self.path in ("/health", "/healthz"):
                self._reply(200, "ok")
            else:
                self._reply(404, "not found")

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

            log.info("webhook ACCEPTED from %s: %s %s [%s] id=%s", client_ip,
                     payload["symbol"], payload["direction"], payload["strategy"],
                     payload["id"])
            sig = enrich_signal(payload, scanner)
            try:
                alerts.webhook_received(sig)
            except Exception:
                log.exception("webhook 'received' alert failed (non-fatal)")
            signal_queue.put(sig)          # main thread will run it through try_enter
            self._reply(200, "accepted")

    return _Handler


def start_webhook_server(signal_queue, scanner, deduper: Deduper | None = None):
    """Start the receiver in a daemon thread. Returns the HTTPServer (or None if
    disabled / misconfigured). Refuses to start an unauthenticated public endpoint."""
    if not config.WEBHOOK_ENABLED:
        log.info("Webhook receiver disabled (WEBHOOK_ENABLED=false)")
        return None
    if not config.WEBHOOK_SECRET:
        log.error("WEBHOOK_ENABLED but WEBHOOK_SECRET is empty — refusing to start "
                  "an unauthenticated public endpoint. Set WEBHOOK_SECRET in .env.")
        return None

    deduper = deduper or Deduper(config.WEBHOOK_DEDUP_SEC)
    handler = make_handler(signal_queue, scanner, deduper)
    try:
        httpd = HTTPServer((config.WEBHOOK_HOST, config.WEBHOOK_PORT), handler)
    except OSError as exc:
        # Never let a webhook bind failure take down the trading loop.
        log.error("Webhook receiver could not bind %s:%d (%s) — continuing WITHOUT it",
                  config.WEBHOOK_HOST, config.WEBHOOK_PORT, exc)
        return None
    thread = threading.Thread(target=httpd.serve_forever, name="webhook", daemon=True)
    thread.start()
    log.info("Webhook receiver listening on %s:%d  (POST /tv-webhook)  "
             "IP allowlist: %s", config.WEBHOOK_HOST, config.WEBHOOK_PORT,
             config.WEBHOOK_IP_ALLOWLIST or "ALL (open)")
    return httpd
