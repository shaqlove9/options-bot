"""Smoke test for webhook.py + main.try_enter — no network, no broker.

Covers the four things that matter for a public, additive entry path:
  1. secret / IP rejection (auth)
  2. dedup of repeated alert ids within the TTL
  3. payload -> scanner.Signal enrichment mapping (incl. advisory fallback)
  4. an ENQUEUED signal flows through the SAME try_enter() pipeline as scanner
     signals, with every risk guard intact (window, halt, cooldown, existing
     position, earnings, budget, ML gate)

Run:  .venv/bin/python test_webhook.py
(Pure fakes — never touches Alpaca, trades.csv, or the model.)
"""
import datetime as dt
import queue as _queue

import pandas as pd

import config

# Webhook auth config (validate_payload reads these at call time).
config.WEBHOOK_SECRET = "s3cret"
config.WEBHOOK_IP_ALLOWLIST = []
config.WEBHOOK_DEDUP_SEC = 60

import earnings                       # noqa: E402  (patched below)
import main                           # noqa: E402
import webhook                        # noqa: E402
from scanner import Signal            # noqa: E402
from utils import ET, now_et          # noqa: E402

OK_IP = "52.89.214.238"


# ===========================================================================
# 1. AUTH — secret + IP allowlist
# ===========================================================================

ded = webhook.Deduper(60)

p, s, _ = webhook.validate_payload(
    {"secret": "wrong", "symbol": "SPY", "direction": "call"}, OK_IP, deduper=ded)
assert p is None and s == 401, f"FAIL: wrong secret should be 401, got {s}"

p, s, _ = webhook.validate_payload(
    {"symbol": "SPY", "direction": "call"}, OK_IP, deduper=ded)
assert p is None and s == 401, f"FAIL: missing secret should be 401, got {s}"

p, s, _ = webhook.validate_payload(
    {"secret": "s3cret", "symbol": "spy", "direction": "call"}, OK_IP, deduper=ded)
assert p is not None and s == 200, f"FAIL: good secret should be accepted, got {s}"
assert p["symbol"] == "SPY", "FAIL: symbol should be upper-cased"
assert p["strategy"] == "scalp", "FAIL: strategy should default to scalp"

# bad direction / strategy
p, s, _ = webhook.validate_payload(
    {"secret": "s3cret", "symbol": "SPY", "direction": "buy"}, OK_IP, deduper=ded)
assert p is None and s == 400, "FAIL: invalid direction should be 400"
p, s, _ = webhook.validate_payload(
    {"secret": "s3cret", "symbol": "SPY", "direction": "put", "strategy": "x"}, OK_IP, deduper=ded)
assert p is None and s == 400, "FAIL: invalid strategy should be 400"

# IP allowlist
config.WEBHOOK_IP_ALLOWLIST = ["9.9.9.9"]
p, s, _ = webhook.validate_payload(
    {"secret": "s3cret", "symbol": "SPY", "direction": "call"}, "1.2.3.4", deduper=ded)
assert p is None and s == 403, f"FAIL: off-allowlist IP should be 403, got {s}"
p, s, _ = webhook.validate_payload(
    {"secret": "s3cret", "symbol": "SPY", "direction": "call"}, "9.9.9.9", deduper=ded)
assert p is not None, "FAIL: allowlisted IP should pass"
config.WEBHOOK_IP_ALLOWLIST = []
print("1. auth (secret + IP allowlist) OK")


# ===========================================================================
# 2. DEDUP
# ===========================================================================

ded2 = webhook.Deduper(60)
base = {"secret": "s3cret", "symbol": "SPY", "direction": "call", "id": "abc"}
p1, _, _ = webhook.validate_payload(dict(base), OK_IP, deduper=ded2)
assert p1 is not None, "FAIL: first alert should be accepted"
p2, s2, m2 = webhook.validate_payload(dict(base), OK_IP, deduper=ded2)
assert p2 is None and s2 == 200 and "duplicate" in m2, "FAIL: repeat id should be ignored"
p3, _, _ = webhook.validate_payload({**base, "id": "def"}, OK_IP, deduper=ded2)
assert p3 is not None, "FAIL: a different id should pass"

# TTL semantics, driven by an injected clock
d = webhook.Deduper(ttl_sec=10)
assert d.seen_recently("x", now=100.0) is False, "FAIL: first sighting is not a dup"
assert d.seen_recently("x", now=105.0) is True, "FAIL: within TTL should be a dup"
assert d.seen_recently("x", now=120.0) is False, "FAIL: past TTL should expire"
assert d.seen_recently(None) is False, "FAIL: missing id must never dedup"
print("2. dedup OK")


# ===========================================================================
# 3. ENRICHMENT — payload -> Signal mapping
# ===========================================================================

# advisory fallback: data client can't provide bars -> still a Signal, flagged
class _FailScanner:
    def _intraday_bars(self, symbol):
        return None

sig = webhook.enrich_signal(
    {"symbol": "QQQ", "direction": "put", "strategy": "runner", "price": 500.0},
    _FailScanner())
assert isinstance(sig, Signal), "FAIL: enrich must return a scanner.Signal"
assert (sig.symbol, sig.direction, sig.strategy) == ("QQQ", "put", "runner"), "FAIL: mapping"
assert sig.advisory is True, "FAIL: missing data -> advisory"
assert sig.source == "tradingview", "FAIL: source tag missing"
assert sig.spot == 500.0, "FAIL: price should seed spot"
assert sig.rsi == 0.0 and sig.rel_volume == 0.0, "FAIL: features default to 0 in advisory"

# happy path: a fake scanner returns real bars; enrichment fills features
_today = now_et().date()
_times = [dt.datetime.combine(_today, dt.time(10, 0)) + dt.timedelta(minutes=15 * i)
          for i in range(10)]
_idx = pd.DatetimeIndex(_times).tz_localize(ET)
_df = pd.DataFrame({
    "open":  [100, 101, 102, 101, 103, 104, 103, 105, 106, 107],
    "high":  [101, 102, 103, 102, 104, 105, 104, 106, 107, 108.5],
    "low":   [99, 100, 101, 100, 102, 103, 102, 104, 105, 106],
    "close": [101, 102, 101, 103, 104, 103, 105, 106, 107, 108],
    "volume": [1000] * 10,
    "vwap":  [100.5] * 10,
}, index=_idx)

class _GoodScanner:
    def _intraday_bars(self, symbol):
        return _df
    def _relative_volume(self, symbol, intraday):
        return 2.3
    def _session_vwap(self, intraday):
        return 99.5

sig = webhook.enrich_signal(
    {"symbol": "SPY", "direction": "call", "strategy": "scalp", "price": None},
    _GoodScanner())
assert sig.advisory is False, "FAIL: good data -> not advisory"
assert sig.spot == 108.0, f"FAIL: spot should be last close, got {sig.spot}"
assert sig.rel_volume == 2.3, "FAIL: rel_volume not wired from scanner"
assert sig.rsi != 0.0, "FAIL: rsi should be computed"
assert abs(sig.vwap_dist_pct - (108.0 - 99.5) / 99.5 * 100) < 1e-6, "FAIL: vwap_dist"
print("3. enrichment mapping (advisory + live) OK")


# ===========================================================================
# 4. PIPELINE — enqueued Signal flows through try_enter with guards intact
# ===========================================================================

# Neutralize the time-of-day / earnings guards deterministically.
main.in_entry_window = lambda t: True
earnings.blocks = lambda s: False


class _FakePick:
    underlying = "SPY"
    option_symbol = "SPY260101C00500000"
    otype = "call"
    strike = 500.0
    iv = 0.3
    spread = 0.05
    cost = 40.0
    expiry = dt.date.today() + dt.timedelta(days=3)


class _FakeChain:
    def __init__(self):
        self.calls = 0
    def find_contract(self, signal):
        self.calls += 1
        return _FakePick()


class _State:
    halted = False


class _FakeRisk:
    def __init__(self):
        self.state = _State()
        self.allow = (True, "ok")
    def can_enter(self, open_positions, trade_cost):
        return self.allow


class _FakeExecutor:
    def __init__(self):
        self.positions = set()
        self.opened = []
    def has_position_in(self, underlying):
        return underlying in self.positions
    def open_count(self):
        return len(self.positions)
    def open_position(self, pick, reason, features, win_prob, tp_pct=None):
        self.opened.append({"pick": pick, "reason": reason,
                            "win_prob": win_prob, "tp": tp_pct})
        self.positions.add(pick.underlying)
        return object()                  # truthy Position


class _FakeLearner:
    def __init__(self, allow=True, prob=0.9):
        self.allow, self.prob, self.calls = allow, prob, 0
    def allows(self, features):
        self.calls += 1
        return self.allow, self.prob


def make_signal(symbol="SPY", strategy="scalp", advisory=False):
    s = Signal(symbol=symbol, direction="call", strategy=strategy, momentum_pct=0.7,
               day_change_pct=1.2, rsi=70.0, rel_volume=2.5, spot=500.0,
               vwap_dist_pct=0.3, time=now_et())
    s.advisory = advisory
    s.source = "tradingview"
    return s


def ctx(**learner_kw):
    return dict(chain=_FakeChain(), risk=_FakeRisk(), executor=_FakeExecutor(),
                learner=_FakeLearner(**learner_kw), cooldowns={}, now=now_et())


# --- happy path, via a real queue (the exact drain the main loop performs) ---
q = _queue.Queue()
q.put(make_signal())
c = ctx()
queued = q.get_nowait()
assert main.try_enter(queued, **c) is True, "FAIL: clean signal should enter"
assert len(c["executor"].opened) == 1, "FAIL: open_position not called"
assert "SPY" in c["executor"].positions, "FAIL: position not tracked"
assert c["cooldowns"]["SPY"] == c["now"], "FAIL: cooldown not recorded"
assert c["learner"].calls == 1, "FAIL: ML gate not consulted for live signal"

# --- guard: daily halt ---
c = ctx(); c["risk"].state.halted = True
assert main.try_enter(make_signal(), **c) is False and not c["executor"].opened, \
    "FAIL: halt must block entry"

# --- guard: symbol cooldown ---
c = ctx(); c["cooldowns"]["SPY"] = c["now"]
assert main.try_enter(make_signal(), **c) is False, "FAIL: cooldown must block"

# --- guard: already holding the underlying ---
c = ctx(); c["executor"].positions.add("SPY")
assert main.try_enter(make_signal(), **c) is False, "FAIL: existing position must block"

# --- guard: earnings block ---
earnings.blocks = lambda s: True
c = ctx()
assert main.try_enter(make_signal(), **c) is False, "FAIL: earnings must block"
earnings.blocks = lambda s: False

# --- guard: risk budget ---
c = ctx(); c["risk"].allow = (False, "budget")
assert main.try_enter(make_signal(), **c) is False and not c["executor"].opened, \
    "FAIL: budget rejection must block"

# --- guard: ML veto on a live (non-advisory) signal ---
c = ctx(allow=False, prob=0.10)
assert main.try_enter(make_signal(), **c) is False and not c["executor"].opened, \
    "FAIL: ML veto must block a live signal"

# --- advisory signal skips ML and still enters (features unreliable) ---
c = ctx(allow=False, prob=0.10)               # learner WOULD veto...
assert main.try_enter(make_signal(advisory=True), **c) is True, \
    "FAIL: advisory signal should enter despite would-be veto"
assert c["learner"].calls == 0, "FAIL: advisory must NOT consult the ML gate"
assert c["executor"].opened[0]["win_prob"] is None, "FAIL: advisory win_prob must be None"

# --- guard: outside entry window ---
main.in_entry_window = lambda t: False
c = ctx()
assert main.try_enter(make_signal(), **c) is False, "FAIL: out-of-window must block"
main.in_entry_window = lambda t: True

# --- runner take-profit is honored through the shared path ---
c = ctx()
main.try_enter(make_signal(strategy="runner"), **c)
assert c["executor"].opened[0]["tp"] == config.RUNNER_TAKE_PROFIT_PCT, \
    "FAIL: runner take-profit not applied"
print("4. try_enter pipeline + all risk guards intact OK")

print("\nALL WEBHOOK TESTS PASSED")
