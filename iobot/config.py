"""config.py — single source of truth for the intraday options bot (iobot).

Every tunable lives here and is overridable via environment variables so the same
code runs in tests, paper, and (eventually) a separate go-live config without
edits. Paths are resolved relative to IOBOT_HOME (or the repo root) — never
hardcoded. Seeds are fixed for reproducible model training.

PAPER-ONLY: this build has no live-execution code. `GO_LIVE` exists as the single
reversible flag the spec calls for, but `broker.py` refuses to run anything but the
paper endpoint regardless — flipping it only records intent.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------- paths ----------------

_PKG_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(os.environ.get("IOBOT_HOME", _PKG_DIR.parent)).resolve()
load_dotenv(BASE_DIR / ".env")  # ALPACA_API_KEY / ALPACA_SECRET_KEY live here

DATA_DIR = Path(os.environ.get("IOBOT_DATA", BASE_DIR / "iobot_data")).resolve()
MODELS_DIR = DATA_DIR / "models"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DB_FILE = str(DATA_DIR / "iobot.db")              # features, trades, rejects, signals
GOVERNOR_STATE_FILE = str(DATA_DIR / "governor_state.json")
STATUS_FILE = str(DATA_DIR / "status.json")       # dashboard heartbeat
LOG_FILE = str(DATA_DIR / "iobot.log")
META_MODEL_FILE = str(MODELS_DIR / "meta_model.joblib")
META_REPORT_FILE = str(MODELS_DIR / "meta_report.json")


# ---------------- env helpers ----------------

def _b(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def _f(name: str, default: float) -> float:
    v = os.environ.get(name)
    return default if v is None else float(v)


def _i(name: str, default: int) -> int:
    v = os.environ.get(name)
    return default if v is None else int(v)


def _s(name: str, default: str) -> str:
    return os.environ.get(name, default)


# ---------------- connection / safety ----------------

ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
TZ = _s("IOBOT_TZ", "America/New_York")
LOG_LEVEL = _s("LOG_LEVEL", "INFO")
SEED = _i("IOBOT_SEED", 42)

# Single reversible paper->live flag (the spec's go-live gate). NO live execution
# is implemented in this build; broker.py hard-pins the paper endpoint.
GO_LIVE = _b("IOBOT_GO_LIVE", False)
ALLOW_LIVE_EXECUTION = False  # compile-time guard; never True in this build


# ---------------- universe / structure ----------------

UNIVERSE = [s.strip().upper() for s in _s("IOBOT_UNIVERSE", "SPY,QQQ").split(",") if s.strip()]

# "single" = slightly-ITM long option (default). "spread" = debit vertical upgrade.
STRUCTURE = _s("IOBOT_STRUCTURE", "single").lower()

# Active entry signal (a key in strategies.REGISTRY). trend_momentum = momentum
# gated to the daily-trend direction; the best backtested candidate (see
# iobot/strategies.py). Set IOBOT_SIGNAL=momentum to revert to the plain baseline.
SIGNAL = _s("IOBOT_SIGNAL", "trend_momentum")

# Slightly-ITM target delta band for the single long leg.
TARGET_DELTA_MIN = _f("IOBOT_DELTA_MIN", 0.60)
TARGET_DELTA_MAX = _f("IOBOT_DELTA_MAX", 0.70)

DTE_MIN = _i("IOBOT_DTE_MIN", 1)
DTE_MAX = _i("IOBOT_DTE_MAX", 5)
ALLOW_0DTE = _b("IOBOT_ALLOW_0DTE", False)        # gamma/theta knife-edge — off by default
QTY = _i("IOBOT_QTY", 1)                           # contracts per position (tiny account: 1)


# ---------------- debit-vertical upgrade path ----------------

SPREAD_ENABLED = _b("IOBOT_SPREAD_ENABLED", False)
# Only take spreads once equity can absorb the spread margin comfortably.
SPREAD_MIN_EQUITY = _f("IOBOT_SPREAD_MIN_EQUITY", 5000.0)
SPREAD_MARGIN_BUFFER = _f("IOBOT_SPREAD_MARGIN_BUFFER", 1000.0)   # cash to reserve per spread
SPREAD_WIDTH_STRIKES = [int(x) for x in _s("IOBOT_SPREAD_WIDTHS", "1,2").split(",")]
SPREAD_PROFIT_TARGET_PCT = _f("IOBOT_SPREAD_TP_PCT", 50.0)        # +% of net debit


# ---------------- entry signal (momentum) ----------------

MOMENTUM_PCT = _f("IOBOT_MOMENTUM_PCT", 0.30)      # min |move| of the trigger 15m candle
RSI_PERIOD = _i("IOBOT_RSI_PERIOD", 5)
RSI_CALL_MIN = _f("IOBOT_RSI_CALL_MIN", 60.0)
RSI_PUT_MAX = _f("IOBOT_RSI_PUT_MAX", 40.0)
REL_VOLUME_MIN = _f("IOBOT_REL_VOLUME_MIN", 1.2)
VWAP_FILTER = _b("IOBOT_VWAP_FILTER", True)        # don't fight the session VWAP
VOLUME_LOOKBACK_DAYS = _i("IOBOT_VOL_LOOKBACK", 20)


# ---------------- session windows (ET, 24h) ----------------

def _hm(name: str, default: tuple[int, int]) -> tuple[int, int]:
    v = os.environ.get(name)
    if not v:
        return default
    h, m = v.split(":")
    return int(h), int(m)


ENTRY_START = _hm("IOBOT_ENTRY_START", (9, 45))    # skip the opening 15 min
ENTRY_END = _hm("IOBOT_ENTRY_END", (15, 30))
FORCE_CLOSE = _hm("IOBOT_FORCE_CLOSE", (15, 50))   # intraday time-stop / EOD flat
SCAN_INTERVAL_SEC = _i("IOBOT_SCAN_INTERVAL", 30)


# ---------------- exits (underlying-based) ----------------

# Stop is a level on the UNDERLYING, expressed as a % move against the entry price.
STOP_UNDERLYING_PCT = _f("IOBOT_STOP_PCT", 0.40)
# Profit target as an R-multiple of the underlying risk distance (target = R * stop).
TARGET_R = _f("IOBOT_TARGET_R", 1.5)


# ---------------- liquidity filter ----------------

MAX_SPREAD_PCT = _f("IOBOT_MAX_SPREAD_PCT", 8.0)   # per-contract bid/ask as % of mid
MIN_OPEN_INTEREST = _i("IOBOT_MIN_OI", 250)
MIN_VOLUME = _i("IOBOT_MIN_VOLUME", 50)


# ---------------- risk governor / kill switches ----------------

RISK_PCT_PER_TRADE = _f("IOBOT_RISK_PCT", 2.0)     # max loss as % of equity (1-3)
DAILY_MAX_LOSS_PCT = _f("IOBOT_DAILY_MAX_LOSS_PCT", 5.0)
TRAILING_DD_PCT = _f("IOBOT_TRAILING_DD_PCT", 20.0)
MAX_TRADES_PER_DAY = _i("IOBOT_MAX_TRADES_DAY", 6)
MAX_CONCURRENT = _i("IOBOT_MAX_CONCURRENT", 1)
MAX_ROUND_TRIPS_PER_DAY = _i("IOBOT_MAX_ROUND_TRIPS", 6)
# Cash account: one full-capital round-trip/day on T+1 settled funds, no leverage.
CASH_ACCOUNT_MODE = _b("IOBOT_CASH_ACCOUNT", False)


# ---------------- order handling ----------------

ENTRY_SLIP = _f("IOBOT_ENTRY_SLIP", 0.03)          # limit padding over mid/ask
ORDER_FILL_TIMEOUT = _i("IOBOT_FILL_TIMEOUT", 20)  # seconds


# ---------------- validation gate ----------------

GATE_MIN_TRADES = _i("IOBOT_GATE_MIN_TRADES", 40)
# Modeled per-leg friction used when scoring expected R net of costs.
MODELED_FEE_PER_CONTRACT = _f("IOBOT_FEE_PER_CONTRACT", 0.65)
MODELED_SLIP_PER_LEG = _f("IOBOT_SLIP_PER_LEG", 0.02)


# ---------------- TradingView webhook receiver ----------------

# Optional inbound alert endpoint (iobot/webhook.py). When enabled a stdlib
# http.server runs in a daemon thread; valid alerts are enqueued and run through
# the SAME _handle_signal pipeline as scanner signals (governor + meta + selection
# + risk) — no parallel trading path. Boot-time / SECURITY settings: the secret
# and IP allowlist come only from the environment, never a settings file.
WEBHOOK_ENABLED = _b("IOBOT_WEBHOOK_ENABLED", False)
WEBHOOK_HOST = _s("IOBOT_WEBHOOK_HOST", "0.0.0.0")
WEBHOOK_PORT = _i("IOBOT_WEBHOOK_PORT", 8080)
WEBHOOK_SECRET = _s("IOBOT_WEBHOOK_SECRET", "")    # required if enabled; from .env
# Comma-separated IPs (empty = allow all). TradingView's published webhook IPs are
# 52.89.214.238,34.212.75.30,54.218.53.128,52.32.178.7 (see webhook.TRADINGVIEW_IPS).
WEBHOOK_IP_ALLOWLIST = [ip.strip() for ip in
                        _s("IOBOT_WEBHOOK_IP_ALLOWLIST", "").split(",") if ip.strip()]
WEBHOOK_DEDUP_SEC = _f("IOBOT_WEBHOOK_DEDUP_SEC", 60.0)  # ignore repeat ids within


# ---------------- meta-labeling layer ----------------

META_MIN_TRADES = _i("IOBOT_META_MIN_TRADES", 40)
META_AUC_BAR = _f("IOBOT_META_AUC_BAR", 0.55)
META_PROBA_THRESHOLD = _f("IOBOT_META_THRESHOLD", 0.50)
META_CV_SPLITS = _i("IOBOT_META_CV_SPLITS", 5)
META_EMBARGO_FRAC = _f("IOBOT_META_EMBARGO", 0.02)   # purged+embargoed walk-forward
# Phase-3 single reversible flag. Even when True the layer only acts if a passing
# model exists AND the trade count / shadow-R preconditions are met (see meta.py).
META_ACTIVE = _b("IOBOT_META_ACTIVE", False)
