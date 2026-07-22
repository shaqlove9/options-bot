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

# Liquid, mid-priced, optionable names where a slightly-ITM contract can fit under
# the $300/position cap. The ETFs (SPY/QQQ/XLF) give VWAP-clean scalps + regime
# context; the rest are high-beta sub-$60 movers with deep weekly chains. The
# liquidity filter + the $300 affordability cap auto-reject anything unsuitable, so
# this list is a generous candidate pool, tunable via one env var.
UNIVERSE = [s.strip().upper() for s in _s(
    "IOBOT_UNIVERSE",
    "SPY,QQQ,AMD,AAPL,F,SOFI,PLTR,INTC,BAC,NIO,CCL,T,GOLD,XLF",
).split(",") if s.strip()]

# "single" = slightly-ITM long option (default). "spread" = debit vertical upgrade.
STRUCTURE = _s("IOBOT_STRUCTURE", "single").lower()

# Active entry signal (a key in strategies.REGISTRY). `confluence` = the senior-trader
# multi-factor TA signal (trend + momentum + structure + volume; see
# iobot/strategies.py). Set IOBOT_SIGNAL=trend_momentum / momentum to revert.
SIGNAL = _s("IOBOT_SIGNAL", "confluence")

# Target delta band for the single long leg. 0.50–0.70 spans near-ATM (more liquid,
# cheaper premium) through slightly-ITM (less theta) — widened from 0.60–0.70 so the
# cheaper, coarser-strike underlyings actually have a qualifying strike.
TARGET_DELTA_MIN = _f("IOBOT_DELTA_MIN", 0.50)
TARGET_DELTA_MAX = _f("IOBOT_DELTA_MAX", 0.70)

DTE_MIN = _i("IOBOT_DTE_MIN", 1)
DTE_MAX = _i("IOBOT_DTE_MAX", 5)
ALLOW_0DTE = _b("IOBOT_ALLOW_0DTE", False)        # gamma/theta knife-edge — off by default
QTY = _i("IOBOT_QTY", 1)                           # base lots (spread path; singles size by $)


# ---------------- sleeve capital / position sizing ----------------

# The bot manages a fixed-size SLEEVE, not the whole (paper) account: all governor
# risk math (per-trade cap, daily-loss kill, trailing DD) is a % of sleeve equity,
# where sleeve equity = SLEEVE_CAPITAL + the sleeve's own realized P&L. Buying-power
# checks still use the real account (paper BP is ample).
SLEEVE_CAPITAL = _f("IOBOT_SLEEVE_CAPITAL", 1000.0)
# Max dollars deployed per single-leg position. Single legs size up to this many
# whole contracts (>=1); a contract whose one-lot cost exceeds it is rejected. This
# is the "max contract price $300" cap.
POSITION_MAX_DOLLARS = _f("IOBOT_POSITION_MAX_DOLLARS", 300.0)


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


# ---------------- confluence signal (senior-trader multi-factor TA) ----------------

# The `confluence` signal scores aligned technical factors and fires only when at
# least CONFLUENCE_MIN of them agree on a direction:
#   trend  : fast EMA over slow EMA + price the right side of session VWAP
#   regime : daily close vs its SMA (trade with the higher-timeframe trend)
#   momentum: MACD histogram in the trade direction + RSI not exhausted
#   structure: breaks the recent intraday swing high/low (continuation)
#   volume : relative volume >= REL_VOLUME_MIN
CONFLUENCE_MIN = _i("IOBOT_CONFLUENCE_MIN", 4)     # of 5 factors
EMA_FAST = _i("IOBOT_EMA_FAST", 9)
EMA_SLOW = _i("IOBOT_EMA_SLOW", 21)
MACD_FAST = _i("IOBOT_MACD_FAST", 12)
MACD_SLOW = _i("IOBOT_MACD_SLOW", 26)
MACD_SIGNAL = _i("IOBOT_MACD_SIGNAL", 9)
CONFLUENCE_SWING_BARS = _i("IOBOT_CONFLUENCE_SWING_BARS", 8)   # intraday breakout lookback
CONFLUENCE_DAILY_SMA = _i("IOBOT_CONFLUENCE_DAILY_SMA", 20)
RSI_OVERBOUGHT = _f("IOBOT_RSI_OVERBOUGHT", 80.0)  # don't chase calls above this
RSI_OVERSOLD = _f("IOBOT_RSI_OVERSOLD", 20.0)      # don't chase puts below this
# ATR-based exits for confluence (carried per-signal as stop_level/target_level):
# stop = ATR_STOP_MULT * ATR from entry; target = ATR_TARGET_R * that risk distance.
ATR_STOP_MULT = _f("IOBOT_ATR_STOP_MULT", 1.2)
ATR_TARGET_R = _f("IOBOT_ATR_TARGET_R", 1.8)

# The `orb_confluence` signal is confluence gated by opening-range-breakout structure
# and a "stocks in play" relative-volume bar (Zarattini/Aziz): the underlying must
# have broken beyond its first-ORB_MINUTES range in the trade direction, on relative
# volume >= ORBCONF_RELVOL_MIN. The higher volume bar exists because the plain
# REL_VOLUME_MIN floor kept letting in quiet-tape chop (live book 2026-07-22:
# winners averaged 1.54x relvol, losers 1.24x).
ORBCONF_RELVOL_MIN = _f("IOBOT_ORBCONF_RELVOL_MIN", 1.5)


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
# Midday no-entry window (ET, half-open [start, end)): block fresh entries during the
# low-conviction lunch chop where the live book bled. Applies to the live engine AND
# the backtest (both gate through clock.in_entry_window). Disable by setting
# IOBOT_NO_ENTRY_START == IOBOT_NO_ENTRY_END (e.g. both 00:00).
NO_ENTRY_START = _hm("IOBOT_NO_ENTRY_START", (11, 0))
NO_ENTRY_END = _hm("IOBOT_NO_ENTRY_END", (13, 0))
SCAN_INTERVAL_SEC = _i("IOBOT_SCAN_INTERVAL", 30)


# ---------------- exits (underlying-based) ----------------

# Stop is a level on the UNDERLYING, expressed as a % move against the entry price.
STOP_UNDERLYING_PCT = _f("IOBOT_STOP_PCT", 0.40)
# Profit target as an R-multiple of the underlying risk distance (target = R * stop).
TARGET_R = _f("IOBOT_TARGET_R", 1.5)
# ORB uses the breakout strategy's own exit (Zarattini/Aziz): stop at the OPPOSITE end
# of the opening range, target a large R-multiple of that adaptive risk distance, else
# flat at EOD. Attached per-signal in strategies.orb (overrides the fixed-% exit above
# for ORB only; other signals keep STOP_UNDERLYING_PCT / TARGET_R).
ORB_TARGET_R = _f("IOBOT_ORB_TARGET_R", 4.0)


# ---------------- overnight holds ----------------

# When True, positions whose thesis is intact (stop not hit) are NOT force-flattened
# at FORCE_CLOSE — they carry overnight. A position is still force-closed at EOD if it
# expires within OVERNIGHT_MIN_DTE_KEEP sessions (avoid expiry/gamma risk) or it has
# been held MAX_HOLD_DAYS calendar days. Stops/targets keep running every tick across
# days, so a broken thesis still exits intraday. Overnight holds REQUIRE position
# persistence (store.positions) so a restart never orphans a live contract.
ALLOW_OVERNIGHT = _b("IOBOT_ALLOW_OVERNIGHT", True)
OVERNIGHT_MIN_DTE_KEEP = _i("IOBOT_OVERNIGHT_MIN_DTE", 1)   # flatten if <= this many DTE
MAX_HOLD_DAYS = _i("IOBOT_MAX_HOLD_DAYS", 5)               # hard time-stop on a carry


# ---------------- liquidity filter ----------------

MAX_SPREAD_PCT = _f("IOBOT_MAX_SPREAD_PCT", 12.0)  # per-contract bid/ask as % of mid
MIN_OPEN_INTEREST = _i("IOBOT_MIN_OI", 100)
# Intraday option volume is a noisy gate (0 early in the day even for liquid names);
# OI + the spread cap are the real liquidity signals, so the volume floor is off.
MIN_VOLUME = _i("IOBOT_MIN_VOLUME", 0)


# ---------------- risk governor / kill switches ----------------

# Aggressive by design (user-chosen): a single $300 position is ~30% of the $1,000
# sleeve, so the per-trade cap is 30%. Kill-switches still bound the downside.
RISK_PCT_PER_TRADE = _f("IOBOT_RISK_PCT", 30.0)    # max loss as % of SLEEVE equity
DAILY_MAX_LOSS_PCT = _f("IOBOT_DAILY_MAX_LOSS_PCT", 25.0)
TRAILING_DD_PCT = _f("IOBOT_TRAILING_DD_PCT", 40.0)
MAX_TRADES_PER_DAY = _i("IOBOT_MAX_TRADES_DAY", 10)
MAX_CONCURRENT = _i("IOBOT_MAX_CONCURRENT", 2)
MAX_ROUND_TRIPS_PER_DAY = _i("IOBOT_MAX_ROUND_TRIPS", 10)
# Post-close re-entry cooldown (minutes): after a position in a given symbol+direction
# CLOSES, block a fresh entry in that same symbol+direction for this many minutes. Kills
# the same-name churn where a just-closed winner/loser is immediately re-traded minutes
# later (the second fire reliably underperformed the first). 0 disables. Note: concurrent
# same-symbol holds are already blocked separately by executor.has_position_in.
SYMBOL_COOLDOWN_MIN = _i("IOBOT_SYMBOL_COOLDOWN_MIN", 0)
# Cash account: one full-capital round-trip/day on T+1 settled funds, no leverage.
CASH_ACCOUNT_MODE = _b("IOBOT_CASH_ACCOUNT", False)


# ---------------- order handling ----------------

ENTRY_SLIP = _f("IOBOT_ENTRY_SLIP", 0.03)          # limit padding over mid/ask
ORDER_FILL_TIMEOUT = _i("IOBOT_FILL_TIMEOUT", 20)  # seconds
# On restart, flatten broker option legs we don't track (orphans from stale fills). An
# untracked position can't be managed/exited by the bot, so it's pure unmanaged risk.
RECONCILE_FLATTEN_ORPHANS = _b("IOBOT_RECONCILE_FLATTEN_ORPHANS", True)


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


# ---------------- notifications (Discord) ----------------

# Optional Discord webhook for entry/exit/halt notifications (iobot/notify.py).
# Fire-and-forget from a daemon thread; a Discord outage never blocks trading.
# No-op (disabled) when unset. Shares the env var name with the legacy alerts.
DISCORD_WEBHOOK_URL = _s("DISCORD_WEBHOOK_URL", "")


# ---------------- meta-labeling layer ----------------

META_MIN_TRADES = _i("IOBOT_META_MIN_TRADES", 40)
META_AUC_BAR = _f("IOBOT_META_AUC_BAR", 0.55)
META_PROBA_THRESHOLD = _f("IOBOT_META_THRESHOLD", 0.50)
META_CV_SPLITS = _i("IOBOT_META_CV_SPLITS", 5)
META_EMBARGO_FRAC = _f("IOBOT_META_EMBARGO", 0.02)   # purged+embargoed walk-forward
# Phase-3 single reversible flag. Even when True the layer only acts if a passing
# model exists AND the trade count / shadow-R preconditions are met (see meta.py).
META_ACTIVE = _b("IOBOT_META_ACTIVE", False)
