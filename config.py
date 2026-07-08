"""Central configuration. All secrets come from .env — never hardcode keys."""
import os

from dotenv import load_dotenv

load_dotenv()

# --- Mode ---
LIVE_MODE = os.getenv("LIVE_MODE", "false").lower() == "true"
PAPER = not LIVE_MODE

# --- Alpaca credentials ---
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")

# --- Alerts (Discord or Slack) ---
# ALERT_BACKEND: "discord", "slack", or "auto" (auto-detect from which URL is set)
ALERT_BACKEND = os.getenv("ALERT_BACKEND", "auto").lower()
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")

# --- Account / risk ---
CAPITAL = 500.00
MAX_TRADE_COST = 50.00          # max premium per trade (ask * 100 * qty)
MAX_CONTRACTS = 1               # contracts per trade
MAX_OPEN_POSITIONS = 3
MAX_DAILY_LOSS = 75.00          # halt all trading for the day if breached
MAX_DAILY_TRADES = None         # None = unlimited; set a number (e.g. 10) before going live
CONSEC_LOSS_PAUSE = 2           # consecutive losses that trigger a pause (ignored in paper)
PAUSE_MINUTES = 30

# --- Universe ---
# NOTE: SPX (CBOE index options) is NOT tradable on Alpaca. SPY/QQQ provide
# the index exposure instead.
UNIVERSE = ["SPY", "QQQ", "NVDA", "TSLA", "AAPL", "AMZN"]

# --- Dynamic universe (screener-discovered symbols) ---
DYNAMIC_UNIVERSE = False                        # default OFF; enable via dashboard
DYNAMIC_UNIVERSE_MAX_SYMBOLS = 15               # cap total universe size
DYNAMIC_UNIVERSE_REFRESH_MIN = 30               # refresh every N minutes
DYNAMIC_UNIVERSE_CORE = ["SPY", "QQQ"]          # always included regardless of screener

# --- Strategy 2: momentum runner (trend continuation) ---
# Catches a stock that's been running all day (not just one big candle) and
# is still pushing to new session highs/lows. Wider profit target; the
# trailing stop manages the ride. All risk limits still apply.
RUNNER_ENABLED = True
RUNNER_DAY_PCT = 2.0            # min % move from today's open to qualify
RUNNER_RSI_MIN = 60             # RSI confirm for upside continuation
RUNNER_RSI_MAX = 40             # RSI confirm for downside continuation
RUNNER_TAKE_PROFIT_PCT = 60.0   # runners get more room than scalps (+40%)
RUNNER_BREAKOUT_TOL = 0.001     # within 0.1% of session high/low counts

# --- Strategy 1: momentum scalp ---
MOMENTUM_PCT = 0.5              # min % move on the latest 15-min candle
RSI_PERIOD = 5
RSI_CALL_MIN = 65               # RSI(5) must exceed this for calls
RSI_PUT_MAX = 35                # RSI(5) must be under this for puts
REL_VOLUME_MIN = 1.5            # today's volume vs time-adjusted 20-day avg
LOOKBACK_MULTIPLIER = 2             # multiplier on VOLUME_LOOKBACK_DAYS for bar fetch
VOLUME_LOOKBACK_DAYS = 20
SYMBOL_COOLDOWN_MIN = 15        # don't re-enter the same underlying for N min

# --- ATR-normalized momentum (3A) ---
USE_ATR_MOMENTUM = True         # use ATR-scaled threshold instead of fixed %
ATR_PERIOD = 14                 # bars for Average True Range computation
ATR_MOMENTUM_MULTIPLE = 0.6    # threshold = (ATR/open * 100) * multiple

# --- Multi-timeframe trend confirmation (3B) ---
MULTI_TF_CONFIRM = True         # require hourly trend to match signal direction

# --- RSI divergence detection (3C) ---
DIVERGENCE_FILTER = True        # skip signals with price/RSI divergence

# --- Market regime filter (3D) ---
VIX_FILTER = True               # skip all signals when market is too volatile
VIX_MAX_DAY_RANGE_PCT = 2.0    # SPY day range % threshold for regime filter

# --- Options contract filters ---
MIN_DTE = 1
MAX_DTE = 7
OTM_MIN_PCT = 1.0               # strike must be at least this % out of the money
OTM_MAX_PCT = 3.5               # ...and at most this % OTM (percent of spot, so
                                # the band scales with the underlying's price)
MAX_IV_RANK = 60.0              # skip overpriced premium
MIN_IV_HISTORY = 20             # IV-rank check needs this many stored sessions
MAX_SPREAD = 0.10               # absolute spread allowance in dollars; a quote
MAX_SPREAD_PCT = 5.0            # passes if spread <= max(MAX_SPREAD, mid * pct)
MIN_OPEN_INTEREST = 500
IV_HISTORY_SESSIONS = 252           # max IV history length (~1 year of sessions)
CHAIN_FETCH_LIMIT = 300             # max contracts per chain fetch request

# --- Exits ---
TAKE_PROFIT_PCT = 40.0          # +40% on option price
STOP_LOSS_PCT = 30.0            # -30% on option price
TRAIL_TRIGGER_PCT = 20.0        # once a position is up this much...
TRAIL_GIVEBACK_PCT = 10.0       # ...exit if it gives back this many points
                                # from its peak (peaked +30% -> exit at +20%)
QUOTE_FAIL_FORCE_CLOSE = 20     # force-close at market after this many consecutive
                                # quote failures (~100s at 5s cadence)
FAILED_EXIT_MAX_RETRIES = 3     # retry failed exit orders this many times
FAILED_EXIT_ALERT_SEC = 60      # seconds between repeated "close manually" alerts
FLATTEN_TIMEOUT_SEC = 60        # hard cap for flatten_all() polling loop
MANAGE_INTERVAL_SEC = 5         # exit-check cadence while positions are open

# --- Entry quality filters ---
VWAP_FILTER = True              # calls only above session VWAP, puts only below
EARNINGS_BLOCK = True           # skip single names with earnings inside the
                                # option's DTE window (IV-crush protection)
ETF_SYMBOLS = {"SPY", "QQQ", "IWM", "DIA", "XSP"}  # exempt from earnings check

# --- Session times (US/Eastern) ---
TZ = "America/New_York"
ENTRY_START = (9, 45)           # no entries in the first 15 minutes
ENTRY_END = (15, 30)            # no entries in the last 30 minutes
FORCE_CLOSE = (15, 45)          # flatten everything — never hold overnight

# --- ML learner (learns from trades.csv) ---
ML_ENABLED = True
ML_MIN_TRADES = 50              # closed trades needed before first training
ML_RETRAIN_EVERY = 10           # retrain after this many new closed trades
ML_WIN_PROB_THRESHOLD = 0.45    # block entries the model scores below this
ML_MIN_AUC = 0.55               # model only gets veto power above this CV AUC;
                                # below it stays advisory (scores logged, no blocks)
ML_N_ESTIMATORS = 100           # GBM number of boosting rounds
ML_MAX_DEPTH = 2                # GBM tree depth (shallow = overfitting protection)
ML_LEARNING_RATE = 0.05         # GBM learning rate
ML_SUBSAMPLE = 0.8              # GBM row subsampling per tree
MODEL_FILE = os.path.join(os.path.dirname(__file__), "model.pkl")

# --- Cadence / files ---
STREAM_CACHE_MAX_AGE_SEC = 1800  # refresh bar cache from REST if no stream
                                 # updates for this long (30 min)
SCAN_INTERVAL_SEC = 30
ENTRY_FILL_TIMEOUT_SEC = 20     # cancel unfilled entry limit orders after this
EXIT_FILL_TIMEOUT_SEC = 30      # cancel unfilled exit limit orders after this
_DIR = os.path.dirname(__file__)
TRADES_CSV = os.path.join(_DIR, "trades.csv")       # legacy — used for migration only
TRADES_DB = os.path.join(_DIR, "trades.db")         # SQLite database (primary store)
IV_HISTORY_FILE = os.path.join(_DIR, "iv_history.json")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# --- AI analyst (Claude) — advisory only, never touches trade decisions ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
AI_ENABLED = os.getenv("AI_ENABLED", "true").lower() == "true"
AI_MODEL = "claude-sonnet-4-20250514"  # ~$0.05-0.15 per report at this usage
AI_BRIEFING_FILE = os.path.join(_DIR, "morning_briefing.md")
AI_REPORT_FILE = os.path.join(_DIR, "daily_report.md")

# --- Dashboard integration (app.py) ---
# On the VM, systemd owns the bot. Set DASHBOARD_MONITOR_ONLY=true there so the
# dashboard hides its Start/Stop/Restart controls and can't fight systemd
# (two main.py instances = double orders). Defaults off for the laptop.
MONITOR_ONLY = os.getenv("DASHBOARD_MONITOR_ONLY", "false").lower() == "true"
STATUS_FILE = os.path.join(_DIR, "status.json")     # bot heartbeat for the UI
STOP_FLAG_FILE = os.path.join(_DIR, "stop.flag")    # UI asks bot to stop gracefully
BOT_LOG_FILE = os.path.join(_DIR, "bot.log")
CONSOLE_LOG_FILE = os.path.join(_DIR, "console.log")
SETTINGS_FILE = os.path.join(_DIR, "settings.json") # UI-saved overrides

# Settings the dashboard may override. Applied last so saved values win;
# anything not in settings.json keeps its default above.
_TUNABLE = {
    "UNIVERSE", "MAX_TRADE_COST", "MAX_OPEN_POSITIONS", "MAX_DAILY_LOSS",
    "MAX_DAILY_TRADES",
    "TAKE_PROFIT_PCT", "STOP_LOSS_PCT", "MOMENTUM_PCT", "REL_VOLUME_MIN",
    "MAX_IV_RANK", "ML_ENABLED", "ML_WIN_PROB_THRESHOLD",
    "OTM_MIN_PCT", "OTM_MAX_PCT", "MAX_SPREAD_PCT",
    "TRAIL_TRIGGER_PCT", "TRAIL_GIVEBACK_PCT", "VWAP_FILTER", "EARNINGS_BLOCK",
    "RUNNER_ENABLED", "RUNNER_DAY_PCT", "RUNNER_TAKE_PROFIT_PCT",
    "EXIT_FILL_TIMEOUT_SEC",
    "USE_ATR_MOMENTUM", "ATR_MOMENTUM_MULTIPLE",
    "MULTI_TF_CONFIRM", "DIVERGENCE_FILTER",
    "VIX_FILTER", "VIX_MAX_DAY_RANGE_PCT",
    "ML_N_ESTIMATORS", "ML_MAX_DEPTH", "ML_LEARNING_RATE", "ML_SUBSAMPLE",
    "DYNAMIC_UNIVERSE", "DYNAMIC_UNIVERSE_MAX_SYMBOLS",
    "DYNAMIC_UNIVERSE_REFRESH_MIN",
}
if os.path.exists(SETTINGS_FILE):
    import json as _json
    import logging as _logging
    _settings_log = _logging.getLogger("config")
    try:
        with open(SETTINGS_FILE) as _f:
            for _k, _v in _json.load(_f).items():
                if _k not in _TUNABLE:
                    continue
                _default = globals().get(_k)
                # Coerce to the type of the hardcoded default so that a
                # dashboard-saved string "50" becomes float 50.0, etc.
                if _default is not None and _v is not None:
                    _expected = type(_default)
                    if not isinstance(_v, _expected):
                        try:
                            _v = _expected(_v)
                        except (ValueError, TypeError):
                            _settings_log.warning(
                                "settings.json: %s=%r cannot be coerced to %s — skipped",
                                _k, _v, _expected.__name__)
                            continue
                globals()[_k] = _v
    except (_json.JSONDecodeError, OSError):
        pass  # bad settings file — fall back to defaults

# Clamp numeric settings to sane ranges. Prevents dashboard-saved garbage
# (e.g. negative stop loss, zero max positions) from breaking the bot.
_RANGE_LIMITS = {
    "MAX_TRADE_COST":       (5.0, 5000.0),
    "MAX_OPEN_POSITIONS":   (1, 20),
    "MAX_DAILY_LOSS":       (10.0, 5000.0),
    "TAKE_PROFIT_PCT":      (5.0, 500.0),
    "STOP_LOSS_PCT":        (5.0, 95.0),
    "MOMENTUM_PCT":         (0.05, 10.0),
    "ML_WIN_PROB_THRESHOLD":(0.1, 0.9),
    "REL_VOLUME_MIN":       (0.1, 50.0),
    "MAX_IV_RANK":          (1.0, 100.0),
    "OTM_MIN_PCT":          (0.1, 20.0),
    "OTM_MAX_PCT":          (0.5, 30.0),
    "MAX_SPREAD_PCT":       (0.5, 50.0),
    "TRAIL_TRIGGER_PCT":    (1.0, 200.0),
    "TRAIL_GIVEBACK_PCT":   (1.0, 100.0),
    "RUNNER_DAY_PCT":       (0.5, 20.0),
    "RUNNER_TAKE_PROFIT_PCT":(5.0, 500.0),
    "EXIT_FILL_TIMEOUT_SEC": (5, 120),
    "ATR_MOMENTUM_MULTIPLE": (0.1, 5.0),
    "VIX_MAX_DAY_RANGE_PCT": (0.5, 10.0),
    "ML_N_ESTIMATORS":      (10, 500),
    "ML_MAX_DEPTH":         (1, 10),
    "ML_LEARNING_RATE":     (0.001, 1.0),
    "ML_SUBSAMPLE":         (0.1, 1.0),
    "DYNAMIC_UNIVERSE_MAX_SYMBOLS": (3, 30),
    "DYNAMIC_UNIVERSE_REFRESH_MIN": (5, 120),
}
for _key, (_lo, _hi) in _RANGE_LIMITS.items():
    _val = globals().get(_key)
    if _val is not None and isinstance(_val, (int, float)):
        _clamped = max(_lo, min(_hi, _val))
        if _clamped != _val:
            import logging as _logging2
            _logging2.getLogger("config").warning(
                "%s=%r out of range [%s, %s] — clamped to %s",
                _key, _val, _lo, _hi, _clamped)
            globals()[_key] = type(_val)(_clamped)
