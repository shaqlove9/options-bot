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

# --- Discord ---
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

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
VOLUME_LOOKBACK_DAYS = 20
SYMBOL_COOLDOWN_MIN = 15        # don't re-enter the same underlying for N min

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

# --- Exits ---
TAKE_PROFIT_PCT = 40.0          # +40% on option price
STOP_LOSS_PCT = 30.0            # -30% on option price
TRAIL_TRIGGER_PCT = 20.0        # once a position is up this much...
TRAIL_GIVEBACK_PCT = 10.0       # ...exit if it gives back this many points
                                # from its peak (peaked +30% -> exit at +20%)
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
MODEL_FILE = os.path.join(os.path.dirname(__file__), "model.pkl")

# --- Cadence / files ---
SCAN_INTERVAL_SEC = 30
ENTRY_FILL_TIMEOUT_SEC = 20     # cancel unfilled entry limit orders after this
_DIR = os.path.dirname(__file__)
TRADES_CSV = os.path.join(_DIR, "trades.csv")
IV_HISTORY_FILE = os.path.join(_DIR, "iv_history.json")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# --- AI analyst (Claude) — advisory only, never touches trade decisions ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
AI_ENABLED = os.getenv("AI_ENABLED", "true").lower() == "true"
AI_MODEL = "claude-fable-5"          # ~$0.05-0.15 per report at this usage
AI_BRIEFING_FILE = os.path.join(_DIR, "morning_briefing.md")
AI_REPORT_FILE = os.path.join(_DIR, "daily_report.md")

# --- TradingView webhook receiver (webhook.py) ---
# Optional inbound alert endpoint. When WEBHOOK_ENABLED, a stdlib http.server runs
# in a daemon thread; valid alerts are enqueued and run through the SAME entry
# pipeline (try_enter) as scanner signals — no parallel trading path. These are
# boot-time / SECURITY settings and are deliberately NOT in _TUNABLE: the secret
# and IP allowlist must never be overridable from settings.json, and the listener
# host/port/enabled are only read once at startup.
WEBHOOK_ENABLED = os.getenv("WEBHOOK_ENABLED", "false").lower() == "true"
WEBHOOK_HOST = os.getenv("WEBHOOK_HOST", "0.0.0.0")
WEBHOOK_PORT = int(os.getenv("WEBHOOK_PORT", "8080"))
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")   # required if enabled; from .env
# Comma-separated IPs (empty = allow all). TradingView's published webhook IPs are
# 52.89.214.238,34.212.75.30,54.218.53.128,52.32.178.7 (see webhook.TRADINGVIEW_IPS).
WEBHOOK_IP_ALLOWLIST = [ip.strip() for ip in
                        os.getenv("WEBHOOK_IP_ALLOWLIST", "").split(",") if ip.strip()]
WEBHOOK_DEDUP_SEC = float(os.getenv("WEBHOOK_DEDUP_SEC", "60"))  # ignore repeat ids within

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
}
if os.path.exists(SETTINGS_FILE):
    import json as _json
    try:
        with open(SETTINGS_FILE) as _f:
            for _k, _v in _json.load(_f).items():
                if _k in _TUNABLE:
                    globals()[_k] = _v
    except (_json.JSONDecodeError, OSError):
        pass  # bad settings file — fall back to defaults
