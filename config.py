"""Central configuration. All secrets come from .env — never hardcode keys."""
import os

from dotenv import load_dotenv

load_dotenv()

# --- Mode ---
LIVE_MODE = os.getenv("LIVE_MODE", "false").lower() == "true"
PAPER = not LIVE_MODE

# --- Instrument ---
# "options" (default, unchanged live behaviour) trades the OTM option via the
# Executor + ChainFetcher. "equity" trades the underlying SHARES directly
# (the validated momentum sleeve) via EquityExecutor — same signal, no option
# pricing/theta drag. Set INSTRUMENT=equity in .env to run the equity sleeve.
INSTRUMENT = os.getenv("INSTRUMENT", "options").lower()

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

# --- Exits (options) ---
TAKE_PROFIT_PCT = 40.0          # +40% on option price
STOP_LOSS_PCT = 30.0            # -30% on option price
TRAIL_TRIGGER_PCT = 20.0        # once a position is up this much...
TRAIL_GIVEBACK_PCT = 10.0       # ...exit if it gives back this many points
                                # from its peak (peaked +30% -> exit at +20%)
MANAGE_INTERVAL_SEC = 5         # exit-check cadence while positions are open

# --- Equity sleeve (INSTRUMENT=equity) ---
# Trades shares of the same UNIVERSE on the same signal. Stops/targets are % of
# the SHARE price (not option premium), so they are ~100x tighter. Values match
# the validated backtest (backtest.py simulate_equity).
EQ_NOTIONAL_PER_TRADE = 4000.0  # $ exposure per position (uses intraday margin)
EQ_ALLOW_SHORT = True           # put signals -> short shares (matches backtest)
EQ_TAKE_PROFIT_PCT = 0.6        # +0.6% move of the share price
EQ_STOP_LOSS_PCT = 0.4          # -0.4% move of the share price
EQ_TRAIL_TRIGGER_PCT = 0.5      # arm trailing stop once up this %
EQ_TRAIL_GIVEBACK_PCT = 0.25    # ...then exit on this much giveback
EQ_MAX_TRADE_RISK = 50.0        # max $ risk/trade for the shared risk gate

# --- Equity meta-labeling layer (decision layer on the share scalper) ---
# A SECONDARY model that learns from our OWN closed trades which signals to skip
# and how big to size. It NEVER invents entries. Capture + shadow-logging run now;
# the model only gates/sizes real (paper) trades after train_meta.py's gate passes
# AND EQ_META_ACTIVE is flipped on (the single reversible activation flag).
EQ_META_ENABLED = True           # capture features + log shadow decisions every signal
EQ_META_ACTIVE = False           # activation flag: let the model gate/size real trades
EQ_META_MIN_TRADES = 40          # labeled closed trades required before the model may activate
EQ_META_AUC_BAR = 0.55           # walk-forward OOS AUC the model must clear to ship
EQ_META_THRESHOLD = 0.50         # operating P(win): when active, skip signals scoring below this
EQ_META_EMBARGO = 2              # trades embargoed between train/test folds (purge leakage)
RANDOM_SEED = 42                 # fixed seed — reproducible training

# --- Equity risk governor (independent of the model) ---
# Kill switches are ACTIVE in paper now (pure downside protection, mirroring the
# trend sleeve's breaker). Vol-targeted sizing is built + shadow-logged but stays
# GATED behind EQ_GOV_SIZING_ACTIVE so the flat EQ_NOTIONAL_PER_TRADE keeps the
# in-flight equity forward-test homogeneous until the gate justifies a change.
EQ_GOV_ENABLED = True
EQ_GOV_DAILY_MAX_LOSS_PCT = 2.0  # flatten + halt for the day at -2% of start-of-day equity
EQ_GOV_TRAILING_DD_PCT = 15.0    # halt if equity falls this far from its peak
EQ_GOV_MAX_CONCURRENT = 3        # hard cap on simultaneous open positions
EQ_GOV_MAX_TRADES_DAY = 20       # hard cap on entries per day
EQ_GOV_RISK_FRAC = 0.0025        # conservative fixed risk/trade (frac of equity) until Kelly measured
EQ_GOV_KELLY_FRAC = 0.25         # fractional-Kelly cap once win-rate/payoff are measured
EQ_GOV_MAX_NOTIONAL_FRAC = 0.5   # vol-target sizing never exceeds this fraction of equity per trade
EQ_GOV_SIZING_ACTIVE = False     # gate: when True, vol-targeted size replaces the flat notional

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
# Equity sleeve logs to its own file so the options trades.csv (and the learner
# trained on it) are never mixed with share trades.
TRADES_CSV = os.path.join(_DIR, "trades_equity.csv" if INSTRUMENT == "equity"
                          else "trades.csv")
IV_HISTORY_FILE = os.path.join(_DIR, "iv_history.json")
# Equity meta-labeling artifacts (equity sleeve only; one row per signal).
META_DB = os.path.join(_DIR, "equity_meta.db")
META_MODEL_FILE = os.path.join(_DIR, "meta_model.pkl")
META_METRICS_FILE = os.path.join(_DIR, "meta_metrics.json")
EQ_GOV_STATE_FILE = os.path.join(_DIR, "equity_governor_state.json")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# --- AI analyst (Claude) — advisory only, never touches trade decisions ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
AI_ENABLED = os.getenv("AI_ENABLED", "true").lower() == "true"
AI_MODEL = "claude-opus-4-8"         # $5/$25 per 1M tok (half of fable-5)
AI_BRIEFING_FILE = os.path.join(_DIR, "morning_briefing.md")
AI_REPORT_FILE = os.path.join(_DIR, "daily_report.md")

# --- Dashboard integration (app.py) ---
# On the VM, systemd owns the bot. Set DASHBOARD_MONITOR_ONLY=true there so the
# dashboard hides its Start/Stop/Restart controls and can't fight systemd
# (two main.py instances = double orders). Defaults off for the laptop.
MONITOR_ONLY = os.getenv("DASHBOARD_MONITOR_ONLY", "false").lower() == "true"
# These IPC/dashboard paths are env-overridable so a SECOND instance (e.g. the
# equity sleeve) can run on the same box without clobbering the options bot's
# status/stop/log files. Unset env = unchanged prod defaults.
STATUS_FILE = os.getenv("STATUS_FILE", os.path.join(_DIR, "status.json"))     # bot heartbeat for the UI
STOP_FLAG_FILE = os.getenv("STOP_FLAG_FILE", os.path.join(_DIR, "stop.flag")) # UI asks bot to stop gracefully
BOT_LOG_FILE = os.getenv("BOT_LOG_FILE", os.path.join(_DIR, "bot.log"))
CONSOLE_LOG_FILE = os.getenv("CONSOLE_LOG_FILE", os.path.join(_DIR, "console.log"))
SETTINGS_FILE = os.getenv("SETTINGS_FILE", os.path.join(_DIR, "settings.json")) # UI-saved overrides

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
    # equity meta-labeling + governor (reversible flags the dashboard may toggle)
    "EQ_META_ENABLED", "EQ_META_ACTIVE", "EQ_META_THRESHOLD",
    "EQ_GOV_ENABLED", "EQ_GOV_SIZING_ACTIVE",
    "EQ_GOV_DAILY_MAX_LOSS_PCT", "EQ_GOV_TRAILING_DD_PCT",
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
