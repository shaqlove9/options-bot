# Options Scalping Bot

Automated options scalper for Alpaca. Trades short-dated (1-7 DTE) OTM calls and puts on momentum signals. Two strategies: **momentum scalp** and **trend runner**. Includes real-time WebSocket streaming, ML-powered entry filtering, a Streamlit dashboard, Discord alerts, and Claude AI analyst.

Paper mode by default. Never holds overnight.

## Risk Warning

Options scalping on a small account is extremely high risk. Short-dated OTM options can lose most of their value in minutes. The $75 daily loss cap is 15% of the $500 account. Run paper mode for weeks before considering live. The backtest uses synthetic Black-Scholes prices — real fills will be worse.

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Broker API | Alpaca (`alpaca-py`) — trading, market data, WebSocket streams |
| Data | REST + WebSocket hybrid (stream-first, REST fallback on disconnect) |
| ML | scikit-learn gradient-boosting classifier |
| Database | SQLite (WAL mode) via `trade_store.py` |
| Dashboard | Streamlit |
| AI Analyst | Claude API (Anthropic) — optional, advisory only |
| Alerts | Discord webhooks — optional, fail-open |
| Deployment | Windows local or Ubuntu EC2 with systemd |

## Quick Start

### Windows

```powershell
git clone https://github.com/shaqlove9/options-bot.git
cd options-bot
setup.bat                        # creates venv, installs deps, copies .env
# Edit .env — fill in ALPACA_API_KEY and ALPACA_SECRET_KEY
"Launch Options Bot.bat"         # opens dashboard at http://localhost:8501
```

### Linux / macOS

```bash
git clone https://github.com/shaqlove9/options-bot.git
cd options-bot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env             # then edit with your Alpaca keys
python main.py                   # or run the dashboard below
```

### Dashboard (recommended way to operate)

```bash
.venv/bin/python -m streamlit run app.py     # Linux
.venv\Scripts\python -m streamlit run app.py  # Windows
```

The dashboard provides: Start/Stop/Restart controls, live P&L + open positions, equity curve + trade history, settings editor, AI analyst reports, and live log viewer. The bot runs as a separate process — closing the browser tab does not stop trading.

### Command line

```bash
python main.py                          # run the bot (paper mode default)
python -m trading.scanner               # one-off scan to check signals
python -m tools.backtest --days 60      # backtest on 60 days of data
python -m tools.diag_probe              # live scanner state per symbol
python -m tools.diag_chain              # contract filter verdicts
python ai_analyst.py briefing           # manual AI morning briefing
python ai_analyst.py report             # manual AI end-of-day report
python -m tools.test_learner            # ML learner smoke test
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ALPACA_API_KEY` | Yes | Alpaca API key (use paper keys first) |
| `ALPACA_SECRET_KEY` | Yes | Alpaca secret key |
| `LIVE_MODE` | No | `true` for real money (default: `false`) |
| `DISCORD_WEBHOOK_URL` | No | Discord webhook for alerts |
| `ANTHROPIC_API_KEY` | No | Claude API key for AI analyst |
| `AI_ENABLED` | No | Enable AI briefings/reports (default: `true`) |
| `LOG_LEVEL` | No | `DEBUG`, `INFO`, `WARNING`, `ERROR` (default: `INFO`) |
| `DASHBOARD_MONITOR_ONLY` | No | `true` on VMs to hide Start/Stop (default: `false`) |

Your Alpaca account needs **options trading enabled** (Level 1 — long calls/puts — is sufficient).

## How It Works

### Data Flow

```
Market Data (Alpaca WebSocket + REST)
  │
  ├── StockBarStreamThread ──── 15-min completed bars ──── queue.Queue
  ├── OptionQuoteStreamThread ── real-time bid/ask ──────── Lock-protected dict
  └── TradingStreamThread ────── order fill events ──────── queue.Queue
  │
  ▼
HybridProviders (stream-first, REST fallback on disconnect)
  │
  ▼
Main Loop (30s scan cadence, 5s exit checks when holding)
  │
  ├── Scanner ──── generates momentum/runner signals
  ├── ChainFetcher ──── finds matching option contracts
  ├── RiskManager ──── position caps, daily loss halt, pause logic
  ├── Learner ──── ML win-probability gate
  └── Executor ──── places orders, manages TP/SL/trailing stop
  │
  ▼
Outputs
  ├── trade_store (SQLite) ──── all closed trade data
  ├── status.json ──── dashboard heartbeat (atomic writes)
  ├── Discord alerts ──── entry/exit/halt/summary
  └── AI reports ──── morning briefing + end-of-day recap
```

### Entry Pipeline

1. **Scanner** detects momentum signal (>0.5% on 15-min candle + RSI + volume + VWAP) or runner signal (>2% day move + new session highs/lows)
2. **ChainFetcher** finds OTM option contract matching filters (DTE, OTM band, spread, OI, IV rank)
3. **RiskManager** checks position limits, daily loss cap, consecutive-loss pause
4. **Learner** scores P(win) — blocks entry if model is gating and score is below threshold
5. **Executor** submits limit buy at ask, cancels if unfilled after 20s

### Exit Management

| Rule | Value |
|---|---|
| Take profit | +40% on premium (+60% for runners) |
| Stop loss | -30% on premium |
| Trailing stop | After +20%, exit if 10 points given back from peak |
| Time stop | Flatten everything at 3:45 PM ET |
| Quote failure | Force-close at market after 20 consecutive failures |
| Exit cadence | Every 5s while holding positions |

### Risk Controls

| Rule | Value |
|---|---|
| Max per trade | $50 premium, 1 contract |
| Max positions | 3 concurrent |
| Daily loss halt | -$75 P&L halts all trading, flattens, alerts Discord |
| Consecutive loss pause | 2 losses in a row triggers 30-min pause |
| Entry window | 9:45 AM - 3:30 PM ET only |
| Earnings block | Skip single names with earnings inside DTE window |

### Crash Recovery

On startup the bot: cancels stray open orders, adopts any option positions still held at Alpaca (so nothing is orphaned overnight), and rebuilds today's P&L from SQLite so the daily loss limit survives restarts.

## Architecture

```
main.py                     Orchestrator loop (entry point)
app.py                      Streamlit dashboard (start/stop, P&L, settings)
config.py                   All tunables (.env + settings.json overrides)
utils.py                    ET time helpers, retry decorator
ai_analyst.py               Claude-powered briefings/reports (advisory only)

trading/                    Core trading logic
├── scanner.py                  Momentum + runner signal detection
├── options_chain.py            Chain fetch, OTM/spread/OI/IV-rank filters
├── executor.py                 Order placement, TP/SL/trailing, crash recovery
├── risk_manager.py             Daily halt, position caps, loss pause
├── learner.py                  Gradient-boosting ML filter
└── earnings.py                 Earnings calendar (IV-crush avoidance)

data/                       Persistence + data providers
├── protocols.py                Protocol interfaces (structural typing)
├── rest.py                     REST data providers (with retry)
├── hybrid.py                   Hybrid providers (stream-first, REST fallback)
├── stream_orders.py            Stream-backed order event provider
└── trade_store.py              SQLite persistence (WAL mode, CSV auto-migration)

streaming/                  WebSocket daemon threads
├── market.py                   Stock bar + option quote streams
└── trading.py                  Trading stream (instant fill events)

alerts/                     Webhook notifications (fail-open)
├── discord.py                  Discord embed backend
└── slack.py                    Slack attachment backend

tools/                      Standalone scripts (not imported by core)
├── backtest.py                 Black-Scholes backtest harness
├── diag_probe.py               Live scanner state diagnostic
├── diag_chain.py               Contract filter diagnostic
└── test_learner.py             ML learner smoke test
```

### Design Patterns

- **Protocol-based DI:** Consumers depend on `typing.Protocol` interfaces, not Alpaca SDK classes. New data backends (WebSocket, mock) added without modifying Scanner/Executor.
- **Hybrid streaming:** Three daemon threads run WebSocket connections. On disconnect, providers fall back to REST seamlessly. On reconnect, streams resume automatically.
- **Fail-open optionals:** Discord, earnings, AI analyst, TA-Lib all degrade gracefully. Trading loop never crashes from optional features.
- **Config override chain:** Hardcoded defaults in `config.py` are overridden by `.env`, then by `settings.json` (dashboard-saved). Range validation clamps values to sane bounds.
- **ML earned gating:** Learner only vetoes entries after proving AUC >= 0.55. Below that it logs scores but doesn't block.
- **Atomic file writes:** All shared files (status.json, settings.json, AI reports) use `.tmp` + `os.replace()` to prevent corruption.

### Multi-Process Communication (Dashboard ↔ Bot)

| File | Purpose |
|---|---|
| `status.json` | Bot heartbeat — written atomically each cycle |
| `settings.json` | Dashboard-saved config overrides |
| `stop.flag` | Dashboard creates to request graceful shutdown |
| `bot.pid` | Bot PID for process management |

## ML Learner

`learner.py` trains a gradient-boosting classifier on your closed trades: signal features at entry mapped to win/loss outcome.

- **Warm-up:** Needs 50 closed trades before first training
- **Earned veto:** Model blocks entries only after cross-validated AUC >= 0.55. Below that it's advisory-only (scores logged, no blocks)
- **Gate threshold:** Entries scoring below `ML_WIN_PROB_THRESHOLD` (0.45) are skipped
- **Retraining:** Automatic after every 10 new closed trades (end of day)
- **Manual:** `python learner.py` to force retrain and see AUC + top features

## AWS Deployment (24/7)

The bot runs 24/7 on AWS EC2 via systemd.

| | |
|---|---|
| Instance | EC2 t3.medium (2 vCPU, 4 GiB), us-east-1 |
| OS | Ubuntu Server 26.04 LTS |
| Cost | ~$36/mo |
| Bot service | `optionsbot.service` (Restart=always) |
| Dashboard | `optionsbot-dashboard.service` (monitoring only, localhost:8501) |

```bash
sudo systemctl status optionsbot          # check status
sudo systemctl restart optionsbot         # restart
sudo systemctl stop optionsbot            # stop
journalctl -u optionsbot -f               # live logs
```

Access the dashboard via SSH tunnel:
```bash
ssh -L 8501:localhost:8501 -i Options-bot.pem ubuntu@<ip>
# then open http://localhost:8501 on your laptop
```

On the VM, `DASHBOARD_MONITOR_ONLY=true` hides Start/Stop buttons so the dashboard can't fight systemd.

## Not Financial Advice

This is software, not investment advice. You are responsible for every order it places. Test extensively in paper mode.
