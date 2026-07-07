# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

An automated options scalping bot for Alpaca that trades short-dated (1-7 DTE) OTM options on a $500 paper account. Two strategies: momentum scalp (>0.5% move on 15-min candles + RSI confirmation) and runner (trend continuation on new session highs/lows). Includes an ML learner that filters entries by win probability, a Streamlit dashboard, Discord/Slack alerts, and Claude AI analyst for briefings/reports.

## Commands

```bash
# Setup
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
# Copy .env.example to .env and fill in ALPACA_API_KEY, ALPACA_SECRET_KEY

# Run bot (paper mode by default)
python main.py

# Run dashboard (recommended way to operate)
.venv\Scripts\python -m streamlit run app.py
# Or double-click "Launch Options Bot.bat"

# Backtest
python -m tools.backtest --days 60

# Diagnostics (read-only, safe to run anytime)
python -m tools.diag_probe      # Live scanner state per symbol
python -m tools.diag_chain      # Contract filter verdicts

# Smoke test
python -m tools.test_learner    # ML learner test with synthetic data

# AI reports (requires ANTHROPIC_API_KEY in .env)
python ai_analyst.py briefing
python ai_analyst.py report
```

No linter, type checker, or test framework is configured. The single test file runs standalone via `python -m tools.test_learner`.

## Architecture

**Package structure:**

```
main.py / app.py / config.py / utils.py / ai_analyst.py   (root entry points + shared)
trading/     signal_scanner, order_manager, contract_selector, risk_manager, ml_filter, earnings_guard
data/        protocols, rest, hybrid, stream_orders, trade_store
streaming/   market (stock bars + option quotes), trading (order fills)
alerts/      discord, slack (webhook backends, fail-open)
tools/       backtest, diag_probe, diag_chain, test_learner
```

**Entry pipeline (sequential):**
`Scanner` (signal) -> `ChainFetcher` (contract) -> `RiskManager` (gate) -> `Learner` (ML filter) -> `Executor` (order)

**Key modules:**

| Module | Role |
|---|---|
| `main.py` | Orchestrator loop — 30s scan cadence, 5s exit management |
| `config.py` | Single source of truth for all tunables; loads `.env` then `settings.json` overrides |
| `trading/signal_scanner.py` | Generates `Signal` objects (momentum + RSI + VWAP + relative volume) |
| `trading/contract_selector.py` | Fetches chains, filters by OTM band/IV rank/spread/OI -> `ContractPick` |
| `trading/order_manager.py` | Places/manages orders, trailing stops, crash recovery (adopts existing positions on startup) |
| `trading/risk_manager.py` | Daily loss halt ($75), position caps (3), consecutive-loss pause, P&L restoration across restarts |
| `trading/ml_filter.py` | Gradient-boosting classifier on trade history; earns veto power only when CV AUC >= 0.55 |
| `data/trade_store.py` | SQLite persistence layer for all trade data (WAL mode); auto-migrates legacy CSV on first run |
| `data/protocols.py` | Protocol interfaces (StockBarProvider, OptionQuoteProvider, etc.) |
| `data/rest.py` | REST implementations of data protocols with retry |
| `data/hybrid.py` | Hybrid providers: stream-first, REST fallback on disconnect |
| `data/stream_orders.py` | Stream-backed OrderEventProvider with REST fallback |
| `streaming/market.py` | StockBarStreamThread + OptionQuoteStreamThread (WebSocket daemon threads) |
| `streaming/trading.py` | TradingStreamThread for instant order fill events |
| `alerts/` | Discord + Slack webhook backends (auto-detect from config) |
| `ai_analyst.py` | Claude-powered pre-market briefings and end-of-day reports (advisory only) |
| `app.py` | Streamlit dashboard — start/stop bot, live P&L, positions, settings editor |

**Multi-process communication (dashboard <-> bot):**
- `status.json` — bot writes heartbeat (atomic write via `.tmp` + rename)
- `settings.json` — dashboard saves config overrides (atomic write, applied at `config.py` import time)
- `stop.flag` — dashboard creates to request graceful shutdown
- `bot.pid` — bot writes its PID for process management

**Runtime files (all gitignored):** `.env`, `trades.db`, `model.pkl`, `iv_history.json`, `status.json`, `settings.json`, `bot.log`, `bot.pid`, `stop.flag`

## Key Design Patterns

- **Protocol-based DI:** Consumers depend on `typing.Protocol` interfaces in `data/protocols.py`, not Alpaca SDK classes. REST, stream, and hybrid backends all satisfy the same contracts.
- **Hybrid streaming:** Three daemon threads (stock bars, option quotes, order fills) run WebSocket connections. On disconnect, providers fall back to REST seamlessly.
- **Fail-open optional dependencies:** Discord/Slack alerts, earnings lookup, Anthropic AI, and TA-Lib all degrade gracefully — the trading loop never crashes from optional features.
- **Crash recovery:** OrderManager reconciles positions at startup (adopts existing holdings); RiskManager restores today's P&L via `trade_store`.
- **Storage abstraction:** All trade persistence goes through `data/trade_store.py` (SQLite). Swapping to Postgres or another backend requires changing only this one file.
- **Config override chain:** hardcoded defaults in `config.py` -> `.env` overrides -> `settings.json` overrides (dashboard-saved). Range validation clamps values to sane bounds. The `_TUNABLE` set controls which keys can be overridden via the dashboard.
- **ML earned gating:** The ML filter only vetoes entries after proving predictive power (AUC >= 0.55). Below that threshold it's advisory-only (scores are logged but trades aren't blocked).
- **Atomic file writes:** All shared files (status.json, settings.json, AI reports) use `.tmp` + `os.replace()` to prevent corruption.

## Environment Variables

Required: `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`
Optional: `LIVE_MODE` (default `false`), `DISCORD_WEBHOOK_URL`, `SLACK_WEBHOOK_URL`, `ALERT_BACKEND` (`auto`/`discord`/`slack`), `ANTHROPIC_API_KEY`, `AI_ENABLED`, `LOG_LEVEL`, `DASHBOARD_MONITOR_ONLY` (set `true` on AWS VM to hide start/stop controls)

## Deployment

Local: Windows with `.venv`. AWS: Ubuntu EC2 with systemd services (`deploy/optionsbot-dashboard.service`). On the VM, `DASHBOARD_MONITOR_ONLY=true` prevents the dashboard from spawning a second bot instance that would conflict with the systemd-managed one.
