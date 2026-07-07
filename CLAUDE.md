# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

An automated options scalping bot for Alpaca that trades short-dated (1-7 DTE) OTM options on a $500 paper account. Two strategies: momentum scalp (>0.5% move on 15-min candles + RSI confirmation) and runner (trend continuation on new session highs/lows). Includes an ML learner that filters entries by win probability, a Streamlit dashboard, Discord alerts, and Claude AI analyst for briefings/reports.

## Commands

```bash
# Setup
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
# Copy .env and fill in ALPACA_API_KEY, ALPACA_SECRET_KEY (minimum required)

# Run bot (paper mode by default)
python main.py

# Run dashboard (recommended way to operate)
.venv\Scripts\python -m streamlit run app.py
# Or double-click "Launch Options Bot.bat"

# Backtest
python backtest.py --days 60

# Diagnostics (read-only, safe to run anytime)
python diag_probe.py            # Live scanner state per symbol
python diag_chain.py            # Contract filter verdicts

# Smoke test
python test_learner.py          # ML learner test with synthetic data

# AI reports (requires ANTHROPIC_API_KEY in .env)
python ai_analyst.py briefing
python ai_analyst.py report
```

No linter, type checker, or test framework is configured. The single test file (`test_learner.py`) runs standalone via `python test_learner.py`.

## Architecture

**Entry pipeline (sequential):**
`Scanner` (signal) → `ChainFetcher` (contract) → `RiskManager` (gate) → `Learner` (ML gate) → `Executor` (order)

**Key modules:**

| Module | Role |
|---|---|
| `main.py` | Orchestrator loop — 30s scan cadence, 5s exit management |
| `config.py` | Single source of truth for all tunables; loads `.env` then `settings.json` overrides |
| `scanner.py` | Generates `Signal` objects (momentum + RSI + VWAP + relative volume) |
| `options_chain.py` | Fetches chains, filters by OTM band/IV rank/spread/OI → `ContractPick` |
| `executor.py` | Places/manages orders, trailing stops, crash recovery (adopts existing positions on startup) |
| `trade_store.py` | SQLite persistence layer for all trade data (WAL mode); auto-migrates legacy CSV on first run |
| `risk_manager.py` | Daily loss halt ($75), position caps (3), consecutive-loss pause, P&L restoration across restarts |
| `learner.py` | Gradient-boosting classifier on trade history; earns veto power only when CV AUC ≥ 0.55 |
| `ai_analyst.py` | Claude-powered pre-market briefings and end-of-day reports (advisory only) |
| `app.py` | Streamlit dashboard — start/stop bot, live P&L, positions, settings editor |
| `alerts.py` | Discord webhook notifications (fail-open: webhook failures never crash the bot) |

**Multi-process communication (dashboard ↔ bot):**
- `status.json` — bot writes heartbeat (atomic write via `.tmp` + rename)
- `settings.json` — dashboard saves config overrides (applied at `config.py` import time)
- `stop.flag` — dashboard creates to request graceful shutdown
- `bot.pid` — bot writes its PID for process management

**Runtime files (all gitignored):** `.env`, `trades.db`, `model.pkl`, `iv_history.json`, `status.json`, `settings.json`, `bot.log`, `bot.pid`, `stop.flag`

## Key Design Patterns

- **Fail-open optional dependencies:** Discord alerts, earnings lookup, Anthropic AI, and TA-Lib all degrade gracefully on failure — the trading loop never crashes from optional features.
- **Crash recovery:** Executor reconciles positions at startup (adopts existing holdings); RiskManager restores today's P&L via `trade_store`.
- **Storage abstraction:** All trade persistence goes through `trade_store.py` (SQLite). Swapping to Postgres or another backend requires changing only this one file.
- **Config override chain:** hardcoded defaults in `config.py` → `.env` overrides → `settings.json` overrides (dashboard-saved). The `_TUNABLE` set controls which keys can be overridden via the dashboard.
- **ML earned gating:** The learner only vetoes entries after proving predictive power (AUC ≥ 0.55). Below that threshold it's advisory-only (scores are logged but trades aren't blocked).

## Environment Variables

Required: `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`
Optional: `LIVE_MODE` (default `false`), `DISCORD_WEBHOOK_URL`, `ANTHROPIC_API_KEY`, `AI_ENABLED`, `LOG_LEVEL`, `DASHBOARD_MONITOR_ONLY` (set `true` on AWS VM to hide start/stop controls)

## Deployment

Local: Windows with `.venv`. AWS: Ubuntu EC2 with systemd services (`deploy/optionsbot-dashboard.service`). On the VM, `DASHBOARD_MONITOR_ONLY=true` prevents the dashboard from spawning a second bot instance that would conflict with the systemd-managed one.
