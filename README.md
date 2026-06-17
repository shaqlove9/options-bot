# Options Scalping Bot — $500 Account

Momentum + IV-based options scalper for Alpaca. **Options only, never holds
overnight, paper mode by default.**

## ⚠️ Read this first

- **Options scalping on a $500 account is extremely high risk.** Short-dated
  OTM options can lose most of their value in minutes. Expect to lose money
  while validating; the $75 daily loss cap is 15% of the account.
- **Run paper mode for weeks before considering live.** The backtest uses
  synthetic Black-Scholes prices — it's an upper bound, not a forecast.
- **SPX is not on Alpaca.** CBOE index options aren't tradable there, so the
  universe uses SPY/QQQ for index exposure (plus NVDA, TSLA, AAPL, AMZN).
- **IV rank warm-up:** Alpaca doesn't provide historical IV, so the bot builds
  its own IV history in `iv_history.json` (one reading per symbol per session).
  The IV-rank > 60 filter activates after 20 sessions; until then it's skipped
  with a warning. More reason to paper trade for a few weeks first.

## Setup

```powershell
cd options-bot
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env       # then fill in your keys
```

Your Alpaca account needs **options trading enabled** (Level 1 — long
calls/puts — is sufficient). Use your **paper** API keys first.

## Run — Dashboard (recommended)

Double-click **`Launch Options Bot.bat`** — it opens the dashboard in your
browser at `http://localhost:8501`:

- ▶ **Start / ⏹ Stop** the bot (stop is graceful — flattens positions first)
- Live daily P&L, open positions with unrealized P&L, win rate, ML model status
- Equity curve + full trade history with CSV download
- ⚙️ Settings panel — universe, $/trade, TP/SL, daily halt, ML threshold
  (saved to `settings.json`; restart the bot to apply)
- Live log viewer

The bot runs as its own process — closing the dashboard tab does **not** stop
trading. Use the ⏹ Stop button for that.

## Run — command line

```powershell
python main.py               # paper mode (default)
python scanner.py            # one-off scan to sanity-check signals
python backtest.py --days 60 # backtest on 60 days of 15-min bars
```

To go live: set `LIVE_MODE=true` in `.env` **and** swap in your live API keys.
Both are required on purpose.

## Deployment — AWS VPS (24/7 paper)

As of **2026-06-15** the bot runs **24/7 in paper mode** on an AWS EC2 instance,
so it no longer depends on the laptop being on.

**Infrastructure**

| | |
|---|---|
| Instance | EC2 **t3.medium** (2 vCPU, 4 GiB), **us-east-1** (closest to Alpaca) |
| OS | Ubuntu Server 26.04 LTS (x86), 30 GiB gp3 root |
| Cost | ~$36/mo (instance ~$30 + EBS ~$2.40 + public IPv4 ~$3.60) |
| App path | `/home/ubuntu/options-bot` |
| Login | `ssh -i Options-bot.pem ubuntu@<public-ip>` (SSH from My IP only) |

The Windows `.venv` does **not** port — it's rebuilt on Linux
(`python3 -m venv .venv && pip install -r requirements.txt`). TA-Lib stays
optional; the pandas RSI fallback is used.

**Runs as a systemd service** (`/etc/systemd/system/optionsbot.service`) that
launches `main.py` directly with `Restart=always` and is `enabled` (auto-starts
on reboot, auto-restarts on crash):

```bash
sudo systemctl status optionsbot          # is it running?
sudo systemctl restart optionsbot         # restart
sudo systemctl stop optionsbot            # stop trading
journalctl -u optionsbot -f               # live logs
```

> **Important:** on the VM, systemd owns the bot. Control it with `systemctl`,
> **not** the dashboard's Start/Stop buttons — those launch/kill `main.py`
> independently and would fight systemd (risking two bot instances = double
> orders).

**Security:** the security group allows **inbound SSH (22) only**. The Streamlit
dashboard (8501) is never exposed to the internet — reach it through an SSH
tunnel: `ssh -L 8501:localhost:8501 -i Options-bot.pem ubuntu@<public-ip>`, then
open `http://localhost:8501` on your laptop.

### Equity sleeve (paper, runs alongside)

As of **2026-06-16** a second instance trades the **underlying shares** on the
same momentum signal (`INSTRUMENT=equity`), as a forward-validation sleeve. It
uses a **separate Alpaca paper account** (keys in `.env.equity`, gitignored) and
separate runtime files (`status_equity.json`, `stop_equity.flag`,
`trades_equity.csv`, etc.) so it never collides with the options bot or its
dashboard. Two systemd units (install commands are in each `.service` file):

```bash
journalctl -u optionsbot-equity -f             # equity bot (shares) live logs
sudo systemctl {status,restart,stop} optionsbot-equity
journalctl -u optionsbot-dashboard-equity -f   # its monitor-only dashboard
```

Its dashboard binds to **127.0.0.1:8502** — add a second forward to the tunnel
(`-L 8502:localhost:8502`) and open `http://localhost:8502`. Port 8501 stays the
options bot.

## Status & next steps

**Done**
- ✅ Migrated from old PC → laptop; `.venv` rebuilt, paper keys verified (Jun 2026).
- ✅ Deployed to AWS EC2, running 24/7 via systemd in paper mode.
- ✅ Smoke-tested on Linux — clean boot, same package versions as laptop.

**Next steps**
- [ ] **Set a billing budget** in AWS (Budgets → ~$45/mo alert) to avoid surprise charges.
- [ ] **Dashboard on the VM (monitoring only).** `app.py`'s `start_bot()` uses the
      Windows-only `subprocess.CREATE_NO_WINDOW` flag, which breaks on Linux — patch
      that, then run the dashboard as a second systemd service and view it over the
      SSH tunnel above.
- [ ] **(Optional) Elastic IP** to pin the public IP so it survives a stop/start.
- [ ] **Let paper mode run for weeks.** Accumulate **50 closed trades** so the ML
      learner can train, and build up IV-rank history (20 sessions) before judging
      signal quality.
- [ ] **Only consider live** after sustained paper validation — and even then, start
      with the smallest possible size. See the risk warnings at the top.

## Strategy

Two entry strategies share all filters and risk rules:

| Rule | Value |
|---|---|
| **Scalp** signal | >0.5% move on a single 15-min candle |
| Scalp confirm | RSI(5) > 65 (calls) / < 35 (puts) |
| **Runner** signal | ±2% from today's open AND still making new session highs/lows |
| Runner confirm | RSI(5) > 60 / < 40; runner take profit is +60% (scalp +40%) |
| Both | volume ≥ 1.5× 20-day avg |
| VWAP filter | calls only above session VWAP, puts only below |
| Earnings | single names skipped when earnings fall inside the DTE window |
| Contract | 1–3.5% OTM, 1–7 DTE, OI > 500, spread ≤ max($0.10, 5% of mid), IV rank ≤ 60 |
| Entry window | 9:45 AM – 3:30 PM ET only |
| Take profit | +40% on premium |
| Stop loss | −30% on premium |
| Trailing stop | after +20%, exit if 10 points are given back from the peak |
| Exit cadence | positions checked every 5s; scans every 30s |
| Time stop | flatten everything at 3:45 PM ET |
| Per trade | $50 max premium, 1 contract |
| Positions | 3 max concurrent |
| Daily halt | −$75 P&L → flatten + halt + Discord alert |
| Loss pause | 2 consecutive losses → 30-min pause |

**Crash recovery:** on startup the bot cancels stray orders, adopts any option
positions still held at Alpaca (so nothing is ever orphaned overnight), and
rebuilds today's P&L from `trades.csv` so the daily loss limit and pause logic
survive restarts.

## Architecture

```
app.py            Streamlit dashboard (start/stop, P&L, positions, settings)
main.py           orchestrator loop (30s cadence)
├── scanner.py        momentum signals (15-min candle + RSI + rel volume)
├── options_chain.py  chain fetch, OI/spread/IV-rank/budget filters
├── executor.py       order placement, TP/SL/time-stop exits, trades.csv log
├── risk_manager.py   daily loss halt, position caps, consecutive-loss pause
├── learner.py        ML win-probability filter trained on trades.csv
├── alerts.py         Discord webhooks (entry/exit/halt/pause/summary)
├── backtest.py       Black-Scholes backtest harness
├── config.py         every tunable in one place
└── utils.py          ET session-time helpers
```

Every closed trade is appended to `trades.csv`: ticker, option symbol, strike,
expiry, entry/exit price, P&L, entry/exit reason — plus the signal features at
entry (momentum, RSI, rel volume, IV, spread, DTE, time of day), which become
the learner's training data.

## ML learner — how the bot learns from its mistakes

`learner.py` trains a gradient-boosting classifier on your own closed trades:
*features at entry → did the trade win?* New signals are scored before entry.

- **Warm-up:** needs 50 closed trades before its first training run. Until
  then it just collects data (another reason to paper trade for a while).
- **Earned veto:** the model only gets to *block* entries once its
  cross-validated AUC clears 0.55 — i.e., it has demonstrated real predictive
  power on held-out trades. Below that it runs in advisory mode: P(win) is
  logged on every entry so you can judge it, but it can't veto. This stops a
  noise-fit model on small data from blocking good trades.
- **Gate:** when gating, entries scoring below `ML_WIN_PROB_THRESHOLD` (0.45)
  are skipped and logged.
- **Retraining:** automatic after every 10 new closed trades (end of day).
  Run `python learner.py` anytime to force a retrain and see AUC + top factors.
- Tune everything in `config.py` (`ML_*`); set `ML_ENABLED = False` to turn
  it off.

## Backtest output

`python backtest.py --days 60` prints: total trades, win rate, avg profit,
avg loss, total P&L, max drawdown, annualized Sharpe, and exit-reason
breakdown. Option prices are synthesized via Black-Scholes at fixed per-symbol
IVs with a $0.06 synthetic spread — real fills will be worse (IV crush, wider
spreads, slippage).

## Not financial advice

This is software, not investment advice. You are responsible for every order
it places. Test in paper mode.
