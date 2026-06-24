# iobot — intraday options scalper (paper)

A "senior-trader" multi-factor **technical-analysis confluence** scalper for Alpaca
options. Trades a fixed **$1,000 sleeve**, aggressively, and learns from its own trade
history. **Paper only** — `broker.py` hard-pins the Alpaca paper endpoint; there is no
live-execution code in this build.

> ⚠️ Aggressive by design: up to 30% of the sleeve on a single position. Defined risk
> (a long option's max loss is the premium), but expect large swings. Not financial advice.

## What it does

Every 30s while the market is open it scans a universe of liquid, mid-priced optionable
names and fires the **confluence** signal only when ≥ `CONFLUENCE_MIN` (default 4 of 5)
technical factors agree on a direction:

| Factor | Bullish when… |
|---|---|
| **Trend** | fast EMA (9) over slow EMA (21) **and** price above session VWAP |
| **Regime** | daily close above its 20-day SMA (trade with the higher timeframe) |
| **Momentum** | MACD histogram positive **and** RSI not exhausted |
| **Structure** | breaks the recent intraday swing high (continuation) |
| **Volume** | relative volume ≥ `REL_VOLUME_MIN` (confirms the leading side) |

It buys a slightly-ITM long call/put (delta 0.60–0.70, 1–5 DTE), **dollar-sized** to
deploy up to `$300` per position (cheap names buy multiple lots; a contract over $300 is
skipped). Exits are ATR-based (stop `1.2·ATR`, target `1.8R`). Positions whose thesis is
intact are **held overnight**; only near-expiry (≤1 DTE) or `MAX_HOLD_DAYS`-old positions
are force-flattened at the EOD time-stop. Open positions are persisted to SQLite and
rehydrated on restart, so a restart never orphans a live contract.

## Run

The bot runs on an AWS EC2 VM as two systemd services (paper, 24/7):

```bash
sudo systemctl status  iobot.service            # the engine ( python -m iobot.engine )
sudo systemctl restart iobot.service            # picks up .env changes
sudo systemctl status  iobot-dashboard.service  # Streamlit monitor (localhost:8501)
tail -f iobot_data/iobot.log                     # live logs
```

Local / offline commands:

```bash
.venv/bin/python -m iobot.engine                 # run the engine
.venv/bin/python -m iobot.cli status             # latest engine status JSON
.venv/bin/python -m iobot.cli train              # retrain + walk-forward validate meta
.venv/bin/python -m iobot.cli gate               # validation-gate read-out (informational)
.venv/bin/python -m iobot.backtest --signal confluence --days 60          # sanity backtest
.venv/bin/python -m iobot.backtest --compare --days 365                   # all signals head-to-head
.venv/bin/python -m pytest iobot/tests -q                                  # test suite
```

The dashboard is monitoring-only and never exposed publicly — reach it via an SSH tunnel
(`ssh -L 8501:localhost:8501 …`).

## Configuration

Everything is a tunable in `iobot/config.py`, overridable by environment variable (set in
`.env`). Keys live under the `IOBOT_*` prefix. Highlights of the current live config:

| Setting | Default | Meaning |
|---|---|---|
| `IOBOT_SIGNAL` | `confluence` | active entry signal (key in `strategies.REGISTRY`) |
| `IOBOT_UNIVERSE` | 14 liquid names | scan list (SPY/QQQ for context; sub-$60 movers for fills) |
| `IOBOT_SLEEVE_CAPITAL` | `1000` | sleeve size; governor risk is a % of this, not the account |
| `IOBOT_POSITION_MAX_DOLLARS` | `300` | max deployed per position / max contract cost |
| `IOBOT_RISK_PCT` | `30` | per-trade max-loss cap, % of sleeve equity |
| `IOBOT_MAX_CONCURRENT` | `2` | concurrent positions |
| `IOBOT_DAILY_MAX_LOSS_PCT` / `IOBOT_TRAILING_DD_PCT` | `25` / `40` | kill-switches |
| `IOBOT_ALLOW_OVERNIGHT` | `true` | carry valid positions overnight |

Risk math runs off **sleeve equity** = `SLEEVE_CAPITAL + realized P&L`, so the bot behaves
as if it has $1,000 growing/shrinking with its own results — independent of the (large)
paper account balance.

## Architecture

```
iobot/
├── engine.py      main 30s loop: signal → features → governor → meta → select → execute
├── signals.py     Signal dataclass + shared TA helpers (RSI, VWAP, ATR, rel-volume)
├── strategies.py  pure entry signals (confluence + momentum/ORB/donchian baselines)
├── chain.py       slightly-ITM contract selection, $300 affordability filter, dollar sizing
├── executor.py    order placement, ATR stop/target, overnight carry, position persistence
├── governor.py    risk caps + kill switches (per-trade %, daily loss, trailing DD, counts)
├── features.py    pre-entry feature capture (leakage-guarded) for the learning loop
├── journal.py     trade log, reject log, shadow log, realized-P&L queries
├── meta.py        meta-labeling model (learns which setups win; shadow until it earns gating)
├── gate.py        informational validation read-out (never blocks paper entries)
├── store.py       single SQLite store (signals, features, trades, positions, …)
├── broker.py      Alpaca clients — PAPER ONLY (refuses any live endpoint)
├── webhook.py     optional TradingView alert receiver → same entry pipeline
├── backtest.py    Black-Scholes backtest harness (sanity / signal comparison)
├── bsm.py         Black-Scholes pricing for the backtest
├── dashboard.py   Streamlit monitor
└── cli.py         offline operator commands (status / train / gate / run)
```

## Learning loop

The bot improves from its own mistakes rather than from a pre-proven edge:

- `features.py` snapshots a leakage-guarded feature vector (including the confluence
  sub-scores) the instant each signal fires.
- `journal.py` logs every closed trade and its label (target hit before stop?).
- `meta.py` trains a meta-labeling model on `features ⋈ outcomes`. It runs in **shadow**
  (logs P(win), never blocks) until its walk-forward AUC clears the bar — then it can size
  and soft-filter entries. The validation **gate is informational only**; capital
  protection comes solely from the risk governor.

## TradingView webhook (optional)

When `IOBOT_WEBHOOK_ENABLED=true`, a stdlib HTTP server runs in a daemon thread
(`POST /tv-webhook`). Alerts are authenticated (mandatory secret, constant-time compare;
optional IP allowlist), de-duplicated, enriched into a `Signal`, and run through the
**same** `_handle_signal` pipeline as scanner signals — governor, meta, selection, and
risk all apply. The receiver thread only reads data and enqueues; it never touches the
executor/governor/meta. Put the endpoint behind TLS (a Cloudflare tunnel is scaffolded in
`deploy/`); the secret travels in the request body.

```bash
curl -X POST http://127.0.0.1:8080/tv-webhook -H 'Content-Type: application/json' \
  -d '{"secret":"YOUR_SECRET","id":"t1","symbol":"SPY","direction":"call"}'
curl http://127.0.0.1:8080/health
```
