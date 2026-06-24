# iobot — intraday options bot (paper-only)

A from-scratch intraday options bot. Daily cadence, **aggressive payoff shape with
every loss defined and capped**. Aggression comes from asymmetric payoff + entry
frequency inside a hard daily loss limit — never from undefined risk, naked shorts,
or oversizing. It assumes **no edge**: it ships paper-only behind a forward-validation
gate and there is **no live-execution code** in this build.

## Strategy
- **Primary structure:** one slightly-ITM long option (call/put by signal), strike
  chosen by delta (~0.60–0.70), short-dated (1–5 DTE; 0DTE off by default), intraday
  only — flat by the close. Risk = premium paid.
- **Upgrade structure (configurable):** debit vertical (bull call / bear put) once
  equity supports the spread margin. Max loss = net debit. Submitted/closed as one
  `mleg` order, never leg-by-leg.
- **Entry signal:** swappable `SignalSource` (default: a 15-min momentum + RSI +
  relative-volume + VWAP signal). No model ever generates signals.
- **Exits:** stop is a LEVEL on the **underlying** (market-sell the option when hit,
  whatever it's worth); target at `TARGET_R` × the risk distance; hard intraday
  time-stop / EOD flat. Spreads have no single-leg stop (risk is structural) — they
  exit on a profit target + time-stop.

## Capital protection (built regardless of signal quality)
Per-trade defined-risk cap (skip, never resize), buying-power / settled-funds check,
daily max-loss kill, sticky trailing-drawdown kill, and caps on trades/day, concurrent
positions, and intraday round-trips. `cash_account_mode` enforces one full-capital
round-trip/day on settled funds. See `governor.py`.

> Account context: the PDT rule was eliminated 2026-06-04 (exposure-based intraday
> margin). Brokers have until 2027-10-20 to implement it — verify your broker, and
> respect `IOBOT_MAX_ROUND_TRIPS`. A large intraday loss can curb the session.

## Meta-labeling layer (filters/sizes only — never signals)
- **Phase 1 (from day one):** snapshot a pre-entry feature vector keyed to `signal_id`.
  A leakage guard refuses any row whose newest input post-dates the capture instant.
- **Phase 2:** label (1 = target before stop), train a regularized classifier, validate
  with **purged + embargoed walk-forward CV**. A model is saved only if OOS AUC ≥ bar
  AND it beats take-every-signal on net E[R].
- **Phase 3:** shadow by default (logs would-skip/size); gates/sizes only behind the
  single `IOBOT_META_ACTIVE` flag once preconditions are met.

## Validation gate
Stays paper until ≥ `IOBOT_GATE_MIN_TRADES` closed trades AND positive expected R
**net of modeled fees + slippage**, recent half non-negative. One reversible
`IOBOT_GO_LIVE` flag records intent only — `broker.py` hard-pins the paper endpoint.

## Run
```bash
pip install -r iobot/requirements.txt
cp .env.iobot.example .env           # fill ALPACA paper keys
python -m iobot.engine               # or: python -m iobot.cli run
python -m iobot.cli train            # retrain + walk-forward validate meta model
python -m iobot.cli gate             # print the validation-gate verdict
streamlit run iobot/dashboard.py --server.port 8501 --server.address 127.0.0.1
pytest iobot/tests -q                 # unit tests (incl. leakage + mleg invariants)
```
Deploy: `deploy/iobot.service` (engine) and `deploy/iobot-dashboard.service` (monitor).
Runtime data lives in `iobot_data/` (sqlite db, logs, governor state, models).

## TradingView webhook (optional)
Set `IOBOT_WEBHOOK_ENABLED=true` (+ a mandatory `IOBOT_WEBHOOK_SECRET`) and the
engine starts a stdlib HTTP receiver in a daemon thread (`POST /tv-webhook`, no new
deps). An alert is authenticated (constant-time secret, optional
`IOBOT_WEBHOOK_IP_ALLOWLIST`), de-duped (`IOBOT_WEBHOOK_DEDUP_SEC`, 60s), enriched
into a `signals.Signal` from live bars, and queued. Each tick drains the queue and
runs every signal through the **same `_handle_signal` path** as scanner signals —
feature capture, governor gate, meta layer, contract selection and the per-trade
risk/BP check all apply. The server thread only reads data and enqueues; it never
touches the executor/governor/meta. Body: `{"secret","id","symbol","direction":"call|put","price"}`
(only `symbol`+`direction` required). If live data can't be fetched the signal is
taken **advisory** — meta gate skipped, all hard risk limits still enforced. Put the
public endpoint behind TLS (the secret travels in the body).

## Layout
`config` settings · `broker` paper clients · `signals` strategy interface ·
`chain` single-leg selection · `spread` vertical selection · `governor` risk/kill
switches · `features` Phase-1 capture · `store`/`journal` sqlite + logs · `executor`
orders + underlying-stops + mleg · `gate` validation · `meta` phases 2–3 ·
`webhook` TradingView receiver · `engine` main loop · `dashboard` monitor · `cli`
operator commands.
