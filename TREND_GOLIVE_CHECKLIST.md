# Trend sleeve — paper → live go-live checklist

**Pre-committed gate. Written before the paper run, so the decision to risk real
money is made on rules, not on a hot streak or impatience.** Trend-following has
long flat/losing stretches by design; the temptation to flip live early (or to
quit during a drawdown) is the main way this fails. Do not edit these thresholds
to rescue a run — only tighten them.

## The strategy being validated
Diversified long-only time-series momentum, 12-month lookback, inverse-vol
weighting, monthly rebalance, on the 12-ETF cross-asset basket. Backtest (2016–
2026, honest costs): Sharpe ~0.9, CAGR ~+7%/yr, max drawdown ~−14%. This is a
MODEST edge — the goal is to capture it with discipline and survive, not to get
rich quick. See [[trend-sleeve]] / `trend_backtest.py`.

## Hard gates — ALL must be true before a single live dollar

- [ ] **Paper run ≥ 6 months / ≥ 6 monthly rebalances** completed via
      `trend_executor.py --execute` (paper). One or two rebalances prove nothing.
- [ ] **Live-paper tracks the backtest** within tolerance: realized return per
      rebalance within ~1.5× the backtest's monthly vol of the modeled return,
      and no unexplained gap (slippage/fills as expected). Build/run the
      forward scoreboard before deciding (mirror `forward_test.py`).
- [ ] **Drawdown stayed within budget**: the circuit breaker (`--max-dd`) never
      tripped from an *execution* problem (only from genuine market drawdown),
      and realized max DD ≤ 1.3× the backtest's (~−14% → ~−18% ceiling).
- [ ] **Execution is clean**: zero order errors across the paper run; fills,
      fractional handling, and rebalance turnover behave as modeled.
- [ ] **Capital is sufficient for the chosen vehicle.** A diversified micro-
      futures book needs ~$10–15k of margin. At $1k you can hold ~1 contract =
      a concentrated, high-variance subset of the validated system. Decide
      explicitly: (a) run concentrated and accept the variance, or (b) wait until
      capital supports ≥4–5 micro markets, or (c) start live on fractional ETFs.
- [ ] **You have sat through one real drawdown in paper without intervening.**
      The skill being tested is *yours*, not just the bot's. If you'd have turned
      it off, you are not ready to run it live.

## Going live is a deliberate, separate change
`trend_executor.py` is PAPER-ONLY by construction (refuses to run under
LIVE_MODE; no live order path exists). Enabling live is a code change that must
be made consciously, with size set to the smallest unit, after every box above
is checked. Initial live size: 1 micro contract (or the ETF-notional equivalent),
regardless of what the account could "afford" — prove it live small first.

## Ongoing discipline once live
- Rebalance on schedule only. No discretionary overrides, no skipping the
  rebalance because a position "feels" wrong.
- The circuit breaker is sacred. If it trips, it trips — review before
  `--reset-halt`, never reflexively.
- Review every rebalance against expectation monthly; log surprises as data.
- Add capital via contributions + compounding. Significant income is a function
  of capital × this modest edge × time — not leverage or a cleverer signal.
