"""pairs_backtest.py — market-neutral pairs (statistical-arbitrage) backtest.

Why this exists
---------------
The strategy review concluded the durable edges for this capital band are
*structural* — mean-reversion / relative-value — not directional price
prediction (which is what both current sleeves do). This is the in-sample gate
for the first structural idea: a classic cointegration pairs trade. Long the
cheap leg, short the rich leg when their spread stretches, unwind when it
reverts. It is dollar- and (approximately) beta-neutral, so it makes money from
the spread reverting rather than from the market going up or down.

It deliberately reuses the EQUITY sleeve's data layer and cost model so the
numbers are directly comparable to the equity backtest and not flattered:

  - bars come from backtest.fetch_bars (15-min RTH bars, same as everything else)
  - every leg pays backtest.EQ_SPREAD_PER_SHARE/2 + EQ_SLIPPAGE_PER_SHARE per
    share, per side — so a pairs ROUND TRIP pays that friction FOUR times
    (open A, open B, close A, close B). Naive pairs backtests ignore this and
    look great; this one does not. Friction is the whole ballgame for pairs.

Pair screening (no statsmodels — numpy/scipy only)
--------------------------------------------------
For every symbol pair we run an Engle-Granger style check on the formation
window: OLS hedge ratio A ~ beta*B, then on the residual spread we compute
  - the Dickey-Fuller t-stat (more negative = more mean-reverting; compared to
    APPROXIMATE critical values, since we have no MacKinnon p-values here), and
  - the Ornstein-Uhlenbeck half-life of mean reversion (bars to decay halfway).
A pair is tradeable if returns are correlated, the DF t-stat clears the
threshold, and the half-life sits in a sane band. These are screened on a
formation slice and then traded out-of-sample on the rest, so the screen does
not peek at the trading period.

Trading rule (walk-forward, strictly causal)
--------------------------------------------
Per bar we use a TRAILING window (shifted one bar, so no same-bar look-ahead)
for the rolling hedge ratio and the spread's mean/std, then the z-score of the
current spread. Enter when |z| >= entry_z, unwind when |z| <= exit_z, bail when
|z| >= stop_z (the relationship broke). Optional max holding bars.

Usage
-----
  python pairs_backtest.py                       # screen + backtest the universe
  python pairs_backtest.py --days 180 --top 5    # trade the 5 best-screened pairs
  python pairs_backtest.py --entry-z 2.5 --exit-z 0.25 --stop-z 4
  python pairs_backtest.py --lookback 96 --max-hold-bars 156
  python pairs_backtest.py --formation-frac 0.4  # 40% of history screens, 60% trades
  python pairs_backtest.py --json out.json
"""
import argparse
import datetime as dt
import itertools
import json
import sys
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

import config
import backtest as bt

# Reuse the equity cost model verbatim so results are comparable + honest.
HALF_SPREAD = bt.EQ_SPREAD_PER_SHARE / 2
SLIP = bt.EQ_SLIPPAGE_PER_SHARE
GROSS_PER_PAIR = bt.EQ_CAPITAL * bt.EQ_LEVERAGE / config.MAX_OPEN_POSITIONS  # ~$1,333/pair leg budget

# Approximate Dickey-Fuller critical values (no constant-only case, large N).
# We have no statsmodels/MacKinnon p-values; these are the textbook asymptotic
# cutoffs and are used only as a coarse screen, never reported as exact.
ADF_CRIT = {"1%": -3.43, "5%": -2.86, "10%": -2.57}


@dataclass
class PairTrade:
    pair: str
    side: str                 # 'long_spread' (long A / short B) or 'short_spread'
    entry_time: dt.datetime
    exit_time: dt.datetime
    beta: float
    z_entry: float
    z_exit: float
    gross_notional: float
    pnl: float
    pnl_pct: float            # P&L as % of gross notional deployed
    exit_reason: str
    bars_held: int


# ---------------------------------------------------------------------------
# stationarity / cointegration screen (numpy only)
# ---------------------------------------------------------------------------

def _ols(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Least squares coefficients for y ~ X (X already includes any constant)."""
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def hedge_ratio(a: np.ndarray, b: np.ndarray) -> float:
    """OLS beta of a on b (no intercept on prices — spread carries the level)."""
    X = np.column_stack([b, np.ones_like(b)])
    beta, _intercept = _ols(a, X)
    return float(beta)


def df_tstat(spread: np.ndarray) -> float:
    """Dickey-Fuller t-stat from regressing Δs_t on s_{t-1} (+ constant).
    The t-stat on the lagged-level coefficient; more negative = more stationary."""
    s = np.asarray(spread, float)
    ds = np.diff(s)
    lag = s[:-1]
    X = np.column_stack([lag, np.ones_like(lag)])
    coef = _ols(ds, X)
    resid = ds - X @ coef
    dof = max(len(ds) - X.shape[1], 1)
    sigma2 = (resid @ resid) / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se_gamma = float(np.sqrt(sigma2 * xtx_inv[0, 0]))
    return float(coef[0] / se_gamma) if se_gamma > 0 else 0.0


def ou_halflife(spread: np.ndarray) -> float:
    """Ornstein-Uhlenbeck half-life of mean reversion, in bars.
    Regress Δs_t on s_{t-1}; lambda<0 => mean-reverting; half-life=-ln2/lambda."""
    s = np.asarray(spread, float)
    ds = np.diff(s)
    lag = s[:-1]
    X = np.column_stack([lag, np.ones_like(lag)])
    lam = _ols(ds, X)[0]
    if lam >= 0:
        return float("inf")          # not mean-reverting
    return float(-np.log(2) / lam)


def screen_pair(a: pd.Series, b: pd.Series) -> dict:
    """Engle-Granger style screen on a formation slice."""
    common = a.index.intersection(b.index)
    a, b = a.reindex(common), b.reindex(common)
    av, bv = a.to_numpy(float), b.to_numpy(float)
    beta = hedge_ratio(av, bv)
    spread = av - beta * bv
    ret_corr = float(np.corrcoef(np.diff(np.log(av)), np.diff(np.log(bv)))[0, 1])
    return {
        "beta": beta,
        "ret_corr": ret_corr,
        "df_tstat": df_tstat(spread),
        "half_life": ou_halflife(spread),
        "n": len(common),
    }


# ---------------------------------------------------------------------------
# walk-forward spread / z-score (strictly causal)
# ---------------------------------------------------------------------------

def rolling_zscore(a: pd.Series, b: pd.Series, lookback: int) -> pd.DataFrame:
    """Causal rolling hedge ratio + spread z-score. Every parameter at bar t is
    computed from data strictly BEFORE t (shift(1)), so there is no look-ahead."""
    common = a.index.intersection(b.index)
    a, b = a.reindex(common), b.reindex(common)
    cov = a.rolling(lookback).cov(b)
    var = b.rolling(lookback).var()
    beta = (cov / var).shift(1)
    spread = a - beta * b
    mu = spread.rolling(lookback).mean().shift(1)
    sd = spread.rolling(lookback).std().shift(1)
    z = (spread - mu) / sd
    return pd.DataFrame({"px_a": a, "px_b": b, "beta": beta,
                         "spread": spread, "mu": mu, "sd": sd, "z": z}).dropna()


# ---------------------------------------------------------------------------
# leg P&L with per-share friction on BOTH ends
# ---------------------------------------------------------------------------

def _leg_pnl(entry_px: float, exit_px: float, is_long: bool, shares: int) -> float:
    if is_long:                                   # buy at ask, later sell at bid
        entry_fill = entry_px + HALF_SPREAD + SLIP
        exit_fill = exit_px - HALF_SPREAD - SLIP
        return (exit_fill - entry_fill) * shares
    # short: sell at bid now, buy back at ask later
    entry_fill = entry_px - HALF_SPREAD - SLIP
    exit_fill = exit_px + HALF_SPREAD + SLIP
    return (entry_fill - exit_fill) * shares


def simulate_pair(pair: str, zdf: pd.DataFrame, *, lookback: int, entry_z: float,
                  exit_z: float, stop_z: float, max_hold_bars: int) -> list[PairTrade]:
    trades: list[PairTrade] = []
    pos = None
    idx = zdf.index.to_list()
    pxa_arr = zdf["px_a"].to_numpy(float)
    pxb_arr = zdf["px_b"].to_numpy(float)
    for i, ts in enumerate(idx):
        row = zdf.loc[ts]
        z, beta, pxa, pxb = row["z"], row["beta"], row["px_a"], row["px_b"]
        last_bar = i == len(idx) - 1

        if pos is not None:
            held = i - pos["i"]
            # Judge the exit on the spread we ACTUALLY hold (entry beta fixed for
            # the position's life, so P&L tracks it) but against a ROLLING mean/std
            # of that same fixed-beta spread, so the reversion target follows the
            # drifting equilibrium instead of a stale entry level.
            sd_i = pos["es_sd"][i]
            held_z = ((pos["es"][i] - pos["es_mu"][i]) / sd_i
                      if np.isfinite(sd_i) and sd_i > 0 else 0.0)
            reason = None
            if abs(held_z) <= exit_z:
                reason = "reversion"
            elif abs(held_z) >= stop_z:
                reason = "stop (spread blew out)"
            elif max_hold_bars and held >= max_hold_bars:
                reason = "time stop"
            elif last_bar:
                reason = "end of data"
            if reason:
                long_a = pos["side"] == "long_spread"
                pnl = (_leg_pnl(pos["pxa"], pxa, long_a, pos["sa"])
                       + _leg_pnl(pos["pxb"], pxb, not long_a, pos["sb"]))
                pnl_pct = pnl / pos["gross"] * 100 if pos["gross"] else 0.0
                trades.append(PairTrade(
                    pair, pos["side"], pos["t"], ts.to_pydatetime(), pos["beta"],
                    pos["z"], float(held_z), pos["gross"], pnl, pnl_pct, reason, held))
                pos = None
            if pos is not None:
                continue

        if last_bar or abs(z) < entry_z or beta <= 0:
            continue
        # size each leg ~half the pair budget; B shares beta-weighted (neutral)
        sa = int((GROSS_PER_PAIR / 2) // pxa)
        sb = int(round(sa * beta))
        if sa < 1 or sb < 1:
            continue
        side = "short_spread" if z > 0 else "long_spread"   # z>0: A rich -> short A
        gross = sa * pxa + sb * pxb
        # Precompute the entry-beta spread + its causal rolling mean/std once, so
        # exits are judged on the spread we actually hold (see the manage block).
        es = pxa_arr - beta * pxb_arr
        es_s = pd.Series(es)
        es_mu = es_s.rolling(lookback).mean().shift(1).to_numpy()
        es_sd = es_s.rolling(lookback).std().shift(1).to_numpy()
        pos = {"side": side, "sa": sa, "sb": sb, "pxa": pxa, "pxb": pxb,
               "gross": gross, "t": ts.to_pydatetime(), "z": float(z),
               "beta": float(beta), "es": es, "es_mu": es_mu, "es_sd": es_sd,
               "i": i}
    return trades


# ---------------------------------------------------------------------------
# stats / reporting
# ---------------------------------------------------------------------------

def compute_stats(trades: list[PairTrade]) -> dict:
    if not trades:
        return {"n": 0}
    df = pd.DataFrame([asdict(t) for t in trades])
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    wins, losses = df[df.pnl > 0], df[df.pnl <= 0]
    gl = abs(losses.pnl.sum())
    daily = df.groupby(df.exit_time.dt.date).pnl_pct.sum() / 100.0
    sharpe = (daily.mean() / daily.std() * np.sqrt(252)
              if len(daily) > 1 and daily.std() > 0 else float("nan"))
    eq = daily.cumsum()
    max_dd = float((eq - eq.cummax()).min() * 100)
    return {
        "n": len(df),
        "period": f"{df.entry_time.min()} → {df.exit_time.max()}",
        "win_rate": len(wins) / len(df) * 100,
        "avg_pnl_pct": df.pnl_pct.mean(),
        "avg_win_pct": wins.pnl_pct.mean() if len(wins) else float("nan"),
        "avg_loss_pct": losses.pnl_pct.mean() if len(losses) else float("nan"),
        "avg_bars_held": df.bars_held.mean(),
        "total_pnl_usd": df.pnl.sum(),
        "profit_factor": (wins.pnl.sum() / gl) if gl > 0 else float("inf"),
        "sharpe": sharpe,
        "max_dd_pct": max_dd,
        "exits": df.exit_reason.value_counts().to_dict(),
        "per_pair": {p: {"n": len(g), "pnl": g.pnl.sum(),
                         "avg_pnl_pct": g.pnl_pct.mean(),
                         "win_rate": (g.pnl > 0).mean() * 100}
                     for p, g in df.groupby("pair")},
    }


def _f(v, spec="{:.2f}"):
    if v is None or (isinstance(v, float) and (np.isnan(v))):
        return "n/a"
    return "∞" if v == float("inf") else spec.format(v)


def render(screen_rows: list[dict], stats: dict, args):
    print("=" * 64)
    print("MARKET-NEUTRAL PAIRS — statistical-arbitrage backtest")
    print("=" * 64)
    print(f"Universe: {', '.join(config.UNIVERSE)}")
    print(f"Formation {args.formation_frac:.0%} / trade {1 - args.formation_frac:.0%}"
          f" | lookback {args.lookback} bars | entry z {args.entry_z} "
          f"exit z {args.exit_z} stop z {args.stop_z}")
    print("-" * 64)
    print("PAIR SCREEN (formation slice; DF crit ≈ -2.86 @5%, approx):")
    print(f"  {'pair':<14}{'corr':>7}{'DF t':>8}{'half-life':>11}  traded?")
    for r in screen_rows:
        hl = "∞" if r["half_life"] == float("inf") else f"{r['half_life']:.0f}"
        print(f"  {r['pair']:<14}{r['ret_corr']:>7.2f}{r['df_tstat']:>8.2f}"
              f"{hl:>11}  {'YES' if r['traded'] else 'no'}")
    print("-" * 64)
    if stats.get("n", 0) == 0:
        print("No trades generated (no pair passed the screen, or z never triggered).")
        print("=" * 64)
        return
    print(f"Trades:          {stats['n']}   ({stats['period']})")
    print(f"Win rate:        {stats['win_rate']:.1f}%")
    print(f"Avg trade:       {_f(stats['avg_pnl_pct'], '{:+.3f}')}% of gross notional")
    print(f"  avg win/loss:  {_f(stats['avg_win_pct'], '{:+.3f}')}% / "
          f"{_f(stats['avg_loss_pct'], '{:+.3f}')}%")
    print(f"Avg hold:        {stats['avg_bars_held']:.1f} bars "
          f"(~{stats['avg_bars_held'] / 26:.1f} sessions)")
    print(f"Profit factor:   {_f(stats['profit_factor'])}")
    print(f"Sharpe (ann.):   {_f(stats['sharpe'])}")
    print(f"Max drawdown:    {_f(stats['max_dd_pct'])}%  (of deployed capital units)")
    print(f"Total P&L:       ${stats['total_pnl_usd']:+.2f}  "
          f"(gross ~${GROSS_PER_PAIR:,.0f}/pair — % rows are the comparable ones)")
    print(f"Exits:           {stats['exits']}")
    print("Per pair:")
    for p, g in stats["per_pair"].items():
        print(f"  {p:<14} {g['n']:>3} trades  {g['win_rate']:5.1f}% win  "
              f"{g['avg_pnl_pct']:+.3f}% avg  ${g['pnl']:+.2f}")
    print("=" * 64)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=180, help="history to pull")
    ap.add_argument("--lookback", type=int, default=120,
                    help="trailing bars for rolling hedge ratio + z-score")
    ap.add_argument("--entry-z", type=float, default=2.0)
    ap.add_argument("--exit-z", type=float, default=0.5)
    ap.add_argument("--stop-z", type=float, default=3.5)
    ap.add_argument("--max-hold-bars", type=int, default=0,
                    help="force-unwind after this many bars (0 = no cap)")
    ap.add_argument("--formation-frac", type=float, default=0.4,
                    help="fraction of history used to SCREEN pairs (rest is traded)")
    ap.add_argument("--min-corr", type=float, default=0.5)
    ap.add_argument("--max-half-life", type=int, default=300,
                    help="reject pairs that revert slower than this (bars)")
    ap.add_argument("--adf", type=float, default=ADF_CRIT["5%"],
                    help="DF t-stat threshold (more negative = stricter)")
    ap.add_argument("--top", type=int, default=0,
                    help="trade only the N best-screened pairs (0 = all that pass)")
    ap.add_argument("--json", metavar="PATH")
    args = ap.parse_args()

    from alpaca.data.historical import StockHistoricalDataClient
    client = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
    closes = {}
    for sym in config.UNIVERSE:
        bars = bt.fetch_bars(client, sym, args.days)
        if not bars.empty:
            closes[sym] = bars["close"]
    if len(closes) < 2:
        print("Need at least two symbols with data.", file=sys.stderr)
        sys.exit(2)

    # Split history: formation (screen) vs trade (out-of-sample).
    all_idx = sorted(set().union(*[s.index for s in closes.values()]))
    split = all_idx[int(len(all_idx) * args.formation_frac)]

    screen_rows, eligible = [], []
    for x, y in itertools.combinations(sorted(closes), 2):
        a, b = closes[x], closes[y]
        form_a, form_b = a[a.index < split], b[b.index < split]
        if len(form_a.index.intersection(form_b.index)) < args.lookback + 5:
            continue
        sc = screen_pair(form_a, form_b)
        passed = (sc["ret_corr"] >= args.min_corr
                  and sc["df_tstat"] <= args.adf
                  and 1 < sc["half_life"] <= args.max_half_life)
        row = {"pair": f"{x}/{y}", "x": x, "y": y, "traded": passed, **sc}
        screen_rows.append(row)
        if passed:
            eligible.append(row)

    # rank eligible pairs by how strongly mean-reverting (most negative DF first)
    eligible.sort(key=lambda r: r["df_tstat"])
    if args.top:
        chosen = eligible[:args.top]
        keep = {r["pair"] for r in chosen}
        for r in screen_rows:
            r["traded"] = r["pair"] in keep
    else:
        chosen = eligible

    all_trades: list[PairTrade] = []
    for r in chosen:
        a, b = closes[r["x"]], closes[r["y"]]
        trade_a, trade_b = a[a.index >= split], b[b.index >= split]
        zdf = rolling_zscore(trade_a, trade_b, args.lookback)
        all_trades += simulate_pair(r["pair"], zdf, lookback=args.lookback,
                                    entry_z=args.entry_z, exit_z=args.exit_z,
                                    stop_z=args.stop_z, max_hold_bars=args.max_hold_bars)

    stats = compute_stats(all_trades)
    render(sorted(screen_rows, key=lambda r: r["df_tstat"]), stats, args)

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"generated": dt.datetime.now().isoformat(timespec="seconds"),
                       "args": vars(args), "screen": screen_rows, "stats": stats},
                      f, indent=2, default=str)
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
