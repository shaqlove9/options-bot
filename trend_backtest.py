"""trend_backtest.py — diversified time-series-momentum (trend-following) backtest.

Why this exists
---------------
The four prior sleeves all died the same way: high-turnover technical signals on
liquid large-caps, where the gross edge is ~0 and/or retail friction eats it.
Trend-following fails for the OPPOSITE reasons and is the most robustly documented
systematic edge there is (Moskowitz-Ooi-Pedersen "Time Series Momentum"; 100+
years, dozens of markets). It is LOW-turnover (monthly), so friction can't
dominate, and it harvests a behavioural/structural premium that persists.

This validates the SIGNAL cheaply on a diversified ETF basket (Alpaca data back
to 2016) that proxies a micro-futures universe across asset classes. A trend is a
trend whether expressed in SPY or /MES — if it survives here with honest costs,
execution translates to micro-futures (lower friction, leverage, 60/40 tax) for
the live account. NOTE on capital: a properly diversified micro-futures book needs
~$10-15k of margin; a $1k account holds ~1 contract, i.e. a high-variance subset
of what this backtest measures. The backtest is the system; small capital is a
concentrated sample of it.

Method (monthly, strictly causal)
---------------------------------
Each month-end: per asset, signal = sign of the trailing L-month return (long/short,
or long-only). Size inverse to trailing realized vol (risk parity), normalise to a
chosen gross leverage. Hold one month; rebalance. Costs charged on turnover. The
decision metrics are portfolio-level and crisis-aware: CAGR, vol, Sharpe, max
drawdown, worst month, % positive months, and returns isolated to the 2018/2020/
2022 crises (trend should be FLAT-to-POSITIVE there — that's its whole value).

Usage
-----
  python trend_backtest.py                       # 12-mo TSMOM, long/short, gross 1.0
  python trend_backtest.py --lookback 6 --mode long-short --gross 2
  python trend_backtest.py --long-only --cost-bps 5
  python trend_backtest.py --json out.json
"""
import argparse
import datetime as dt
import json
import sys

import numpy as np
import pandas as pd

import config

UNIVERSE = ["SPY", "QQQ", "IWM",        # US equity indices
            "EFA", "EEM",               # intl / EM equity
            "TLT", "IEF",               # long / mid US bonds
            "GLD", "SLV",               # precious metals
            "DBC", "USO",               # broad commodities / oil
            "UUP"]                      # US dollar
CRISES = {
    "2018 Q4 selloff": ("2018-10-01", "2018-12-31"),
    "2020 COVID crash": ("2020-02-01", "2020-03-31"),
    "2022 bear market": ("2022-01-01", "2022-10-31"),
}


def fetch_closes(client, symbols, start):
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    df = client.get_stock_bars(StockBarsRequest(
        symbol_or_symbols=symbols, timeframe=TimeFrame.Day,
        start=dt.datetime.combine(start, dt.time()))).df
    closes = {}
    for s in symbols:
        if s in df.index.get_level_values(0):
            x = df.loc[s]["close"].copy()
            x.index = pd.to_datetime(x.index).tz_localize(None).normalize()
            closes[s] = x
    return pd.DataFrame(closes).sort_index()


def compute_weights(mom_row: pd.Series, vol_row: pd.Series, mode: str,
                    gross: float) -> pd.Series:
    """Inverse-vol risk-parity weights from a momentum row, normalised to `gross`
    leverage. The SINGLE source of truth for the signal — backtest and live
    executor both call this, so live can't drift from what was validated."""
    sig = np.sign(mom_row)
    if mode == "long-only":
        sig = sig.clip(lower=0)
    raw = (sig / vol_row).replace([np.inf, -np.inf], np.nan)
    raw = raw.where(raw.notna() & mom_row.notna(), 0.0)
    gnorm = raw.abs().sum()
    return (raw / gnorm * gross) if gnorm > 0 else raw * 0.0


def latest_target_weights(closes: pd.DataFrame, *, lookback: int, mode: str,
                          gross: float, vol_window: int) -> pd.Series:
    """Target weights as of the most recent month-end — what the live executor
    rebalances toward. Same computation the backtest uses each step."""
    vol = closes.pct_change().rolling(vol_window).std() * np.sqrt(252)
    me = closes.resample("ME").last()
    me_vol = vol.reindex(me.index, method="ffill")
    mom = me / me.shift(lookback) - 1.0
    return compute_weights(mom.iloc[-1], me_vol.iloc[-1], mode, gross)


def backtest(closes: pd.DataFrame, *, lookback: int, mode: str, gross: float,
             vol_window: int, cost_bps: float) -> pd.DataFrame:
    """Returns a monthly frame with portfolio return (net of cost) and turnover."""
    daily_ret = closes.pct_change()
    # trailing annualised vol, sampled later at month ends (causal: uses data <= t)
    vol = daily_ret.rolling(vol_window).std() * np.sqrt(252)

    me = closes.resample("ME").last()                 # month-end prices
    me_vol = vol.reindex(me.index, method="ffill")
    mom = me / me.shift(lookback) - 1.0               # trailing L-month return

    rows = []
    prev_w = pd.Series(0.0, index=closes.columns)
    for i in range(lookback, len(me) - 1):
        t, t1 = me.index[i], me.index[i + 1]
        w = compute_weights(mom.iloc[i], me_vol.iloc[i], mode, gross)

        nxt = (me.loc[t1] / me.loc[t] - 1.0).reindex(w.index).fillna(0.0)
        gross_ret = float((w * nxt).sum())
        turnover = float((w - prev_w).abs().sum())
        cost = turnover * cost_bps / 1e4
        rows.append({"date": t1, "ret": gross_ret - cost,
                     "gross_ret": gross_ret, "turnover": turnover,
                     "n_long": int((w > 0).sum()), "n_short": int((w < 0).sum())})
        prev_w = w
    return pd.DataFrame(rows).set_index("date")


def stats(m: pd.DataFrame) -> dict:
    r = m["ret"]
    n_years = len(r) / 12
    eq = (1 + r).cumprod()
    cagr = eq.iloc[-1] ** (1 / n_years) - 1
    vol = r.std() * np.sqrt(12)
    sharpe = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else float("nan")
    dd = eq / eq.cummax() - 1
    crisis = {}
    for name, (a, b) in CRISES.items():
        seg = r[(r.index >= a) & (r.index <= b)]
        crisis[name] = float((1 + seg).prod() - 1) if len(seg) else float("nan")
    return {
        "months": len(r), "years": n_years,
        "cagr": float(cagr), "vol": float(vol), "sharpe": float(sharpe),
        "max_dd": float(dd.min()), "worst_month": float(r.min()),
        "best_month": float(r.max()), "pct_positive": float((r > 0).mean() * 100),
        "avg_turnover": float(m["turnover"].mean()),
        "final_mult": float(eq.iloc[-1]), "crisis": crisis,
    }


def bench_spy(closes: pd.DataFrame) -> dict:
    me = closes["SPY"].resample("ME").last()
    r = me.pct_change().dropna()
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    return {"cagr": float(eq.iloc[-1] ** (12 / len(r)) - 1),
            "sharpe": float(r.mean() / r.std() * np.sqrt(12)),
            "max_dd": float(dd.min())}


def render(s: dict, bench: dict, args):
    print("=" * 64)
    print("DIVERSIFIED TREND-FOLLOWING (time-series momentum) — ETF proxy")
    print("=" * 64)
    print(f"Universe: {len(UNIVERSE)} ETFs across equity/bond/metal/commodity/FX")
    print(f"{args.lookback}-mo lookback | {args.mode} | gross {args.gross:.1f}x "
          f"| vol-window {args.vol_window}d | cost {args.cost_bps:.0f}bps/turnover")
    print(f"Period: {s['months']} months (~{s['years']:.1f} yrs)")
    print("-" * 64)
    print(f"CAGR:            {s['cagr']*100:+.1f}%/yr   (×{s['final_mult']:.2f} over the run)")
    print(f"Volatility:      {s['vol']*100:.1f}%/yr")
    print(f"Sharpe:          {s['sharpe']:.2f}")
    print(f"Max drawdown:    {s['max_dd']*100:.1f}%")
    print(f"Worst month:     {s['worst_month']*100:+.1f}%   best {s['best_month']*100:+.1f}%")
    print(f"% positive mo:   {s['pct_positive']:.0f}%")
    print(f"Avg turnover:    {s['avg_turnover']:.2f}/mo  (×{args.cost_bps:.0f}bps = cost drag)")
    print("- - - crisis behaviour (trend should be FLAT-to-POSITIVE here) - - -")
    for name, v in s["crisis"].items():
        tag = "✓" if (np.isnan(v) or v >= -0.05) else "✗"
        print(f"  {tag} {name:<20} {v*100:+.1f}%")
    print("- - - vs buy-and-hold SPY (same window) - - -")
    print(f"  SPY: CAGR {bench['cagr']*100:+.1f}%/yr | Sharpe {bench['sharpe']:.2f} "
          f"| max DD {bench['max_dd']*100:.1f}%")
    print("=" * 64)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lookback", type=int, default=12, help="momentum lookback in months")
    ap.add_argument("--mode", choices=["long-short", "long-only"], default="long-short")
    ap.add_argument("--long-only", action="store_true", help="alias for --mode long-only")
    ap.add_argument("--gross", type=float, default=1.0, help="gross leverage (sum|w|)")
    ap.add_argument("--vol-window", type=int, default=60, help="trailing days for vol sizing")
    ap.add_argument("--cost-bps", type=float, default=5.0, help="cost per unit turnover (bps)")
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--json", metavar="PATH")
    args = ap.parse_args()
    if args.long_only:
        args.mode = "long-only"

    from alpaca.data.historical.stock import StockHistoricalDataClient
    client = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
    closes = fetch_closes(client, UNIVERSE, dt.date.fromisoformat(args.start))
    if closes.empty:
        print("No data.", file=sys.stderr)
        sys.exit(2)

    m = backtest(closes, lookback=args.lookback, mode=args.mode, gross=args.gross,
                 vol_window=args.vol_window, cost_bps=args.cost_bps)
    s = stats(m)
    bench = bench_spy(closes)
    render(s, bench, args)

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"generated": dt.datetime.now().isoformat(timespec="seconds"),
                       "args": vars(args), "stats": s, "benchmark": bench}, f,
                      indent=2, default=str)
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
