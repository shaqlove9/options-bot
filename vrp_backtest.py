"""vrp_backtest.py — volatility-risk-premium sleeve: defined-risk short put spreads on SPY.

Why this exists
---------------
The strategy review's durable structural edge that we had NOT tested is the
volatility risk premium: option implied vol is, on average, richer than the
realized vol that follows, so selling insurance pays — regardless of market
direction. This is the in-sample gate for capturing it the safe way: a
DEFINED-RISK short put spread (sell a put ~1σ OTM, buy a further-OTM wing). Max
loss = width − credit, known at entry, so a vol spike can hurt but cannot ruin.

Honesty notes (read once — VRP lies differently than everything else)
---------------------------------------------------------------------
  - Short-vol makes small steady gains then occasionally loses a lot. Win rate
    and Sharpe FLATTER it worst of all — an 85% win rate over a sample with no
    crash is not an edge, it's a risk you haven't been paid to see yet. So the
    decision metrics here are TAIL-aware: max drawdown, worst trade, CVaR(95%),
    and P&L isolated to the vol-spike windows. Win rate is shown but labelled
    NON-DECISIVE.
  - Fills use REAL historical Alpaca option prices (not Black-Scholes with a
    guessed IV, which would fabricate the very premium we're measuring). The
    known approximation: daily option bars are TRADE prices, not NBBO quotes, so
    we model crossing a slippage on each of the 4 legs per round trip. Option
    spreads are the dominant cost in premium selling — an optimistic fill model
    is how VRP backtests fool you. Stress --slip to see the sensitivity.
  - Strikes are chosen by a realized-vol estimate of the 1σ move (no greeks),
    rounded to listed $5 increments; if a contract has no data we nudge the
    strike and retry, else the entry is skipped.

Usage
-----
  python vrp_backtest.py                       # default 45-DTE, 50%/21-DTE/2x mgmt
  python vrp_backtest.py --start 2024-02-01 --cadence-days 7
  python vrp_backtest.py --dte 45 --width 5 --k-sd 1.0 --slip 0.03
  python vrp_backtest.py --profit-target 0.5 --stop-mult 2 --manage-dte 21
  python vrp_backtest.py --json out.json
"""
import argparse
import datetime as dt
import json
import sys
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

import config

ROOT = "SPY"
BARS_PER_YEAR = 252
CONTRACT_MULT = 100          # one option contract = 100 shares
# Alpaca options launched ~Feb 2024; no usable option history before this.
DATA_FLOOR = dt.date(2024, 2, 1)


@dataclass
class VrpTrade:
    entry_date: dt.date
    exit_date: dt.date
    expiry: dt.date
    spot_entry: float
    short_strike: float
    long_strike: float
    contracts: int
    credit: float            # per-spread credit received (after slippage), $
    exit_value: float        # per-spread cost to close (after slippage), $
    max_loss: float          # defined risk per spread, $ (width - credit)
    pnl: float               # total $ across contracts
    pnl_pct_of_risk: float   # pnl / (max_loss * contracts) — comparable across sizes
    dte_entry: int
    days_held: int
    exit_reason: str


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def fetch_daily(client, symbol: str, start: dt.date, end: dt.date) -> pd.Series:
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    bars = client.get_stock_bars(StockBarsRequest(
        symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
        start=dt.datetime.combine(start, dt.time()),
        end=dt.datetime.combine(end, dt.time()))).df
    if bars.empty:
        return pd.Series(dtype=float)
    bars = bars.droplevel("symbol") if "symbol" in bars.index.names else bars
    s = bars["close"].copy()
    s.index = pd.to_datetime(s.index).date
    return s


def occ_symbol(root: str, expiry: dt.date, otype: str, strike: float) -> str:
    return f"{root}{expiry:%y%m%d}{'P' if otype == 'put' else 'C'}{int(round(strike * 1000)):08d}"


def fetch_option_closes(oclient, symbols: list[str], start: dt.date,
                        end: dt.date) -> dict[str, pd.Series]:
    from alpaca.data.requests import OptionBarsRequest
    from alpaca.data.timeframe import TimeFrame
    out: dict[str, pd.Series] = {}
    try:
        df = oclient.get_option_bars(OptionBarsRequest(
            symbol_or_symbols=symbols, timeframe=TimeFrame.Day,
            start=dt.datetime.combine(start, dt.time()),
            end=dt.datetime.combine(end, dt.time()))).df
    except Exception:
        return out
    if df.empty:
        return out
    for sym in symbols:
        if sym in df.index.get_level_values(0):
            s = df.loc[sym]["close"].copy()
            s.index = pd.to_datetime(s.index).date
            out[sym] = s
    return out


# ---------------------------------------------------------------------------
# scheduling / strike selection
# ---------------------------------------------------------------------------

def third_fridays(start: dt.date, end: dt.date) -> list[dt.date]:
    out, y, m = [], start.year, start.month
    while dt.date(y, m, 1) <= end:
        d = dt.date(y, m, 15)                       # 3rd Friday is the 15th-21st
        d += dt.timedelta((4 - d.weekday()) % 7)
        if start <= d <= end:
            out.append(d)
        m, y = (m % 12 + 1, y + (m == 12))
    return out


def realized_vol(closes: pd.Series, asof_idx: int, window: int = 20) -> float:
    """Annualized realized vol from the trailing `window` daily returns."""
    if asof_idx < window + 1:
        return float("nan")
    rets = np.diff(np.log(closes.iloc[asof_idx - window:asof_idx + 1].to_numpy()))
    return float(np.std(rets, ddof=1) * np.sqrt(BARS_PER_YEAR))


def pick_strikes(spot: float, rvol: float, dte: int, k_sd: float,
                 width: float) -> tuple[float, float]:
    """Short strike ~k_sd standard deviations OTM (put side), wing `width` below.
    Rounded to $5 increments (dense, reliably listed for SPY)."""
    sigma_move = spot * rvol * np.sqrt(dte / BARS_PER_YEAR)
    short = round((spot - k_sd * sigma_move) / 5) * 5
    return float(short), float(short - width)


# ---------------------------------------------------------------------------
# simulation
# ---------------------------------------------------------------------------

def simulate(spot: pd.Series, oclient, *, start: dt.date, cadence_days: int,
             dte: int, dte_min: int, dte_max: int, k_sd: float, width: float,
             slip: float, commission: float, profit_target: float,
             stop_mult: float, manage_dte: int, risk_per_trade: float,
             max_concurrent: int) -> list[VrpTrade]:
    trades: list[VrpTrade] = []
    dates = list(spot.index)
    fridays = third_fridays(start, dates[-1] + dt.timedelta(days=dte_max + 5))
    open_until: list[dt.date] = []          # expiries of live positions (concurrency)
    last_entry: dt.date | None = None

    for i, today in enumerate(dates):
        open_until = [e for e in open_until if e >= today]
        if today < max(start, DATA_FLOOR):
            continue
        if last_entry and (today - last_entry).days < cadence_days:
            continue
        if len(open_until) >= max_concurrent:
            continue
        rvol = realized_vol(spot, i)
        if not np.isfinite(rvol) or rvol <= 0:
            continue

        # nearest 3rd-Friday expiry within the DTE band
        cands = [f for f in fridays if dte_min <= (f - today).days <= dte_max]
        if not cands:
            continue
        expiry = min(cands, key=lambda f: abs((f - today).days - dte))
        dte_entry = (expiry - today).days
        spot0 = float(spot.iloc[i])

        # build the spread, retrying nearby strikes if data is missing
        entered = False
        for nudge in (0, -5, 5, -10, 10):
            short_k, long_k = pick_strikes(spot0, rvol, dte_entry, k_sd, width)
            short_k += nudge
            long_k = short_k - width
            sym_s = occ_symbol(ROOT, expiry, "put", short_k)
            sym_l = occ_symbol(ROOT, expiry, "put", long_k)
            closes = fetch_option_closes(oclient, [sym_s, sym_l], today, expiry)
            if sym_s not in closes or sym_l not in closes:
                continue
            cs, cl = closes[sym_s], closes[sym_l]
            if today not in cs.index or today not in cl.index:
                continue

            # entry: we SELL the spread -> receive credit, give up slippage/leg
            credit = (cs.loc[today] - slip) - (cl.loc[today] + slip)
            if credit <= 0:
                continue
            max_loss = width - credit
            if max_loss <= 0:
                continue
            contracts = int(risk_per_trade // (max_loss * CONTRACT_MULT))
            if contracts < 1:
                continue

            # manage day by day to expiry
            mgmt_dates = [d for d in cs.index if today < d <= expiry and d in cl.index]
            exit_value, exit_date, reason = None, expiry, "expiry"
            for d in mgmt_dates:
                # cost to BUY the spread back, paying slippage on each leg
                val = (cs.loc[d] + slip) - (cl.loc[d] - slip)
                val = max(val, 0.0)
                profit = credit - val
                dte_left = (expiry - d).days
                if profit >= profit_target * credit:
                    exit_value, exit_date, reason = val, d, "profit target"
                    break
                if profit <= -stop_mult * credit:
                    exit_value, exit_date, reason = min(val, width), d, "stop"
                    break
                if dte_left <= manage_dte:
                    exit_value, exit_date, reason = val, d, f"{manage_dte}-DTE roll"
                    break
            if exit_value is None:                  # held to expiry -> settle intrinsic
                s_exp = float(spot.loc[expiry]) if expiry in spot.index else spot0
                exit_value = max(0.0, min(short_k - s_exp, width))
                exit_date, reason = expiry, "expiry"

            pnl = (credit - exit_value) * CONTRACT_MULT * contracts
            pnl -= commission * 2 * 2 * contracts   # open+close, 2 legs
            risk_dollars = max_loss * CONTRACT_MULT * contracts
            trades.append(VrpTrade(
                today, exit_date, expiry, spot0, short_k, long_k, contracts,
                round(credit, 2), round(exit_value, 2), round(max_loss, 2),
                round(pnl, 2), pnl / risk_dollars if risk_dollars else 0.0,
                dte_entry, (exit_date - today).days, reason))
            open_until.append(expiry)
            last_entry = today
            entered = True
            break
        _ = entered
    return trades


# ---------------------------------------------------------------------------
# tail-aware stats
# ---------------------------------------------------------------------------

def spike_windows(spot: pd.Series, down_pct: float = 3.0) -> set[dt.date]:
    """Dates inside a vol spike: any day with a <= -down_pct% daily move, plus
    the 5 trading days after (the bleed)."""
    rets = spot.pct_change() * 100
    hits = [i for i, r in enumerate(rets) if r <= -down_pct]
    idx = list(spot.index)
    out: set[dt.date] = set()
    for h in hits:
        for j in range(h, min(h + 6, len(idx))):
            out.add(idx[j])
    return out


def compute_stats(trades: list[VrpTrade], spot: pd.Series) -> dict:
    if not trades:
        return {"n": 0}
    df = pd.DataFrame([asdict(t) for t in trades])
    pnl = df.pnl.to_numpy()
    wins = df[df.pnl > 0]
    spikes = spike_windows(spot)
    in_spike = df[df.entry_date.isin(spikes) | df.exit_date.isin(spikes)]
    # daily equity curve from per-trade pnl booked on exit date
    daily = df.groupby("exit_date").pnl.sum().sort_index()
    eq = daily.cumsum()
    dd = (eq - eq.cummax())
    var95 = float(np.percentile(pnl, 5))
    cvar95 = float(pnl[pnl <= var95].mean()) if (pnl <= var95).any() else var95
    total_risk = float((df.max_loss * CONTRACT_MULT * df.contracts).sum())
    return {
        "n": len(df),
        "period": f"{df.entry_date.min()} → {df.exit_date.max()}",
        "win_rate": len(wins) / len(df) * 100,        # NON-DECISIVE (see header)
        "total_pnl": float(pnl.sum()),
        "avg_pnl": float(pnl.mean()),
        "avg_pct_of_risk": float(df.pnl_pct_of_risk.mean() * 100),
        "worst_trade": float(pnl.min()),
        "best_trade": float(pnl.max()),
        "cvar95_trade": cvar95,                        # avg of worst 5% of trades
        "max_drawdown": float(dd.min()),
        "profit_factor": (wins.pnl.sum() / abs(df[df.pnl <= 0].pnl.sum())
                          if (df.pnl <= 0).any() else float("inf")),
        "return_on_risk_pct": pnl.sum() / total_risk * 100 if total_risk else 0.0,
        "avg_days_held": float(df.days_held.mean()),
        "exits": df.exit_reason.value_counts().to_dict(),
        "spike": {
            "n": len(in_spike),
            "pnl": float(in_spike.pnl.sum()) if len(in_spike) else 0.0,
            "worst": float(in_spike.pnl.min()) if len(in_spike) else 0.0,
        },
    }


def _f(v, spec="{:.2f}"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return "∞" if v == float("inf") else spec.format(v)


def render(stats: dict, args):
    print("=" * 64)
    print("VRP SLEEVE — defined-risk SHORT PUT SPREADS on SPY")
    print("=" * 64)
    print(f"{args.dte}-DTE target | width ${args.width:.0f} | short ~{args.k_sd:.1f}σ OTM "
          f"| slip ${args.slip:.2f}/leg")
    print(f"manage: {args.profit_target:.0%} profit / {args.manage_dte}-DTE / "
          f"{args.stop_mult:.0f}× stop | risk ${args.risk_per_trade:.0f}/trade "
          f"| cadence {args.cadence_days}d")
    print("-" * 64)
    if stats.get("n", 0) == 0:
        print("No trades generated (no option data in window, or no qualifying entry).")
        print("=" * 64)
        return
    print(f"Trades:          {stats['n']}   ({stats['period']})")
    print(f"Total P&L:       ${stats['total_pnl']:+,.2f}")
    print(f"Return on risk:  {_f(stats['return_on_risk_pct'], '{:+.1f}')}%  "
          f"(P&L ÷ total defined risk deployed)")
    print(f"Avg trade:       ${stats['avg_pnl']:+,.2f}   "
          f"({_f(stats['avg_pct_of_risk'], '{:+.1f}')}% of risk)")
    print(f"Avg days held:   {stats['avg_days_held']:.0f}")
    print(f"Win rate:        {stats['win_rate']:.1f}%   <- NON-DECISIVE for short-vol")
    print("- - - tail (this is the decision-relevant part) - - -")
    print(f"Worst trade:     ${stats['worst_trade']:+,.2f}")
    print(f"CVaR(95%) trade: ${stats['cvar95_trade']:+,.2f}   (avg of worst 5%)")
    print(f"Max drawdown:    ${stats['max_drawdown']:+,.2f}")
    print(f"Profit factor:   {_f(stats['profit_factor'])}")
    sp = stats["spike"]
    print(f"Vol-spike trades:{sp['n']:>3}   P&L ${sp['pnl']:+,.2f}   "
          f"worst ${sp['worst']:+,.2f}   <- did defined risk hold?")
    print(f"Exits:           {stats['exits']}")
    print("=" * 64)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2024-02-01", help="backtest start (YYYY-MM-DD)")
    ap.add_argument("--end", default="", help="backtest end (default: today)")
    ap.add_argument("--cadence-days", type=int, default=7, help="min days between entries")
    ap.add_argument("--dte", type=int, default=45, help="target days to expiry")
    ap.add_argument("--dte-min", type=int, default=25)
    ap.add_argument("--dte-max", type=int, default=60)
    ap.add_argument("--k-sd", type=float, default=1.0, help="short strike σ OTM (1≈16Δ)")
    ap.add_argument("--width", type=float, default=5.0, help="spread width in $")
    ap.add_argument("--slip", type=float, default=0.03, help="$ slippage per leg per side")
    ap.add_argument("--commission", type=float, default=0.0, help="$ per contract per leg")
    ap.add_argument("--profit-target", type=float, default=0.5, help="close at this fraction of credit")
    ap.add_argument("--stop-mult", type=float, default=2.0, help="stop at this × credit loss")
    ap.add_argument("--manage-dte", type=int, default=21, help="roll/close at this DTE")
    ap.add_argument("--risk-per-trade", type=float, default=500.0, help="$ defined risk budget")
    ap.add_argument("--max-concurrent", type=int, default=4, help="max overlapping positions")
    ap.add_argument("--json", metavar="PATH")
    args = ap.parse_args()

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end) if args.end else dt.date.today()
    if start < DATA_FLOOR:
        print(f"[note] clamping start to {DATA_FLOOR} (no Alpaca option data before).",
              file=sys.stderr)
        start = DATA_FLOOR

    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.historical.option import OptionHistoricalDataClient
    sclient = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
    oclient = OptionHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)

    spot = fetch_daily(sclient, ROOT, start - dt.timedelta(days=40), end)
    if spot.empty:
        print("No SPY data.", file=sys.stderr)
        sys.exit(2)

    trades = simulate(
        spot, oclient, start=start, cadence_days=args.cadence_days, dte=args.dte,
        dte_min=args.dte_min, dte_max=args.dte_max, k_sd=args.k_sd, width=args.width,
        slip=args.slip, commission=args.commission, profit_target=args.profit_target,
        stop_mult=args.stop_mult, manage_dte=args.manage_dte,
        risk_per_trade=args.risk_per_trade, max_concurrent=args.max_concurrent)

    stats = compute_stats(trades, spot)
    render(stats, args)

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"generated": dt.datetime.now().isoformat(timespec="seconds"),
                       "args": vars(args), "stats": stats,
                       "trades": [asdict(t) for t in trades]}, f, indent=2, default=str)
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
