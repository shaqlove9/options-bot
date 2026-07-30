"""validate_swing.py — re-run the 45-DTE vol-gated credit spread on REAL option prices.

`backtest.py --swing` prices both legs with Black-Scholes off a realized-vol
estimate. That is fine for ranking structures against each other, but it cannot
answer the only question that matters before capital moves: **does the market
actually pay the credit the model assumes?** A BSM backtest of premium selling
fabricates the very premium it is trying to measure.

So this re-runs the SAME entry rule — vol-regime gate (top pctile of trailing
20d realized vol) + daily SMA regime — against REAL Alpaca historical option
bars, and reports the same net-E[R] yardstick so the two are directly comparable.

Deliberate differences from the BSM run, both of which make this HARDER, not
easier:
  * Alpaca daily option bars are TRADE prices, not NBBO quotes, so a per-leg
    `--slip` is crossed on all four legs. Premium selling dies on fills; an
    optimistic fill model is exactly how a VRP backtest fools you.
  * Stops are checked on the daily CLOSE of the actual spread, not against an
    intraday extreme of the underlying. The BSM run assumed a fill AT the stop
    level whenever the day's adverse extreme touched it — real short premium
    gaps straight through that. This removes that flattery.

HARD LIMIT: Alpaca options history starts ~Feb 2024, so this covers a shorter
window (and fewer trades) than the 4.25y BSM run. Fewer samples, but real prices.

Run:
  python -m iobot.validate_swing --universe SPY,QQQ,IWM --iv-rank-min 0.7
  python -m iobot.validate_swing --slip 0.05 --width 5 --k-sd 1.0
"""
from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
from alpaca.data.requests import OptionBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from iobot import bsm, strategies
from iobot.broker import build_clients

BARS_PER_YEAR = 252
CONTRACT_MULT = 100
DATA_FLOOR = dt.date(2024, 2, 1)      # Alpaca options history floor


@dataclass
class RealTrade:
    symbol: str
    direction: str
    otype: str
    entry_date: dt.date
    exit_date: dt.date
    expiry: dt.date
    short_strike: float
    long_strike: float
    credit: float
    debit: float
    pnl: float
    r_multiple: float
    max_loss: float
    exit_reason: str
    vol_rank: float


# ---------------- data ----------------

def fetch_daily_closes(client, symbol: str, start: dt.date, end: dt.date) -> pd.Series:
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
    """OCC 21-char option symbol, e.g. SPY240719P00530000."""
    return (f"{root}{expiry:%y%m%d}{'P' if otype == 'put' else 'C'}"
            f"{int(round(strike * 1000)):08d}")


def fetch_option_closes(oclient, symbols: list[str], start: dt.date,
                        end: dt.date) -> dict[str, pd.Series]:
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


# ---------------- scheduling / strikes ----------------

def third_fridays(start: dt.date, end: dt.date) -> list[dt.date]:
    out, y, m = [], start.year, start.month
    while dt.date(y, m, 1) <= end:
        d = dt.date(y, m, 15)
        d += dt.timedelta((4 - d.weekday()) % 7)
        if start <= d <= end:
            out.append(d)
        m, y = (m % 12 + 1, y + (m == 12))
    return out


def realized_vol(closes: pd.Series, i: int, window: int = 20) -> float:
    if i < window + 1:
        return float("nan")
    rets = np.diff(np.log(closes.iloc[i - window:i + 1].to_numpy()))
    return float(np.std(rets, ddof=1) * np.sqrt(BARS_PER_YEAR))


def vol_rank_at(closes: pd.Series, i: int, lookback: int = 252) -> float | None:
    """Percentile of today's 20d realized vol within its trailing history.
    Reuses `strategies._vol_rank` so the gate is byte-identical to the BSM run."""
    prior = pd.DataFrame({"close": closes.iloc[:i + 1].to_numpy()})
    if len(prior) < 40:
        return None
    saved = strategies.EM_IV_LOOKBACK
    strategies.EM_IV_LOOKBACK = lookback
    try:
        return strategies._vol_rank(prior)
    finally:
        strategies.EM_IV_LOOKBACK = saved


def pick_strikes(spot: float, rvol: float, dte: int, k_sd: float, width: float,
                 otype: str, inc: float, *, mode: str = "sd",
                 target_delta: float = 0.16, vrp: float = 1.1) -> tuple[float, float]:
    """Short strike, wing `width` further OTM.

    mode="sd"    -> k_sd realized-vol standard deviations OTM.
    mode="delta" -> the SAME `bsm.strike_for_delta` call `backtest.py --swing`
                    makes, with the same sigma (realized * vrp). This matters:
                    selecting by sd and selecting by delta give DIFFERENT strikes
                    (implied > realized, so a 1.0-realized-sd strike sits further
                    out than a 0.16-delta one and collects less). Only in "delta"
                    mode is the real-quote run a like-for-like validation of the
                    BSM run, isolating PRICE SOURCE as the single variable.
    """
    if mode == "delta":
        T = dte / 365.0
        sigma = max(1e-6, rvol * vrp)
        short = bsm.strike_for_delta(spot, T, sigma, target_delta, otype, increment=inc)
    else:
        move = spot * rvol * np.sqrt(dte / BARS_PER_YEAR)
        short = (round((spot - k_sd * move) / inc) * inc if otype == "put"
                 else round((spot + k_sd * move) / inc) * inc)
    return (float(short), float(short - width)) if otype == "put" \
        else (float(short), float(short + width))


# ---------------- simulation ----------------

def simulate_symbol(symbol, closes: pd.Series, oclient, *, dte, dte_min, dte_max,
                    k_sd, width, slip, commission, profit_target, stop_mult,
                    manage_dte, iv_rank_min, iv_rank_max, strike_inc,
                    strike_mode="sd", target_delta=0.16, vrp=1.1,
                    verbose=False) -> list[RealTrade]:
    trades: list[RealTrade] = []
    dates = list(closes.index)
    fridays = third_fridays(DATA_FLOOR, dates[-1] + dt.timedelta(days=dte_max + 40))
    busy_until: dt.date | None = None

    for i, today in enumerate(dates):
        if today < DATA_FLOOR or i < 60:
            continue
        if busy_until and today <= busy_until:
            continue
        rvol = realized_vol(closes, i)
        if not np.isfinite(rvol) or rvol <= 0:
            continue

        rank = vol_rank_at(closes, i)
        if rank is None or not (iv_rank_min <= rank <= iv_rank_max):
            continue

        spot0 = float(closes.iloc[i])
        sma = float(closes.iloc[max(0, i - strategies.EM_REGIME_SMA):i].mean())
        direction = "call" if spot0 > sma else "put"      # bullish -> sell puts
        otype = "put" if direction == "call" else "call"

        cands = [f for f in fridays if dte_min <= (f - today).days <= dte_max]
        if not cands:
            continue
        expiry = min(cands, key=lambda f: abs((f - today).days - dte))
        dte_entry = (expiry - today).days

        for nudge in (0, -1, 1, -2, 2, -5, 5):
            short_k, long_k = pick_strikes(spot0, rvol, dte_entry, k_sd, width,
                                           otype, strike_inc, mode=strike_mode,
                                           target_delta=target_delta, vrp=vrp)
            short_k += nudge * strike_inc
            long_k = short_k - width if otype == "put" else short_k + width
            if short_k <= 0 or long_k <= 0:
                continue
            sym_s = occ_symbol(symbol, expiry, otype, short_k)
            sym_l = occ_symbol(symbol, expiry, otype, long_k)
            px = fetch_option_closes(oclient, [sym_s, sym_l], today, expiry)
            if sym_s not in px or sym_l not in px:
                continue
            cs, cl = px[sym_s], px[sym_l]
            if today not in cs.index or today not in cl.index:
                continue

            # SELL the spread: hit the bid on the short, lift the ask on the wing
            credit = (float(cs.loc[today]) - slip) - (float(cl.loc[today]) + slip)
            if credit <= 0:
                break
            max_loss = width - credit
            if max_loss <= 0:
                break

            tgt = credit * (1.0 - profit_target)
            stp = min(width, credit * (1.0 + stop_mult))

            exit_date, debit, reason = expiry, None, "settled at expiry"
            fwd = [d for d in cs.index if d > today and d in cl.index]
            for d in fwd:
                days_left = (expiry - d).days
                val = float(cs.loc[d]) - float(cl.loc[d])       # mid-ish, trade px
                # stop first (conservative), then target — same order as the BSM run
                if val >= stp:
                    exit_date, debit, reason = d, val + 2 * slip, "stop (spread 2x credit)"
                    break
                if val <= tgt:
                    exit_date, debit, reason = d, val + 2 * slip, "profit target"
                    break
                if days_left <= manage_dte:
                    exit_date, debit, reason = d, val + 2 * slip, f"closed at {manage_dte}DTE"
                    break
            if debit is None:       # rode to expiry: settle at intrinsic, no closing legs
                last = fwd[-1] if fwd else today
                sT = float(closes.loc[last]) if last in closes.index else spot0
                intrinsic = (max(0.0, short_k - sT) - max(0.0, long_k - sT)) if otype == "put" \
                    else (max(0.0, sT - short_k) - max(0.0, sT - long_k))
                debit = min(width, max(0.0, intrinsic))
                exit_date = expiry
                legs = 2
            else:
                legs = 4
            debit = max(0.0, min(width, debit))

            fees = commission * legs
            pnl = (credit - debit) * CONTRACT_MULT - fees
            ml = max_loss * CONTRACT_MULT + fees
            trades.append(RealTrade(
                symbol, direction, otype, today, exit_date, expiry, short_k, long_k,
                credit, debit, pnl, pnl / ml if ml > 0 else 0.0, ml, reason, rank))
            busy_until = exit_date
            if verbose:
                print(f"    {symbol} {today} {otype} {short_k:g}/{long_k:g} exp {expiry} "
                      f"cr ${credit*100:.0f} -> {reason} P&L ${pnl:+.0f}")
            break
    return trades


def summarize(trades: list[RealTrade]) -> dict:
    if not trades:
        return {"n": 0}
    r = np.array([t.r_multiple for t in trades])
    pnl = np.array([t.pnl for t in trades])
    eq = np.cumsum(pnl)
    dd = float((np.maximum.accumulate(eq) - eq).max()) if len(eq) else 0.0
    wins = pnl > 0
    gl = -pnl[~wins].sum()
    reasons: dict[str, int] = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    return {"n": len(trades), "win_rate": float(wins.mean()),
            "expected_r": float(r.mean()), "median_r": float(np.median(r)),
            "total_pnl": float(pnl.sum()), "avg_pnl": float(pnl.mean()),
            "profit_factor": float(pnl[wins].sum() / gl) if gl > 0 else float("inf"),
            "max_dd": dd, "avg_win": float(pnl[wins].mean()) if wins.any() else 0.0,
            "avg_loss": float(pnl[~wins].mean()) if (~wins).any() else 0.0,
            "avg_credit": float(np.mean([t.credit for t in trades])) * 100,
            "cvar95": float(np.mean(np.sort(pnl)[:max(1, len(pnl) // 20)])),
            "worst": float(pnl.min()), "exit_reasons": reasons}


def report(s: dict, args):
    if s["n"] == 0:
        print("\nNo trades — no option data matched, or the gate admitted nothing.")
        return
    verdict = ("EDGE (net+)" if s["expected_r"] > 0.03 and s["total_pnl"] > 0
               else "NO EDGE (net<=0)" if s["expected_r"] <= 0
               else "MARGINAL (~breakeven, treat as no edge)")
    print(f"\n{'='*64}\nREAL-QUOTE RESULT  (slip=${args.slip}/leg, "
          f"comm=${args.commission}, gate {args.iv_rank_min:.0%}-{args.iv_rank_max:.0%})"
          f"\n{'='*64}")
    print(f"  trades            {s['n']}")
    print(f"  win rate          {s['win_rate']*100:.1f}%")
    print(f"  expected R (net)  {s['expected_r']:+.3f}   <- compare to the BSM run")
    print(f"  median R          {s['median_r']:+.3f}")
    print(f"  profit factor     {s['profit_factor']:.2f}")
    print(f"  total P&L         ${s['total_pnl']:+,.0f}  (avg ${s['avg_pnl']:+.1f}/trade)")
    print(f"  avg credit        ${s['avg_credit']:+.2f}   <- does the market PAY it?")
    print(f"  avg win / loss    ${s['avg_win']:+.1f} / ${s['avg_loss']:+.1f}")
    print(f"  max drawdown      ${s['max_dd']:,.0f}")
    print(f"  worst trade       ${s['worst']:+,.0f}")
    print(f"  CVaR(95%)         ${s['cvar95']:+,.0f}   <- tail, the number short premium lies about")
    print(f"  exits             {s['exit_reasons']}")
    print(f"  VERDICT           {verdict}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="45DTE vol-gated credit spread on REAL option prices")
    ap.add_argument("--universe", default="SPY,QQQ,IWM")
    ap.add_argument("--dte", type=int, default=45)
    ap.add_argument("--dte-min", type=int, default=30)
    ap.add_argument("--dte-max", type=int, default=60)
    ap.add_argument("--k-sd", type=float, default=1.0, help="short strike sd OTM (1.0 ~ 0.16 delta)")
    ap.add_argument("--width", type=float, default=5.0)
    ap.add_argument("--strike-inc", type=float, default=1.0)
    ap.add_argument("--slip", type=float, default=0.02, help="$ per leg crossed")
    ap.add_argument("--commission", type=float, default=0.65)
    ap.add_argument("--profit-target", type=float, default=0.50)
    ap.add_argument("--stop-mult", type=float, default=2.0)
    ap.add_argument("--manage-dte", type=int, default=21)
    ap.add_argument("--iv-rank-min", type=float, default=0.7)
    ap.add_argument("--iv-rank-max", type=float, default=1.0)
    ap.add_argument("--strike-mode", choices=["sd", "delta"], default="delta",
                    help="delta = same strikes backtest.py --swing picks (like-for-like)")
    ap.add_argument("--delta", type=float, default=0.16, help="target short-leg delta")
    ap.add_argument("--vrp", type=float, default=1.1, help="IV = realized * vrp, as in backtest.py")
    ap.add_argument("--slip-sweep", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--csv", default="")
    a = ap.parse_args(argv)

    clients = build_clients()
    universe = [s.strip().upper() for s in a.universe.split(",") if s.strip()]
    start = DATA_FLOOR - dt.timedelta(days=500)
    end = dt.date.today()

    print(f"Real-quote validation: {universe}  {a.dte}DTE  {a.k_sd}sd  ${a.width:g} wide  "
          f"vol gate {a.iv_rank_min:.0%}-{a.iv_rank_max:.0%}")
    print(f"Alpaca options history floor: {DATA_FLOOR}\n")

    all_trades: list[RealTrade] = []
    for sym in universe:
        closes = fetch_daily_closes(clients.stock_data, sym, start, end)
        if closes.empty:
            print(f"  {sym}: no underlying data, skipped")
            continue
        t = simulate_symbol(sym, closes, clients.option_data, dte=a.dte,
                            dte_min=a.dte_min, dte_max=a.dte_max, k_sd=a.k_sd,
                            width=a.width, slip=a.slip, commission=a.commission,
                            profit_target=a.profit_target, stop_mult=a.stop_mult,
                            manage_dte=a.manage_dte, iv_rank_min=a.iv_rank_min,
                            iv_rank_max=a.iv_rank_max, strike_inc=a.strike_inc,
                            strike_mode=a.strike_mode, target_delta=a.delta,
                            vrp=a.vrp, verbose=a.verbose)
        print(f"  {sym}: {len(t)} real-quote spreads")
        all_trades += t

    s = summarize(all_trades)
    report(s, a)

    if a.slip_sweep and all_trades:
        print(f"\n{'='*64}\nSLIPPAGE SENSITIVITY (re-simulated per level)\n{'='*64}")
        print(f"{'slip/leg':>10}{'n':>6}{'win%':>8}{'E[R]':>9}{'totP&L':>11}")
        for sl in (0.01, 0.02, 0.03, 0.05, 0.10):
            tt: list[RealTrade] = []
            for sym in universe:
                closes = fetch_daily_closes(clients.stock_data, sym, start, end)
                if closes.empty:
                    continue
                tt += simulate_symbol(sym, closes, clients.option_data, dte=a.dte,
                                      dte_min=a.dte_min, dte_max=a.dte_max, k_sd=a.k_sd,
                                      width=a.width, slip=sl, commission=a.commission,
                                      profit_target=a.profit_target, stop_mult=a.stop_mult,
                                      manage_dte=a.manage_dte, iv_rank_min=a.iv_rank_min,
                                      iv_rank_max=a.iv_rank_max, strike_inc=a.strike_inc,
                                      strike_mode=a.strike_mode, target_delta=a.delta,
                                      vrp=a.vrp)
            st = summarize(tt)
            if st["n"]:
                print(f"{sl:>10.2f}{st['n']:>6}{st['win_rate']*100:>7.1f}%"
                      f"{st['expected_r']:>+9.3f}{st['total_pnl']:>+11,.0f}")

    if a.csv and all_trades:
        pd.DataFrame([t.__dict__ for t in all_trades]).to_csv(a.csv, index=False)
        print(f"\nwrote {len(all_trades)} trades -> {a.csv}")
    return s


if __name__ == "__main__":
    main()
