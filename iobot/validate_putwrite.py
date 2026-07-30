"""validate_putwrite.py — SHORT PUTS (no wing) on REAL option prices.

Why this exists, when `validate_swing.py` already tests credit spreads: the
published long-run evidence for premium selling is for NAKED / cash-secured put
writing (the CBOE PUT index, history to 1986), NOT for credit spreads. The
documented reason is the wing — index put skew prices the further-OTM option
you BUY at a HIGHER implied vol than the one you sell, so the protective leg is
relatively expensive and eats the credit. `validate_swing.py` measured exactly
that: the market paid 62% of the flat-BSM credit.

So this drops the wing and tests the structure the evidence actually supports:
  * ONE leg, not four per round trip — the friction term that killed every
    spread run is cut by ~4x before anything else changes.
  * Same high-IV entry gate (`strategies._vol_rank`), which independently
    matches published SPX work finding that avoiding low-IV entries is where
    the improvement comes from.
  * **Fill quality is an explicit variable.** Every earlier model in this repo
    charged for CROSSING the spread (taking liquidity). The VRP literature is
    blunt that the premium is capturable mainly by "a patient trader who sells
    liquidity to the market, not takes it from the market". `--slip 0` models a
    patient mid fill; raise it to model paying up.

HONESTY REQUIREMENTS specific to naked puts — this structure lies differently:
  * Downside is large and one-sided. Win rate and average P&L flatter it worst.
    So the tail numbers (worst trade, CVaR95, max drawdown) are printed as
    decision metrics, not footnotes.
  * The benchmark is NOT zero, it is BUY-AND-HOLD. The PUT index's actual claim
    is equity-LIKE returns at lower volatility — a risk transformation, not
    alpha. A put-write that trails buy-and-hold on return AND on drawdown has
    no case at all, so both are reported side by side.
  * Capital is the real constraint: one SPY put ties up ~strike x 100 in
    collateral. Return is therefore reported on collateral, not per contract.

HARD LIMIT: Alpaca options history starts ~Feb 2024.

Run:
  python -m iobot.validate_putwrite --universe SPY --iv-rank-min 0.7
  python -m iobot.validate_putwrite --delta 0.30 --slip 0.0 --fill-sweep
"""
from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from iobot import bsm, strategies
from iobot.broker import build_clients
from iobot.validate_swing import (BARS_PER_YEAR, CONTRACT_MULT, DATA_FLOOR,
                                  fetch_daily_closes, fetch_option_closes,
                                  occ_symbol, realized_vol, third_fridays,
                                  vol_rank_at)


@dataclass
class PutTrade:
    symbol: str
    entry_date: dt.date
    exit_date: dt.date
    expiry: dt.date
    strike: float
    spot_entry: float
    spot_exit: float
    credit: float          # per share, received
    debit: float           # per share, paid to close (0 if expired worthless)
    collateral: float      # strike * 100 — the cash actually tied up
    pnl: float
    ret_on_collateral: float
    exit_reason: str
    vol_rank: float


def simulate_symbol(symbol, closes: pd.Series, oclient, *, dte, dte_min, dte_max,
                    target_delta, slip, commission, profit_target, manage_dte,
                    iv_rank_min, iv_rank_max, strike_inc, vrp, hold_to_expiry,
                    verbose=False) -> list[PutTrade]:
    trades: list[PutTrade] = []
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

        cands = [f for f in fridays if dte_min <= (f - today).days <= dte_max]
        if not cands:
            continue
        expiry = min(cands, key=lambda f: abs((f - today).days - dte))
        dte_entry = (expiry - today).days
        spot0 = float(closes.iloc[i])
        T = dte_entry / 365.0
        sigma = max(1e-6, rvol * vrp)
        strike = bsm.strike_for_delta(spot0, T, sigma, target_delta, "put",
                                      increment=strike_inc)
        if strike <= 0:
            continue

        entered = False
        for nudge in (0, -1, 1, -2, 2):
            k = strike + nudge * strike_inc
            sym = occ_symbol(symbol, expiry, "put", k)
            px = fetch_option_closes(oclient, [sym], today, expiry)
            if sym not in px or today not in px[sym].index:
                continue
            ps = px[sym]
            # SELL to open. slip=0 models a patient mid fill (providing liquidity);
            # raise it to model lifting/hitting the quote.
            credit = float(ps.loc[today]) - slip
            if credit <= 0.01:
                break
            collateral = k * CONTRACT_MULT

            exit_date, debit, reason, legs = expiry, None, "expired worthless", 1
            fwd = [d for d in ps.index if d > today]
            if not hold_to_expiry:
                tgt = credit * (1.0 - profit_target)
                for d in fwd:
                    if float(ps.loc[d]) <= tgt:
                        exit_date, debit, reason, legs = d, float(ps.loc[d]) + slip, \
                            "profit target", 2
                        break
                    if (expiry - d).days <= manage_dte:
                        exit_date, debit, reason, legs = d, float(ps.loc[d]) + slip, \
                            f"closed at {manage_dte}DTE", 2
                        break
            if debit is None:
                # Rode to expiry: settle at intrinsic. OTM = keep the whole credit
                # and pay no closing leg — the structural advantage of holding.
                sT = float(closes.loc[exit_date]) if exit_date in closes.index else \
                    float(closes.iloc[-1])
                debit = max(0.0, k - sT)
                reason = "expired worthless" if debit <= 0 else "assigned ITM"
                legs = 1
            sT = float(closes.loc[exit_date]) if exit_date in closes.index else spot0

            fees = commission * legs
            pnl = (credit - debit) * CONTRACT_MULT - fees
            trades.append(PutTrade(symbol, today, exit_date, expiry, k, spot0, sT,
                                   credit, debit, collateral, pnl,
                                   pnl / collateral, reason, rank))
            busy_until = exit_date
            entered = True
            if verbose:
                print(f"    {symbol} {today} {k:g}P exp {expiry} cr ${credit*100:.0f} "
                      f"-> {reason} P&L ${pnl:+.0f}")
            break
        if not entered:
            continue
    return trades


def buy_and_hold(closes: pd.Series, start: dt.date, end: dt.date) -> dict:
    """Benchmark: the PUT index claim is equity-LIKE returns at lower risk, so
    the put-write must be judged against holding the thing, not against zero."""
    s = closes[(closes.index >= start) & (closes.index <= end)]
    if len(s) < 2:
        return {}
    eq = s / s.iloc[0]
    peak = np.maximum.accumulate(eq.to_numpy())
    dd = float(((peak - eq.to_numpy()) / peak).max())
    yrs = max(1e-9, (s.index[-1] - s.index[0]).days / 365.25)
    total = float(eq.iloc[-1] - 1.0)
    rets = np.diff(np.log(s.to_numpy()))
    return {"total_return": total,
            "annualized": (1 + total) ** (1 / yrs) - 1,
            "max_dd": dd, "years": yrs,
            "vol": float(np.std(rets, ddof=1) * np.sqrt(BARS_PER_YEAR))}


def summarize(trades: list[PutTrade]) -> dict:
    if not trades:
        return {"n": 0}
    # Trades arrive grouped BY SYMBOL. Two things must be fixed before any
    # return or drawdown number means anything:
    #  1. order chronologically — otherwise the equity curve replays one symbol's
    #     whole history before the next starts, and the drawdown is fiction.
    #  2. capital is NOT one contract's collateral. Each symbol runs its own
    #     book continuously, so the portfolio must fund every symbol AT ONCE.
    #     Dividing total P&L by a single contract's collateral overstates the
    #     return by roughly the number of symbols traded.
    trades = sorted(trades, key=lambda t: t.exit_date)
    pnl = np.array([t.pnl for t in trades])
    eq = np.cumsum(pnl)
    peak = np.maximum.accumulate(eq)
    dd = float((peak - eq).max()) if len(eq) else 0.0
    wins = pnl > 0
    gl = -pnl[~wins].sum()
    per_symbol: dict[str, list[float]] = {}
    for t in trades:
        per_symbol.setdefault(t.symbol, []).append(t.collateral)
    avg_coll = float(sum(np.mean(v) for v in per_symbol.values()))
    yrs = max(1e-9, (trades[-1].exit_date - trades[0].entry_date).days / 365.25)
    total_ret = float(pnl.sum()) / avg_coll
    reasons: dict[str, int] = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    return {"n": len(trades), "win_rate": float(wins.mean()),
            "total_pnl": float(pnl.sum()), "avg_pnl": float(pnl.mean()),
            "avg_collateral": avg_coll,
            "total_return": total_ret,
            "annualized": (1 + total_ret) ** (1 / yrs) - 1 if total_ret > -1 else -1.0,
            "years": yrs,
            "profit_factor": float(pnl[wins].sum() / gl) if gl > 0 else float("inf"),
            "max_dd": dd, "max_dd_pct": dd / avg_coll,
            "worst": float(pnl.min()),
            "cvar95": float(np.mean(np.sort(pnl)[:max(1, len(pnl) // 20)])),
            "avg_credit": float(np.mean([t.credit for t in trades])) * 100,
            "exit_reasons": reasons}


def report(s: dict, bh: dict, args):
    if s["n"] == 0:
        print("\nNo trades — no option data matched, or the gate admitted nothing.")
        return
    print(f"\n{'='*68}\nPUT-WRITE ON REAL QUOTES  (delta {args.delta}, slip ${args.slip}/leg, "
          f"gate {args.iv_rank_min:.0%}-{args.iv_rank_max:.0%})\n{'='*68}")
    print(f"  trades            {s['n']}  over {s['years']:.2f} years")
    print(f"  win rate          {s['win_rate']*100:.1f}%")
    print(f"  avg credit        ${s['avg_credit']:+.2f}")
    print(f"  total P&L         ${s['total_pnl']:+,.0f}  (avg ${s['avg_pnl']:+.1f}/trade)")
    print(f"  capital required  ${s['avg_collateral']:,.0f}   <- ALL symbols funded concurrently")
    print(f"  return on collat  {s['total_return']*100:+.2f}%   "
          f"({s['annualized']*100:+.2f}%/yr)")
    print(f"  profit factor     {s['profit_factor']:.2f}")
    print(f"\n  --- TAIL (the numbers short premium lies about) ---")
    print(f"  max drawdown      ${s['max_dd']:,.0f}  ({s['max_dd_pct']*100:.1f}% of collateral)")
    print(f"  worst trade       ${s['worst']:+,.0f}")
    print(f"  CVaR(95%)         ${s['cvar95']:+,.0f}")
    print(f"  exits             {s['exit_reasons']}")
    if bh:
        print(f"\n  --- BENCHMARK: buy & hold the underlying, same window ---")
        print(f"  buy-hold return   {bh['total_return']*100:+.2f}%  "
              f"({bh['annualized']*100:+.2f}%/yr), vol {bh['vol']*100:.1f}%")
        print(f"  buy-hold max DD   {bh['max_dd']*100:.1f}%")
        beat_ret = s["annualized"] > bh["annualized"]
        beat_dd = s["max_dd_pct"] < bh["max_dd"]
        if beat_ret and beat_dd:
            v = "BEATS buy-and-hold on BOTH return and drawdown"
        elif beat_dd:
            v = "lower drawdown but LOWER return — a risk transformation, judge vs your goal"
        elif beat_ret:
            v = "higher return but DEEPER drawdown — not the PUT-index profile"
        else:
            v = "LOSES to buy-and-hold on both — no case"
        print(f"  VERDICT           {v}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="short put writing on REAL option prices")
    ap.add_argument("--universe", default="SPY")
    ap.add_argument("--dte", type=int, default=45)
    ap.add_argument("--dte-min", type=int, default=25)
    ap.add_argument("--dte-max", type=int, default=60)
    ap.add_argument("--delta", type=float, default=0.30, help="short put delta (PUT index ~ATM)")
    ap.add_argument("--strike-inc", type=float, default=1.0)
    ap.add_argument("--slip", type=float, default=0.0,
                    help="$/leg. 0 = patient mid fill (PROVIDING liquidity)")
    ap.add_argument("--commission", type=float, default=0.65)
    ap.add_argument("--profit-target", type=float, default=0.50)
    ap.add_argument("--manage-dte", type=int, default=21)
    ap.add_argument("--hold-to-expiry", action="store_true",
                    help="PUT-index style: no management, ride to settlement")
    ap.add_argument("--iv-rank-min", type=float, default=0.0)
    ap.add_argument("--iv-rank-max", type=float, default=1.0)
    ap.add_argument("--vrp", type=float, default=1.1)
    ap.add_argument("--fill-sweep", action="store_true",
                    help="sweep fill quality: mid -> paying up")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--csv", default="")
    a = ap.parse_args(argv)

    clients = build_clients()
    universe = [s.strip().upper() for s in a.universe.split(",") if s.strip()]
    start = DATA_FLOOR - dt.timedelta(days=500)
    end = dt.date.today()

    print(f"Put-write validation: {universe}  {a.dte}DTE  {a.delta} delta  "
          f"gate {a.iv_rank_min:.0%}-{a.iv_rank_max:.0%}  "
          f"{'hold to expiry' if a.hold_to_expiry else 'managed 50%/21DTE'}")
    print(f"Alpaca options history floor: {DATA_FLOOR}\n")

    all_trades: list[PutTrade] = []
    bh_all: dict = {}
    for sym in universe:
        closes = fetch_daily_closes(clients.stock_data, sym, start, end)
        if closes.empty:
            print(f"  {sym}: no underlying data, skipped")
            continue
        t = simulate_symbol(sym, closes, clients.option_data, dte=a.dte,
                            dte_min=a.dte_min, dte_max=a.dte_max,
                            target_delta=a.delta, slip=a.slip,
                            commission=a.commission, profit_target=a.profit_target,
                            manage_dte=a.manage_dte, iv_rank_min=a.iv_rank_min,
                            iv_rank_max=a.iv_rank_max, strike_inc=a.strike_inc,
                            vrp=a.vrp, hold_to_expiry=a.hold_to_expiry,
                            verbose=a.verbose)
        print(f"  {sym}: {len(t)} put-writes")
        all_trades += t
        if t and not bh_all:
            bh_all = buy_and_hold(closes, t[0].entry_date, t[-1].exit_date)

    s = summarize(all_trades)
    report(s, bh_all, a)

    if a.fill_sweep and all_trades:
        print(f"\n{'='*68}\nFILL-QUALITY SENSITIVITY — the lever never modelled before"
              f"\n{'='*68}")
        print(f"{'slip/leg':>10}{'n':>6}{'win%':>8}{'ret/yr':>10}{'maxDD%':>9}{'worst':>10}")
        for sl in (0.0, 0.02, 0.05, 0.10, 0.25):
            tt: list[PutTrade] = []
            for sym in universe:
                closes = fetch_daily_closes(clients.stock_data, sym, start, end)
                if closes.empty:
                    continue
                tt += simulate_symbol(sym, closes, clients.option_data, dte=a.dte,
                                      dte_min=a.dte_min, dte_max=a.dte_max,
                                      target_delta=a.delta, slip=sl,
                                      commission=a.commission,
                                      profit_target=a.profit_target,
                                      manage_dte=a.manage_dte,
                                      iv_rank_min=a.iv_rank_min,
                                      iv_rank_max=a.iv_rank_max,
                                      strike_inc=a.strike_inc, vrp=a.vrp,
                                      hold_to_expiry=a.hold_to_expiry)
            st = summarize(tt)
            if st["n"]:
                print(f"{sl:>10.2f}{st['n']:>6}{st['win_rate']*100:>7.1f}%"
                      f"{st['annualized']*100:>+9.2f}%{st['max_dd_pct']*100:>8.1f}%"
                      f"{st['worst']:>+10,.0f}")

    if a.csv and all_trades:
        pd.DataFrame([t.__dict__ for t in all_trades]).to_csv(a.csv, index=False)
        print(f"\nwrote {len(all_trades)} trades -> {a.csv}")
    return s


if __name__ == "__main__":
    main()
