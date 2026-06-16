"""backtest_vol.py — systematic short-volatility (VRP harvest) backtest.

Sells DEFINED-RISK option structures (iron condor or put credit spread) on a
liquid ETF at a regular cadence, prices the entry with Black-Scholes at an
assumed IMPLIED vol, and settles at expiry against the ACTUAL underlying close.

The volatility risk premium is real *if* the IV we sell at exceeds the move the
market actually realizes — so the result is highly sensitive to ENTRY_IV. That
is the whole ballgame, so we SWEEP it (run with --sweep) rather than trusting a
single assumption. Synthetic pricing; entry friction only (held to expiry).

Usage:  python backtest_vol.py --days 250 --structure condor --sweep
"""
import argparse
import datetime as dt
import math

import numpy as np
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

import config
from backtest import black_scholes  # reuse the same Black-Scholes pricer

RISK_FREE = 0.045

# ---- knobs (sweepable) ----
UNDERLYING = "SPY"
ENTRY_IV = 0.16              # implied vol we SELL at (the key assumption)
TENOR_DAYS = 7              # calendar days to expiry (weekly)
SHORT_SIGMA = 1.0           # short strikes this many std devs from spot (~16 delta)
WING_SIGMA = 1.0           # long (protection) strikes this many more sigma out
CONTRACTS = 1
SPREAD_FRAC = 0.04          # option bid/ask as a fraction of mid
MIN_SPREAD = 0.05
SLIPPAGE_PER_SIDE = 0.02
CAPITAL = 1000.0


def _leg_credit(spot, strike, tte, iv, otype):
    """Premium RECEIVED selling one option (mid - half spread - slippage)."""
    mid = black_scholes(spot, strike, tte, iv, otype)
    spread = max(MIN_SPREAD, mid * SPREAD_FRAC)
    return max(0.0, mid - spread / 2 - SLIPPAGE_PER_SIDE)


def _leg_debit(spot, strike, tte, iv, otype):
    """Premium PAID buying one protection leg (mid + half spread + slippage)."""
    mid = black_scholes(spot, strike, tte, iv, otype)
    spread = max(MIN_SPREAD, mid * SPREAD_FRAC)
    return mid + spread / 2 + SLIPPAGE_PER_SIDE


def fetch_daily(client, symbol, days):
    start = dt.datetime.now() - dt.timedelta(days=days + 10)
    bars = client.get_stock_bars(StockBarsRequest(
        symbol_or_symbols=symbol, timeframe=TimeFrame.Day, start=start)).df
    if bars.empty:
        return bars
    bars = bars.droplevel("symbol") if "symbol" in bars.index.names else bars
    return bars


def simulate_vol(daily: pd.DataFrame, structure: str = "condor"):
    """Open one defined-risk short-premium position per (non-overlapping) tenor,
    hold to expiry, settle at the actual underlying close. Returns list of dicts."""
    closes = daily["close"]
    dates = list(closes.index)
    trades = []
    tte = TENOR_DAYS / 365.0
    i = 0
    while i < len(dates):
        entry_date = dates[i]
        spot = float(closes.iloc[i])
        # find the bar at/after expiry
        expiry_target = entry_date + pd.Timedelta(days=TENOR_DAYS)
        future = [j for j in range(i + 1, len(dates)) if dates[j] >= expiry_target]
        if not future:
            break
        j = future[0]
        expiry_px = float(closes.iloc[j])

        sigma = spot * ENTRY_IV * math.sqrt(tte)           # 1 std move over tenor
        credit = 0.0
        max_loss_width = 0.0

        # put credit spread (bullish or the put wing of a condor)
        sp_k = spot - SHORT_SIGMA * sigma                  # short put
        lp_k = spot - (SHORT_SIGMA + WING_SIGMA) * sigma   # long put (protection)
        credit += _leg_credit(spot, sp_k, tte, ENTRY_IV, "put")
        credit -= _leg_debit(spot, lp_k, tte, ENTRY_IV, "put")
        put_width = sp_k - lp_k
        put_loss = max(0.0, sp_k - expiry_px) - max(0.0, lp_k - expiry_px)
        put_loss = min(put_loss, put_width)

        call_loss = 0.0
        call_width = 0.0
        if structure == "condor":
            sc_k = spot + SHORT_SIGMA * sigma              # short call
            lc_k = spot + (SHORT_SIGMA + WING_SIGMA) * sigma  # long call
            credit += _leg_credit(spot, sc_k, tte, ENTRY_IV, "call")
            credit -= _leg_debit(spot, lc_k, tte, ENTRY_IV, "call")
            call_width = lc_k - sc_k
            call_loss = max(0.0, expiry_px - sc_k) - max(0.0, expiry_px - lc_k)
            call_loss = min(call_loss, call_width)

        max_loss_width = max(put_width, call_width)
        pnl = (credit - put_loss - call_loss) * 100 * CONTRACTS
        trades.append({
            "entry": entry_date, "expiry": dates[j], "spot": spot,
            "expiry_px": expiry_px, "credit": credit * 100 * CONTRACTS,
            "pnl": pnl, "max_risk": (max_loss_width - credit) * 100 * CONTRACTS,
        })
        i = j  # non-overlapping: next position opens at this expiry
    return trades


def report(trades, label):
    if not trades:
        print(f"[{label}] no positions."); return
    df = pd.DataFrame(trades)
    wins = df[df.pnl > 0]
    total = df.pnl.sum()
    equity = CAPITAL + df.pnl.cumsum()
    dd = (equity - equity.cummax()).min()
    print(f"{label:18s} n={len(df):3d}  win={len(wins)/len(df)*100:4.1f}%  "
          f"avg=${df.pnl.mean():+6.2f}  total=${total:+8.2f} ({total/CAPITAL*100:+5.1f}%)  "
          f"worst=${df.pnl.min():+7.2f}  maxDD=${dd:+7.2f}  "
          f"avg_credit=${df.credit.mean():.0f} avg_risk=${df.max_risk.mean():.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=250)
    ap.add_argument("--structure", choices=["condor", "put_spread"], default="condor")
    ap.add_argument("--sweep", action="store_true", help="sweep ENTRY_IV")
    args = ap.parse_args()

    client = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
    daily = fetch_daily(client, UNDERLYING, args.days)
    print(f"{UNDERLYING}: {len(daily)} daily bars  "
          f"{daily.index.min().date()} → {daily.index.max().date()}  "
          f"({args.structure}, {TENOR_DAYS}d tenor, short@{SHORT_SIGMA}σ)")
    realized = daily["close"].pct_change().std() * math.sqrt(252)
    print(f"realized annualized vol over window ≈ {realized*100:.1f}%\n")

    global ENTRY_IV
    ivs = [0.10, 0.13, 0.16, 0.20, 0.25] if args.sweep else [ENTRY_IV]
    for iv in ivs:
        ENTRY_IV = iv
        report(simulate_vol(daily, args.structure), f"sell IV={iv:.0%}")


if __name__ == "__main__":
    main()
