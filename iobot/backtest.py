"""backtest.py — fast signal-iteration backtest for the iobot single-leg structure.

Replays the EXACT live signal (`signals.evaluate`) and exit levels
(`executor.single_exit_levels`) over historical underlying bars, and prices the
slightly-ITM long option with Black-Scholes (`bsm`) from the underlying path + an
IV assumption. Output is per-trade R-multiples and expected R *net of modeled
costs*, plus a cost sensitivity sweep — the same yardstick as the live gate, so a
backtest KEEP/KILL is comparable to the live one.

Run:
  python -m iobot.backtest --days 90 --path-tf 5 --dte 3 --half-spread 0.02 --vrp 1.1
  python -m iobot.backtest --days 365 --path-tf 15 --sweep
  python -m iobot.backtest --days 90 --csv bt_trades.csv

Caveats: BSM repricing approximates absolute P&L (no real option quotes); trust the
SIGN and the sweep, not the exact dollars. Fills assume the stop/target underlying
level is reached (slightly optimistic) — the cost knobs counter this.
"""
from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from iobot import bsm, config, strategies
from iobot.broker import build_clients
from iobot.clock import ET, _at, in_entry_window
from iobot.executor import single_exit_levels

R_FLOOR_IV, R_CAP_IV = 0.08, 1.0


@dataclass
class BTParams:
    days: int = 90
    path_tf: int = 5            # minutes for the intra-hold path
    dte: int = 3
    target_delta: float = 0.65
    half_spread: float = 0.02   # fraction of option price, each side
    fee_per_contract: float = 0.65
    vrp: float = 1.1            # IV = realized_vol * vrp
    strike_inc: float = 1.0


@dataclass
class Trade:
    symbol: str
    direction: str
    entry_time: dt.datetime
    exit_time: dt.datetime
    entry_underlying: float
    exit_underlying: float
    strike: float
    iv: float
    entry_opt: float
    exit_opt: float
    pnl: float
    r_multiple: float
    exit_reason: str


# ---------------- data ----------------

def _fetch(client, symbol: str, days: int, tf: TimeFrame) -> pd.DataFrame:
    start = dt.datetime.now(tz=ET) - dt.timedelta(days=days)
    df = client.get_stock_bars(StockBarsRequest(
        symbol_or_symbols=symbol, timeframe=tf, start=start)).df
    if df.empty:
        return df
    df = df.droplevel("symbol") if "symbol" in df.index.names else df
    df.index = df.index.tz_convert(ET)
    return df.between_time("09:30", "16:00")


def _fetch_daily(client, symbol: str, days: int) -> pd.DataFrame:
    start = dt.datetime.now(tz=ET) - dt.timedelta(days=days + config.VOLUME_LOOKBACK_DAYS * 2)
    df = client.get_stock_bars(StockBarsRequest(
        symbol_or_symbols=symbol, timeframe=TimeFrame.Day, start=start)).df
    return df.droplevel("symbol") if "symbol" in df.index.names else df


def _iv_by_day(daily: pd.DataFrame, vrp: float) -> dict[dt.date, float]:
    """Annualized realized vol (20d) * vrp, as-of the PRIOR close (no lookahead)."""
    rets = np.log(daily["close"] / daily["close"].shift(1))
    rv = rets.rolling(20).std() * np.sqrt(252) * vrp
    rv = rv.shift(1).clip(R_FLOOR_IV, R_CAP_IV)     # use yesterday's estimate today
    return {d.date(): float(v) for d, v in rv.items() if pd.notna(v)}


# ---------------- simulation ----------------

def _walk_exit(direction, stop, target, path: pd.DataFrame, force_close_t):
    """First stop/target touch on the path (stop checked first within a bar),
    else force-close at the last bar. Returns (exit_underlying, exit_time, reason)."""
    for ts, bar in path.iterrows():
        if direction == "call":
            if bar["low"] <= stop:
                return stop, ts, "stop (underlying)"
            if bar["high"] >= target:
                return target, ts, "profit target (underlying)"
        else:
            if bar["high"] >= stop:
                return stop, ts, "stop (underlying)"
            if bar["low"] <= target:
                return target, ts, "profit target (underlying)"
    if not path.empty:
        return float(path.iloc[-1]["close"]), path.index[-1], "time stop (EOD flat)"
    return None


def _simulate_symbol(symbol, sig_bars, path_bars, daily, p: BTParams, signal_fn) -> list[Trade]:
    iv_map = _iv_by_day(daily, p.vrp)
    trades: list[Trade] = []
    tf = dt.timedelta(minutes=15)
    for day in sorted({d for d in sig_bars.index.date}):
        day_bars = sig_bars[sig_bars.index.date == day]
        force_close_t = _at(dt.datetime.combine(day, dt.time()).replace(tzinfo=ET),
                            config.FORCE_CLOSE)
        sigma = iv_map.get(day)
        if sigma is None:
            continue
        trades_today = 0
        cursor = None  # ignore signals until after the last exit (one position at a time)
        for start in day_bars.index:
            close_time = start + tf
            if cursor is not None and close_time <= cursor:
                continue
            if not in_entry_window(close_time) or trades_today >= config.MAX_TRADES_PER_DAY:
                continue
            intraday = sig_bars[sig_bars.index <= start]
            daily_ctx = daily[daily.index.date < day]
            sig = signal_fn(symbol, intraday, daily_ctx, close_time)
            if sig is None:
                continue

            S0, otype = sig.spot, sig.direction
            T0 = p.dte / 365.0
            K = bsm.strike_for_delta(S0, T0, sigma, p.target_delta, otype, increment=p.strike_inc)
            entry_opt = bsm.price(S0, K, T0, sigma, otype)
            if entry_opt <= 0.01:
                continue
            if sig.stop_level is not None and sig.target_level is not None:
                stop, target = sig.stop_level, sig.target_level   # e.g. range scalp
            else:
                stop, target = single_exit_levels(otype, S0)
            path = path_bars[(path_bars.index > close_time)
                             & (path_bars.index <= force_close_t)
                             & (path_bars.index.date == day)]
            res = _walk_exit(otype, stop, target, path, force_close_t)
            if res is None:
                continue
            S_exit, exit_time, reason = res
            held = max(0.0, (exit_time - close_time).total_seconds())
            T_exit = max(0.0, T0 - held / 86400.0 / 365.0)
            exit_opt = bsm.price(S_exit, K, T_exit, sigma, otype)

            entry_fill = entry_opt * (1 + p.half_spread)
            exit_fill = exit_opt * (1 - p.half_spread)
            fees = p.fee_per_contract * 2 * config.QTY
            pnl = (exit_fill - entry_fill) * 100 * config.QTY - fees
            max_loss = entry_fill * 100 * config.QTY + p.fee_per_contract * config.QTY
            r = pnl / max_loss if max_loss > 0 else 0.0
            trades.append(Trade(symbol, otype, close_time, exit_time, S0, S_exit, K,
                                sigma, entry_fill, exit_fill, pnl, r, reason))
            trades_today += 1
            cursor = exit_time
    return trades


def load_data(p: BTParams) -> dict:
    """Fetch bars once so multiple strategies can be compared without refetching."""
    clients = build_clients()
    sig_tf = TimeFrame(15, TimeFrameUnit.Minute)
    path_tf = TimeFrame(p.path_tf, TimeFrameUnit.Minute)
    data = {}
    for symbol in config.UNIVERSE:
        sig_bars = _fetch(clients.stock_data, symbol, p.days, sig_tf)
        path_bars = _fetch(clients.stock_data, symbol, p.days, path_tf)
        daily = _fetch_daily(clients.stock_data, symbol, p.days)
        if sig_bars.empty or path_bars.empty or daily.empty:
            print(f"  {symbol}: insufficient data, skipped")
            continue
        data[symbol] = (sig_bars, path_bars, daily)
    return data


def run(p: BTParams, signal_fn, data: dict | None = None, verbose: bool = True) -> list[Trade]:
    data = data if data is not None else load_data(p)
    all_trades: list[Trade] = []
    for symbol, (sig_bars, path_bars, daily) in data.items():
        t = _simulate_symbol(symbol, sig_bars, path_bars, daily, p, signal_fn)
        if verbose:
            print(f"  {symbol}: {len(t)} trades over "
                  f"{sig_bars.index.min().date()}..{sig_bars.index.max().date()}")
        all_trades += t
    return all_trades


# ---------------- metrics ----------------

def summarize(trades: list[Trade]) -> dict:
    if not trades:
        return {"n": 0}
    r = np.array([t.r_multiple for t in trades])
    pnl = np.array([t.pnl for t in trades])
    equity = np.cumsum(pnl)
    peak = np.maximum.accumulate(equity)
    max_dd = float((peak - equity).max()) if len(equity) else 0.0
    wins = pnl > 0
    gross_win = pnl[wins].sum()
    gross_loss = -pnl[~wins].sum()
    reasons: dict[str, int] = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    return {
        "n": len(trades),
        "win_rate": float(wins.mean()),
        "expected_r": float(r.mean()),
        "median_r": float(np.median(r)),
        "r_std": float(r.std()),
        "total_pnl": float(pnl.sum()),
        "avg_pnl": float(pnl.mean()),
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "max_drawdown_$": max_dd,
        "avg_winner": float(pnl[wins].mean()) if wins.any() else 0.0,
        "avg_loser": float(pnl[~wins].mean()) if (~wins).any() else 0.0,
        "exit_reasons": reasons,
    }


def _print_summary(s: dict, p: BTParams):
    if s["n"] == 0:
        print("No trades.")
        return
    # Require a real margin, not noise: meaningful E[R] AND positive total P&L.
    if s["expected_r"] > 0.03 and s["total_pnl"] > 0:
        verdict = "EDGE (net+)"
    elif s["expected_r"] <= 0:
        verdict = "NO EDGE (net<=0)"
    else:
        verdict = "MARGINAL (~breakeven, treat as no edge)"
    print(f"\n{'='*60}\nRESULT  (dte={p.dte} half_spread={p.half_spread} vrp={p.vrp} "
          f"fee=${p.fee_per_contract})\n{'='*60}")
    print(f"  trades            {s['n']}")
    print(f"  win rate          {s['win_rate']*100:.1f}%")
    print(f"  expected R (net)  {s['expected_r']:+.3f}   <- the number that matters")
    print(f"  median R          {s['median_r']:+.3f}")
    print(f"  profit factor     {s['profit_factor']:.2f}")
    print(f"  total P&L         ${s['total_pnl']:+,.0f}  (avg ${s['avg_pnl']:+.1f}/trade)")
    print(f"  avg win / loss    ${s['avg_winner']:+.1f} / ${s['avg_loser']:+.1f}")
    print(f"  max drawdown      ${s['max_drawdown_$']:,.0f}")
    print(f"  exits             {s['exit_reasons']}")
    print(f"  VERDICT           {verdict}")


def _sweep(trades_base: list[Trade], p: BTParams):
    """Re-score the SAME entries/exits under different cost assumptions.
    Costs are applied post-hoc to the modeled option prices, so no re-sim needed."""
    print(f"\n{'='*60}\nCOST SENSITIVITY — expected R (net)\n{'='*60}")
    spreads = [0.01, 0.015, 0.02, 0.03, 0.05]
    print("half_spread:  " + "  ".join(f"{s:>6.3f}" for s in spreads))
    # group base trades by (entry_opt_raw, exit_opt_raw) — recover raw from filled.
    rows = []
    for t in trades_base:
        raw_entry = t.entry_opt / (1 + p.half_spread)
        raw_exit = t.exit_opt / (1 - p.half_spread)
        rows.append((raw_entry, raw_exit))
    er_line = []
    for hs in spreads:
        rs = []
        for raw_entry, raw_exit in rows:
            ef = raw_entry * (1 + hs)
            xf = raw_exit * (1 - hs)
            fees = p.fee_per_contract * 2 * config.QTY
            pnl = (xf - ef) * 100 * config.QTY - fees
            ml = ef * 100 * config.QTY + p.fee_per_contract * config.QTY
            rs.append(pnl / ml if ml > 0 else 0.0)
        er_line.append(float(np.mean(rs)) if rs else 0.0)
    print("expected R:   " + "  ".join(f"{e:>+6.3f}" for e in er_line))
    print("(flips to no-edge where expected R <= 0)")


def _compare(p: BTParams):
    """Run every registered strategy on the same data and rank by net expected R."""
    data = load_data(p)
    rows = []
    for name, fn in strategies.REGISTRY.items():
        s = summarize(run(p, fn, data=data, verbose=False))
        rows.append((name, s))
    rows.sort(key=lambda r: r[1].get("expected_r", -9), reverse=True)
    print(f"\n{'='*72}\nSTRATEGY COMPARISON  ({p.days}d, dte={p.dte}, "
          f"half_spread={p.half_spread})\n{'='*72}")
    print(f"{'strategy':<18}{'trades':>7}{'win%':>7}{'E[R]net':>9}{'PF':>6}"
          f"{'totP&L':>10}")
    for name, s in rows:
        if s["n"] == 0:
            print(f"{name:<18}{'0':>7}  (no trades)")
            continue
        print(f"{name:<18}{s['n']:>7}{s['win_rate']*100:>6.1f}{s['expected_r']:>+9.3f}"
              f"{s['profit_factor']:>6.2f}{s['total_pnl']:>+10.0f}")
    print("\n(E[R]net > ~0.05 across the cost sweep = worth graduating to live paper)")


def main(argv=None):
    ap = argparse.ArgumentParser(description="iobot single-leg backtest")
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--path-tf", type=int, default=5, help="intra-hold path tf (min)")
    ap.add_argument("--dte", type=int, default=3)
    ap.add_argument("--delta", type=float, default=0.65)
    ap.add_argument("--half-spread", type=float, default=0.02)
    ap.add_argument("--fee", type=float, default=0.65)
    ap.add_argument("--vrp", type=float, default=1.1)
    ap.add_argument("--signal", type=str, default="momentum",
                    choices=list(strategies.REGISTRY))
    ap.add_argument("--compare", action="store_true", help="run all strategies head-to-head")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--csv", type=str, default="")
    a = ap.parse_args(argv)
    p = BTParams(days=a.days, path_tf=a.path_tf, dte=a.dte, target_delta=a.delta,
                 half_spread=a.half_spread, fee_per_contract=a.fee, vrp=a.vrp)
    print(f"Backtest: universe={config.UNIVERSE} days={p.days} signal=15m path={p.path_tf}m")
    if a.compare:
        _compare(p)
        return
    print(f"Strategy: {a.signal}")
    trades = run(p, strategies.REGISTRY[a.signal])
    s = summarize(trades)
    _print_summary(s, p)
    if a.sweep:
        _sweep(trades, p)
    if a.csv and trades:
        pd.DataFrame([t.__dict__ for t in trades]).to_csv(a.csv, index=False)
        print(f"\nwrote {len(trades)} trades -> {a.csv}")
    return s


if __name__ == "__main__":
    main()
