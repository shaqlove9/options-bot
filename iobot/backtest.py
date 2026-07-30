"""backtest.py — fast signal-iteration backtest for the iobot single-leg structure.

Replays the EXACT live signal (`signals.evaluate`) and exit levels
(`executor.single_exit_levels`) over historical underlying bars, and prices the
slightly-ITM long option with Black-Scholes (`bsm`) from the underlying path + an
IV assumption. Output is per-trade R-multiples and expected R *net of modeled
costs*, plus a cost sensitivity sweep — the same yardstick as the live gate, so a
backtest KEEP/KILL is comparable to the live one.

`--mode credit` swaps the long option for a SHORT VERTICAL (see `credit.py`) on the
SAME entries, so the two structures are directly comparable: same signal, same
window, same one-position cursor, only the payoff differs. Short premium is scored
on the same net-E[R] yardstick, with a friction autopsy because that is where
credit spreads historically die (4 legs crossed instead of 2).

Run:
  python -m iobot.backtest --days 90 --path-tf 5 --dte 3 --half-spread 0.02 --vrp 1.1
  python -m iobot.backtest --days 365 --path-tf 15 --sweep
  python -m iobot.backtest --days 90 --csv bt_trades.csv
  python -m iobot.backtest --days 180 --mode credit --struct-sweep --sweep

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

from iobot import bsm, config, credit as creditmod, features as featmod, meta, strategies
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
    zero_dte: bool = False      # expiry = 16:00 TODAY, time-to-expiry from the clock
    ignore_entry_window: bool = False   # let a scheduled signal set its own entry time
    # Floor on per-leg half-spread in DOLLARS. A percentage alone flatters cheap
    # 0DTE legs, which cannot trade tighter than a penny-wide market.
    min_half_spread_abs: float = 0.005


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
    features: dict = field(default_factory=dict)   # pre-entry meta features (phase-1)


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


# ---------------- credit (short vertical) simulation ----------------

@dataclass
class CreditTrade:
    """A short-vertical round trip. Field names that `summarize` reads
    (pnl, r_multiple, exit_reason, entry_time) match `Trade` deliberately."""
    symbol: str
    direction: str            # the SIGNAL direction ("call" = bullish)
    otype: str                # leg type actually sold ("put" = bull put spread)
    entry_time: dt.datetime
    exit_time: dt.datetime
    entry_underlying: float
    exit_underlying: float
    short_strike: float
    long_strike: float
    width: float
    iv: float
    credit: float             # per share, RECEIVED (net of entry half-spread)
    debit: float              # per share, PAID to close (net of exit half-spread)
    mid_credit: float         # per share at MID — the frictionless ideal
    mid_debit: float          # per share at MID
    pnl: float
    r_multiple: float
    max_loss: float
    exit_reason: str

    @property
    def spread_friction(self) -> float:
        """Dollars lost to crossing the bid/ask on all four legs."""
        return ((self.mid_credit - self.credit) + (self.debit - self.mid_debit)) * 100.0

    @property
    def theory_pnl(self) -> float:
        """P&L if every leg filled at mid and fees were zero — the edge the
        structure would have in a frictionless market."""
        return (self.mid_credit - self.mid_debit) * 100.0


def _walk_exit_credit(q, credit: float, p: BTParams, cp, path: pd.DataFrame,
                      entry_time, T0: float, sigma: float):
    """Walk the path repricing the SPREAD (not the underlying) against
    credit-based thresholds. Stop is checked before target within a bar — the
    conservative convention `_walk_exit` uses for the single leg.

    Returns (exit_underlying, exit_time, debit_paid, mid_debit, reason).

    With `cp.manage=False` the position is held to expiry: no intraday target or
    stop, and the close is a SETTLEMENT — no bid/ask crossed and no closing fee,
    which is a genuine (and favourable) advantage of holding rather than trading
    out. `_simulate_symbol_credit` mirrors that by charging opening fees only.
    """
    tgt = creditmod.target_debit(credit, cp)
    stp = creditmod.stop_debit(credit, cp, q.width)
    if not cp.manage:
        if path.empty:
            return None
        ts = path.index[-1]
        S_end = float(path.iloc[-1]["close"])
        held = max(0.0, (ts - entry_time).total_seconds())
        T_now = max(0.0, T0 - held / 86400.0 / 365.0)
        intrinsic = creditmod.value_at(S_end, T_now, sigma, q)
        return S_end, ts, intrinsic, intrinsic, "settled at expiry"
    for ts, bar in path.iterrows():
        held = max(0.0, (ts - entry_time).total_seconds())
        T_now = max(0.0, T0 - held / 86400.0 / 365.0)

        # adverse extreme first: does the open loss hit the stop?
        S_bad = creditmod.adverse_price(bar, q.otype)
        mid_bad = creditmod.value_at(S_bad, T_now, sigma, q)
        if mid_bad >= stp:
            d = creditmod.exit_debit(stp, q, S_bad, T_now, sigma, p.half_spread, p.min_half_spread_abs)
            return S_bad, ts, d, mid_bad, "stop (spread 2x credit)"

        # favorable extreme: can we buy it back at the profit target?
        S_good = creditmod.favorable_price(bar, q.otype)
        mid_good = creditmod.value_at(S_good, T_now, sigma, q)
        if mid_good <= tgt:
            d = creditmod.exit_debit(tgt, q, S_good, T_now, sigma, p.half_spread, p.min_half_spread_abs)
            return S_good, ts, d, mid_good, f"profit target ({cp.profit_target_pct:.0f}% of credit)"

    if not path.empty:
        ts = path.index[-1]
        S_end = float(path.iloc[-1]["close"])
        held = max(0.0, (ts - entry_time).total_seconds())
        T_now = max(0.0, T0 - held / 86400.0 / 365.0)
        d = creditmod.exit_debit(0.0, q, S_end, T_now, sigma, p.half_spread, p.min_half_spread_abs)
        mid_end = creditmod.value_at(S_end, T_now, sigma, q)
        return S_end, ts, d, mid_end, "time stop (EOD flat)"
    return None


def _simulate_symbol_credit(symbol, sig_bars, path_bars, daily, p: BTParams,
                            cp, signal_fn) -> list[CreditTrade]:
    """Same entries as `_simulate_symbol` — same signal, same window, same
    one-position-at-a-time cursor — but expressed as a short vertical."""
    iv_map = _iv_by_day(daily, p.vrp)
    trades: list[CreditTrade] = []
    tf = dt.timedelta(minutes=15)
    for day in sorted({d for d in sig_bars.index.date}):
        day_bars = sig_bars[sig_bars.index.date == day]
        force_close_t = _at(dt.datetime.combine(day, dt.time()).replace(tzinfo=ET),
                            config.FORCE_CLOSE)
        sigma = iv_map.get(day)
        if sigma is None:
            continue
        trades_today = 0
        cursor = None
        for start in day_bars.index:
            close_time = start + tf
            if cursor is not None and close_time <= cursor:
                continue
            if not p.ignore_entry_window and not in_entry_window(close_time):
                continue
            if trades_today >= config.MAX_TRADES_PER_DAY:
                continue
            intraday = sig_bars[sig_bars.index <= start]
            daily_ctx = daily[daily.index.date < day]
            sig = signal_fn(symbol, intraday, daily_ctx, close_time)
            if sig is None:
                continue

            S0 = sig.spot
            if p.zero_dte:
                # 0DTE: real time-to-expiry off the clock. p.dte/365 collapses to
                # zero here, which would price every leg at intrinsic (no premium).
                expiry_dt = _at(dt.datetime.combine(day, dt.time()).replace(tzinfo=ET),
                                (16, 0))
                T0 = max(0.0, (expiry_dt - close_time).total_seconds()) / (365.0 * 86400.0)
                if T0 <= 0:
                    continue
            else:
                T0 = p.dte / 365.0

            em = creditmod.expected_move(S0, sigma) if cp.strike_mode == "expected_move" else None
            q = creditmod.select(S0, T0, sigma, sig.direction, cp, em=em)
            if q is None:
                continue
            credit = creditmod.entry_credit(q, p.half_spread, p.min_half_spread_abs)
            if credit <= 0.01:      # nothing left after crossing 2 spreads
                continue

            path = path_bars[(path_bars.index > close_time)
                             & (path_bars.index <= force_close_t)
                             & (path_bars.index.date == day)]
            res = _walk_exit_credit(q, credit, p, cp, path, close_time, T0, sigma)
            if res is None:
                continue
            S_exit, exit_time, debit, mid_debit, reason = res

            # Held to expiry the position settles: 2 opening fees, no closing leg.
            # Traded out it costs 2 legs on the way in and 2 on the way out.
            legs_charged = 2 if reason == "settled at expiry" else 4
            fees = p.fee_per_contract * legs_charged * config.QTY
            pnl = (credit - debit) * 100 * config.QTY - fees
            ml = creditmod.max_loss(credit, q.width, fees, config.QTY)
            r = pnl / ml if ml > 0 else 0.0
            trades.append(CreditTrade(
                symbol, sig.direction, q.otype, close_time, exit_time, S0, S_exit,
                q.short_strike, q.long_strike, q.width, sigma, credit, debit,
                q.mid_value, mid_debit, pnl, r, ml, reason))
            trades_today += 1
            cursor = exit_time
    return trades


def run_credit(p: BTParams, cp, signal_fn, data: dict | None = None,
               verbose: bool = True) -> list[CreditTrade]:
    data = data if data is not None else load_data(p)
    all_trades: list[CreditTrade] = []
    for symbol, (sig_bars, path_bars, daily) in data.items():
        t = _simulate_symbol_credit(symbol, sig_bars, path_bars, daily, p, cp, signal_fn)
        if verbose:
            print(f"  {symbol}: {len(t)} credit spreads over "
                  f"{sig_bars.index.min().date()}..{sig_bars.index.max().date()}")
        all_trades += t
    return all_trades


# ---------------- swing (multi-day) credit simulation ----------------
#
# The variance risk premium is a TERM-STRUCTURE effect: it is documented on
# ~30-45 day options, where there is time for implied to be wrong about realized.
# 0DTE has almost none. This path holds a 45-DTE short vertical across days and
# manages it like the classic structure (50% of credit, 2x stop, close at
# MANAGE_DTE remaining) — the horizon where premium selling is supposed to earn.
#
# Two modelling choices deliberately do NOT flatter the seller:
#   * the spread is repriced with the CURRENT day's vol, not the entry day's, so
#     a vol spike raises the buy-back cost and hurts the short exactly as it does
#     live (vega risk). Holding vol constant would hide the seller's worst risk.
#   * the stop is checked against the day's ADVERSE extreme (low for a bull put),
#     before the target, so an intraday breach counts even if the close recovers.

def _swing_exit(q, credit, p: BTParams, cp, daily_fwd, entry_sigma, iv_map,
                dte: int, manage_dte: int):
    """Walk forward day by day. Returns (S_exit, exit_date, debit, mid_debit,
    reason, days_held)."""
    tgt = creditmod.target_debit(credit, cp)
    stp = creditmod.stop_debit(credit, cp, q.width)
    for k, (ts, bar) in enumerate(daily_fwd.iterrows(), start=1):
        days_left = dte - k
        T_now = max(0.0, days_left / 365.0)
        sigma_now = iv_map.get(ts.date(), entry_sigma)

        if cp.manage:
            S_bad = creditmod.adverse_price(bar, q.otype)
            mid_bad = creditmod.value_at(S_bad, T_now, sigma_now, q)
            if mid_bad >= stp:
                d = creditmod.exit_debit(stp, q, S_bad, T_now, sigma_now,
                                         p.half_spread, p.min_half_spread_abs)
                return S_bad, ts, d, mid_bad, "stop (spread 2x credit)", k
            S_good = creditmod.favorable_price(bar, q.otype)
            mid_good = creditmod.value_at(S_good, T_now, sigma_now, q)
            if mid_good <= tgt:
                d = creditmod.exit_debit(tgt, q, S_good, T_now, sigma_now,
                                         p.half_spread, p.min_half_spread_abs)
                return S_good, ts, d, mid_good, "profit target (50% of credit)", k

        # management rule: flatten with `manage_dte` days left (gamma gets ugly after)
        if days_left <= manage_dte:
            S_end = float(bar["close"])
            mid = creditmod.value_at(S_end, T_now, sigma_now, q)
            d = creditmod.exit_debit(mid, q, S_end, T_now, sigma_now,
                                     p.half_spread, p.min_half_spread_abs)
            return S_end, ts, d, mid, f"closed at {manage_dte}DTE", k
        if days_left <= 0:
            S_end = float(bar["close"])
            mid = creditmod.value_at(S_end, 0.0, sigma_now, q)
            return S_end, ts, mid, mid, "settled at expiry", k
    return None


def _simulate_symbol_swing(symbol, daily, p: BTParams, cp, dte: int,
                           manage_dte: int) -> list[CreditTrade]:
    """One 45-DTE short vertical at a time, entered at the daily close when the
    vol gate and the SMA regime both allow it."""
    iv_map = _iv_by_day(daily, p.vrp)
    trades: list[CreditTrade] = []
    dates = list(daily.index)
    cursor_i = -1
    for i, ts in enumerate(dates):
        if i <= cursor_i or i < 60:
            continue
        day = ts.date()
        sigma = iv_map.get(day)
        if sigma is None:
            continue
        prior = daily.iloc[:i]                      # strictly before today: no lookahead
        if len(prior) < strategies.EM_REGIME_SMA:
            continue
        if strategies.EM_IV_RANK_MIN > 0.0 or strategies.EM_IV_RANK_MAX < 1.0:
            rank = strategies._vol_rank(prior)
            if rank is None or not (strategies.EM_IV_RANK_MIN <= rank
                                    <= strategies.EM_IV_RANK_MAX):
                continue

        S0 = float(daily.iloc[i]["close"])
        sma = float(prior["close"].tail(strategies.EM_REGIME_SMA).mean())
        direction = "call" if S0 > sma else "put"
        T0 = dte / 365.0
        # 45-DTE expected move scales with the horizon, not one session.
        em = (creditmod.expected_move(S0, sigma, sessions=dte * 252 / 365)
              if cp.strike_mode == "expected_move" else None)
        q = creditmod.select(S0, T0, sigma, direction, cp, em=em)
        if q is None:
            continue
        credit = creditmod.entry_credit(q, p.half_spread, p.min_half_spread_abs)
        if credit <= 0.01:
            continue

        fwd = daily.iloc[i + 1:]
        if fwd.empty:
            break
        res = _swing_exit(q, credit, p, cp, fwd, sigma, iv_map, dte, manage_dte)
        if res is None:
            break                                    # ran out of forward data
        S_exit, exit_ts, debit, mid_debit, reason, held = res

        legs_charged = 2 if reason == "settled at expiry" else 4
        fees = p.fee_per_contract * legs_charged * config.QTY
        pnl = (credit - debit) * 100 * config.QTY - fees
        ml = creditmod.max_loss(credit, q.width, fees, config.QTY)
        trades.append(CreditTrade(
            symbol, direction, q.otype, ts, exit_ts, S0, S_exit,
            q.short_strike, q.long_strike, q.width, sigma, credit, debit,
            q.mid_value, mid_debit, pnl, pnl / ml if ml > 0 else 0.0, ml, reason))
        cursor_i = i + held                          # one position at a time
    return trades


def run_swing(p: BTParams, cp, dte: int, manage_dte: int, data: dict,
              verbose: bool = True) -> list[CreditTrade]:
    all_trades: list[CreditTrade] = []
    for symbol, (_sig, _path, daily) in data.items():
        t = _simulate_symbol_swing(symbol, daily, p, cp, dte, manage_dte)
        if verbose:
            print(f"  {symbol}: {len(t)} swing spreads over "
                  f"{daily.index.min().date()}..{daily.index.max().date()}")
        all_trades += t
    return all_trades


def _capture_features(sig, all_daily: dict, day, trades_today: int) -> dict:
    """Build the SAME pre-entry feature vector the live bot captures, using only
    data observable at the signal instant (daily frames sliced to before `day`).
    Sequence features (win/loss streak) are filled later in global trade order; here
    they are 0. `featmod.build` enforces the leakage guard."""
    daily_ctx = {s: df[df.index.date < day] for s, df in all_daily.items()}
    ctx = featmod.RegimeContext(daily=daily_ctx, vix=None,
                                win_streak=0, loss_streak=0, trades_today=trades_today)
    return featmod.build(sig, ctx).features


def _simulate_symbol(symbol, sig_bars, path_bars, daily, p: BTParams, signal_fn,
                     all_daily: dict | None = None) -> list[Trade]:
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
            feats = _capture_features(sig, all_daily, day, trades_today) if all_daily else {}
            trades.append(Trade(symbol, otype, close_time, exit_time, S0, S_exit, K,
                                sigma, entry_fill, exit_fill, pnl, r, reason, feats))
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


def load_daily_only(p: BTParams, extra_days: int = 420) -> dict:
    """Swing mode needs no intraday bars but a LOT more daily history: the vol
    percentile looks back 252 sessions before the window even starts. Shaped like
    `load_data` so the same runners consume it."""
    clients = build_clients()
    data = {}
    for symbol in config.UNIVERSE:
        daily = _fetch_daily(clients.stock_data, symbol, p.days + extra_days)
        if daily.empty:
            print(f"  {symbol}: insufficient data, skipped")
            continue
        daily.index = pd.to_datetime(daily.index)
        data[symbol] = (None, None, daily)
    return data


def run(p: BTParams, signal_fn, data: dict | None = None, verbose: bool = True) -> list[Trade]:
    data = data if data is not None else load_data(p)
    all_daily = {sym: d[2] for sym, d in data.items()}
    all_trades: list[Trade] = []
    for symbol, (sig_bars, path_bars, daily) in data.items():
        t = _simulate_symbol(symbol, sig_bars, path_bars, daily, p, signal_fn, all_daily)
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
    dte_lbl = "0 (intraday clock)" if p.zero_dte else str(p.dte)
    print(f"\n{'='*60}\nRESULT  (dte={dte_lbl} half_spread={p.half_spread} vrp={p.vrp} "
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


# ---------------- credit reporting ----------------

def _print_credit_summary(trades: list, s: dict, p: BTParams, cp):
    """Standard summary plus the friction autopsy — for short premium the
    question is never 'did it win', it is 'did the premium survive the spread'."""
    if s["n"] == 0:
        print("No credit spreads taken.")
        return
    _print_summary(s, p)
    theory = float(np.mean([t.theory_pnl for t in trades]))
    friction = float(np.mean([t.spread_friction for t in trades]))
    # Fees ACTUALLY charged: settled-at-expiry trades pay opening legs only, so
    # this must be derived from the trades, not assumed to be 4 legs.
    legs = [2 if t.exit_reason == "settled at expiry" else 4 for t in trades]
    fees = p.fee_per_contract * float(np.mean(legs)) * config.QTY
    avg_credit = float(np.mean([t.credit for t in trades])) * 100
    avg_ml = float(np.mean([t.max_loss for t in trades]))
    print(f"\n{'-'*60}\nFRICTION AUTOPSY  (short_delta={cp.short_delta} "
          f"width=${cp.width:g} tp={cp.profit_target_pct:.0f}% stop={cp.stop_mult:g}x)"
          f"\n{'-'*60}")
    print(f"  avg credit taken  ${avg_credit:+.2f}   (max profit per spread)")
    print(f"  avg capital risk  ${avg_ml:.2f}   (width - credit, the R denominator)")
    print(f"  avg P&L at MID    ${theory:+.2f}   <- frictionless edge of the structure")
    print(f"  avg spread cost   ${-friction:+.2f}   (4 legs crossed)")
    print(f"  fees per spread   ${-fees:+.2f}   ({np.mean(legs):.1f} legs x ${p.fee_per_contract})")
    print(f"  avg NET P&L       ${theory - friction - fees:+.2f}")
    if theory > 0:
        pct = (friction + fees) / theory * 100
        print(f"  friction eats     {pct:.0f}% of the frictionless edge")
    else:
        print("  (structure is negative even at mid — friction is not the binding issue)")


def _credit_struct_sweep(p: BTParams, cp, signal_fn, data: dict):
    """Grid over the two knobs that define a short vertical: how far OTM the
    short leg sits, and how wide the protection is. The VRP backtest found its
    only positive result on a single narrow ridge, so the SHAPE of this grid
    matters more than any one cell — a lone positive surrounded by negatives is
    sampling variance, not alpha."""
    from dataclasses import replace
    deltas = [0.16, 0.20, 0.30, 0.40]
    widths = [1.0, 2.0, 5.0]
    print(f"\n{'='*72}\nCREDIT STRUCTURAL SWEEP — net E[R]  (n in parens)\n{'='*72}")
    print(f"{'short_delta':<14}" + "".join(f"{'w=$'+format(w,'g'):>18}" for w in widths))
    for d in deltas:
        cells = []
        for w in widths:
            cpx = replace(cp, short_delta=d, width=w)
            t = run_credit(p, cpx, signal_fn, data=data, verbose=False)
            st = summarize(t)
            cells.append(f"{st['expected_r']:+.3f} ({st['n']})" if st["n"] else "-- (0)")
        print(f"{d:<14.2f}" + "".join(f"{c:>18}" for c in cells))
    print("\n(a positive cell isolated among negatives = variance, not edge)")


def _credit_em_sweep(p: BTParams, cp, signal_fn, data: dict):
    """How far outside the expected move the short strike sits, x width. This is
    the knob the replicated strategy actually turns — closer strikes collect more
    premium but get breached more often."""
    from dataclasses import replace
    fracs = [0.5, 0.75, 1.0, 1.5, 2.0]
    widths = [1.0, 2.0, 3.0]
    print(f"\n{'='*76}\nEXPECTED-MOVE SWEEP — net E[R]  (win% / n)\n{'='*76}")
    print(f"{'em_frac':<10}" + "".join(f"{'w=$'+format(w,'g'):>22}" for w in widths))
    for fr in fracs:
        cells = []
        for w in widths:
            st = summarize(run_credit(p, replace(cp, em_frac=fr, width=w), signal_fn,
                                      data=data, verbose=False))
            cells.append(f"{st['expected_r']:+.3f} ({st['win_rate']*100:.0f}%/{st['n']})"
                         if st["n"] else "-- (0)")
        print(f"{fr:<10.2f}" + "".join(f"{c:>22}" for c in cells))
    print("\n(further out = higher win rate, smaller credit — the tradeoff that decides it)")


def _credit_vol_sweep(p: BTParams, cp, signal_fn, data: dict):
    """Vol-regime gate x strike distance. Short premium is compensation for
    bearing variance risk, and that compensation is richest when vol is elevated
    relative to its own history — so this is the sweep that matters most for
    whether premium selling has an edge at all.

    n is printed because a high gate leaves few trades: a good-looking cell on a
    handful of samples is noise, and must be read as such.
    """
    from dataclasses import replace
    gates = [(0.0, 1.0), (0.5, 1.0), (0.7, 1.0), (0.8, 1.0), (0.5, 0.9), (0.0, 0.5)]
    fracs = [0.5, 0.6, 0.75]
    print(f"\n{'='*84}\nVOL-REGIME GATE x STRIKE DISTANCE — net E[R]  (win% / n)\n{'='*84}")
    print(f"{'vol pctile':<16}" + "".join(f"{'em='+format(fr,'g'):>22}" for fr in fracs))
    saved = (strategies.EM_IV_RANK_MIN, strategies.EM_IV_RANK_MAX)
    try:
        for lo, hi in gates:
            strategies.EM_IV_RANK_MIN, strategies.EM_IV_RANK_MAX = lo, hi
            cells = []
            for fr in fracs:
                st = summarize(run_credit(p, replace(cp, em_frac=fr), signal_fn,
                                          data=data, verbose=False))
                cells.append(f"{st['expected_r']:+.3f} ({st['win_rate']*100:.0f}%/{st['n']})"
                             if st["n"] else "-- (0)")
            label = f"{lo:.0%}-{hi:.0%}" + (" (all)" if (lo, hi) == (0.0, 1.0) else "")
            print(f"{label:<16}" + "".join(f"{c:>22}" for c in cells))
    finally:
        strategies.EM_IV_RANK_MIN, strategies.EM_IV_RANK_MAX = saved
    print("\n(small n = noise; a cell needs BOTH a positive E[R] and enough trades to believe)")


def _swing_sweep(p: BTParams, cp, dte: int, manage_dte: int, data: dict):
    """The test this whole thread has been building toward: the vol-regime gate
    applied at the 45-DTE horizon, where the variance risk premium actually lives."""
    from dataclasses import replace
    gates = [(0.0, 1.0), (0.5, 1.0), (0.7, 1.0), (0.8, 1.0), (0.0, 0.5)]
    deltas = [0.10, 0.16, 0.30]
    print(f"\n{'='*84}\nSWING {dte}DTE — VOL GATE x SHORT DELTA — net E[R]  (win% / n)\n{'='*84}")
    print(f"{'vol pctile':<16}" + "".join(f"{'d='+format(d,'g'):>22}" for d in deltas))
    saved = (strategies.EM_IV_RANK_MIN, strategies.EM_IV_RANK_MAX)
    try:
        for lo, hi in gates:
            strategies.EM_IV_RANK_MIN, strategies.EM_IV_RANK_MAX = lo, hi
            cells = []
            for d in deltas:
                st = summarize(run_swing(p, replace(cp, short_delta=d), dte,
                                         manage_dte, data, verbose=False))
                cells.append(f"{st['expected_r']:+.3f} ({st['win_rate']*100:.0f}%/{st['n']})"
                             if st["n"] else "-- (0)")
            label = f"{lo:.0%}-{hi:.0%}" + (" (all)" if (lo, hi) == (0.0, 1.0) else "")
            print(f"{label:<16}" + "".join(f"{c:>22}" for c in cells))
    finally:
        strategies.EM_IV_RANK_MIN, strategies.EM_IV_RANK_MAX = saved
    print("\n(small n = noise; needs BOTH positive E[R] and enough trades to believe)")


def _credit_cost_sweep(p: BTParams, cp, signal_fn, data: dict):
    """Re-simulate under harsher per-leg spreads. Unlike the single-leg sweep
    this MUST re-run: the half-spread changes the credit taken, which changes
    the target/stop thresholds and therefore the exits themselves."""
    from dataclasses import replace
    spreads = [0.01, 0.015, 0.02, 0.03, 0.05]
    print(f"\n{'='*72}\nCREDIT COST SENSITIVITY — net E[R]\n{'='*72}")
    print("half_spread:  " + "  ".join(f"{s:>7.3f}" for s in spreads))
    ers, ns = [], []
    for hs in spreads:
        st = summarize(run_credit(replace(p, half_spread=hs), cp, signal_fn,
                                  data=data, verbose=False))
        ers.append(st.get("expected_r", 0.0) if st["n"] else 0.0)
        ns.append(st["n"])
    print("expected R:   " + "  ".join(f"{e:>+7.3f}" for e in ers))
    print("trades:       " + "  ".join(f"{n:>7}" for n in ns))
    print("(trade count falls as the credit stops clearing the spread at all)")


# ---------------- meta-filter (phase 2 validation, in-backtest) ----------------

def _meta_frame(trades: list[Trade]):
    """Time-order trades by entry and assemble (X, y, net_r, ordered_trades) for the
    walk-forward. Sequence features (win/loss streak, trades_today) are filled here
    from the realized order — the live analogue of journal.recent_streaks/governor.
    Labels mirror journal.label_from_reason (1 iff profit target hit first). net_r is
    the backtest r_multiple, which is ALREADY net of modeled spread+fees (so, unlike
    meta._net_r on live rows, no extra friction is subtracted here)."""
    trades = sorted(trades, key=lambda t: t.entry_time)
    win = loss = 0
    day_counts: dict = {}
    for t in trades:
        d = t.entry_time.date()
        t.features["win_streak"] = float(win)
        t.features["loss_streak"] = float(loss)
        t.features["trades_today"] = float(day_counts.get(d, 0))
        day_counts[d] = day_counts.get(d, 0) + 1
        if t.pnl > 0:
            win, loss = win + 1, 0
        else:
            loss, win = loss + 1, 0
    cols = sorted({k for t in trades for k in t.features})
    X = pd.DataFrame([[t.features.get(c, 0.0) for c in cols] for t in trades], columns=cols)
    y = np.array([1 if t.exit_reason.lower().startswith("profit target") else 0
                  for t in trades], dtype=int)
    net_r = np.array([t.r_multiple for t in trades])
    return X, y, net_r, trades


def run_meta(p: BTParams, signal_fn, data: dict | None = None,
             thresholds=None, verbose: bool = True) -> dict:
    """Purged+embargoed walk-forward meta-labeling on this signal's backtest trades.
    Scores model-filtered net E[R] vs the take-every-signal baseline out-of-sample —
    the same machinery (meta.purged_walk_forward_splits / _make_model) the live trainer
    uses, so an in-backtest 'meta helps' is comparable to the live gate's verdict."""
    from sklearn.metrics import roc_auc_score

    trades = run(p, signal_fn, data=data, verbose=verbose)
    X, y, net_r, trades = _meta_frame(trades)
    n = len(trades)
    out = {"n": n}
    if n < config.META_MIN_TRADES:
        print(f"\nMETA: {n}/{config.META_MIN_TRADES} trades — insufficient to validate.")
        return out
    if len(np.unique(y)) < 2:
        print(f"\nMETA: labels not both-class ({y.sum()}/{n} wins) — cannot train.")
        return out

    oos = np.full(n, np.nan)
    for tr, te in meta.purged_walk_forward_splits(n, config.META_CV_SPLITS,
                                                  config.META_EMBARGO_FRAC):
        if len(np.unique(y[tr])) < 2:
            continue
        m = meta._make_model()
        m.fit(X.iloc[tr], y[tr])
        oos[te] = m.predict_proba(X.iloc[te])[:, 1]

    mask = ~np.isnan(oos)
    if mask.sum() < config.META_CV_SPLITS or len(np.unique(y[mask])) < 2:
        print("\nMETA: insufficient out-of-sample coverage to validate.")
        return out

    yv, pv, rv = y[mask], oos[mask], net_r[mask]
    auc = float(roc_auc_score(yv, pv))
    baseline_er = meta._expected_r(rv, np.ones_like(yv, dtype=bool))
    thresholds = thresholds or [0.45, 0.50, 0.55, 0.60]

    print(f"\n{'='*66}\nMETA-FILTER — purged walk-forward (signal={signal_fn.__name__ if hasattr(signal_fn,'__name__') else 'signal'}, "
          f"hs={p.half_spread})\n{'='*66}")
    print(f"  OOS samples       {int(mask.sum())} of {n} trades")
    print(f"  OOS AUC           {auc:.3f}   (bar {config.META_AUC_BAR})")
    print(f"  take-all E[R]     {baseline_er:+.3f}   (the no-filter baseline)")
    print(f"  {'thr':>5}{'kept':>7}{'keep%':>7}{'win%':>7}{'E[R]net':>9}{'vs base':>9}")
    best = {"er": baseline_er, "thr": None}
    for thr in thresholds:
        take = pv >= thr
        if take.sum() == 0:
            print(f"  {thr:>5.2f}{0:>7}   (none kept)")
            continue
        er = meta._expected_r(rv, take)
        wr = float(yv[take].mean())
        print(f"  {thr:>5.2f}{int(take.sum()):>7}{take.mean()*100:>6.0f}%"
              f"{wr*100:>6.1f}%{er:>+9.3f}{er-baseline_er:>+9.3f}")
        if er > best["er"]:
            best = {"er": er, "thr": thr}

    passes = auc >= config.META_AUC_BAR and best["thr"] is not None
    if best["thr"] is None:
        verdict = "NO LIFT (filter never beats take-all)"
    elif passes:
        verdict = (f"META HELPS — thr={best['thr']:.2f} lifts E[R] {baseline_er:+.3f}"
                   f" -> {best['er']:+.3f} (AUC clears bar)")
    else:
        verdict = (f"WEAK — best thr={best['thr']:.2f} E[R] {best['er']:+.3f} but "
                   f"AUC {auc:.3f} < bar {config.META_AUC_BAR} (likely overfit)")
    print(f"  VERDICT           {verdict}")

    # Coefficient read (model fit on all data) — directional, scaled features.
    final = meta._make_model()
    final.fit(X, y)
    coefs = final.named_steps["clf"].coef_[0]
    top = sorted(zip(X.columns, coefs), key=lambda c: abs(c[1]), reverse=True)[:6]
    print("  top features      " + ", ".join(f"{c}:{w:+.2f}" for c, w in top))
    out.update({"oos_auc": auc, "baseline_er": baseline_er,
                "best_er": best["er"], "best_thr": best["thr"], "passes": passes})
    return out


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
    ap.add_argument("--signal", type=str, default="confluence",
                    choices=list(strategies.REGISTRY))
    ap.add_argument("--compare", action="store_true", help="run all strategies head-to-head")
    ap.add_argument("--meta", action="store_true",
                    help="validate the meta-filter (walk-forward) on the chosen signal")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--csv", type=str, default="")
    # --- credit (short vertical) mode ---
    ap.add_argument("--mode", choices=["single", "credit"], default="single",
                    help="single = long option (live structure); credit = short vertical")
    ap.add_argument("--short-delta", type=float, default=0.30,
                    help="credit mode: |delta| of the short leg")
    ap.add_argument("--width", type=float, default=1.0,
                    help="credit mode: dollars between short and long strike")
    ap.add_argument("--tp-pct", type=float, default=50.0,
                    help="credit mode: buy back at this %% of credit captured")
    ap.add_argument("--stop-mult", type=float, default=2.0,
                    help="credit mode: stop when open loss = N x credit")
    ap.add_argument("--struct-sweep", action="store_true",
                    help="credit mode: sweep short_delta x width")
    ap.add_argument("--zero-dte", action="store_true",
                    help="credit mode: expiry 16:00 today, time-to-expiry from the clock")
    ap.add_argument("--strike-mode", choices=["delta", "expected_move"], default="delta",
                    help="credit mode: how the short strike is placed")
    ap.add_argument("--em-frac", type=float, default=0.5,
                    help="expected_move mode: short strike at this fraction of the EM")
    ap.add_argument("--no-manage", action="store_true",
                    help="credit mode: hold to expiry (no intraday target/stop)")
    ap.add_argument("--strike-inc", type=float, default=1.0,
                    help="strike grid increment of the underlying")
    ap.add_argument("--universe", type=str, default="",
                    help="comma-separated override (does NOT touch the live .env)")
    ap.add_argument("--any-time", action="store_true",
                    help="ignore the live entry window (for scheduled signals)")
    ap.add_argument("--min-half-spread", type=float, default=0.005,
                    help="dollar floor on per-leg half-spread (0.005 = penny-wide market)")
    ap.add_argument("--em-sweep", action="store_true",
                    help="expected_move mode: sweep how far out the short strike sits")
    ap.add_argument("--iv-rank-min", type=float, default=0.0,
                    help="em_regime: only sell when vol percentile >= this (0 = always)")
    ap.add_argument("--iv-rank-max", type=float, default=1.0,
                    help="em_regime: only sell when vol percentile <= this")
    ap.add_argument("--vol-sweep", action="store_true",
                    help="sweep the vol-regime gate x strike distance")
    ap.add_argument("--swing", action="store_true",
                    help="credit mode: multi-day hold (45DTE style) instead of intraday")
    ap.add_argument("--manage-dte", type=int, default=21,
                    help="swing mode: flatten with this many days to expiry left")
    ap.add_argument("--swing-sweep", action="store_true",
                    help="swing mode: sweep vol gate x short delta")
    a = ap.parse_args(argv)
    if a.universe:
        config.UNIVERSE = [s.strip().upper() for s in a.universe.split(",") if s.strip()]
    strategies.EM_IV_RANK_MIN = a.iv_rank_min
    strategies.EM_IV_RANK_MAX = a.iv_rank_max
    p = BTParams(days=a.days, path_tf=a.path_tf, dte=a.dte, target_delta=a.delta,
                 half_spread=a.half_spread, fee_per_contract=a.fee, vrp=a.vrp,
                 strike_inc=a.strike_inc, zero_dte=a.zero_dte,
                 ignore_entry_window=a.any_time,
                 min_half_spread_abs=a.min_half_spread)
    print(f"Backtest: universe={config.UNIVERSE} days={p.days} signal=15m path={p.path_tf}m")
    if a.compare:
        _compare(p)
        return
    if a.meta:
        print(f"Strategy: {a.signal}  (meta-filter validation)")
        return run_meta(p, strategies.REGISTRY[a.signal])

    if a.mode == "credit":
        cp = creditmod.CreditParams(short_delta=a.short_delta, width=a.width,
                                    profit_target_pct=a.tp_pct, stop_mult=a.stop_mult,
                                    strike_inc=a.strike_inc, strike_mode=a.strike_mode,
                                    em_frac=a.em_frac, manage=not a.no_manage)
        placing = (f"short {a.short_delta:.2f}d" if a.strike_mode == "delta"
                   else f"short at {a.em_frac:g}x expected move")
        if a.swing:
            print(f"Structure: CREDIT VERTICAL SWING ({placing}, ${a.width:g} wide, "
                  f"{a.dte}DTE entry, flatten at {a.manage_dte}DTE, "
                  f"{'managed' if not a.no_manage else 'no intraday management'}, "
                  f"vol gate {a.iv_rank_min:.0%}-{a.iv_rank_max:.0%})")
            data = load_daily_only(p)
            trades = run_swing(p, cp, a.dte, a.manage_dte, data)
            s = summarize(trades)
            _print_credit_summary(trades, s, p, cp)
            if a.swing_sweep:
                _swing_sweep(p, cp, a.dte, a.manage_dte, data)
            if a.csv and trades:
                pd.DataFrame([t.__dict__ for t in trades]).to_csv(a.csv, index=False)
                print(f"\nwrote {len(trades)} swing spreads -> {a.csv}")
            return s

        print(f"Strategy: {a.signal}  structure=CREDIT VERTICAL "
              f"({placing}, ${a.width:g} wide, "
              f"{'0DTE' if a.zero_dte else str(a.dte)+'DTE'}, "
              f"{'hold to expiry' if a.no_manage else 'managed'})")
        data = load_data(p)
        trades = run_credit(p, cp, strategies.REGISTRY[a.signal], data=data)
        s = summarize(trades)
        _print_credit_summary(trades, s, p, cp)
        if a.struct_sweep:
            _credit_struct_sweep(p, cp, strategies.REGISTRY[a.signal], data)
        if a.em_sweep:
            _credit_em_sweep(p, cp, strategies.REGISTRY[a.signal], data)
        if a.vol_sweep:
            _credit_vol_sweep(p, cp, strategies.REGISTRY[a.signal], data)
        if a.sweep:
            _credit_cost_sweep(p, cp, strategies.REGISTRY[a.signal], data)
        if a.csv and trades:
            pd.DataFrame([t.__dict__ for t in trades]).to_csv(a.csv, index=False)
            print(f"\nwrote {len(trades)} credit spreads -> {a.csv}")
        return s

    print(f"Strategy: {a.signal}")
    trades = run(p, strategies.REGISTRY[a.signal])
    s = summarize(trades)
    _print_summary(s, p)
    if a.sweep:
        _sweep(trades, p)
    if a.csv and trades:
        (pd.DataFrame([t.__dict__ for t in trades])
         .drop(columns=["features"], errors="ignore").to_csv(a.csv, index=False))
        print(f"\nwrote {len(trades)} trades -> {a.csv}")
    return s


if __name__ == "__main__":
    main()
