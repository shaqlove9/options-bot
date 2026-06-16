"""backtest.py — backtest harness using underlying OHLCV + Black-Scholes
estimated option prices.

Methodology / limitations (read before trusting numbers):
  - Underlying 15-min bars come from Alpaca historical data.
  - Option prices are SYNTHESIZED with Black-Scholes at a fixed per-symbol IV.
    Still synthetic (no real OI, no true IV term structure), but the cost
    model is deliberately conservative — see "realistic cost modeling" below.
  - Contract selection MIRRORS the live bot (options_chain.find_contract):
    the nearest strike inside the OTM_MIN..OTM_MAX band (percent of spot) that
    fits MAX_TRADE_COST. The old version priced a single 1-strike-OTM contract,
    which rejected the high-IV names on budget and undercounted trades badly.
  - Spread is a percent of mid (not a flat nickel), plus per-side slippage, so
    fills approximate what a retail account actually pays. Optional exit IV
    haircut models momentum-pop IV fade. These pull results toward reality but
    are still an estimate — validate in paper before trusting them.
  - Same rules as live: momentum + RSI(5) + rel volume, +40%/-30%, 3:45 flat,
    daily loss halt, 2-loss 30-min pause.

Usage:  python backtest.py --days 60
"""
import argparse
import datetime as dt
import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

import config
from scanner import rsi_last
from utils import ET

log = logging.getLogger("backtest")

RISK_FREE = 0.045
SIM_DTE = 3                       # synthetic contracts expire in ~3 days

# --- realistic cost modeling (was a flat $0.06 spread, which made fills a
#     fantasy: 100% backtest win rate vs ~14% live). Short-dated OTM options
#     trade with wide percentage spreads; the retail account crosses them on
#     entry AND exit and eats slippage on top. ---
SPREAD_FRAC = 0.04                # synthetic bid/ask width as a fraction of mid
MIN_SPREAD = 0.05                 # ...but never tighter than a nickel
SLIPPAGE_PER_SIDE = 0.02          # adverse fill vs mid, charged on entry and exit
EXIT_IV_HAIRCUT = 0.0             # optional IV-crush proxy on exits (relative,
                                  # e.g. 0.05 = exit IV 5% below entry); 0 = off

# --- debit-spread variant ---------------------------------------------------
# Same momentum/RSI entries as the single-long scalp, but instead of buying one
# OTM option we buy the near-the-money leg AND sell a further-OTM leg of the
# same expiry. This cuts the net debit (cheaper, less theta/vega drag) but caps
# the gain at the strike width and DOUBLES the number of fills crossed. Whether
# that nets out positive is exactly what the head-to-head is meant to answer.
SPREAD_WIDTH_STEPS = 2            # short leg this many strikes further OTM

# --- equity day-trading variant ---------------------------------------------
# Same signal, but trade the UNDERLYING SHARES instead of options. No synthetic
# pricing at all — fills are real historical prices ± a small known friction —
# so this read is far more faithful than the options sim. Trade-off: no option
# leverage, so we apply Reg-T intraday margin and price-based (% of underlying)
# stops in place of the premium-based ones. Tune these to sweep.
EQ_CAPITAL = 1000.0               # account size for the equity test
EQ_LEVERAGE = 4.0                 # Reg-T intraday buying power (4:1)
EQ_SPREAD_PER_SHARE = 0.01        # bid/ask width on a liquid name (~a penny)
EQ_SLIPPAGE_PER_SHARE = 0.005     # adverse fill vs mid, per side
EQ_COMMISSION = 0.0               # Alpaca equities are commission-free
EQ_TAKE_PCT = 0.6                 # take profit, as a % move of the underlying
EQ_STOP_PCT = 0.4                 # stop loss, as a % move of the underlying
EQ_TRAIL_TRIGGER_PCT = 0.5        # arm the trailing stop once up this %
EQ_TRAIL_GIVEBACK_PCT = 0.25      # ...then exit after giving back this much
EQ_MAX_DAILY_LOSS = 0.06 * EQ_CAPITAL   # halt the day past this $ loss

ASSUMED_IV = {                    # rough per-symbol IV for pricing
    "SPY": 0.18, "QQQ": 0.22, "NVDA": 0.50,
    "TSLA": 0.55, "AAPL": 0.28, "AMZN": 0.35,
}
STRIKE_STEP = {                   # typical near-dated strike increments
    "SPY": 1.0, "QQQ": 1.0, "NVDA": 2.5,
    "TSLA": 2.5, "AAPL": 2.5, "AMZN": 2.5,
}


# ---------------- Black-Scholes ----------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes(spot: float, strike: float, t_years: float, iv: float,
                  otype: str, r: float = RISK_FREE) -> float:
    """European option price; floors at intrinsic value near expiry."""
    if t_years <= 0:
        return max(0.0, spot - strike) if otype == "call" else max(0.0, strike - spot)
    d1 = (math.log(spot / strike) + (r + iv ** 2 / 2) * t_years) / (iv * math.sqrt(t_years))
    d2 = d1 - iv * math.sqrt(t_years)
    if otype == "call":
        return spot * _norm_cdf(d1) - strike * math.exp(-r * t_years) * _norm_cdf(d2)
    return strike * math.exp(-r * t_years) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


# ---------------- simulation ----------------

@dataclass
class SimTrade:
    symbol: str
    direction: str
    entry_time: dt.datetime
    exit_time: dt.datetime
    strike: float
    entry_price: float
    exit_price: float
    pnl: float
    exit_reason: str
    strategy: str = "scalp"
    short_strike: float | None = None     # set only for debit-spread trades


@dataclass
class EqTrade:
    symbol: str
    direction: str                        # 'call'=long shares, 'put'=short shares
    entry_time: dt.datetime
    exit_time: dt.datetime
    shares: int
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float                        # P&L as % of position notional
    exit_reason: str
    strategy: str = "scalp"


def fetch_bars(client, symbol: str, days: int) -> pd.DataFrame:
    start = dt.datetime.now(tz=ET) - dt.timedelta(days=days + config.VOLUME_LOOKBACK_DAYS * 2)
    bars = client.get_stock_bars(StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(15, TimeFrameUnit.Minute),
        start=start,
    )).df
    if bars.empty:
        return bars
    bars = bars.droplevel("symbol") if "symbol" in bars.index.names else bars
    bars.index = bars.index.tz_convert(ET)
    bars = bars.between_time("09:30", "16:00")
    # Cumulative session VWAP per day (mirrors scanner._session_vwap)
    px = bars["vwap"] if "vwap" in bars else bars[["high", "low", "close"]].mean(axis=1)
    grp = bars.index.date
    pv = (px * bars["volume"]).groupby(grp).cumsum()
    vv = bars["volume"].groupby(grp).cumsum()
    bars = bars.assign(session_vwap=pv / vv.replace(0, float("nan")))
    return bars


def avg_daily_volume(bars: pd.DataFrame) -> pd.Series:
    """Rolling 20-day average daily volume, shifted so 'today' is excluded."""
    daily = bars["volume"].groupby(bars.index.date).sum()
    return daily.rolling(config.VOLUME_LOOKBACK_DAYS).mean().shift(1)


def quote(spot: float, strike: float, tte: float, iv: float,
          otype: str) -> tuple[float, float]:
    """Synthetic mid price and full bid/ask width for one option."""
    mid = black_scholes(spot, strike, tte, iv, otype)
    spread = max(MIN_SPREAD, mid * SPREAD_FRAC)
    return mid, spread


def _ask(spot: float, strike: float, tte: float, iv: float, otype: str) -> float:
    """Cross to buy: mid + half-spread + slippage."""
    mid, spread = quote(spot, strike, tte, iv, otype)
    return mid + spread / 2 + SLIPPAGE_PER_SIDE


def _bid(spot: float, strike: float, tte: float, iv: float, otype: str) -> float:
    """Cross to sell: mid - half-spread - slippage, floored at a penny."""
    mid, spread = quote(spot, strike, tte, iv, otype)
    return max(0.01, mid - spread / 2 - SLIPPAGE_PER_SIDE)


def entry_ask(spot: float, strike: float, tte: float, iv: float, otype: str) -> float:
    """What the account pays to open a single long: mid + half-spread + slippage."""
    return _ask(spot, strike, tte, iv, otype)


def exit_bid(spot: float, strike: float, tte: float, iv: float, otype: str) -> float:
    """What the account receives to close a single long, with an optional
    IV-crush haircut applied to the exit mid."""
    return _bid(spot, strike, tte, iv * (1 - EXIT_IV_HAIRCUT), otype)


def short_leg_strike(symbol: str, long_strike: float, direction: str) -> float:
    """The further-OTM leg we sell against the long: above for calls, below for
    puts, SPREAD_WIDTH_STEPS grid increments away."""
    step = STRIKE_STEP.get(symbol, 1.0)
    offset = SPREAD_WIDTH_STEPS * step
    return long_strike + offset if direction == "call" else long_strike - offset


def spread_entry_debit(spot: float, long_strike: float, short_strike: float,
                       tte: float, iv: float, direction: str) -> float:
    """Net debit to open the spread: pay the ask on the long, collect the bid on
    the short. (Both legs cross at full entry IV.)"""
    return (_ask(spot, long_strike, tte, iv, direction)
            - _bid(spot, short_strike, tte, iv, direction))


def spread_exit_credit(spot: float, long_strike: float, short_strike: float,
                       tte: float, iv: float, direction: str) -> float:
    """Net credit to close the spread: sell the long (bid), buy back the short
    (ask). Both legs priced at the faded exit IV. Floored at a penny."""
    faded = iv * (1 - EXIT_IV_HAIRCUT)
    val = (_bid(spot, long_strike, tte, faded, direction)
           - _ask(spot, short_strike, tte, faded, direction))
    return max(0.01, val)


def select_contract(symbol: str, spot: float, direction: str,
                    entry_dt: dt.datetime) -> tuple[float, dt.datetime, float] | None:
    """Mirror options_chain.find_contract: among grid strikes inside the
    OTM_MIN..OTM_MAX band (percent of spot) that fit MAX_TRADE_COST, take the
    one nearest the money. (Live tie-breaks on tightest spread then nearer
    strike; with a uniform synthetic spread the nearest strike wins outright.)
    Returns (strike, expiry, entry_ask) or None if nothing in band fits budget.
    """
    step = STRIKE_STEP.get(symbol, 1.0)
    iv = ASSUMED_IV[symbol]
    expiry = (entry_dt + dt.timedelta(days=SIM_DTE)).replace(hour=16, minute=0)
    tte = (expiry - entry_dt).total_seconds() / (365 * 86400)
    if direction == "call":
        lo, hi = spot * (1 + config.OTM_MIN_PCT / 100), spot * (1 + config.OTM_MAX_PCT / 100)
    else:
        lo, hi = spot * (1 - config.OTM_MAX_PCT / 100), spot * (1 - config.OTM_MIN_PCT / 100)

    strikes: list[float] = []
    k = math.ceil(lo / step) * step
    while k <= hi + 1e-9:
        strikes.append(round(k, 4))
        k += step
    strikes.sort(key=lambda s: abs(s - spot))   # nearest the money first

    for strike in strikes:
        ask = entry_ask(spot, strike, tte, iv, direction)
        if ask >= 0.05 and ask * 100 * config.MAX_CONTRACTS <= config.MAX_TRADE_COST:
            return strike, expiry, ask
    return None


def select_spread(symbol: str, spot: float, direction: str,
                  entry_dt: dt.datetime) -> tuple[float, float, dt.datetime, float] | None:
    """Pick the SAME long leg the single-long bot would buy, then sell a leg
    SPREAD_WIDTH_STEPS further OTM. Returns (long_strike, short_strike, expiry,
    net_debit) or None if no long fits budget or the net debit is degenerate.
    Using the single-long leg as the anchor keeps the head-to-head fair.
    """
    sel = select_contract(symbol, spot, direction, entry_dt)
    if sel is None:
        return None
    long_strike, expiry, _ = sel
    short_strike = short_leg_strike(symbol, long_strike, direction)
    if short_strike <= 0:
        return None
    iv = ASSUMED_IV[symbol]
    tte = (expiry - entry_dt).total_seconds() / (365 * 86400)
    debit = spread_entry_debit(spot, long_strike, short_strike, tte, iv, direction)
    if debit < 0.05 or debit * 100 * config.MAX_CONTRACTS > config.MAX_TRADE_COST:
        return None
    return long_strike, short_strike, expiry, debit


def detect_signal(sym: str, ts, row, all_bars: dict[str, pd.DataFrame],
                  adv: dict[str, pd.Series], day, t,
                  style: str = "momentum") -> tuple[str, str] | None:
    """The instrument-independent entry decision. Returns (direction, strategy)
    where direction is 'call'(=long/bullish) or 'put'(=short/bearish), or None.
    Shared by the options and equity engines so the signal is identical.

    style='momentum' (default): the live bot's logic — buy strength (up candle +
      RSI>CALL_MIN), short weakness, trend-aligned VWAP filter, plus the runner.
    style='reversion': a CONTROLLED INVERSION for the mean-reversion test — same
      trigger candles, OPPOSITE direction (fade the move: buy the oversold dip,
      short the overbought spike) and flipped VWAP sense (favour stretched-from-
      VWAP entries). Runner is skipped (fading breakouts is a different bet)."""
    bars = all_bars[sym]
    history = bars[bars.index <= ts]
    if len(history) < config.RSI_PERIOD + 2 or row["open"] <= 0:
        return None
    momentum = (row["close"] - row["open"]) / row["open"] * 100
    rsi = rsi_last(history["close"], config.RSI_PERIOD)

    # Strategy 1: scalp (one strong candle). Momentum follows it, reversion fades.
    direction, strategy = None, None
    if abs(momentum) >= config.MOMENTUM_PCT:
        up_strong = momentum > 0 and rsi > config.RSI_CALL_MIN
        down_weak = momentum < 0 and rsi < config.RSI_PUT_MAX
        if style == "reversion":
            if down_weak:                       # oversold dip -> buy the bounce
                direction, strategy = "call", "revert"
            elif up_strong:                     # overbought spike -> short the fade
                direction, strategy = "put", "revert"
        else:
            if up_strong:
                direction, strategy = "call", "scalp"
            elif down_weak:
                direction, strategy = "put", "scalp"

    # Strategy 2: runner (day trend + new session high/low) — momentum style only
    today_hist = history[history.index.date == day]
    if (style != "reversion" and direction is None and config.RUNNER_ENABLED
            and len(today_hist) >= 3 and today_hist["open"].iloc[0] > 0):
        day_change = ((row["close"] - today_hist["open"].iloc[0])
                      / today_hist["open"].iloc[0] * 100)
        tol = config.RUNNER_BREAKOUT_TOL
        if (day_change >= config.RUNNER_DAY_PCT
                and row["close"] > row["open"]
                and row["close"] >= today_hist["high"].iloc[:-1].max() * (1 - tol)
                and rsi > config.RUNNER_RSI_MIN):
            direction, strategy = "call", "runner"
        elif (day_change <= -config.RUNNER_DAY_PCT
                and row["close"] < row["open"]
                and row["close"] <= today_hist["low"].iloc[:-1].min() * (1 + tol)
                and rsi < config.RUNNER_RSI_MAX):
            direction, strategy = "put", "runner"
    if direction is None:
        return None

    # VWAP direction filter. Momentum wants trend-aligned (call above VWAP);
    # reversion wants stretched-from-VWAP (buy the dip below it), so flip it.
    if config.VWAP_FILTER:
        vwap = row.get("session_vwap")
        if vwap is None or pd.isna(vwap):
            return None
        below = row["close"] <= vwap
        above = row["close"] >= vwap
        if style == "reversion":
            if direction == "call" and above:   # only buy dips below VWAP
                return None
            if direction == "put" and below:    # only short spikes above VWAP
                return None
        else:
            if direction == "call" and below:
                return None
            if direction == "put" and above:
                return None

    # Relative volume (time-adjusted vs 20-day avg)
    avg = adv[sym].get(day)
    if avg is None or pd.isna(avg) or avg <= 0:
        return None
    today_so_far = history[history.index.date == day]["volume"].sum()
    elapsed = max(((t - t.replace(hour=9, minute=30)).total_seconds()) / 23400, 0.02)
    if today_so_far / (avg * elapsed) < config.REL_VOLUME_MIN:
        return None
    return direction, strategy


def simulate(all_bars: dict[str, pd.DataFrame], mode: str = "long") -> list[SimTrade]:
    """mode='long' buys a single OTM option per signal; mode='spread' buys the
    same leg and sells one SPREAD_WIDTH_STEPS further OTM (defined-risk debit
    spread). Entry signals, risk halts, and exit rules are identical between the
    two so the only variable is the contract structure."""
    trades: list[SimTrade] = []
    adv = {sym: avg_daily_volume(b) for sym, b in all_bars.items()}
    days = sorted({d for b in all_bars.values() for d in set(b.index.date)})

    for day in days:
        daily_pnl = 0.0
        halted = False
        consec_losses = 0
        paused_until: dt.datetime | None = None
        open_pos: dict[str, dict] = {}      # symbol -> position state
        cooldown: dict[str, dt.datetime] = {}

        # Merge each symbol's bars for the day into one time-ordered stream.
        day_bars = []
        for sym, bars in all_bars.items():
            b = bars[bars.index.date == day]
            for ts, row in b.iterrows():
                day_bars.append((ts, sym, row))
        day_bars.sort(key=lambda x: x[0])

        for ts, sym, row in day_bars:
            t = ts.to_pydatetime()
            force_close = t.time() >= dt.time(15, 45)

            # ---- manage an open position in this symbol ----
            if sym in open_pos:
                pos = open_pos[sym]
                tte = max((pos["expiry"] - t).total_seconds() / (365 * 86400), 1e-6)
                if pos.get("short_strike") is not None:
                    bid = spread_exit_credit(row["close"], pos["strike"],
                                             pos["short_strike"], tte,
                                             ASSUMED_IV[sym], pos["direction"])
                else:
                    bid = exit_bid(row["close"], pos["strike"], tte,
                                   ASSUMED_IV[sym], pos["direction"])
                chg = (bid - pos["entry_price"]) / pos["entry_price"] * 100
                pos["peak_pct"] = max(pos.get("peak_pct", 0.0), chg)
                reason = None
                if chg >= pos.get("tp", config.TAKE_PROFIT_PCT):
                    reason = "take profit"
                elif chg <= -config.STOP_LOSS_PCT:
                    reason = "stop loss"
                elif (pos["peak_pct"] >= config.TRAIL_TRIGGER_PCT
                      and pos["peak_pct"] - chg >= config.TRAIL_GIVEBACK_PCT):
                    reason = "trailing stop"
                elif force_close:
                    reason = "time stop 3:45"
                if reason:
                    pnl = (bid - pos["entry_price"]) * 100
                    daily_pnl += pnl
                    trades.append(SimTrade(sym, pos["direction"], pos["entry_time"], t,
                                           pos["strike"], pos["entry_price"], bid,
                                           pnl, reason, pos.get("strategy", "scalp"),
                                           pos.get("short_strike")))
                    del open_pos[sym]
                    if pnl < 0:
                        consec_losses += 1
                        if consec_losses >= config.CONSEC_LOSS_PAUSE:
                            paused_until = t + dt.timedelta(minutes=config.PAUSE_MINUTES)
                            consec_losses = 0
                    else:
                        consec_losses = 0
                    if daily_pnl <= -config.MAX_DAILY_LOSS:
                        halted = True

            # ---- entries ----
            if (halted or force_close or sym in open_pos
                    or len(open_pos) >= config.MAX_OPEN_POSITIONS
                    or not dt.time(9, 45) <= t.time() <= dt.time(15, 30)
                    or (paused_until and t < paused_until)
                    or (sym in cooldown
                        and (t - cooldown[sym]).total_seconds() < config.SYMBOL_COOLDOWN_MIN * 60)):
                continue

            sig = detect_signal(sym, ts, row, all_bars, adv, day, t)
            if sig is None:
                continue
            direction, strategy = sig

            # Select + price the synthetic contract(s) (mirrors live chain logic)
            short_strike = None
            if mode == "spread":
                sel = select_spread(sym, row["close"], direction, t)
                if sel is None:
                    continue
                strike, short_strike, expiry, cost = sel
            else:
                sel = select_contract(sym, row["close"], direction, t)
                if sel is None:
                    continue
                strike, expiry, cost = sel
            if daily_pnl - cost * 100 * config.STOP_LOSS_PCT / 100 <= -config.MAX_DAILY_LOSS:
                continue

            open_pos[sym] = {"direction": direction, "strike": strike,
                             "short_strike": short_strike,
                             "entry_price": cost, "entry_time": t, "expiry": expiry,
                             "strategy": strategy,
                             "tp": (config.RUNNER_TAKE_PROFIT_PCT if strategy == "runner"
                                    else config.TAKE_PROFIT_PCT)}
            cooldown[sym] = t

    return trades


def simulate_equity(all_bars: dict[str, pd.DataFrame],
                    style: str = "momentum") -> list[EqTrade]:
    """Same signal/guards/halts as simulate(), but trades the underlying shares.
    'call' -> long, 'put' -> short. Fills are real historical prices ± a penny
    spread and slippage (no synthetic option pricing). Exits are price-based
    (% move of the underlying) since premium-based stops don't map to shares."""
    trades: list[EqTrade] = []
    adv = {sym: avg_daily_volume(b) for sym, b in all_bars.items()}
    days = sorted({d for b in all_bars.values() for d in set(b.index.date)})
    per_pos_notional = EQ_CAPITAL * EQ_LEVERAGE / config.MAX_OPEN_POSITIONS
    half_spread = EQ_SPREAD_PER_SHARE / 2

    for day in days:
        daily_pnl = 0.0
        halted = False
        consec_losses = 0
        paused_until: dt.datetime | None = None
        open_pos: dict[str, dict] = {}
        cooldown: dict[str, dt.datetime] = {}

        day_bars = []
        for sym, bars in all_bars.items():
            b = bars[bars.index.date == day]
            for ts, row in b.iterrows():
                day_bars.append((ts, sym, row))
        day_bars.sort(key=lambda x: x[0])

        for ts, sym, row in day_bars:
            t = ts.to_pydatetime()
            force_close = t.time() >= dt.time(15, 45)
            px = row["close"]

            # ---- manage an open position ----
            if sym in open_pos:
                pos = open_pos[sym]
                ref = pos["ref_price"]
                move = ((px - ref) if pos["direction"] == "call"
                        else (ref - px)) / ref * 100
                pos["peak_pct"] = max(pos.get("peak_pct", 0.0), move)
                reason = None
                if move >= EQ_TAKE_PCT:
                    reason = "take profit"
                elif move <= -EQ_STOP_PCT:
                    reason = "stop loss"
                elif (pos["peak_pct"] >= EQ_TRAIL_TRIGGER_PCT
                      and pos["peak_pct"] - move >= EQ_TRAIL_GIVEBACK_PCT):
                    reason = "trailing stop"
                elif force_close:
                    reason = "time stop 3:45"
                if reason:
                    if pos["direction"] == "call":          # sell the long at bid
                        exit_fill = px - half_spread - EQ_SLIPPAGE_PER_SHARE
                        pnl = (exit_fill - pos["entry_fill"]) * pos["shares"]
                    else:                                   # buy back the short at ask
                        exit_fill = px + half_spread + EQ_SLIPPAGE_PER_SHARE
                        pnl = (pos["entry_fill"] - exit_fill) * pos["shares"]
                    pnl -= EQ_COMMISSION * 2
                    notional = pos["entry_fill"] * pos["shares"]
                    pnl_pct = pnl / notional * 100 if notional else 0.0
                    daily_pnl += pnl
                    trades.append(EqTrade(sym, pos["direction"], pos["entry_time"], t,
                                          pos["shares"], pos["entry_fill"], exit_fill,
                                          pnl, pnl_pct, reason, pos.get("strategy", "scalp")))
                    del open_pos[sym]
                    if pnl < 0:
                        consec_losses += 1
                        if consec_losses >= config.CONSEC_LOSS_PAUSE:
                            paused_until = t + dt.timedelta(minutes=config.PAUSE_MINUTES)
                            consec_losses = 0
                    else:
                        consec_losses = 0
                    if daily_pnl <= -EQ_MAX_DAILY_LOSS:
                        halted = True

            # ---- entries (guards identical to the options engine) ----
            if (halted or force_close or sym in open_pos
                    or len(open_pos) >= config.MAX_OPEN_POSITIONS
                    or not dt.time(9, 45) <= t.time() <= dt.time(15, 30)
                    or (paused_until and t < paused_until)
                    or (sym in cooldown
                        and (t - cooldown[sym]).total_seconds() < config.SYMBOL_COOLDOWN_MIN * 60)):
                continue

            sig = detect_signal(sym, ts, row, all_bars, adv, day, t, style=style)
            if sig is None:
                continue
            direction, strategy = sig

            shares = int(per_pos_notional // px)
            if shares < 1:                                  # share too pricey for sizing
                continue
            entry_fill = (px + half_spread + EQ_SLIPPAGE_PER_SHARE if direction == "call"
                          else px - half_spread - EQ_SLIPPAGE_PER_SHARE)
            open_pos[sym] = {"direction": direction, "shares": shares,
                             "entry_fill": entry_fill, "ref_price": px,
                             "entry_time": t, "strategy": strategy}
            cooldown[sym] = t

    return trades


# ---------------- stats ----------------

def report(trades: list[SimTrade], label: str = "single-long"):
    if not trades:
        print(f"[{label}] No trades generated — try more --days or loosen filters.")
        return
    df = pd.DataFrame([vars(t) for t in trades])
    wins = df[df.pnl > 0]
    losses = df[df.pnl <= 0]

    daily = df.groupby(df.exit_time.apply(lambda t: t.date())).pnl.sum()
    equity = config.CAPITAL + daily.cumsum()
    drawdown = equity - equity.cummax()
    daily_ret = daily / config.CAPITAL
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
              if len(daily_ret) > 1 and daily_ret.std() > 0 else float("nan"))

    print("=" * 52)
    print(f"BACKTEST RESULTS — {label}  (synthetic Black-Scholes pricing)")
    print("=" * 52)
    print(f"Avg trade:       ${df.pnl.mean():+.2f}")
    print(f"Period:          {df.entry_time.min().date()} → {df.exit_time.max().date()}")
    print(f"Total trades:    {len(df)}")
    print(f"Win rate:        {len(wins) / len(df) * 100:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"Avg profit:      ${wins.pnl.mean():+.2f}" if len(wins) else "Avg profit:      n/a")
    print(f"Avg loss:        ${losses.pnl.mean():+.2f}" if len(losses) else "Avg loss:        n/a")
    print(f"Total P&L:       ${df.pnl.sum():+.2f}")
    print(f"Max drawdown:    ${drawdown.min():.2f}  ({drawdown.min() / config.CAPITAL * 100:.1f}% of capital)")
    print(f"Sharpe (ann.):   {sharpe:.2f}")
    print(f"Exits:           {df.exit_reason.value_counts().to_dict()}")
    for strat, grp in df.groupby("strategy"):
        gw = (grp.pnl > 0).sum()
        print(f"  {strat:7s}: {len(grp)} trades, {gw / len(grp) * 100:.0f}% win, "
              f"${grp.pnl.sum():+.2f}")
    print("=" * 52)
    print("⚠️  Synthetic option prices — real spreads, IV moves, and slippage")
    print("    will reduce these numbers. Validate in paper mode first.")


def report_equity(trades: list[EqTrade], label: str = "equity (shares)"):
    if not trades:
        print(f"[{label}] No trades generated.")
        return
    df = pd.DataFrame([vars(t) for t in trades])
    wins = df[df.pnl > 0]
    losses = df[df.pnl <= 0]

    daily = df.groupby(df.exit_time.apply(lambda t: t.date())).pnl.sum()
    equity = EQ_CAPITAL + daily.cumsum()
    drawdown = equity - equity.cummax()
    daily_ret = daily / EQ_CAPITAL
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
              if len(daily_ret) > 1 and daily_ret.std() > 0 else float("nan"))
    total = df.pnl.sum()

    print("=" * 52)
    print(f"BACKTEST RESULTS — {label}  (real historical share prices)")
    print("=" * 52)
    print(f"Capital/BP:      ${EQ_CAPITAL:,.0f} / ${EQ_CAPITAL * EQ_LEVERAGE:,.0f}  ({EQ_LEVERAGE:.0f}:1 intraday)")
    print(f"Period:          {df.entry_time.min().date()} → {df.exit_time.max().date()}")
    print(f"Total trades:    {len(df)}")
    print(f"Win rate:        {len(wins) / len(df) * 100:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"Avg trade:       ${df.pnl.mean():+.2f}   ({df.pnl_pct.mean():+.3f}% of notional)")
    print(f"Avg profit:      ${wins.pnl.mean():+.2f}" if len(wins) else "Avg profit:      n/a")
    print(f"Avg loss:        ${losses.pnl.mean():+.2f}" if len(losses) else "Avg loss:        n/a")
    print(f"Total P&L:       ${total:+.2f}   ({total / EQ_CAPITAL * 100:+.1f}% on ${EQ_CAPITAL:,.0f})")
    print(f"Max drawdown:    ${drawdown.min():.2f}  ({drawdown.min() / EQ_CAPITAL * 100:.1f}% of capital)")
    print(f"Sharpe (ann.):   {sharpe:.2f}")
    print(f"Exits:           {df.exit_reason.value_counts().to_dict()}")
    for strat, grp in df.groupby("strategy"):
        gw = (grp.pnl > 0).sum()
        print(f"  {strat:7s}: {len(grp)} trades, {gw / len(grp) * 100:.0f}% win, "
              f"${grp.pnl.sum():+.2f}")
    print("=" * 52)
    print("✓ Real share prices; only friction (penny spread + slippage) modeled.")
    print("  In-sample on a short window — confirm forward in paper before trusting.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=60, help="trading days to test")
    parser.add_argument("--mode", choices=["long", "spread", "both"], default="long",
                        help="options structure: single-long, debit-spread, or both")
    parser.add_argument("--instrument", choices=["options", "equity", "both"],
                        default="both", help="options, equity shares, or both head-to-head")
    args = parser.parse_args()

    client = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
    all_bars = {}
    for sym in config.UNIVERSE:
        if sym not in ASSUMED_IV:
            log.warning("%s: no assumed IV configured — skipping", sym)
            continue
        bars = fetch_bars(client, sym, args.days)
        if not bars.empty:
            all_bars[sym] = bars
            print(f"Loaded {len(bars)} bars for {sym}")

    if args.instrument in ("options", "both"):
        runs = ["long", "spread"] if args.mode == "both" else [args.mode]
        labels = {"long": "options single-long",
                  "spread": f"options debit-spread ({SPREAD_WIDTH_STEPS}-wide)"}
        for m in runs:
            report(simulate(all_bars, mode=m), label=labels[m])
    if args.instrument in ("equity", "both"):
        report_equity(simulate_equity(all_bars))


if __name__ == "__main__":
    main()
