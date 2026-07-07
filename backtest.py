"""backtest.py — backtest harness using underlying OHLCV + Black-Scholes
estimated option prices.

Methodology / limitations (read before trusting numbers):
  - Underlying 15-min bars come from Alpaca historical data.
  - Option prices are SYNTHESIZED with Black-Scholes at a fixed per-symbol IV
    (no IV crush/expansion modeling, no real spreads/OI). Real fills will be
    worse — treat results as an upper bound on the strategy, not a forecast.
  - Entry at synthetic ask (mid + half spread), exit at synthetic bid.
  - Same rules as live: momentum + RSI(5) + rel volume, +40%/-30%, 3:45 flat,
    daily -$75 halt, 2-loss 30-min pause.

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
SIM_SPREAD = 0.06                 # synthetic bid/ask spread (passes the $0.10 filter)
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


def pick_strike(symbol: str, spot: float, direction: str) -> float:
    """1 strike OTM on the symbol's typical strike grid."""
    step = STRIKE_STEP.get(symbol, 1.0)
    if direction == "call":
        return math.floor(spot / step) * step + step
    return math.ceil(spot / step) * step - step


def simulate(all_bars: dict[str, pd.DataFrame]) -> list[SimTrade]:
    trades: list[SimTrade] = []
    adv = {sym: avg_daily_volume(b) for sym, b in all_bars.items()}

    # Pre-merge all symbols into one sorted DataFrame with a "sym" column.
    # Much faster than per-day iterrows() + list sort.
    frames = []
    for sym, bars in all_bars.items():
        f = bars.copy()
        f["sym"] = sym
        frames.append(f)
    if not frames:
        return trades
    merged = pd.concat(frames).sort_index()
    merged["_date"] = merged.index.date
    days = sorted(merged["_date"].unique())

    for day in days:
        daily_pnl = 0.0
        halted = False
        consec_losses = 0
        paused_until: dt.datetime | None = None
        open_pos: dict[str, dict] = {}      # symbol -> position state
        cooldown: dict[str, dt.datetime] = {}

        day_slice = merged[merged["_date"] == day]

        for row in day_slice.itertuples():
            ts = row.Index
            t = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
            sym = row.sym
            force_close = t.time() >= dt.time(15, 45)

            # ---- manage an open position in this symbol ----
            if sym in open_pos:
                pos = open_pos[sym]
                tte = max((pos["expiry"] - t).total_seconds() / (365 * 86400), 1e-6)
                mid = black_scholes(row.close, pos["strike"], tte,
                                    ASSUMED_IV[sym], pos["direction"])
                bid = max(0.01, mid - SIM_SPREAD / 2)
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
                                           pnl, reason, pos.get("strategy", "scalp")))
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

            bars = all_bars[sym]
            history = bars[bars.index <= ts]
            if len(history) < config.RSI_PERIOD + 2 or row.open <= 0:
                continue
            momentum = (row.close - row.open) / row.open * 100
            rsi = rsi_last(history["close"], config.RSI_PERIOD)

            # Strategy 1: scalp (one strong candle)
            direction, strategy = None, None
            if abs(momentum) >= config.MOMENTUM_PCT:
                if momentum > 0 and rsi > config.RSI_CALL_MIN:
                    direction, strategy = "call", "scalp"
                elif momentum < 0 and rsi < config.RSI_PUT_MAX:
                    direction, strategy = "put", "scalp"

            # Strategy 2: runner (day trend + new session high/low)
            today_hist = history[history.index.date == day]
            if (direction is None and config.RUNNER_ENABLED
                    and len(today_hist) >= 3 and today_hist["open"].iloc[0] > 0):
                day_change = ((row.close - today_hist["open"].iloc[0])
                              / today_hist["open"].iloc[0] * 100)
                tol = config.RUNNER_BREAKOUT_TOL
                if (day_change >= config.RUNNER_DAY_PCT
                        and row.close > row.open
                        and row.close >= today_hist["high"].iloc[:-1].max() * (1 - tol)
                        and rsi > config.RUNNER_RSI_MIN):
                    direction, strategy = "call", "runner"
                elif (day_change <= -config.RUNNER_DAY_PCT
                        and row.close < row.open
                        and row.close <= today_hist["low"].iloc[:-1].min() * (1 + tol)
                        and rsi < config.RUNNER_RSI_MAX):
                    direction, strategy = "put", "runner"
            if direction is None:
                continue

            # VWAP direction filter (mirrors scanner)
            if config.VWAP_FILTER:
                vwap = getattr(row, "session_vwap", None)
                if vwap is None or pd.isna(vwap):
                    continue
                if direction == "call" and row.close <= vwap:
                    continue
                if direction == "put" and row.close >= vwap:
                    continue

            # Relative volume (time-adjusted vs 20-day avg)
            avg = adv[sym].get(day)
            if avg is None or pd.isna(avg) or avg <= 0:
                continue
            today_so_far = history[history.index.date == day]["volume"].sum()
            elapsed = max(((t - t.replace(hour=9, minute=30)).total_seconds()) / 23400, 0.02)
            if today_so_far / (avg * elapsed) < config.REL_VOLUME_MIN:
                continue

            # Price the synthetic contract
            strike = pick_strike(sym, row.close, direction)
            expiry = (t + dt.timedelta(days=SIM_DTE)).replace(hour=16, minute=0)
            tte = (expiry - t).total_seconds() / (365 * 86400)
            mid = black_scholes(row.close, strike, tte, ASSUMED_IV[sym], direction)
            ask = mid + SIM_SPREAD / 2
            if ask * 100 > config.MAX_TRADE_COST or ask < 0.05:
                continue
            if daily_pnl - ask * 100 * config.STOP_LOSS_PCT / 100 <= -config.MAX_DAILY_LOSS:
                continue

            open_pos[sym] = {"direction": direction, "strike": strike,
                             "entry_price": ask, "entry_time": t, "expiry": expiry,
                             "strategy": strategy,
                             "tp": (config.RUNNER_TAKE_PROFIT_PCT if strategy == "runner"
                                    else config.TAKE_PROFIT_PCT)}
            cooldown[sym] = t

    return trades


# ---------------- stats ----------------

def report(trades: list[SimTrade]):
    if not trades:
        print("No trades generated — try more --days or loosen filters.")
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
    print("BACKTEST RESULTS  (synthetic Black-Scholes pricing)")
    print("=" * 52)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=60, help="trading days to test")
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

    trades = simulate(all_bars)
    report(trades)


if __name__ == "__main__":
    main()
