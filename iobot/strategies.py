"""strategies.py — candidate entry signals to A/B in the backtest.

Each is a PURE function with the same signature as `signals.evaluate`:
    fn(symbol, intraday, daily, now) -> Signal | None
so it drops straight into the live `MomentumSignal`-style scan and the backtest with
no other changes. Exits/costs are held constant across all of them — we're isolating
the ENTRY edge.

Long options need real directional follow-through to beat theta, so the candidates
lean toward breakout / trend-continuation rather than mean reversion.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from iobot import config, signals
from iobot.signals import Signal

# ---- tunables (kept local so the backtest can iterate without touching config) ----
ORB_MINUTES = 30          # opening-range window
DONCHIAN_BARS = 8         # intraday breakout lookback (~2h of 15m bars)
TREND_SMA = 20            # daily SMA for the regime filter

# range_scalp: fade the edges of an established intraday channel
RANGE_MIN_BARS = 8        # need ~2h of session before a range is "established"
RANGE_MIN_PCT = 0.30      # day range too small below this (no room to scalp)
RANGE_MAX_PCT = 1.60      # above this it's an expansion/trend day, not a channel
RANGE_EDGE_FRAC = 0.20    # "at the edge" = within this fraction of the range
RANGE_TARGET_FRAC = 0.55  # take profit ~mid/opposite side of the range
RANGE_STOP_FRAC = 0.15    # stop this fraction of the range beyond the edge (range break)


def _today(intraday: pd.DataFrame, now: dt.datetime) -> pd.DataFrame:
    return intraday[intraday.index.date == now.date()]


def _mk(symbol, direction, intraday, daily, now, source) -> Signal | None:
    """Assemble a Signal with the shared feature helpers (spot = last close)."""
    spot = float(intraday.iloc[-1]["close"])
    rel_vol = signals._relative_volume(intraday, daily, now)
    if rel_vol < config.REL_VOLUME_MIN:
        return None
    vwap = signals._session_vwap(intraday, now.date())
    if vwap is None:
        return None
    vwap_dist = (spot - vwap) / vwap * 100
    candle = intraday.iloc[-1]
    momentum = (candle["close"] - candle["open"]) / candle["open"] * 100 if candle["open"] else 0.0
    return Signal(symbol=symbol, direction=direction, spot=spot, momentum_pct=momentum,
                  rsi=signals._wilder_rsi(intraday["close"], config.RSI_PERIOD),
                  rel_volume=rel_vol, vwap_dist_pct=vwap_dist,
                  atr_pct=signals._atr_pct(intraday, spot), time=now, source=source)


# ---------------- candidates ----------------

def momentum(symbol, intraday, daily, now):
    """Baseline: the live 15-min momentum + RSI + relvol + VWAP signal."""
    return signals.evaluate(symbol, intraday, daily, now)


def trend_momentum(symbol, intraday, daily, now):
    """Momentum, but only WITH the daily trend (regime filter) — drops counter-trend
    chop that long premium can't pay theta on."""
    sig = signals.evaluate(symbol, intraday, daily, now)
    if sig is None or daily is None or len(daily) < TREND_SMA + 1:
        return None
    prior = daily[daily.index.date < now.date()]
    if len(prior) < TREND_SMA:
        return None
    sma = float(prior["close"].tail(TREND_SMA).mean())
    trend_up = float(prior["close"].iloc[-1]) > sma
    if (sig.direction == "call") != trend_up:     # keep only trend-aligned
        return None
    sig.source = "trend_momentum"
    return sig


def orb(symbol, intraday, daily, now):
    """Opening-range breakout: after the first ORB_MINUTES, enter on the bar that
    breaks the opening range high (call) / low (put), with relvol + VWAP alignment."""
    today = _today(intraday, now)
    if len(today) < 3:
        return None
    open_t = today.index[0]
    or_end = open_t + dt.timedelta(minutes=ORB_MINUTES)
    if now <= or_end + dt.timedelta(minutes=15):
        return None
    opening = today[today.index < or_end]
    if opening.empty:
        return None
    or_high, or_low = float(opening["high"].max()), float(opening["low"].min())
    spot = float(today.iloc[-1]["close"])
    prev = float(today.iloc[-2]["close"])
    direction = None
    if prev <= or_high < spot:
        direction = "call"
    elif prev >= or_low > spot:
        direction = "put"
    if direction is None:
        return None
    sig = _mk(symbol, direction, intraday, daily, now, "orb")
    if sig is None:
        return None
    if config.VWAP_FILTER and ((direction == "call" and spot <= signals._session_vwap(intraday, now.date()))
                               or (direction == "put" and spot >= signals._session_vwap(intraday, now.date()))):
        return None
    return sig


def donchian(symbol, intraday, daily, now):
    """Intraday Donchian breakout: close breaks the high/low of the last
    DONCHIAN_BARS (prior bars), with relvol + VWAP alignment."""
    today = _today(intraday, now)
    if len(today) < DONCHIAN_BARS + 2:
        return None
    window = today.iloc[-(DONCHIAN_BARS + 1):-1]
    hi, lo = float(window["high"].max()), float(window["low"].min())
    spot = float(today.iloc[-1]["close"])
    prev = float(today.iloc[-2]["close"])
    direction = None
    if prev <= hi < spot:
        direction = "call"
    elif prev >= lo > spot:
        direction = "put"
    if direction is None:
        return None
    sig = _mk(symbol, direction, intraday, daily, now, "donchian")
    if sig is None:
        return None
    vwap = signals._session_vwap(intraday, now.date())
    if config.VWAP_FILTER and ((direction == "call" and spot <= vwap)
                               or (direction == "put" and spot >= vwap)):
        return None
    return sig


def trend_donchian(symbol, intraday, daily, now):
    """Donchian breakout filtered to the daily trend direction."""
    sig = donchian(symbol, intraday, daily, now)
    if sig is None or daily is None:
        return None
    prior = daily[daily.index.date < now.date()]
    if len(prior) < TREND_SMA:
        return None
    sma = float(prior["close"].tail(TREND_SMA).mean())
    trend_up = float(prior["close"].iloc[-1]) > sma
    if (sig.direction == "call") != trend_up:
        return None
    sig.source = "trend_donchian"
    return sig


def range_scalp(symbol, intraday, daily, now):
    """Channel/mean-reversion scalp: once an intraday range is established, fade a
    tag of its edge that closes back inside (rejection), targeting the middle/other
    side. Exits are RANGE-AWARE: target ~mid-range, stop just beyond the edge (range
    break). Counter-trend by design, so it carries its own exit levels."""
    today = _today(intraday, now)
    if len(today) < RANGE_MIN_BARS:
        return None
    dh, dl = float(today["high"].max()), float(today["low"].min())
    rng = dh - dl
    candle = today.iloc[-1]
    spot = float(candle["close"])
    if rng <= 0 or spot <= 0:
        return None
    rng_pct = rng / spot * 100
    if not (RANGE_MIN_PCT <= rng_pct <= RANGE_MAX_PCT):
        return None
    band = RANGE_EDGE_FRAC * rng
    rsi = signals._wilder_rsi(intraday["close"], config.RSI_PERIOD)

    direction = stop = target = None
    # at support: dipped to the low but closed back up inside, oversold -> long call
    if (candle["low"] <= dl + band and candle["close"] > dl
            and candle["close"] >= candle["open"] and rsi < config.RSI_PUT_MAX):
        direction = "call"
        target = dl + RANGE_TARGET_FRAC * rng
        stop = dl - RANGE_STOP_FRAC * rng
    # at resistance: tagged the high but closed back down inside, overbought -> long put
    elif (candle["high"] >= dh - band and candle["close"] < dh
          and candle["close"] <= candle["open"] and rsi > config.RSI_CALL_MIN):
        direction = "put"
        target = dh - RANGE_TARGET_FRAC * rng
        stop = dh + RANGE_STOP_FRAC * rng
    if direction is None:
        return None
    # sanity: target in front, stop behind, with real room
    if direction == "call" and not (stop < spot < target):
        return None
    if direction == "put" and not (target < spot < stop):
        return None

    momentum_pct = (candle["close"] - candle["open"]) / candle["open"] * 100 if candle["open"] else 0.0
    vwap = signals._session_vwap(intraday, now.date())
    return Signal(symbol=symbol, direction=direction, spot=spot, momentum_pct=momentum_pct,
                  rsi=rsi, rel_volume=signals._relative_volume(intraday, daily, now),
                  vwap_dist_pct=(spot - vwap) / vwap * 100 if vwap else 0.0,
                  atr_pct=signals._atr_pct(intraday, spot), time=now, source="range_scalp",
                  stop_level=stop, target_level=target)


REGISTRY = {
    "momentum": momentum,
    "trend_momentum": trend_momentum,
    "orb": orb,
    "donchian": donchian,
    "trend_donchian": trend_donchian,
    "range_scalp": range_scalp,
}
