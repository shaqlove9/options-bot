"""signals.py — swappable entry-signal interface + a momentum signal.

A SignalSource turns market data into at most one Signal per underlying per scan.
The bot only opens a position on a confirmed Signal. Swapping strategies means
implementing `SignalSource.scan()`; nothing downstream changes.

The meta-labeling layer NEVER lives here — it only filters/sizes the signals this
module emits. No model generates signals.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from iobot import config
from iobot.clock import ET, now_et, session_elapsed_fraction

log = logging.getLogger("signals")


def _wilder_rsi(closes: pd.Series, period: int) -> float:
    delta = closes.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0.0, float("nan"))
    return float((100 - 100 / (1 + rs)).iloc[-1])


@dataclass
class Signal:
    symbol: str
    direction: str          # "call" | "put"
    spot: float             # latest underlying price (entry reference for stops)
    momentum_pct: float
    rsi: float
    rel_volume: float
    vwap_dist_pct: float
    atr_pct: float
    time: dt.datetime
    source: str = "momentum"

    @property
    def signal_id(self) -> str:
        """Deterministic id linking the pre-entry feature snapshot to the trade."""
        return f"{self.symbol}-{self.time.strftime('%Y%m%dT%H%M%S')}-{self.direction}"

    def reason(self) -> str:
        return (f"{self.momentum_pct:+.2f}% 15m, RSI(5)={self.rsi:.1f}, "
                f"relvol {self.rel_volume:.1f}x, VWAP {self.vwap_dist_pct:+.2f}%")


class SignalSource:
    """Interface. Implement scan() to plug in a different strategy."""

    name = "base"

    def scan(self) -> list[Signal]:
        raise NotImplementedError


class MomentumSignal(SignalSource):
    """A strong 15-minute momentum candle confirmed by RSI, relative volume, and
    VWAP alignment. Long call on bullish thrust, long put on bearish."""

    name = "momentum"

    def __init__(self, stock_data: StockHistoricalDataClient):
        self.data = stock_data

    def _intraday(self, symbol: str) -> pd.DataFrame | None:
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(15, TimeFrameUnit.Minute),
            start=now_et() - dt.timedelta(days=5))
        try:
            bars = self.data.get_stock_bars(req).df
        except Exception as exc:
            log.warning("%s: intraday bars failed: %s", symbol, exc)
            return None
        if bars.empty:
            return None
        bars = bars.droplevel("symbol") if "symbol" in bars.index.names else bars
        bars.index = bars.index.tz_convert(ET)
        return bars.between_time("09:30", "16:00")

    def _daily(self, symbol: str) -> pd.DataFrame | None:
        req = StockBarsRequest(
            symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
            start=now_et() - dt.timedelta(days=config.VOLUME_LOOKBACK_DAYS * 2))
        try:
            bars = self.data.get_stock_bars(req).df
        except Exception as exc:
            log.warning("%s: daily bars failed: %s", symbol, exc)
            return None
        if bars is None or bars.empty:
            return None
        return bars.droplevel("symbol") if "symbol" in bars.index.names else bars

    @staticmethod
    def _vwap(intraday: pd.DataFrame) -> float | None:
        today = now_et().date()
        bars = intraday[intraday.index.date == today]
        vol = bars["volume"].sum()
        if bars.empty or vol <= 0:
            return None
        px = bars["vwap"] if "vwap" in bars else bars[["high", "low", "close"]].mean(axis=1)
        return float((px * bars["volume"]).sum() / vol)

    @staticmethod
    def _atr_pct(intraday: pd.DataFrame, spot: float) -> float:
        """ATR(14) over 15m bars as a % of spot — a realized-vol proxy."""
        if len(intraday) < 15 or spot <= 0:
            return 0.0
        h, l, c = intraday["high"], intraday["low"], intraday["close"].shift(1)
        tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        return float(atr / spot * 100) if pd.notna(atr) else 0.0

    def _rel_volume(self, symbol: str, intraday: pd.DataFrame) -> float:
        daily = self._daily(symbol)
        if daily is None or len(daily) < config.VOLUME_LOOKBACK_DAYS:
            return 0.0
        today = now_et().date()
        hist = daily[daily.index.date < today].tail(config.VOLUME_LOOKBACK_DAYS)
        if hist.empty:
            return 0.0
        avg = float(hist["volume"].mean())
        today_vol = float(intraday[intraday.index.date == today]["volume"].sum())
        elapsed = session_elapsed_fraction(now_et())
        if elapsed <= 0 or avg <= 0:
            return 0.0
        return today_vol / (avg * elapsed)

    def _check(self, symbol: str) -> Signal | None:
        intraday = self._intraday(symbol)
        if intraday is None or len(intraday) < config.RSI_PERIOD + 2:
            return None
        candle = intraday.iloc[-1]
        if candle["open"] <= 0:
            return None
        spot = float(candle["close"])
        momentum = (candle["close"] - candle["open"]) / candle["open"] * 100
        rsi = _wilder_rsi(intraday["close"], config.RSI_PERIOD)

        direction = None
        if abs(momentum) >= config.MOMENTUM_PCT:
            if momentum > 0 and rsi > config.RSI_CALL_MIN:
                direction = "call"
            elif momentum < 0 and rsi < config.RSI_PUT_MAX:
                direction = "put"
        if direction is None:
            return None

        rel_vol = self._rel_volume(symbol, intraday)
        if rel_vol < config.REL_VOLUME_MIN:
            return None

        vwap = self._vwap(intraday)
        if vwap is None:
            return None
        vwap_dist = (spot - vwap) / vwap * 100
        if config.VWAP_FILTER:
            if direction == "call" and spot <= vwap:
                return None
            if direction == "put" and spot >= vwap:
                return None

        return Signal(
            symbol=symbol, direction=direction, spot=spot, momentum_pct=momentum,
            rsi=rsi, rel_volume=rel_vol, vwap_dist_pct=vwap_dist,
            atr_pct=self._atr_pct(intraday, spot), time=now_et())

    def scan(self) -> list[Signal]:
        out = []
        for symbol in config.UNIVERSE:
            try:
                sig = self._check(symbol)
            except Exception:
                log.exception("%s: scan error", symbol)
                continue
            if sig:
                log.info("SIGNAL %s %s — %s", sig.symbol, sig.direction.upper(), sig.reason())
                out.append(sig)
        return out
