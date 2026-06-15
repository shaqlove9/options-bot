"""scanner.py — scans the universe every 30s for momentum signals.

Signal rules (all must pass):
  - Latest 15-min candle moved > MOMENTUM_PCT (close vs open)
  - RSI(5) on 15-min closes: > 65 for calls, < 35 for puts
  - Underlying volume today >= 1.5x its time-adjusted 20-day average
"""
import datetime as dt
import logging
import time
from dataclasses import dataclass

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

import config
from utils import ET, now_et, session_elapsed_fraction

log = logging.getLogger("scanner")

# RSI: TA-Lib if available, otherwise a pure-pandas Wilder RSI (identical math).
try:
    import talib

    def rsi_last(closes: pd.Series, period: int) -> float:
        values = talib.RSI(closes.to_numpy(dtype=float), timeperiod=period)
        return float(values[-1])
except ImportError:
    def rsi_last(closes: pd.Series, period: int) -> float:
        delta = closes.diff()
        gain = delta.clip(lower=0.0).ewm(alpha=1 / period, adjust=False).mean()
        loss = (-delta.clip(upper=0.0)).ewm(alpha=1 / period, adjust=False).mean()
        rs = gain / loss.replace(0.0, float("nan"))
        return float((100 - 100 / (1 + rs)).iloc[-1])


@dataclass
class Signal:
    symbol: str
    direction: str          # "call" | "put"
    strategy: str           # "scalp" (one strong candle) | "runner" (day trend)
    momentum_pct: float     # % move of the triggering 15-min candle
    day_change_pct: float   # % move from today's open
    rsi: float
    rel_volume: float
    spot: float             # latest close of the underlying
    vwap_dist_pct: float    # % distance of spot from session VWAP
    time: dt.datetime

    def reason(self) -> str:
        if self.strategy == "runner":
            return (f"[runner] {self.day_change_pct:+.1f}% on the day, new session "
                    f"{'high' if self.direction == 'call' else 'low'}, "
                    f"RSI(5)={self.rsi:.1f}, rel vol {self.rel_volume:.1f}x")
        return (f"[scalp] {self.momentum_pct:+.2f}% 15m candle, RSI(5)={self.rsi:.1f}, "
                f"rel vol {self.rel_volume:.1f}x, VWAP {self.vwap_dist_pct:+.2f}%")


class Scanner:
    _HEARTBEAT_INTERVAL = 300  # log a "still scanning" line every 5 min when quiet

    def __init__(self, data_client: StockHistoricalDataClient):
        self.data = data_client
        self._last_heartbeat: float = 0.0

    # ---------------- data fetch ----------------

    def _intraday_bars(self, symbol: str) -> pd.DataFrame | None:
        """15-min bars covering today plus enough history for RSI(5)."""
        start = now_et() - dt.timedelta(days=5)
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(15, TimeFrameUnit.Minute),
            start=start,
        )
        try:
            bars = self.data.get_stock_bars(req).df
        except Exception as exc:
            log.warning("%s: intraday bars fetch failed: %s", symbol, exc)
            return None
        if bars.empty:
            return None
        bars = bars.droplevel("symbol") if "symbol" in bars.index.names else bars
        bars.index = bars.index.tz_convert(ET)
        # Regular session only — overnight bars distort RSI and volume.
        return bars.between_time("09:30", "16:00")

    def _daily_bars(self, symbol: str) -> pd.DataFrame | None:
        start = now_et() - dt.timedelta(days=config.VOLUME_LOOKBACK_DAYS * 2)
        req = StockBarsRequest(
            symbol_or_symbols=symbol, timeframe=TimeFrame.Day, start=start,
        )
        try:
            bars = self.data.get_stock_bars(req).df
        except Exception as exc:
            log.warning("%s: daily bars fetch failed: %s", symbol, exc)
            return None
        if bars.empty:
            return None
        return bars.droplevel("symbol") if "symbol" in bars.index.names else bars

    # ---------------- signal checks ----------------

    @staticmethod
    def _session_vwap(intraday: pd.DataFrame) -> float | None:
        """Volume-weighted average price over today's session so far."""
        today = now_et().date()
        bars = intraday[intraday.index.date == today]
        vol = bars["volume"].sum()
        if bars.empty or vol <= 0:
            return None
        # Alpaca bars carry per-bar VWAP; fall back to typical price if absent.
        px = bars["vwap"] if "vwap" in bars else bars[["high", "low", "close"]].mean(axis=1)
        return float((px * bars["volume"]).sum() / vol)

    def _relative_volume(self, symbol: str, intraday: pd.DataFrame) -> float:
        """Today's cumulative volume vs the time-adjusted 20-day daily average."""
        daily = self._daily_bars(symbol)
        if daily is None or len(daily) < config.VOLUME_LOOKBACK_DAYS:
            return 0.0
        today = now_et().date()
        # Exclude today from the average if present.
        hist = daily[daily.index.date < today].tail(config.VOLUME_LOOKBACK_DAYS)
        if hist.empty:
            return 0.0
        avg_daily = float(hist["volume"].mean())
        today_vol = float(intraday[intraday.index.date == today]["volume"].sum())
        elapsed = session_elapsed_fraction(now_et())
        if elapsed <= 0 or avg_daily <= 0:
            return 0.0
        return today_vol / (avg_daily * elapsed)

    def _check_symbol(self, symbol: str) -> Signal | None:
        intraday = self._intraday_bars(symbol)
        if intraday is None or len(intraday) < config.RSI_PERIOD + 2:
            return None

        candle = intraday.iloc[-1]
        if candle["open"] <= 0:
            return None
        spot = float(candle["close"])
        momentum = (candle["close"] - candle["open"]) / candle["open"] * 100
        rsi = rsi_last(intraday["close"], config.RSI_PERIOD)

        today_bars = intraday[intraday.index.date == now_et().date()]
        if today_bars.empty or today_bars["open"].iloc[0] <= 0:
            return None
        day_change = (spot - today_bars["open"].iloc[0]) / today_bars["open"].iloc[0] * 100

        direction, strategy = self._scalp_check(momentum, rsi)
        if direction is None:
            direction, strategy = self._runner_check(today_bars, spot, rsi, day_change)
        if direction is None:
            return None

        rel_vol = self._relative_volume(symbol, intraday)
        if rel_vol < config.REL_VOLUME_MIN:
            log.debug("%s: rel volume %.2fx below %.1fx", symbol, rel_vol, config.REL_VOLUME_MIN)
            return None

        # VWAP direction filter — don't fight the session's average price.
        vwap = self._session_vwap(intraday)
        if vwap is None:
            return None
        vwap_dist = (spot - vwap) / vwap * 100
        if config.VWAP_FILTER:
            if direction == "call" and spot <= vwap:
                log.debug("%s: bullish signal below VWAP (%.2f%%) — skipped",
                          symbol, vwap_dist)
                return None
            if direction == "put" and spot >= vwap:
                log.debug("%s: bearish signal above VWAP (%.2f%%) — skipped",
                          symbol, vwap_dist)
                return None

        return Signal(
            symbol=symbol,
            direction=direction,
            strategy=strategy,
            momentum_pct=momentum,
            day_change_pct=day_change,
            rsi=rsi,
            rel_volume=rel_vol,
            spot=spot,
            vwap_dist_pct=vwap_dist,
            time=now_et(),
        )

    @staticmethod
    def _scalp_check(momentum: float, rsi: float) -> tuple[str | None, str]:
        """Strategy 1: one strong 15-min candle with RSI confirmation."""
        if abs(momentum) >= config.MOMENTUM_PCT:
            if momentum > 0 and rsi > config.RSI_CALL_MIN:
                return "call", "scalp"
            if momentum < 0 and rsi < config.RSI_PUT_MAX:
                return "put", "scalp"
        return None, ""

    @staticmethod
    def _runner_check(today_bars: pd.DataFrame, spot: float, rsi: float,
                      day_change: float) -> tuple[str | None, str]:
        """Strategy 2: stock running hard all day AND still breaking to new
        session highs/lows — bet on momentum continuation."""
        if not config.RUNNER_ENABLED or len(today_bars) < 3:
            return None, ""
        candle = today_bars.iloc[-1]
        tol = config.RUNNER_BREAKOUT_TOL
        if day_change >= config.RUNNER_DAY_PCT:
            still_pushing = (candle["close"] > candle["open"]
                             and spot >= today_bars["high"].iloc[:-1].max() * (1 - tol))
            if still_pushing and rsi > config.RUNNER_RSI_MIN:
                return "call", "runner"
        elif day_change <= -config.RUNNER_DAY_PCT:
            still_pushing = (candle["close"] < candle["open"]
                             and spot <= today_bars["low"].iloc[:-1].min() * (1 + tol))
            if still_pushing and rsi < config.RUNNER_RSI_MAX:
                return "put", "runner"
        return None, ""

    def scan(self) -> list[Signal]:
        """One pass over the universe. Called every SCAN_INTERVAL_SEC by main."""
        signals = []
        for symbol in config.UNIVERSE:
            try:
                sig = self._check_symbol(symbol)
            except Exception:
                log.exception("%s: scan error", symbol)
                continue
            if sig:
                log.info("SIGNAL %s %s — %s", sig.symbol, sig.direction.upper(), sig.reason())
                signals.append(sig)
        now = time.monotonic()
        if not signals and now - self._last_heartbeat >= self._HEARTBEAT_INTERVAL:
            self._last_heartbeat = now
            log.info("Scan heartbeat — no signals in last 5 min, market quiet "
                     "(universe: %s)", " ".join(config.UNIVERSE))
        elif signals:
            self._last_heartbeat = now
        return signals


if __name__ == "__main__":
    # Quick manual test: python scanner.py
    client = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
    for s in Scanner(client).scan():
        print(s)
