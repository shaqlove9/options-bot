"""features.py — Phase-1 feature capture for meta-labeling.

At the MOMENT a signal fires (pre-entry) we snapshot a feature vector keyed to the
signal_id and store it append-only. This runs from day one so data accrues before
the model is ever trained.

LABEL-LEAKAGE GUARD: every feature is derived only from data observable at or
before the capture instant. `build()` tracks the newest input timestamp
(`max_asof_ts`); `capture()` refuses to write a row whose `max_asof_ts` is after
`captured_at_ts`. No post-entry information (fills, exits, outcome) ever enters a
feature row — those become the *label*, attached later by journal.py.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import pandas as pd

from iobot import config
from iobot.clock import minutes_to_close, now_et

log = logging.getLogger("features")


class LeakageError(ValueError):
    """A feature row would include data timestamped after the capture instant."""


@dataclass
class FeatureRow:
    signal_id: str
    captured_at_ts: float
    max_asof_ts: float
    features: dict = field(default_factory=dict)


def assert_no_leakage(captured_at_ts: float, max_asof_ts: float, tol: float = 1.0):
    """Hard guard: no input may be newer than the capture instant (1s tolerance)."""
    if max_asof_ts > captured_at_ts + tol:
        raise LeakageError(
            f"feature input as-of {max_asof_ts:.0f} is after capture "
            f"{captured_at_ts:.0f} (+{max_asof_ts - captured_at_ts:.0f}s) — "
            f"would leak future data into the row")


def _ts(value) -> float:
    """Epoch seconds for a pandas/py timestamp (tz-aware or naive treated as UTC)."""
    t = pd.Timestamp(value)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.timestamp()


@dataclass
class RegimeContext:
    """Pre-fetched daily frames + scalars used for regime/streak features.
    Kept as plain inputs so build() is pure and unit-testable."""
    daily: dict[str, pd.DataFrame]   # symbol -> daily OHLCV (index = dates)
    vix: float | None = None
    vix_asof: object | None = None   # timestamp of the vix reading, if any
    win_streak: int = 0
    loss_streak: int = 0
    trades_today: int = 0


def build(signal, ctx: RegimeContext) -> FeatureRow:
    """PURE: assemble the pre-entry feature vector and the newest input timestamp."""
    captured_at = signal.time
    captured_ts = _ts(captured_at)
    asof = captured_ts  # intraday signal features are as-of the signal instant

    feats: dict[str, float] = {
        "minute_of_day": captured_at.hour * 60 + captured_at.minute,
        "minutes_to_close": minutes_to_close(captured_at),
        "direction_is_call": 1.0 if signal.direction == "call" else 0.0,
        "momentum_pct": signal.momentum_pct,
        "rsi": signal.rsi,
        "rel_volume": signal.rel_volume,
        "vwap_dist_pct": signal.vwap_dist_pct,
        "atr_pct": signal.atr_pct,          # recent realized-vol proxy
        "win_streak": float(ctx.win_streak),
        "loss_streak": float(ctx.loss_streak),
        "trades_today": float(ctx.trades_today),
    }

    # Regime: each underlying's last close vs its 20/50 SMA.
    for sym in config.UNIVERSE:
        df = ctx.daily.get(sym)
        if df is None or df.empty or "close" not in df:
            feats[f"{sym}_above_sma20"] = 0.0
            feats[f"{sym}_above_sma50"] = 0.0
            continue
        close = df["close"]
        last = float(close.iloc[-1])
        sma20 = float(close.tail(20).mean())
        sma50 = float(close.tail(50).mean())
        feats[f"{sym}_above_sma20"] = 1.0 if last > sma20 else 0.0
        feats[f"{sym}_above_sma50"] = 1.0 if last > sma50 else 0.0
        asof = max(asof, _ts(df.index[-1]))

    # Volatility regime: VIX if supplied, else a realized-vol proxy from the signal.
    if ctx.vix is not None:
        feats["vix"] = float(ctx.vix)
        if ctx.vix_asof is not None:
            asof = max(asof, _ts(ctx.vix_asof))
    else:
        feats["vix"] = signal.atr_pct * 16.0   # documented proxy when VIX absent

    return FeatureRow(signal_id=signal.signal_id, captured_at_ts=captured_ts,
                      max_asof_ts=asof, features=feats)


def capture(conn, signal, row: FeatureRow):
    """Persist the signal + its feature row, after the leakage guard passes."""
    assert_no_leakage(row.captured_at_ts, row.max_asof_ts)
    conn.execute(
        "INSERT OR IGNORE INTO signals(signal_id, captured_at, symbol, direction, "
        "source, spot) VALUES (?,?,?,?,?,?)",
        (signal.signal_id, signal.time.isoformat(timespec="seconds"), signal.symbol,
         signal.direction, signal.source, signal.spot))
    conn.execute(
        "INSERT OR REPLACE INTO features(signal_id, captured_at_ts, max_asof_ts, "
        "features_json) VALUES (?,?,?,?)",
        (row.signal_id, row.captured_at_ts, row.max_asof_ts, json.dumps(row.features)))
    conn.commit()
    log.info("captured features for %s (%d fields)", row.signal_id, len(row.features))


class FeatureBuilder:
    """Fetches the regime context and builds+captures a feature row per signal."""

    def __init__(self, stock_data):
        self.data = stock_data

    def _daily(self, symbol: str) -> pd.DataFrame | None:
        import datetime as dt

        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
                               start=now_et() - dt.timedelta(days=90))
        try:
            bars = self.data.get_stock_bars(req).df
        except Exception as exc:
            log.warning("%s: regime daily fetch failed: %s", symbol, exc)
            return None
        if bars is None or bars.empty:
            return None
        return bars.droplevel("symbol") if "symbol" in bars.index.names else bars

    def context(self, win_streak: int, loss_streak: int, trades_today: int) -> RegimeContext:
        daily = {s: self._daily(s) for s in config.UNIVERSE}
        daily = {k: v for k, v in daily.items() if v is not None}
        return RegimeContext(daily=daily, win_streak=win_streak,
                             loss_streak=loss_streak, trades_today=trades_today)
