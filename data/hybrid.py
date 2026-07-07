"""Hybrid data providers: stream-first with REST fallback.

HybridStockBarProvider — enriches a REST-bootstrapped bar cache with stream
    events for faster signal detection.
HybridOptionQuoteProvider — reads from the option quote stream, falls back to
    REST for symbols without a stream quote.
"""
from __future__ import annotations

import logging
import time

import pandas as pd

import config
from data.protocols import StockBarProvider, OptionQuoteProvider
from data.rest import RestStockBarProvider, RestOptionQuoteProvider
from streaming.market import StockBarStreamThread, OptionQuoteStreamThread
from utils import ET, now_et

log = logging.getLogger("data.hybrid")


class HybridStockBarProvider:
    """StockBarProvider that merges stream bars into a REST-bootstrapped cache."""

    # Cache TTL comes from config.STREAM_CACHE_MAX_AGE_SEC

    def __init__(self, stream: StockBarStreamThread,
                 rest: RestStockBarProvider):
        self._stream = stream
        self._rest = rest
        self._cache: dict[str, pd.DataFrame] = {}
        self._cache_time: dict[str, float] = {}  # symbol -> monotonic timestamp

    def get_bars(self, symbols: list[str], timeframe_minutes: int = 15,
                 lookback_days: int = 5) -> dict[str, pd.DataFrame]:
        # Daily bars always come from REST (stream only provides intraday)
        if timeframe_minutes >= 1440:
            return self._rest.get_bars(symbols, timeframe_minutes, lookback_days)

        # Drain new stream bars into cache
        for bar in self._stream.drain_bars():
            if bar.symbol not in self._cache:
                continue
            ts = pd.Timestamp(bar.timestamp).tz_convert(ET)
            row = pd.DataFrame({
                "open": [bar.open], "high": [bar.high],
                "low": [bar.low], "close": [bar.close],
                "volume": [bar.volume],
                "vwap": [bar.vwap if bar.vwap else bar.close],
            }, index=pd.DatetimeIndex([ts]))
            existing = self._cache[bar.symbol]
            # Replace if same timestamp, append if new
            if ts in existing.index:
                existing.loc[ts] = row.iloc[0]
            else:
                self._cache[bar.symbol] = pd.concat([existing, row]).sort_index()
            self._cache_time[bar.symbol] = time.monotonic()

        # Expire stale cache entries
        now_mono = time.monotonic()
        stale = [s for s in symbols
                 if s in self._cache
                 and now_mono - self._cache_time.get(s, 0) > config.STREAM_CACHE_MAX_AGE_SEC]
        for s in stale:
            del self._cache[s]
            self._cache_time.pop(s, None)

        # Bootstrap from REST if cache is empty or stream disconnected
        need_bootstrap = [s for s in symbols if s not in self._cache]
        if need_bootstrap or not self._stream.is_connected():
            rest_bars = self._rest.get_bars(symbols, timeframe_minutes, lookback_days)
            for sym, bars in rest_bars.items():
                self._cache[sym] = bars
                self._cache_time[sym] = now_mono
            if not self._stream.is_connected():
                return rest_bars

        return {s: self._cache[s] for s in symbols if s in self._cache}


class HybridOptionQuoteProvider:
    """OptionQuoteProvider that reads from stream, REST fallback for misses."""

    def __init__(self, stream: OptionQuoteStreamThread,
                 rest: RestOptionQuoteProvider):
        self._stream = stream
        self._rest = rest

    def get_latest_quotes(self, symbols: list[str]) -> dict[str, tuple[float, float]]:
        if not symbols:
            return {}
        # Try stream first
        result = self._stream.get_quotes(symbols)
        # Fall back to REST for missing symbols
        missing = [s for s in symbols if s not in result]
        if missing:
            rest_quotes = self._rest.get_latest_quotes(missing)
            result.update(rest_quotes)
        return result

    def get_latest_quote(self, symbol: str) -> tuple[float, float] | None:
        quote = self._stream.get_quote(symbol)
        if quote is not None:
            return quote
        return self._rest.get_latest_quote(symbol)

    def subscribe(self, symbols: list[str]):
        self._stream.subscribe(symbols)

    def unsubscribe(self, symbols: list[str]):
        self._stream.unsubscribe(symbols)
