"""REST implementations of the data provider protocols.

Each class wraps existing Alpaca SDK clients and applies the @retry decorator
for transient network errors.
"""
from __future__ import annotations

import datetime as dt
import logging

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import (OptionLatestQuoteRequest,
                                  OptionSnapshotRequest, StockBarsRequest)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient

from utils import ET, now_et, retry

log = logging.getLogger("data.rest")


class RestStockBarProvider:
    def __init__(self, client: StockHistoricalDataClient):
        self._client = client

    @retry(max_attempts=3, delay=2.0, backoff=2.0,
           exceptions=(ConnectionError, OSError))
    def get_bars(self, symbols: list[str], timeframe_minutes: int = 15,
                 lookback_days: int = 5) -> dict[str, pd.DataFrame]:
        start = now_et() - dt.timedelta(days=lookback_days)
        is_daily = timeframe_minutes >= 1440
        tf = TimeFrame.Day if is_daily else TimeFrame(timeframe_minutes, TimeFrameUnit.Minute)
        req = StockBarsRequest(
            symbol_or_symbols=symbols, timeframe=tf, start=start,
        )
        all_bars = self._client.get_stock_bars(req).df
        if all_bars.empty:
            return {}
        result: dict[str, pd.DataFrame] = {}
        if "symbol" in all_bars.index.names:
            for sym in all_bars.index.get_level_values("symbol").unique():
                bars = all_bars.xs(sym, level="symbol")
                bars.index = bars.index.tz_convert(ET)
                if not is_daily:
                    bars = bars.between_time("09:30", "16:00")
                if not bars.empty:
                    result[sym] = bars
        else:
            all_bars.index = all_bars.index.tz_convert(ET)
            if not is_daily:
                all_bars = all_bars.between_time("09:30", "16:00")
            if not all_bars.empty and len(symbols) == 1:
                result[symbols[0]] = all_bars
        return result


class RestOptionQuoteProvider:
    def __init__(self, client: OptionHistoricalDataClient):
        self._client = client

    @retry(max_attempts=3, delay=2.0, backoff=2.0,
           exceptions=(ConnectionError, OSError))
    def get_latest_quotes(self, symbols: list[str]) -> dict[str, tuple[float, float]]:
        if not symbols:
            return {}
        quotes = self._client.get_option_latest_quote(
            OptionLatestQuoteRequest(symbol_or_symbols=symbols))
        result = {}
        for sym, q in quotes.items():
            bid = float(q.bid_price or 0)
            ask = float(q.ask_price or 0)
            if bid > 0 or ask > 0:
                result[sym] = (bid, ask)
        return result

    def get_latest_quote(self, symbol: str) -> tuple[float, float] | None:
        result = self.get_latest_quotes([symbol])
        return result.get(symbol)


class RestOptionSnapshotProvider:
    def __init__(self, client: OptionHistoricalDataClient):
        self._client = client

    @retry(max_attempts=3, delay=2.0, backoff=2.0,
           exceptions=(ConnectionError, OSError))
    def get_snapshots(self, symbols: list[str]) -> dict:
        return self._client.get_option_snapshot(
            OptionSnapshotRequest(symbol_or_symbols=symbols))


class RestOrderEventProvider:
    def __init__(self, client: TradingClient):
        self._client = client

    def get_order_status(self, order_id: str) -> object | None:
        try:
            return self._fetch_order(order_id)
        except Exception:
            return None

    @retry(max_attempts=3, delay=2.0, backoff=2.0,
           exceptions=(ConnectionError, OSError))
    def _fetch_order(self, order_id: str):
        return self._client.get_order_by_id(order_id)
