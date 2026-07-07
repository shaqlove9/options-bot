"""Protocol interfaces for data providers.

Consumers depend on these behavior contracts instead of concrete Alpaca SDK
classes.  REST, WebSocket, and hybrid implementations all satisfy the same
protocols via structural subtyping (no inheritance required).
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class StockBarProvider(Protocol):
    def get_bars(self, symbols: list[str], timeframe_minutes: int = 15,
                 lookback_days: int = 5) -> dict[str, pd.DataFrame]: ...


@runtime_checkable
class OptionQuoteProvider(Protocol):
    def get_latest_quotes(self, symbols: list[str]) -> dict[str, tuple[float, float]]:
        """Returns {symbol: (bid, ask)} for each symbol with a valid quote."""
        ...

    def get_latest_quote(self, symbol: str) -> tuple[float, float] | None:
        """Returns (bid, ask) or None if quote unavailable."""
        ...


@runtime_checkable
class OptionSnapshotProvider(Protocol):
    def get_snapshots(self, symbols: list[str]) -> dict: ...


@runtime_checkable
class OrderEventProvider(Protocol):
    def get_order_status(self, order_id: str) -> object | None: ...
