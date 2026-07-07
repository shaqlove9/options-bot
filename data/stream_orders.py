"""Stream-backed OrderEventProvider with REST fallback.

Consumes events from TradingStreamThread and serves them to the Executor
via the OrderEventProvider protocol. Falls back to REST when the stream
is disconnected.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

from alpaca.trading.enums import OrderStatus

from data.protocols import OrderEventProvider
from data.rest import RestOrderEventProvider
from streaming.trading import TradingStreamThread

log = logging.getLogger("data.stream_orders")


class StreamOrderEventProvider:
    """OrderEventProvider that prefers stream events, REST fallback."""

    def __init__(self, stream: TradingStreamThread,
                 rest_fallback: RestOrderEventProvider):
        self._stream = stream
        self._rest = rest_fallback
        self._event_cache: dict[str, object] = {}

    _STATUS_MAP = {
        "filled": OrderStatus.FILLED,
        "canceled": OrderStatus.CANCELED,
        "rejected": OrderStatus.REJECTED,
        "expired": OrderStatus.EXPIRED,
        "suspended": OrderStatus.SUSPENDED,
        "new": OrderStatus.NEW,
        "accepted": OrderStatus.ACCEPTED,
        "partially_filled": OrderStatus.PARTIALLY_FILLED,
    }

    def consume_stream_events(self):
        """Drain the stream queue into the local cache. Call once per cycle."""
        for evt in self._stream.drain_events():
            # Map string status to OrderStatus enum so Executor comparisons
            # (e.g. order.status == OrderStatus.FILLED) work identically.
            status = self._STATUS_MAP.get(evt.status, evt.status)
            obj = SimpleNamespace(
                status=status,
                filled_avg_price=evt.filled_avg_price,
            )
            self._event_cache[evt.order_id] = obj

    def get_order_status(self, order_id: str) -> object | None:
        # 1. Check stream event cache
        cached = self._event_cache.pop(order_id, None)
        if cached is not None:
            return cached

        # 2. Stream connected but no event yet — order still pending
        if self._stream.is_connected():
            return None

        # 3. Stream disconnected — fall back to REST
        log.debug("Stream disconnected, falling back to REST for order %s",
                  order_id[:8])
        return self._rest.get_order_status(order_id)
