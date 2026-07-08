"""Background daemon thread for the Alpaca TradingStream.

Pushes order fill/cancel/reject events to a thread-safe queue so the main
loop can consume them without polling get_order_by_id().
"""
from __future__ import annotations

import logging
import queue
from dataclasses import dataclass

from alpaca.trading.stream import TradingStream

import config
from streaming.base import BaseStreamThread

log = logging.getLogger("streaming.trading")


@dataclass
class StreamOrderEvent:
    order_id: str
    status: str          # "filled", "canceled", "rejected", etc.
    filled_avg_price: float | None
    raw: dict


class TradingStreamThread(BaseStreamThread):

    _thread_name = "TradingStream"

    def __init__(self):
        super().__init__()
        self._queue: queue.Queue[StreamOrderEvent] = queue.Queue()

    def drain_events(self) -> list[StreamOrderEvent]:
        events = []
        while True:
            try:
                events.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return events

    def _run_stream(self):
        stream = TradingStream(
            config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY,
            paper=config.PAPER)

        async def on_trade_update(data):
            try:
                event_data = data if isinstance(data, dict) else data.__dict__
                order_data = event_data.get("order", event_data)

                order_id = str(order_data.get("id", ""))
                event_type = str(event_data.get("event", "")).lower()

                filled_avg_price = None
                if event_type in ("fill", "filled"):
                    raw_price = (event_data.get("price")
                                 or order_data.get("filled_avg_price"))
                    if raw_price is not None:
                        filled_avg_price = float(raw_price)

                status_map = {
                    "fill": "filled", "filled": "filled",
                    "canceled": "canceled", "cancelled": "canceled",
                    "rejected": "rejected", "expired": "expired",
                    "suspended": "suspended",
                    "partial_fill": "partially_filled",
                    "new": "new", "accepted": "accepted",
                }
                status = status_map.get(event_type, event_type)

                evt = StreamOrderEvent(
                    order_id=order_id,
                    status=status,
                    filled_avg_price=filled_avg_price,
                    raw=event_data if isinstance(event_data, dict) else {},
                )
                self._queue.put(evt)
                if status in ("filled", "canceled", "rejected", "expired"):
                    log.info("Stream event: %s order %s (fill $%s)",
                             status, order_id[:8],
                             f"{filled_avg_price:.2f}" if filled_avg_price else "n/a")
            except Exception:
                log.exception("Error processing trade_update event")

        stream.subscribe_trade_updates(on_trade_update)
        self._connected.set()
        log.info("TradingStream connected")

        try:
            stream.run()
        finally:
            self._connected.clear()
