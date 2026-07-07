"""Background daemon thread for the Alpaca TradingStream.

Pushes order fill/cancel/reject events to a thread-safe queue so the main
loop can consume them without polling get_order_by_id().
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass

from alpaca.trading.stream import TradingStream

import config

log = logging.getLogger("streaming.trading")


@dataclass
class StreamOrderEvent:
    order_id: str
    status: str          # "filled", "canceled", "rejected", etc.
    filled_avg_price: float | None
    raw: dict


class TradingStreamThread:
    def __init__(self):
        self._queue: queue.Queue[StreamOrderEvent] = queue.Queue()
        self._connected = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        self._thread = threading.Thread(
            target=self._run, name="TradingStream", daemon=True)
        self._thread.start()
        log.info("TradingStreamThread started")

    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        log.info("TradingStreamThread stopped")

    def is_connected(self) -> bool:
        return self._connected.is_set()

    def drain_events(self) -> list[StreamOrderEvent]:
        events = []
        while True:
            try:
                events.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return events

    def _run(self):
        while not self._stop_event.is_set():
            try:
                self._run_stream()
            except Exception:
                self._connected.clear()
                if self._stop_event.is_set():
                    break
                log.exception("TradingStream error — reconnecting in 5s")
                time.sleep(5)

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
