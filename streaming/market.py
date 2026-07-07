"""Background daemon threads for Alpaca stock bar and option quote streams.

StockBarStreamThread — receives completed 15-min bars via StockDataStream.
OptionQuoteStreamThread — receives real-time option quotes via OptionDataStream.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass

from alpaca.data.live.stock import StockDataStream
from alpaca.data.live.option import OptionDataStream

import config

log = logging.getLogger("streaming.market")


@dataclass
class StreamBar:
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float | None
    timestamp: object  # pd.Timestamp or datetime


class StockBarStreamThread:
    """Daemon thread that subscribes to completed bars for config.UNIVERSE."""

    def __init__(self):
        self._queue: queue.Queue[StreamBar] = queue.Queue()
        self._connected = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        self._thread = threading.Thread(
            target=self._run, name="StockDataStream", daemon=True)
        self._thread.start()
        log.info("StockBarStreamThread started")

    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        log.info("StockBarStreamThread stopped")

    def is_connected(self) -> bool:
        return self._connected.is_set()

    def drain_bars(self) -> list[StreamBar]:
        bars = []
        while True:
            try:
                bars.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return bars

    def _run(self):
        while not self._stop_event.is_set():
            try:
                self._run_stream()
            except Exception:
                self._connected.clear()
                if self._stop_event.is_set():
                    break
                log.exception("StockDataStream error — reconnecting in 5s")
                time.sleep(5)

    def _run_stream(self):
        stream = StockDataStream(
            config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)

        async def on_bar(bar):
            try:
                sb = StreamBar(
                    symbol=bar.symbol,
                    open=float(bar.open),
                    high=float(bar.high),
                    low=float(bar.low),
                    close=float(bar.close),
                    volume=float(bar.volume),
                    vwap=float(bar.vwap) if hasattr(bar, "vwap") and bar.vwap else None,
                    timestamp=bar.timestamp,
                )
                self._queue.put(sb)
            except Exception:
                log.exception("Error processing stock bar event")

        stream.subscribe_bars(on_bar, *list(config.UNIVERSE))
        self._connected.set()
        log.info("StockDataStream connected (universe: %s)",
                 " ".join(config.UNIVERSE))

        try:
            stream.run()
        finally:
            self._connected.clear()


class OptionQuoteStreamThread:
    """Daemon thread that subscribes to real-time option quotes.

    Latest bid/ask stored in a dict protected by a Lock (consumers want the
    *latest* value, not every historical tick).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._quotes: dict[str, tuple[float, float]] = {}
        self._connected = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._stream: OptionDataStream | None = None
        self._subscribed: set[str] = set()
        self._pending_subs: list[str] = []
        self._pending_unsubs: list[str] = []
        self._sub_lock = threading.Lock()

    def start(self):
        self._thread = threading.Thread(
            target=self._run, name="OptionDataStream", daemon=True)
        self._thread.start()
        log.info("OptionQuoteStreamThread started")

    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        log.info("OptionQuoteStreamThread stopped")

    def is_connected(self) -> bool:
        return self._connected.is_set()

    def get_quote(self, symbol: str) -> tuple[float, float] | None:
        with self._lock:
            return self._quotes.get(symbol)

    def get_quotes(self, symbols: list[str]) -> dict[str, tuple[float, float]]:
        with self._lock:
            return {s: self._quotes[s] for s in symbols if s in self._quotes}

    def subscribe(self, symbols: list[str]):
        with self._sub_lock:
            new = [s for s in symbols if s not in self._subscribed]
            if new:
                self._pending_subs.extend(new)
                self._subscribed.update(new)
                log.info("Option quote subscribe queued: %s", new)

    def unsubscribe(self, symbols: list[str]):
        with self._sub_lock:
            removing = [s for s in symbols if s in self._subscribed]
            if removing:
                self._pending_unsubs.extend(removing)
                self._subscribed -= set(removing)
                with self._lock:
                    for s in removing:
                        self._quotes.pop(s, None)
                log.info("Option quote unsubscribe queued: %s", removing)

    def _run(self):
        while not self._stop_event.is_set():
            try:
                self._run_stream()
            except Exception:
                self._connected.clear()
                if self._stop_event.is_set():
                    break
                log.exception("OptionDataStream error — reconnecting in 5s")
                time.sleep(5)

    def _run_stream(self):
        stream = OptionDataStream(
            config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
        self._stream = stream

        async def on_quote(quote):
            try:
                bid = float(quote.bid_price or 0)
                ask = float(quote.ask_price or 0)
                if bid > 0 or ask > 0:
                    with self._lock:
                        self._quotes[quote.symbol] = (bid, ask)
            except Exception:
                log.exception("Error processing option quote event")

        # On (re)connect, subscribe to all symbols that should be active.
        # This handles both initial connection and reconnection after failure.
        with self._sub_lock:
            initial = list(self._subscribed)
            self._pending_subs.clear()
            self._pending_unsubs.clear()
        if initial:
            stream.subscribe_quotes(on_quote, *initial)
            log.info("OptionDataStream subscribed to %d symbols on connect",
                     len(initial))

        self._connected.set()
        log.info("OptionDataStream connected")

        try:
            stream.run()
        finally:
            self._connected.clear()
            self._stream = None
