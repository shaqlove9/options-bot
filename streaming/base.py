"""Base class for reconnecting WebSocket daemon threads."""
from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod

log = logging.getLogger("streaming.base")


class BaseStreamThread(ABC):
    """Shared skeleton: init, start, stop, is_connected, reconnect loop.

    Subclasses override ``_run_stream()`` with SDK-specific logic and
    set ``_thread_name`` for the daemon thread.
    """

    _thread_name: str = "StreamThread"

    def __init__(self):
        self._connected = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        self._thread = threading.Thread(
            target=self._run, name=self._thread_name, daemon=True)
        self._thread.start()
        log.info("%s started", self._thread_name)

    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        log.info("%s stopped", self._thread_name)

    def is_connected(self) -> bool:
        return self._connected.is_set()

    def _run(self):
        while not self._stop_event.is_set():
            try:
                self._run_stream()
            except Exception:
                self._connected.clear()
                if self._stop_event.is_set():
                    break
                log.exception("%s error — reconnecting in 5s", self._thread_name)
                time.sleep(5)

    @abstractmethod
    def _run_stream(self):
        """Connect and run the SDK stream. Must set/clear ``_connected``."""
