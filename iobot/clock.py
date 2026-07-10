"""clock.py — time/session helpers in market (ET) time + logging setup."""
from __future__ import annotations

import datetime as dt
import logging
import logging.handlers
from zoneinfo import ZoneInfo

from iobot import config

ET = ZoneInfo(config.TZ)

_root = logging.getLogger()
if not _root.handlers:
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.handlers.RotatingFileHandler(
                config.LOG_FILE, maxBytes=2_000_000, backupCount=2, encoding="utf-8"),
        ],
    )


def now_et() -> dt.datetime:
    return dt.datetime.now(tz=ET)


def _at(t: dt.datetime, hm: tuple[int, int]) -> dt.datetime:
    return t.replace(hour=hm[0], minute=hm[1], second=0, microsecond=0)


def in_entry_window(t: dt.datetime) -> bool:
    """Inside the configured entry window (default 9:45-15:30 ET, weekday), and NOT
    inside the midday no-entry window (default 11:00-13:00 ET). Market-holiday gating
    is handled by the Alpaca clock in the engine loop."""
    if not (t.weekday() < 5 and _at(t, config.ENTRY_START) <= t <= _at(t, config.ENTRY_END)):
        return False
    ns, ne = _at(t, config.NO_ENTRY_START), _at(t, config.NO_ENTRY_END)
    if ns < ne and ns <= t < ne:   # half-open; disabled when start == end
        return False
    return True


def past_force_close(t: dt.datetime) -> bool:
    """At/after the intraday time-stop — everything must be flat."""
    return t >= _at(t, config.FORCE_CLOSE)


def minutes_to_close(t: dt.datetime) -> float:
    return max(0.0, (_at(t, (16, 0)) - t).total_seconds() / 60.0)


def session_elapsed_fraction(t: dt.datetime) -> float:
    """Fraction of the 9:30-16:00 regular session elapsed (0..1)."""
    open_ = t.replace(hour=9, minute=30, second=0, microsecond=0)
    close = t.replace(hour=16, minute=0, second=0, microsecond=0)
    if t <= open_:
        return 0.0
    if t >= close:
        return 1.0
    return (t - open_).total_seconds() / (close - open_).total_seconds()
