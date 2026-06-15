"""Shared time/session helpers."""
import datetime as dt
import logging
import logging.handlers
from zoneinfo import ZoneInfo

import config

ET = ZoneInfo(config.TZ)

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.handlers.RotatingFileHandler(
            config.BOT_LOG_FILE, maxBytes=2_000_000, backupCount=2,
            encoding="utf-8"),
    ],
)


def now_et() -> dt.datetime:
    return dt.datetime.now(tz=ET)


def _at(t: dt.datetime, hm: tuple[int, int]) -> dt.datetime:
    return t.replace(hour=hm[0], minute=hm[1], second=0, microsecond=0)


def is_market_day(t: dt.datetime) -> bool:
    return t.weekday() < 5  # holidays handled by Alpaca clock in main loop


def in_entry_window(t: dt.datetime) -> bool:
    """9:45 AM - 3:30 PM ET — no first/last 15 minutes."""
    return is_market_day(t) and _at(t, config.ENTRY_START) <= t <= _at(t, config.ENTRY_END)


def past_force_close(t: dt.datetime) -> bool:
    return t >= _at(t, config.FORCE_CLOSE)


def session_elapsed_fraction(t: dt.datetime) -> float:
    """Fraction of the 9:30-16:00 regular session that has elapsed."""
    open_ = t.replace(hour=9, minute=30, second=0, microsecond=0)
    close = t.replace(hour=16, minute=0, second=0, microsecond=0)
    if t <= open_:
        return 0.0
    if t >= close:
        return 1.0
    return (t - open_).total_seconds() / (close - open_).total_seconds()
