"""Entry-window gate incl. the midday no-entry exclusion (clock.in_entry_window)."""
from __future__ import annotations

import datetime as dt

from iobot import config
from iobot.clock import ET, in_entry_window


def _et(h, m, weekday_offset=0):
    # 2026-07-06 is a Monday.
    return dt.datetime(2026, 7, 6 + weekday_offset, h, m, tzinfo=ET)


def test_morning_open_drive_allowed():
    assert in_entry_window(_et(9, 45))
    assert in_entry_window(_et(10, 30))


def test_afternoon_allowed():
    assert in_entry_window(_et(13, 0))     # exactly at NO_ENTRY_END is allowed (half-open)
    assert in_entry_window(_et(14, 30))


def test_midday_window_blocked():
    assert not in_entry_window(_et(11, 0))    # inclusive start
    assert not in_entry_window(_et(11, 30))
    assert not in_entry_window(_et(12, 59))


def test_outside_session_blocked():
    assert not in_entry_window(_et(9, 30))    # before ENTRY_START
    assert not in_entry_window(_et(15, 45))   # after ENTRY_END


def test_weekend_blocked():
    assert not in_entry_window(_et(10, 0, weekday_offset=5))   # Saturday


def test_disabled_when_start_equals_end(monkeypatch):
    monkeypatch.setattr(config, "NO_ENTRY_START", (0, 0))
    monkeypatch.setattr(config, "NO_ENTRY_END", (0, 0))
    assert in_entry_window(_et(11, 30))   # no exclusion when window is empty
