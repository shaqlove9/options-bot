"""Tests for the orb_confluence signal: confluence gated by opening-range breakout
structure + a stocks-in-play relative-volume bar (2026-07-22 tightening)."""
from __future__ import annotations

import datetime as dt

from iobot import config, strategies

from .test_confluence_sleeve import _daily_uptrend, _intraday_uptrend


def _fixture(monkeypatch, relvol_min=0.0):
    monkeypatch.setattr(config, "REL_VOLUME_MIN", 0.0)
    monkeypatch.setattr(config, "ORBCONF_RELVOL_MIN", relvol_min)
    monkeypatch.setattr(config, "CONFLUENCE_MIN", 4)
    intraday, today, now = _intraday_uptrend()
    return intraday, _daily_uptrend(today), now


def test_fires_on_breakout_with_volume(monkeypatch):
    intraday, daily, now = _fixture(monkeypatch)
    sig = strategies.orb_confluence("SPY", intraday, daily, now)
    assert sig is not None
    assert sig.direction == "call"
    assert sig.source == "orb_confluence"
    # keeps confluence's ATR exit geometry
    assert sig.stop_level < sig.spot < sig.target_level
    # sanity: spot really is beyond the opening-range high
    today_bars = intraday[intraday.index.date == now.date()]
    or_end = today_bars.index[0] + dt.timedelta(minutes=strategies.ORB_MINUTES)
    or_high = float(today_bars[today_bars.index < or_end]["high"].max())
    assert sig.spot > or_high


def test_blocks_low_relative_volume(monkeypatch):
    intraday, daily, now = _fixture(monkeypatch, relvol_min=99.0)
    # confluence itself fires — the relvol gate is what blocks it
    assert strategies.confluence("SPY", intraday, daily, now) is not None
    assert strategies.orb_confluence("SPY", intraday, daily, now) is None


def test_blocks_inside_opening_range(monkeypatch):
    intraday, daily, now = _fixture(monkeypatch)
    # spike the first bar of today so the whole session trades inside the range
    intraday = intraday.copy()
    today_mask = intraday.index.date == now.date()
    first_today = intraday.index[today_mask][0]
    intraday.loc[first_today, "high"] = float(intraday["close"].iloc[-1]) + 50.0
    assert strategies.confluence("SPY", intraday, daily, now) is not None
    assert strategies.orb_confluence("SPY", intraday, daily, now) is None


def test_waits_for_opening_range_to_complete(monkeypatch):
    intraday, daily, _ = _fixture(monkeypatch)
    today = intraday.index[-1].date()
    early = dt.datetime(today.year, today.month, today.day, 10, 10,
                        tzinfo=intraday.index[-1].tzinfo)
    # 10:10 < opening-range end (10:00) + one 15m bar → no entry regardless of bars
    assert strategies.orb_confluence("SPY", intraday, daily, early) is None
