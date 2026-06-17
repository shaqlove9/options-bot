"""Shared test fixtures."""
from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import pytest

from iobot import store
from iobot.clock import ET
from iobot.signals import Signal


@pytest.fixture
def conn():
    c = store.connect(":memory:")
    yield c
    c.close()


@pytest.fixture
def account():
    return SimpleNamespace(equity=10_000.0, cash=10_000.0, buying_power=20_000.0,
                           options_buying_power=10_000.0)


def make_signal(symbol="SPY", direction="call", spot=500.0, when=None) -> Signal:
    when = when or dt.datetime(2026, 6, 17, 11, 0, tzinfo=ET)
    return Signal(symbol=symbol, direction=direction, spot=spot, momentum_pct=0.5,
                  rsi=70.0 if direction == "call" else 30.0, rel_volume=1.5,
                  vwap_dist_pct=0.3, atr_pct=0.8, time=when)
