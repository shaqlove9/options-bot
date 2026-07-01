"""Shared test fixtures.

Environment is redirected BEFORE any iobot import (config resolves paths and
the Discord URL at import time, and load_dotenv never overrides pre-set vars):
  - IOBOT_DATA -> a temp dir, so test runs don't write into the production
    iobot_data/ (the prod log was collecting fake CRITICAL "SLEEVE HALTED"
    lines from the governor kill-switch tests);
  - DISCORD_WEBHOOK_URL -> blank, so those same halt tests can't send real
    Discord alerts. An autouse fixture re-blanks the config attr per-test as
    a guard against any future test that sets it.
"""
from __future__ import annotations

import os
import tempfile

os.environ["IOBOT_DATA"] = tempfile.mkdtemp(prefix="iobot-test-")
os.environ["DISCORD_WEBHOOK_URL"] = ""

import datetime as dt
from types import SimpleNamespace

import pytest

from iobot import config, store
from iobot.clock import ET
from iobot.signals import Signal


@pytest.fixture(autouse=True)
def _no_discord(monkeypatch):
    monkeypatch.setattr(config, "DISCORD_WEBHOOK_URL", "")


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
