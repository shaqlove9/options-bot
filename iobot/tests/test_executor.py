"""Underlying-based exit levels for the single long leg."""
from __future__ import annotations

import pytest

from iobot import config, executor


def test_call_levels(monkeypatch):
    monkeypatch.setattr(config, "STOP_UNDERLYING_PCT", 0.40)
    monkeypatch.setattr(config, "TARGET_R", 1.5)
    stop, target = executor.single_exit_levels("call", 100.0)
    assert stop == pytest.approx(99.60)        # 100 - 0.40
    assert target == pytest.approx(100.60)     # 100 + 1.5 * 0.40


def test_put_levels(monkeypatch):
    monkeypatch.setattr(config, "STOP_UNDERLYING_PCT", 0.40)
    monkeypatch.setattr(config, "TARGET_R", 1.5)
    stop, target = executor.single_exit_levels("put", 100.0)
    assert stop == pytest.approx(100.40)
    assert target == pytest.approx(99.40)


def test_stop_and_target_hit_logic():
    assert executor.single_stop_hit("call", 99.5, 99.6)
    assert not executor.single_stop_hit("call", 99.7, 99.6)
    assert executor.single_target_hit("call", 100.7, 100.6)
    assert executor.single_stop_hit("put", 100.5, 100.4)
    assert executor.single_target_hit("put", 99.3, 99.4)


def test_open_legs_two_to_open():
    from types import SimpleNamespace
    pick = SimpleNamespace(
        long_leg=SimpleNamespace(option_symbol="L"),
        short_leg=SimpleNamespace(option_symbol="S"))
    legs = executor.open_legs(pick)
    assert len(legs) == 2
    intents = {lg.symbol: str(lg.position_intent).lower() for lg in legs}
    assert "buy_to_open" in intents["L"] and "sell_to_open" in intents["S"]
