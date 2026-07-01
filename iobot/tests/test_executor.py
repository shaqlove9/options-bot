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


# ---------------- reconcile / rehydrate (broker drift) ----------------

import datetime as dt

from iobot import store
from iobot.executor import Executor, SingleLegPosition


class _FakeBrokerPos:
    def __init__(self, symbol):
        self.symbol = symbol
        # Mirror the real Alpaca enum's str form ("AssetClass.US_OPTION") so this test
        # guards the case-insensitive asset_class filter in reconcile().
        self.asset_class = "AssetClass.US_OPTION"


class _FakeTrading:
    """Minimal trading client: reports a fixed held set, records close_position calls."""
    def __init__(self, held):
        self._held = list(held)
        self.closed: list[str] = []

    def cancel_orders(self):
        pass

    def get_all_positions(self):
        return [_FakeBrokerPos(s) for s in self._held]

    def close_position(self, symbol):
        self.closed.append(symbol)
        self._held = [s for s in self._held if s != symbol]


def _mk_exec(held):
    conn = store.connect(":memory:")
    ex = Executor(_FakeTrading(held), None, None, conn)
    return ex, conn


def _persist_single(ex, option_symbol="SOFI260702C00018000"):
    pos = SingleLegPosition(
        underlying="SOFI", direction="call", option_symbol=option_symbol, strike=18.0,
        expiry=dt.date(2026, 7, 2), qty=5, entry_fill=0.58, entry_mid=0.59,
        entry_underlying=18.4, stop_level=18.23, target_level=18.75, max_loss=290.0,
        entry_time=dt.datetime(2026, 7, 1, 10, 15), signal_id="s1")
    ex._persist(pos)
    return pos


def test_rehydrate_recovers_broker_held_position():
    sym = "SOFI260702C00018000"
    ex, conn = _mk_exec(held=[sym])
    _persist_single(ex, sym)
    ex.reconcile()
    assert [p.option_symbol for p in ex.positions] == [sym]
    # still persisted (not dropped)
    assert conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0] == 1


def test_rehydrate_drops_position_broker_no_longer_holds():
    # Persisted locally but broker holds nothing -> dropped (and unpersisted), not recovered.
    ex, conn = _mk_exec(held=[])
    _persist_single(ex)
    ex.reconcile()
    assert ex.positions == []
    assert conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0] == 0


def test_reconcile_flattens_untracked_orphans(monkeypatch):
    monkeypatch.setattr(config, "RECONCILE_FLATTEN_ORPHANS", True)
    orphan = "PLTR260702P00116000"
    ex, _ = _mk_exec(held=[orphan])          # broker holds it, nothing persisted locally
    ex.reconcile()
    assert ex.trading.closed == [orphan]     # auto-flattened


def test_reconcile_leaves_orphans_when_flatten_disabled(monkeypatch):
    monkeypatch.setattr(config, "RECONCILE_FLATTEN_ORPHANS", False)
    orphan = "PLTR260702P00116000"
    ex, _ = _mk_exec(held=[orphan])
    ex.reconcile()
    assert ex.trading.closed == []           # warned, not closed
