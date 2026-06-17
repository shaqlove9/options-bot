"""Risk governor: per-trade cap, BP/settled-funds, kill switches, caps, cash mode."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from iobot import config, governor


@pytest.fixture
def gov(tmp_path):
    g = governor.RiskGovernor(state_file=str(tmp_path / "gov.json"))
    g.begin_day(10_000.0)
    return g


def acct(equity=10_000, cash=10_000, bp=20_000, obp=10_000):
    return SimpleNamespace(equity=equity, cash=cash, buying_power=bp, options_buying_power=obp)


def test_per_trade_cap_rejects_oversize(gov, monkeypatch):
    monkeypatch.setattr(config, "RISK_PCT_PER_TRADE", 2.0)   # cap = $200
    ok, why = gov.approve_risk(max_loss=300, required_bp=300, account=acct())
    assert not ok and "cap" in why
    ok, _ = gov.approve_risk(max_loss=150, required_bp=150, account=acct())
    assert ok


def test_buying_power_check(gov):
    ok, why = gov.approve_risk(max_loss=100, required_bp=15_000, account=acct(obp=5_000))
    assert not ok and "buying power" in why


def test_cash_account_needs_settled_funds(gov, monkeypatch):
    monkeypatch.setattr(config, "CASH_ACCOUNT_MODE", True)
    ok, why = gov.approve_risk(max_loss=100, required_bp=1_500, account=acct(cash=1_000))
    assert not ok and "settled" in why


def test_daily_loss_kill(gov, monkeypatch):
    monkeypatch.setattr(config, "DAILY_MAX_LOSS_PCT", 5.0)
    ok, why = gov.can_open(equity=9_400, open_count=0)   # -6% on the day
    assert not ok and "daily" in why.lower()


def test_trailing_dd_kill_is_sticky(gov, monkeypatch):
    monkeypatch.setattr(config, "TRAILING_DD_PCT", 20.0)
    gov.update_equity(12_000)            # peak
    gov.update_equity(9_000)             # -25% off peak
    assert gov.state.sleeve_halted
    ok, why = gov.can_open(equity=9_000, open_count=0)
    assert not ok and "halted" in why


def test_concurrent_and_trade_caps(gov, monkeypatch):
    monkeypatch.setattr(config, "MAX_CONCURRENT", 1)
    monkeypatch.setattr(config, "MAX_TRADES_PER_DAY", 2)
    ok, why = gov.can_open(equity=10_000, open_count=1)
    assert not ok and "concurrent" in why
    gov.record_entry(); gov.record_entry()
    ok, why = gov.can_open(equity=10_000, open_count=0)
    assert not ok and "trades/day" in why


def test_round_trip_cap(gov, monkeypatch):
    monkeypatch.setattr(config, "MAX_ROUND_TRIPS_PER_DAY", 1)
    gov.record_exit(5.0)
    ok, why = gov.can_open(equity=10_000, open_count=0)
    assert not ok and "round-trips" in why


def test_cash_mode_one_round_trip(gov, monkeypatch):
    monkeypatch.setattr(config, "CASH_ACCOUNT_MODE", True)
    gov.record_entry()
    ok, why = gov.can_open(equity=10_000, open_count=0)
    assert not ok and "cash account" in why
