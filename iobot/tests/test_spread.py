"""Debit-vertical selection + the always-two-legs close invariant."""
from __future__ import annotations

import datetime as dt

import pytest

from iobot import config, executor, spread
from iobot.tests.conftest import make_signal

EXP = dt.date(2026, 6, 19)


def leg(strike, bid, ask, oi=1000, vol=500):
    return spread.Leg(option_symbol=f"O{strike}", otype="call", strike=strike,
                      expiry=EXP, bid=bid, ask=ask, open_interest=oi, volume=vol)


def call_legs():
    # SPY ~500, strikes 498..503 step 1. Long NTM = 500.
    return {
        498: leg(498, 3.8, 4.0), 499: leg(499, 2.9, 3.1), 500: leg(500, 2.0, 2.2),
        501: leg(501, 1.3, 1.5), 502: leg(502, 0.8, 1.0), 503: leg(503, 0.5, 0.7),
    }


def test_selects_long_ntm_and_otm_short(monkeypatch):
    monkeypatch.setattr(config, "SPREAD_WIDTH_STRIKES", [1, 2])
    monkeypatch.setattr(config, "RISK_PCT_PER_TRADE", 2.0)   # cap = $200 at $10k
    monkeypatch.setattr(config, "QTY", 1)
    monkeypatch.setattr(config, "MAX_SPREAD_PCT", 20.0)      # cheap legs run wider %
    sig = make_signal(direction="call", spot=500)
    pick, why = spread.select_spread(sig, call_legs(), equity=10_000, expiry=EXP)
    assert why == "ok"
    assert pick.long_leg.strike == 500
    assert pick.short_leg.strike in (501, 502)
    # max loss = net debit * 100, and within the cap
    assert pick.max_loss == pytest.approx(pick.net_debit * 100)
    assert pick.max_loss <= 200


def test_risk_cap_rejects_when_equity_tiny(monkeypatch):
    monkeypatch.setattr(config, "SPREAD_WIDTH_STRIKES", [1, 2])
    monkeypatch.setattr(config, "RISK_PCT_PER_TRADE", 2.0)   # cap = $2 at $100
    monkeypatch.setattr(config, "MAX_SPREAD_PCT", 20.0)
    sig = make_signal(direction="call", spot=500)
    pick, why = spread.select_spread(sig, call_legs(), equity=100, expiry=EXP)
    assert pick is None and "max loss" in why


def test_liquidity_rejects_short_leg(monkeypatch):
    monkeypatch.setattr(config, "SPREAD_WIDTH_STRIKES", [1])
    monkeypatch.setattr(config, "MIN_OPEN_INTEREST", 250)
    monkeypatch.setattr(config, "MAX_SPREAD_PCT", 20.0)
    legs = call_legs()
    legs[501] = leg(501, 1.3, 1.5, oi=5)        # illiquid short
    sig = make_signal(direction="call", spot=500)
    pick, why = spread.select_spread(sig, legs, equity=10_000, expiry=EXP)
    assert pick is None and "OI" in why


def test_close_legs_always_two_combined():
    pos = executor.SpreadPosition(
        underlying="SPY", direction="call", long_symbol="L", short_symbol="S",
        long_strike=500, short_strike=502, expiry=EXP, width=2, qty=1,
        entry_debit=1.0, entry_mid=1.0, max_loss=100, entry_underlying=500,
        entry_time=dt.datetime(2026, 6, 17, 11, 0))
    legs = executor.close_legs(pos)
    assert len(legs) == 2                        # never single-leg
    sides = {lg.symbol: str(lg.side).lower() for lg in legs}
    intents = {lg.symbol: str(lg.position_intent).lower() for lg in legs}
    assert "sell" in sides["L"] and "buy" in sides["S"]
    assert "sell_to_close" in intents["L"] and "buy_to_close" in intents["S"]
