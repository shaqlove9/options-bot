"""Single-leg selection: slightly-ITM by delta + liquidity filter."""
from __future__ import annotations

import datetime as dt

import pytest

from iobot import chain, config
from iobot.tests.conftest import make_signal

EXP = dt.date(2026, 6, 19)


def cand(strike, delta, bid=1.4, ask=1.5, oi=1000, vol=500):
    # default premium $1.50 -> $150/lot, under the $300 per-position cap
    return chain.Candidate(option_symbol=f"O{strike}", strike=strike, expiry=EXP,
                           bid=bid, ask=ask, delta=delta, iv=0.2,
                           open_interest=oi, volume=vol)


def test_picks_delta_nearest_band_centre(monkeypatch):
    monkeypatch.setattr(config, "TARGET_DELTA_MIN", 0.60)
    monkeypatch.setattr(config, "TARGET_DELTA_MAX", 0.70)
    sig = make_signal(direction="call", spot=500)
    # 0.75 out of band (too ITM), 0.65 in band (best), 0.50 out of band (OTM).
    pick, why = chain.select_contract(sig, [cand(495, 0.75), cand(498, 0.65), cand(500, 0.50)])
    assert why == "ok" and pick.strike == 498 and abs(pick.delta) == 0.65


def test_rejects_when_no_strike_in_band(monkeypatch):
    monkeypatch.setattr(config, "TARGET_DELTA_MIN", 0.60)
    monkeypatch.setattr(config, "TARGET_DELTA_MAX", 0.70)
    sig = make_signal(direction="call", spot=500)
    pick, why = chain.select_contract(sig, [cand(500, 0.50), cand(490, 0.85)])
    assert pick is None and "delta out of band" in why


def test_liquidity_filter_rejects_with_reason(monkeypatch):
    monkeypatch.setattr(config, "TARGET_DELTA_MIN", 0.60)
    monkeypatch.setattr(config, "TARGET_DELTA_MAX", 0.70)
    monkeypatch.setattr(config, "MIN_OPEN_INTEREST", 250)
    monkeypatch.setattr(config, "MAX_SPREAD_PCT", 8.0)
    sig = make_signal(direction="call", spot=500)
    # in-band delta but illiquid: tiny OI and a very wide quote.
    pick, why = chain.select_contract(sig, [cand(498, 0.65, bid=1.0, ask=2.0, oi=10)])
    assert pick is None and ("OI" in why or "wide spread" in why)


def test_put_uses_abs_delta(monkeypatch):
    monkeypatch.setattr(config, "TARGET_DELTA_MIN", 0.60)
    monkeypatch.setattr(config, "TARGET_DELTA_MAX", 0.70)
    sig = make_signal(direction="put", spot=500)
    pick, why = chain.select_contract(sig, [cand(503, -0.65), cand(500, -0.50)])
    assert why == "ok" and pick.strike == 503


def test_dollar_sizing_and_max_loss(monkeypatch):
    # $1.50 premium under a $300 cap -> 2 lots, max loss = full premium across lots.
    monkeypatch.setattr(config, "POSITION_MAX_DOLLARS", 300.0)
    sig = make_signal(direction="call", spot=500)
    pick, _ = chain.select_contract(sig, [cand(498, 0.65, bid=1.4, ask=1.5)])
    assert pick.qty == 2
    assert pick.max_loss == pytest.approx(1.5 * 100 * 2)


def test_contract_over_cap_rejected(monkeypatch):
    # a $4.00 premium ($400/lot) exceeds the $300 cap -> filtered out, no pick.
    monkeypatch.setattr(config, "POSITION_MAX_DOLLARS", 300.0)
    monkeypatch.setattr(config, "TARGET_DELTA_MIN", 0.60)
    monkeypatch.setattr(config, "TARGET_DELTA_MAX", 0.70)
    sig = make_signal(direction="call", spot=500)
    pick, why = chain.select_contract(sig, [cand(498, 0.65, bid=3.9, ask=4.0)])
    assert pick is None and "cap" in why
