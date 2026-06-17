"""Black-Scholes sanity: parity, delta bounds, strike-for-delta round-trip."""
from __future__ import annotations

import pytest

from iobot import bsm

S, T, SIG = 500.0, 3 / 365, 0.18


def test_atm_call_equals_put():
    assert bsm.price(S, S, T, SIG, "call") == pytest.approx(bsm.price(S, S, T, SIG, "put"))


def test_intrinsic_at_expiry():
    assert bsm.price(510, 500, 0, SIG, "call") == 10
    assert bsm.price(490, 500, 0, SIG, "put") == 10
    assert bsm.price(490, 500, 0, SIG, "call") == 0


def test_delta_bounds():
    assert 0 < bsm.delta(S, S, T, SIG, "call") < 1
    assert -1 < bsm.delta(S, S, T, SIG, "put") < 0


def test_strike_for_delta_round_trip():
    for otype in ("call", "put"):
        K = bsm.strike_for_delta(S, T, SIG, 0.65, otype, increment=1.0)
        assert abs(bsm.delta(S, K, T, SIG, otype)) == pytest.approx(0.65, abs=0.03)
    # slightly-ITM: call strike below spot, put strike above spot
    assert bsm.strike_for_delta(S, T, SIG, 0.65, "call") < S
    assert bsm.strike_for_delta(S, T, SIG, 0.65, "put") > S
