"""Short-vertical (credit spread) construction, cost asymmetry and exit math.

Config is pinned inside each test that depends on it — the live .env tuning must
never be able to void these (see commit 63078ae, where exactly that silently
disabled a leakage guard).
"""
from __future__ import annotations

import pytest

from iobot import credit

S = 100.0
T = 3 / 365.0
SIGMA = 0.30


def params(**kw) -> credit.CreditParams:
    base = dict(short_delta=0.30, width=1.0, profit_target_pct=50.0,
                stop_mult=2.0, strike_inc=1.0)
    base.update(kw)
    return credit.CreditParams(**base)


# ---------------- structure ----------------

def test_bullish_signal_sells_a_put_spread_below_spot():
    """A 'call' (bullish) signal must become a BULL PUT spread: both legs puts,
    short above long, and the whole structure below spot."""
    q = credit.select(S, T, SIGMA, "call", params())
    assert q is not None
    assert q.otype == "put"
    assert q.short_strike > q.long_strike      # protection is further OTM
    assert q.short_strike < S                  # short leg is OTM


def test_bearish_signal_sells_a_call_spread_above_spot():
    q = credit.select(S, T, SIGMA, "put", params())
    assert q is not None
    assert q.otype == "call"
    assert q.short_strike < q.long_strike
    assert q.short_strike > S


def test_width_is_respected_on_both_directions():
    for direction, w in [("call", 2.0), ("put", 5.0)]:
        q = credit.select(S, T, SIGMA, direction, params(width=w))
        assert q is not None
        assert abs(q.short_strike - q.long_strike) == pytest.approx(w)
        assert q.width == pytest.approx(w)


def test_short_leg_tracks_the_requested_delta():
    """A 0.16-delta short leg must sit further OTM than a 0.40-delta one."""
    far = credit.select(S, T, SIGMA, "call", params(short_delta=0.16))
    near = credit.select(S, T, SIGMA, "call", params(short_delta=0.40))
    assert far.short_strike < near.short_strike      # puts: lower strike = further OTM


def test_spread_value_never_exceeds_width():
    """A vertical is worth at most its width — even deep against us."""
    q = credit.select(S, T, SIGMA, "call", params(width=1.0))
    crashed = credit.value_at(S * 0.5, T, SIGMA, q)   # underlying halves
    assert 0.0 <= crashed <= q.width


def test_degenerate_geometry_rejected():
    assert credit.select(S, T, SIGMA, "call", params(width=0.0)) is None


# ---------------- cost asymmetry (the point of the exercise) ----------------

def test_credit_received_is_worse_than_mid():
    """You sell the short leg at the bid and buy protection at the ask, so the
    credit collected is strictly below the mid value of the spread."""
    q = credit.select(S, T, SIGMA, "call", params())
    assert credit.entry_credit(q, half_spread=0.02) < q.mid_value


def test_exit_debit_is_worse_than_mid():
    q = credit.select(S, T, SIGMA, "call", params())
    mid = credit.value_at(S, T, SIGMA, q)
    paid = credit.exit_debit(mid, q, S, T, SIGMA, half_spread=0.02)
    assert paid > mid


def test_wider_spreads_cost_strictly_more_to_trade():
    """Friction must scale with the half-spread — this is the knob the whole
    credit-vs-single question turns on."""
    q = credit.select(S, T, SIGMA, "call", params())
    cheap = credit.entry_credit(q, half_spread=0.01)
    dear = credit.entry_credit(q, half_spread=0.05)
    assert dear < cheap


def test_leg_cost_floors_at_the_penny_tick():
    """A 2% half-spread on a $0.12 0DTE leg is a quarter of a cent — below the
    tightest market that can exist. The dollar floor must win."""
    assert credit.leg_cost(0.12, 0.02, min_abs=0.005) == pytest.approx(0.005)


def test_leg_cost_uses_percentage_when_it_exceeds_the_floor():
    assert credit.leg_cost(5.00, 0.02, min_abs=0.005) == pytest.approx(0.10)


def test_tick_floor_reduces_the_credit_on_cheap_legs():
    """Same spread, same percentage — applying the floor must collect less."""
    q = credit.select(S, T, SIGMA, "call", params(short_delta=0.10))
    assert credit.entry_credit(q, 0.02, min_abs=0.005) < credit.entry_credit(q, 0.02, 0.0)


def test_zero_spread_recovers_the_mid():
    q = credit.select(S, T, SIGMA, "call", params())
    assert credit.entry_credit(q, half_spread=0.0) == pytest.approx(q.mid_value)


# ---------------- exit thresholds ----------------

def test_target_debit_is_half_the_credit_at_50pct():
    assert credit.target_debit(0.40, params(profit_target_pct=50.0)) == pytest.approx(0.20)


def test_stop_debit_is_three_times_credit_at_2x_loss():
    """Stop at 2x credit LOST means buying back for 3x credit (1x returns the
    premium, the extra 2x is the loss)."""
    assert credit.stop_debit(0.40, params(stop_mult=2.0), width=5.0) == pytest.approx(1.20)


def test_stop_debit_capped_at_width():
    """Loss can never exceed the width, so neither can the stop threshold."""
    assert credit.stop_debit(0.40, params(stop_mult=2.0), width=1.0) == pytest.approx(1.0)


# ---------------- risk denominator ----------------

def test_max_loss_is_width_minus_credit_plus_fees():
    assert credit.max_loss(0.30, 1.0, 2.60, 1) == pytest.approx((1.0 - 0.30) * 100 + 2.60)


def test_max_loss_never_negative_on_overwide_credit():
    assert credit.max_loss(2.0, 1.0, 0.0, 1) == pytest.approx(0.0)


# ---------------- adverse/favorable direction ----------------

def test_short_puts_are_hurt_by_the_low_and_helped_by_the_high():
    bar = {"low": 95.0, "high": 105.0}
    assert credit.adverse_price(bar, "put") == 95.0
    assert credit.favorable_price(bar, "put") == 105.0


def test_short_calls_are_hurt_by_the_high_and_helped_by_the_low():
    bar = {"low": 95.0, "high": 105.0}
    assert credit.adverse_price(bar, "call") == 105.0
    assert credit.favorable_price(bar, "call") == 95.0


# ---------------- expected-move strike placement ----------------

def em_params(**kw):
    return params(strike_mode="expected_move", **kw)


def test_expected_move_scales_with_vol_and_price():
    lo = credit.expected_move(100.0, 0.20)
    hi = credit.expected_move(100.0, 0.40)
    assert hi == pytest.approx(2 * lo)
    assert credit.expected_move(200.0, 0.20) == pytest.approx(2 * lo)


def test_expected_move_is_a_one_session_move():
    """sigma is annualized; the EM must be the 1-of-252-sessions scaling."""
    assert credit.expected_move(100.0, 0.252) == pytest.approx(
        100.0 * 0.252 * (1 / 252) ** 0.5)


def test_em_mode_places_short_strike_outside_the_move():
    """Bullish -> short put sits BELOW spot by em_frac x expected move."""
    em = credit.expected_move(S, SIGMA)
    q = credit.select(S, T, SIGMA, "call", em_params(em_frac=0.5), em=em)
    assert q is not None and q.otype == "put"
    assert q.short_strike == pytest.approx(round((S - 0.5 * em) / 1.0) * 1.0)
    assert q.short_strike < S


def test_em_mode_bearish_places_short_call_above_spot():
    em = credit.expected_move(S, SIGMA)
    q = credit.select(S, T, SIGMA, "put", em_params(em_frac=0.5), em=em)
    assert q is not None and q.otype == "call"
    assert q.short_strike > S


def test_larger_em_frac_moves_the_short_strike_further_out():
    em = credit.expected_move(S, SIGMA)
    near = credit.select(S, T, SIGMA, "call", em_params(em_frac=0.5), em=em)
    far = credit.select(S, T, SIGMA, "call", em_params(em_frac=2.0), em=em)
    assert far.short_strike < near.short_strike


def test_em_mode_requires_an_expected_move():
    assert credit.select(S, T, SIGMA, "call", em_params(), em=None) is None
    assert credit.select(S, T, SIGMA, "call", em_params(), em=0.0) is None


def test_em_mode_rejects_strike_rounded_back_through_spot():
    """A tiny expected move rounds the short strike onto/through spot — that is
    not an OTM credit spread and must be refused, not silently sold."""
    assert credit.select(S, T, SIGMA, "call", em_params(strike_inc=10.0),
                         em=0.01) is None


def test_adverse_move_increases_spread_value():
    """Sanity-check the whole chain: a bull put spread must get MORE expensive
    to buy back when the underlying falls."""
    q = credit.select(S, T, SIGMA, "call", params(width=2.0))
    here = credit.value_at(S, T, SIGMA, q)
    lower = credit.value_at(S - 5.0, T, SIGMA, q)
    assert lower > here
