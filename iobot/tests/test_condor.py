"""Iron condor pricing/selection sanity + the P&L-sign decomposition that proves
the 'no edge intraday' finding is real (not a bug):
  - held to near-expiry with no move => profit ~= full credit (theta works);
  - same-day exit captures almost no theta, and 4-leg costs swamp it.
"""
from __future__ import annotations

from iobot import condor

S, DL, DH, SIG = 752.5, 750.0, 755.0, 0.15
T0 = 3 / 365


def _pick():
    pick, why = condor.select_condor("SPY", S, DL, DH, SIG, T0)
    assert why == "ok"
    return pick


def test_strikes_ordered_around_spot():
    p = _pick()
    assert p.long_put_k < p.short_put_k < S < p.short_call_k < p.long_call_k
    assert 0 < p.raw_credit < p.wing


def test_rejects_when_spot_outside_range_strikes():
    # spot above the short call strike -> not a valid mid-range condor
    pick, why = condor.select_condor("SPY", 760, DL, DH, SIG, T0)
    assert pick is None


def _pnl(pick, T_exit, spot_exit, hs):
    entry = condor.credit_with_costs(pick.sp, pick.lp, pick.sc, pick.lc, hs)
    ex = condor.leg_prices(spot_exit, T_exit, SIG, pick)
    return (entry - condor.cost_to_close(*ex, hs)) * 100


def test_profit_when_held_to_expiry_no_move():
    p = _pick()
    assert _pnl(p, 0.0005 / 365, S, 0.0) > 50      # captures ~full credit


def test_intraday_theta_tiny_and_costs_dominate():
    p = _pick()
    same_day_T = (3 - 0.2) / 365
    assert 0 <= _pnl(p, same_day_T, S, 0.0) < 10   # ~no theta in a few hours
    assert _pnl(p, same_day_T, S, 0.02) < 0        # 4-leg round-trip cost swamps it
