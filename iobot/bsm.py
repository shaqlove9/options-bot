"""bsm.py — minimal Black-Scholes used by the backtest to price the option leg.

We don't have deep intraday option-quote history, so the backtest reprices the
chosen contract from the underlying path with an IV assumption. BSM captures the
two things that decide whether an intraday options edge survives: convexity
(gamma) on the move and theta over the hold. Treat absolute P&L as approximate and
lean on the cost SWEEP — the question is "does the signal beat realistic costs",
not "what is the exact dollar P&L".
"""
from __future__ import annotations

from math import erf, exp, log, sqrt
from statistics import NormalDist

_N = NormalDist()


def _cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _d1(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    return (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))


def price(S: float, K: float, T: float, sigma: float, otype: str, r: float = 0.0) -> float:
    """European option price. Falls back to intrinsic value as T->0 or sigma->0."""
    intrinsic = max(0.0, S - K) if otype == "call" else max(0.0, K - S)
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return intrinsic
    d1 = _d1(S, K, T, sigma, r)
    d2 = d1 - sigma * sqrt(T)
    call = S * _cdf(d1) - K * exp(-r * T) * _cdf(d2)
    return call if otype == "call" else call - S + K * exp(-r * T)  # put via parity


def delta(S: float, K: float, T: float, sigma: float, otype: str, r: float = 0.0) -> float:
    if T <= 0 or sigma <= 0:
        itm = (S > K) if otype == "call" else (S < K)
        base = 1.0 if itm else 0.0
        return base if otype == "call" else -base
    d1 = _d1(S, K, T, sigma, r)
    return _cdf(d1) if otype == "call" else _cdf(d1) - 1.0


def strike_for_delta(S: float, T: float, sigma: float, target_delta: float,
                     otype: str, r: float = 0.0, increment: float = 1.0) -> float:
    """Strike whose |delta| ~= target_delta, rounded to the strike increment.
    call: N(d1)=target; put: N(d1)=1-target."""
    target = abs(target_delta)
    if T <= 0 or sigma <= 0:
        return round(S / increment) * increment
    d1 = _N.inv_cdf(target if otype == "call" else 1.0 - target)
    K = S * exp((r + 0.5 * sigma * sigma) * T - d1 * sigma * sqrt(T))
    return round(K / increment) * increment
