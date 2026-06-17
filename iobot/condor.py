"""condor.py — defined-risk iron condor for trading an established intraday range.

Sell an OTM put spread + an OTM call spread: short strikes just OUTSIDE the day's
range, long wings further out for defined risk. You collect a net credit and profit
if price stays in the channel (theta works FOR you); max loss = wing width − credit,
capped by structure. This is the structurally-correct way to monetize a range with
options (vs buying premium, which fights theta).

`select_condor(...)` is PURE (BSM-priced from spot/IV) so it is unit-testable; the
backtest applies bid/ask + fees per leg. Live execution would submit all four legs
as ONE mleg order (not built until the backtest validates an edge).
"""
from __future__ import annotations

from dataclasses import dataclass

from iobot import bsm

# ---- tunables ----
CONDOR_EDGE_BUFFER = 0.15   # short strikes placed this fraction of range beyond each edge
CONDOR_WING = 2.0           # wing width ($) = defined risk per side
CONDOR_TARGET_FRAC = 0.50   # take profit once 50% of the credit has decayed away
CONDOR_STOP_MULT = 1.0      # stop when the buy-back cost reaches (1+mult)x the credit
CONDOR_MID_TOL = 0.35       # only enter when spot is within this frac of range mid


@dataclass
class CondorPick:
    symbol: str
    spot: float
    short_put_k: float
    long_put_k: float
    short_call_k: float
    long_call_k: float
    wing: float
    sp: float                # raw (mid) leg prices at entry
    lp: float
    sc: float
    lc: float

    @property
    def raw_credit(self) -> float:
        return (self.sp - self.lp) + (self.sc - self.lc)

    def describe(self) -> str:
        return (f"{self.symbol} IC {self.long_put_k:g}/{self.short_put_k:g}-"
                f"{self.short_call_k:g}/{self.long_call_k:g} credit ${self.raw_credit:.2f} "
                f"maxloss ${(self.wing - self.raw_credit) * 100:.0f}")


def select_condor(symbol, spot, dl, dh, sigma, T, increment: float = 1.0
                  ) -> tuple[CondorPick | None, str]:
    """PURE: build a condor with shorts just outside [dl, dh] and fixed wings."""
    rng = dh - dl
    if rng <= 0 or spot <= 0 or T <= 0 or sigma <= 0:
        return None, "bad inputs"
    sp_k = round((dl - CONDOR_EDGE_BUFFER * rng) / increment) * increment
    sc_k = round((dh + CONDOR_EDGE_BUFFER * rng) / increment) * increment
    lp_k = sp_k - CONDOR_WING
    lc_k = sc_k + CONDOR_WING
    if not (lp_k < sp_k < spot < sc_k < lc_k):
        return None, "strikes not ordered around spot"
    sp = bsm.price(spot, sp_k, T, sigma, "put")
    lp = bsm.price(spot, lp_k, T, sigma, "put")
    sc = bsm.price(spot, sc_k, T, sigma, "call")
    lc = bsm.price(spot, lc_k, T, sigma, "call")
    pick = CondorPick(symbol, spot, sp_k, lp_k, sc_k, lc_k, CONDOR_WING, sp, lp, sc, lc)
    if pick.raw_credit <= 0:
        return None, "non-positive credit"
    if pick.raw_credit >= CONDOR_WING:
        return None, "credit >= wing (mispriced)"
    return pick, "ok"


def leg_prices(spot: float, T: float, sigma: float, pick: CondorPick
               ) -> tuple[float, float, float, float]:
    """Current (sp, lp, sc, lc) mid prices — used to value the condor for exits."""
    return (bsm.price(spot, pick.short_put_k, T, sigma, "put"),
            bsm.price(spot, pick.long_put_k, T, sigma, "put"),
            bsm.price(spot, pick.short_call_k, T, sigma, "call"),
            bsm.price(spot, pick.long_call_k, T, sigma, "call"))


def credit_with_costs(sp, lp, sc, lc, hs: float) -> float:
    """Net credit RECEIVED to open: sell shorts at bid, buy wings at ask."""
    return sp * (1 - hs) - lp * (1 + hs) + sc * (1 - hs) - lc * (1 + hs)


def cost_to_close(sp, lp, sc, lc, hs: float) -> float:
    """Net debit PAID to close: buy back shorts at ask, sell wings at bid."""
    return sp * (1 + hs) - lp * (1 - hs) + sc * (1 + hs) - lc * (1 - hs)
