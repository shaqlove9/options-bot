"""credit.py — SHORT (credit) vertical spreads, for backtest evaluation.

The mirror image of `spread.py` (which is debit-only). Driven by the SAME
directional signal the live bot uses, but expressed as short premium:

  signal "call" (bullish) -> BULL PUT spread:  SELL put at ~`short_delta`,
                                               BUY  put `width` lower  (protection)
  signal "put"  (bearish) -> BEAR CALL spread: SELL call at ~`short_delta`,
                                               BUY  call `width` higher (protection)

You collect a credit up front; max profit = credit, max loss = width - credit.
Both legs are priced with `bsm` from the underlying path, exactly like the
single-leg backtest, so the two structures are comparable on identical entries.

Everything here is PURE (no I/O, no clients) so the width/credit/threshold math
is unit-testable — same convention as `spread.select_spread`.

THE COST POINT THIS MODULE EXISTS TO MEASURE: a credit vertical crosses the
bid/ask FOUR times per round trip (2 legs open + 2 legs close) versus TWO for a
single long option, and pays 4x the per-contract fee. `entry_credit` and
`exit_debit` deliberately apply the half-spread in the PENALISING direction on
every leg, so the modeled fill is always worse than mid.

Caveat, and it points one way: BSM is European. Real short legs carry early
assignment and pin risk, which make live results WORSE than modeled, never
better. Read a negative verdict here as an upper bound on the real thing.
"""
from __future__ import annotations

from dataclasses import dataclass

from iobot import bsm


@dataclass
class CreditParams:
    """Structural knobs for a short vertical."""
    short_delta: float = 0.30      # |delta| of the SHORT leg (0.30 = classic)
    width: float = 1.0             # dollars between short and long strike
    profit_target_pct: float = 50.0   # close when we can buy back for <= this % of credit gone
    stop_mult: float = 2.0         # close when open loss reaches stop_mult x credit
    strike_inc: float = 1.0        # strike grid increment
    strike_mode: str = "delta"     # "delta" | "expected_move"
    em_frac: float = 0.5           # short strike at this fraction of the expected move
    manage: bool = True            # False = hold to expiry, no intraday target/stop


@dataclass
class CreditQuote:
    """A selected short vertical at a point in time."""
    otype: str            # the option type of BOTH legs ("put" = bull put, "call" = bear call)
    short_strike: float
    long_strike: float
    width: float
    short_px: float       # BSM mid of the short leg
    long_px: float        # BSM mid of the long leg

    @property
    def mid_value(self) -> float:
        """Mid value of the spread = what it costs to buy back at mid. In [0, width]."""
        return max(0.0, self.short_px - self.long_px)


def expected_move(S: float, sigma: float, sessions: float = 1.0) -> float:
    """One-session expected move implied by `sigma` (annualized IV).

    The strategy this replicates uses VIX1D/sqrt(252) * 0.5 * price. VIX1D is not
    on the Alpaca feed, so `sigma` (realized vol * vrp, the same estimate the rest
    of the backtest prices with) stands in for it. Same shape, different vol input
    — a documented substitution, not the original signal.
    """
    return S * sigma * ((sessions / 252.0) ** 0.5)


def select(S: float, T: float, sigma: float, direction: str,
           p: CreditParams, em: float | None = None) -> CreditQuote | None:
    """Pick the short vertical for a directional signal.

    `direction` is the SIGNAL's direction ("call" = bullish, "put" = bearish);
    the returned spread is the opposite-type short vertical that profits from it.
    In "expected_move" mode the short strike is placed `em_frac` of `em` away from
    spot instead of at a target delta.
    Returns None if the strikes collapse or the geometry is degenerate.
    """
    # Bullish signal -> sell PUTS below spot. Bearish -> sell CALLS above spot.
    otype = "put" if direction == "call" else "call"

    if p.strike_mode == "expected_move":
        if em is None or em <= 0:
            return None
        offset = em * p.em_frac
        raw = S - offset if otype == "put" else S + offset
        short_K = round(raw / p.strike_inc) * p.strike_inc
        # The strike grid can round the short leg back through spot on tiny moves.
        if (otype == "put" and short_K >= S) or (otype == "call" and short_K <= S):
            return None
    else:
        short_K = bsm.strike_for_delta(S, T, sigma, p.short_delta, otype,
                                       increment=p.strike_inc)
    long_K = short_K - p.width if otype == "put" else short_K + p.width

    # Degenerate geometry: protection at/through the money, or non-positive strike.
    if long_K <= 0 or short_K <= 0 or p.width <= 0:
        return None

    short_px = bsm.price(S, short_K, T, sigma, otype)
    long_px = bsm.price(S, long_K, T, sigma, otype)
    q = CreditQuote(otype=otype, short_strike=short_K, long_strike=long_K,
                    width=p.width, short_px=short_px, long_px=long_px)
    if q.mid_value <= 0:
        return None
    return q


def value_at(S: float, T: float, sigma: float, q: CreditQuote) -> float:
    """Buy-back (mid) value of the same spread at a later underlying/time.
    Clamped to [0, width] — a vertical can never be worth more than its width."""
    short_px = bsm.price(S, q.short_strike, T, sigma, q.otype)
    long_px = bsm.price(S, q.long_strike, T, sigma, q.otype)
    return min(q.width, max(0.0, short_px - long_px))


def leg_cost(px: float, half_spread: float, min_abs: float = 0.0) -> float:
    """Half the bid/ask on ONE leg, in dollars per share.

    A pure percentage understates cheap options badly: 0DTE legs trade for
    pennies, and a 2% half-spread on a $0.12 option is $0.0024 — a quarter of a
    cent, when the market cannot be tighter than one cent wide. `min_abs` is the
    floor (0.005 = a penny-wide market), so short-dated cheap legs are charged
    what they actually cost to cross.
    """
    return max(px * half_spread, min_abs)


def entry_credit(q: CreditQuote, half_spread: float, min_abs: float = 0.0) -> float:
    """Credit actually RECEIVED per share: sell the short leg at its bid, buy the
    long leg at its ask. Always <= mid_value — you never collect the mid."""
    sell = q.short_px - leg_cost(q.short_px, half_spread, min_abs)
    buy = q.long_px + leg_cost(q.long_px, half_spread, min_abs)
    return sell - buy


def exit_debit(spread_mid: float, q: CreditQuote, S: float, T: float,
               sigma: float, half_spread: float, min_abs: float = 0.0) -> float:
    """Debit actually PAID per share to close: buy the short leg back at its ask,
    sell the long leg at its bid. Always >= mid — you never close at the mid.

    `spread_mid` is passed in so callers that already computed it don't reprice.
    """
    short_px = bsm.price(S, q.short_strike, T, sigma, q.otype)
    long_px = bsm.price(S, q.long_strike, T, sigma, q.otype)
    buy = short_px + leg_cost(short_px, half_spread, min_abs)
    sell = long_px - leg_cost(long_px, half_spread, min_abs)
    return max(0.0, min(q.width, buy - sell))


def target_debit(credit: float, p: CreditParams) -> float:
    """Buy-back price at which the profit target is met (take profit at
    `profit_target_pct` of the credit collected)."""
    return credit * (1.0 - p.profit_target_pct / 100.0)


def stop_debit(credit: float, p: CreditParams, width: float) -> float:
    """Buy-back price at which the stop trips (open loss = stop_mult x credit).
    Capped at the width — beyond that the spread is fully lost anyway."""
    return min(width, credit * (1.0 + p.stop_mult))


def adverse_price(bar, otype: str) -> float:
    """The extreme of a bar that HURTS a short vertical.
    Short puts are hurt by the underlying falling; short calls by it rising."""
    return float(bar["low"]) if otype == "put" else float(bar["high"])


def favorable_price(bar, otype: str) -> float:
    """The extreme of a bar that HELPS a short vertical."""
    return float(bar["high"]) if otype == "put" else float(bar["low"])


def max_loss(credit: float, width: float, fees: float, qty: int = 1) -> float:
    """Capital genuinely at risk = (width - credit) per share, plus fees.
    This is the R denominator, so credit-spread R is directly comparable to the
    single-leg backtest's R (both are P&L over defined max loss)."""
    return max(0.0, (width - credit)) * 100.0 * qty + fees
