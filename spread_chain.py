"""spread_chain.py — builds a directional DEBIT VERTICAL spread from a signal.

Bull call (signal "call"): BUY a near-the-money call, SELL a higher-strike call.
Bear put  (signal "put"):  BUY a near-the-money put,  SELL a lower-strike put.

The spread's max loss is the net debit paid — DEFINED at entry. We pick the long
leg nearest the money and a short leg `SPREAD_WIDTH_STRIKES` strikes further OTM,
choosing the width whose max loss fits the per-trade risk cap, and reject the
signal entirely (logging the reason) if no width passes liquidity + risk.

Split into:
  - select_spread(...)  — PURE: takes already-fetched legs; unit-testable.
  - build_spread(...)   — fetches the chain + quotes for one signal, then selects.
"""
import datetime as dt
import logging
from dataclasses import dataclass

import config
from utils import now_et

log = logging.getLogger("spread_chain")


@dataclass
class Leg:
    option_symbol: str
    otype: str               # "call" | "put"
    strike: float
    expiry: dt.date
    bid: float
    ask: float
    open_interest: int
    volume: int | None = None

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread_pct(self) -> float:
        m = self.mid
        return (self.ask - self.bid) / m * 100 if m > 0 else float("inf")

    def liquid(self) -> tuple[bool, str]:
        if self.bid <= 0 or self.ask <= 0:
            return False, "no quote"
        if self.spread_pct > config.SPREAD_MAX_LEG_SPREAD_PCT:
            return False, f"wide spread {self.spread_pct:.0f}%"
        if self.open_interest < config.SPREAD_MIN_OI:
            return False, f"low OI {self.open_interest}"
        if config.SPREAD_MIN_VOLUME and (self.volume or 0) < config.SPREAD_MIN_VOLUME:
            return False, f"low vol {self.volume}"
        return True, "ok"


@dataclass
class SpreadPick:
    underlying: str
    direction: str            # "call" = bull call | "put" = bear put
    long_leg: Leg
    short_leg: Leg
    expiry: dt.date
    width: float              # |strike difference| in dollars
    net_debit: float          # per share (long.ask - short.bid)

    @property
    def max_loss(self) -> float:
        return self.net_debit * 100          # one contract per leg

    @property
    def max_profit(self) -> float:
        return (self.width - self.net_debit) * 100

    @property
    def reward_risk(self) -> float:
        return self.max_profit / self.max_loss if self.max_loss > 0 else 0.0

    def describe(self) -> str:
        return (f"{self.underlying} {'BULL CALL' if self.direction == 'call' else 'BEAR PUT'} "
                f"{self.long_leg.strike:g}/{self.short_leg.strike:g} exp {self.expiry} "
                f"debit ${self.net_debit:.2f} maxloss ${self.max_loss:.0f} "
                f"maxprofit ${self.max_profit:.0f} R:R {self.reward_risk:.2f}")


def _strike_increment(strikes: list[float]) -> float:
    diffs = [round(b - a, 4) for a, b in zip(sorted(strikes), sorted(strikes)[1:])
             if b - a > 1e-6]
    return min(diffs) if diffs else 1.0


def select_spread(signal, legs_by_strike: dict[float, Leg], equity: float,
                  expiry: dt.date) -> tuple[SpreadPick | None, str]:
    """PURE selection from a {strike: Leg} map for a single expiry. Returns
    (pick, reason); pick is None when nothing qualifies and reason says why."""
    if len(legs_by_strike) < 2:
        return None, "fewer than 2 strikes available"
    strikes = sorted(legs_by_strike)
    incr = _strike_increment(strikes)
    bullish = signal.direction == "call"

    # Long leg = nearest strike to spot (near-the-money).
    long_strike = min(strikes, key=lambda s: abs(s - signal.spot))
    long_leg = legs_by_strike[long_strike]
    ok, why = long_leg.liquid()
    if not ok:
        return None, f"long leg illiquid ({why})"

    risk_cap = config.SPREAD_MAX_RISK_PCT / 100 * equity
    candidates: list[SpreadPick] = []
    reasons: list[str] = []
    for w in config.SPREAD_WIDTH_STRIKES:
        short_strike = round(long_strike + w * incr * (1 if bullish else -1), 4)
        short_leg = legs_by_strike.get(short_strike)
        if short_leg is None:
            reasons.append(f"w{w}: no strike {short_strike:g}")
            continue
        ok, why = short_leg.liquid()
        if not ok:
            reasons.append(f"w{w}: short leg {why}")
            continue
        width = abs(short_strike - long_strike)
        net_debit = long_leg.ask - short_leg.bid           # cross the spread on both
        if net_debit <= 0:
            reasons.append(f"w{w}: non-positive debit")
            continue
        if net_debit >= width:
            reasons.append(f"w{w}: debit ${net_debit:.2f} >= width ${width:.2f} (no edge room)")
            continue
        pick = SpreadPick(underlying=signal.symbol, direction=signal.direction,
                          long_leg=long_leg, short_leg=short_leg, expiry=expiry,
                          width=width, net_debit=net_debit)
        if pick.max_loss > risk_cap:
            reasons.append(f"w{w}: max loss ${pick.max_loss:.0f} > cap ${risk_cap:.0f}")
            continue
        candidates.append(pick)

    if not candidates:
        return None, "; ".join(reasons) or "no qualifying width"
    # Best reward:risk; tie-break the cheaper debit (less capital at risk).
    candidates.sort(key=lambda p: (-p.reward_risk, p.net_debit))
    return candidates[0], "ok"


# ---------------- chain fetch (impure) ----------------

def _fetch_legs(signal, trading, data):
    """Pull the nearest-expiry candidate legs for the signal as {strike: Leg}."""
    from alpaca.data.requests import OptionSnapshotRequest
    from alpaca.trading.enums import AssetStatus, ContractType
    from alpaca.trading.requests import GetOptionContractsRequest

    today = now_et().date()
    eff_min = config.SPREAD_DTE_MIN if config.SPREAD_ALLOW_0DTE else max(config.SPREAD_DTE_MIN, 1)
    ctype = ContractType.CALL if signal.direction == "call" else ContractType.PUT
    if signal.direction == "call":
        lo, hi = signal.spot * 0.99, signal.spot * 1.06
    else:
        lo, hi = signal.spot * 0.94, signal.spot * 1.01
    req = GetOptionContractsRequest(
        underlying_symbols=[signal.symbol], status=AssetStatus.ACTIVE, type=ctype,
        expiration_date_gte=today + dt.timedelta(days=eff_min),
        expiration_date_lte=today + dt.timedelta(days=config.SPREAD_DTE_MAX),
        strike_price_gte=str(round(lo, 2)), strike_price_lte=str(round(hi, 2)),
        limit=300)
    contracts = list(trading.get_option_contracts(req).option_contracts or [])
    if not contracts:
        return None, None, "no contracts in DTE window"

    # Nearest expiry that has at least two candidate strikes.
    by_exp: dict[dt.date, list] = {}
    for c in contracts:
        by_exp.setdefault(c.expiration_date, []).append(c)
    expiry = None
    for e in sorted(by_exp):
        if len(by_exp[e]) >= 2:
            expiry = e
            break
    if expiry is None:
        return None, None, "no expiry with >=2 strikes"
    chosen = {c.symbol: c for c in by_exp[expiry]}

    snaps = data.get_option_snapshot(OptionSnapshotRequest(symbol_or_symbols=list(chosen)))
    legs: dict[float, Leg] = {}
    for sym, c in chosen.items():
        snap = snaps.get(sym)
        q = getattr(snap, "latest_quote", None) if snap else None
        if not q:
            continue
        vol = None
        db = getattr(snap, "daily_bar", None)
        if db is not None:
            vol = int(getattr(db, "volume", 0) or 0)
        legs[float(c.strike_price)] = Leg(
            option_symbol=sym, otype=signal.direction, strike=float(c.strike_price),
            expiry=expiry, bid=float(q.bid_price or 0), ask=float(q.ask_price or 0),
            open_interest=int(c.open_interest or 0), volume=vol)
    if not legs:
        return None, None, "no quotes for chosen expiry"
    return legs, expiry, "ok"


def build_spread(signal, trading, data, equity: float) -> tuple[SpreadPick | None, str]:
    """Fetch the chain for `signal` and select a defined-risk vertical, or
    (None, reason). Underlyings outside SPREAD_UNIVERSE are hard-rejected."""
    if signal.symbol not in config.SPREAD_UNIVERSE:
        return None, f"{signal.symbol} not in spread whitelist"
    try:
        legs, expiry, why = _fetch_legs(signal, trading, data)
    except Exception as exc:
        log.warning("%s: chain fetch failed: %s", signal.symbol, exc)
        return None, f"chain fetch error: {exc}"
    if legs is None:
        return None, why
    return select_spread(signal, legs, equity, expiry)
