"""chain.py — selects the slightly-ITM single long option for a signal.

Primary structure: ONE long call (bullish) or long put (bearish), strike chosen by
delta (default 0.60-0.70) so it is slightly in-the-money — cutting extrinsic/theta
bleed versus OTM lottery tickets. Short-dated (configurable DTE window; 0DTE off by
default). Risk = premium paid, defined at entry.

Liquidity filter rejects (never trades) a contract whose bid/ask, open interest, or
volume fail the configured minimums; the failing reason is returned for logging.

`select_contract(...)` is PURE (operates on already-fetched candidates) so strike
selection and the liquidity gate are unit-testable without the API.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass

from iobot import config
from iobot.clock import now_et

log = logging.getLogger("chain")


@dataclass
class Candidate:
    """One quoted contract considered for the long leg."""
    option_symbol: str
    strike: float
    expiry: dt.date
    bid: float
    ask: float
    delta: float | None
    iv: float | None
    open_interest: int
    volume: int

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
        if self.spread_pct > config.MAX_SPREAD_PCT:
            return False, f"wide spread {self.spread_pct:.0f}%"
        if self.open_interest < config.MIN_OPEN_INTEREST:
            return False, f"low OI {self.open_interest}"
        if self.volume < config.MIN_VOLUME:
            return False, f"low vol {self.volume}"
        return True, "ok"


@dataclass
class ContractPick:
    option_symbol: str
    underlying: str
    otype: str               # "call" | "put"
    strike: float
    expiry: dt.date
    bid: float
    ask: float
    delta: float | None
    iv: float | None
    open_interest: int
    volume: int

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def max_loss(self) -> float:
        """Defined risk of a long option = premium paid (worst case = full ask)."""
        return self.ask * 100 * config.QTY

    def describe(self) -> str:
        d = f"{self.delta:+.2f}" if self.delta is not None else "n/a"
        return (f"{self.underlying} LONG {self.otype.upper()} {self.strike:g} "
                f"exp {self.expiry} delta {d} ask ${self.ask:.2f} "
                f"maxloss ${self.max_loss:.0f}")


def select_contract(signal, candidates: list[Candidate]
                    ) -> tuple[ContractPick | None, str]:
    """PURE: pick the slightly-ITM long leg from quoted candidates of one expiry.

    Strike chosen by |delta| nearest the centre of the target band; only candidates
    with delta inside the band and passing the liquidity filter qualify. Returns
    (pick, reason); reason explains a None so the engine can log the rejection.
    """
    if not candidates:
        return None, "no contracts in DTE window"

    lo, hi = config.TARGET_DELTA_MIN, config.TARGET_DELTA_MAX
    centre = (lo + hi) / 2
    in_band: list[tuple[float, Candidate]] = []
    counts = {"no delta": 0, "delta out of band": 0, "illiquid": 0}
    illiquid_reasons: list[str] = []
    for c in candidates:
        if c.delta is None:
            counts["no delta"] += 1
            continue
        ad = abs(c.delta)
        if not (lo <= ad <= hi):
            counts["delta out of band"] += 1
            continue
        ok, why = c.liquid()
        if not ok:
            counts["illiquid"] += 1
            illiquid_reasons.append(why)
            continue
        in_band.append((abs(ad - centre), c))

    if not in_band:
        detail = ", ".join(f"{n} {k}" for k, n in counts.items() if n)
        if illiquid_reasons:
            detail += f" ({'; '.join(sorted(set(illiquid_reasons)))})"
        return None, f"no qualifying strike — {detail or 'no candidates'}"

    # Closest delta to band centre; tie-break tighter quoted spread.
    in_band.sort(key=lambda t: (t[0], t[1].spread_pct))
    c = in_band[0][1]
    pick = ContractPick(
        option_symbol=c.option_symbol, underlying=signal.symbol, otype=signal.direction,
        strike=c.strike, expiry=c.expiry, bid=c.bid, ask=c.ask, delta=c.delta,
        iv=c.iv, open_interest=c.open_interest, volume=c.volume)
    return pick, "ok"


# ---------------- chain fetch (impure) ----------------

def _dte_floor() -> int:
    return config.DTE_MIN if config.ALLOW_0DTE else max(config.DTE_MIN, 1)


def _fetch_candidates(signal, trading, data) -> tuple[list[Candidate], str]:
    from alpaca.data.requests import OptionSnapshotRequest
    from alpaca.trading.enums import AssetStatus, ContractType
    from alpaca.trading.requests import GetOptionContractsRequest

    today = now_et().date()
    ctype = ContractType.CALL if signal.direction == "call" else ContractType.PUT
    # ITM band: calls below spot, puts above. Generous net; delta picks the strike.
    if signal.direction == "call":
        lo, hi = signal.spot * 0.90, signal.spot * 1.01
    else:
        lo, hi = signal.spot * 0.99, signal.spot * 1.10
    req = GetOptionContractsRequest(
        underlying_symbols=[signal.symbol], status=AssetStatus.ACTIVE, type=ctype,
        expiration_date_gte=today + dt.timedelta(days=_dte_floor()),
        expiration_date_lte=today + dt.timedelta(days=config.DTE_MAX),
        strike_price_gte=str(round(lo, 2)), strike_price_lte=str(round(hi, 2)),
        limit=400)
    contracts = list(trading.get_option_contracts(req).option_contracts or [])
    if not contracts:
        return [], "no contracts in DTE/strike window"

    # Trade the nearest expiry in the window (shortest DTE for an intraday hold).
    by_exp: dict[dt.date, list] = {}
    for c in contracts:
        by_exp.setdefault(c.expiration_date, []).append(c)
    expiry = min(by_exp)
    chosen = {c.symbol: c for c in by_exp[expiry]}

    snaps = data.get_option_snapshot(OptionSnapshotRequest(symbol_or_symbols=list(chosen)))
    out: list[Candidate] = []
    for sym, c in chosen.items():
        snap = snaps.get(sym)
        q = getattr(snap, "latest_quote", None) if snap else None
        if not q:
            continue
        greeks = getattr(snap, "greeks", None)
        delta = float(greeks.delta) if greeks and greeks.delta is not None else None
        iv = float(snap.implied_volatility) if snap and snap.implied_volatility else None
        db = getattr(snap, "daily_bar", None)
        vol = int(getattr(db, "volume", 0) or 0) if db is not None else 0
        out.append(Candidate(
            option_symbol=sym, strike=float(c.strike_price), expiry=expiry,
            bid=float(q.bid_price or 0), ask=float(q.ask_price or 0),
            delta=delta, iv=iv, open_interest=int(c.open_interest or 0), volume=vol))
    if not out:
        return [], "no quotes for nearest expiry"
    return out, "ok"


def build_contract(signal, trading, data) -> tuple[ContractPick | None, str]:
    """Fetch the chain for `signal` and select the slightly-ITM long leg, or
    (None, reason). Underlyings outside the whitelist are hard-rejected."""
    if signal.symbol not in config.UNIVERSE:
        return None, f"{signal.symbol} not in whitelist"
    try:
        candidates, why = _fetch_candidates(signal, trading, data)
    except Exception as exc:
        log.warning("%s: chain fetch failed: %s", signal.symbol, exc)
        return None, f"chain fetch error: {exc}"
    if not candidates:
        return None, why
    return select_contract(signal, candidates)
