"""options_chain.py — pulls chain data and filters by spread / OI / IV.

Contract selection for a signal:
  1. Contracts 1-7 DTE, calls for bullish / puts for bearish
  2. Strike 1-3.5% out of the money (percent of spot, so the band scales
     with the underlying's price instead of counting strike increments)
  3. Open interest > 500
  4. Bid/ask spread <= max($0.10, 5% of the option's mid price)
  5. IV rank <= 60 (rank built from a locally persisted ATM-IV history; the
     check is skipped with a warning until MIN_IV_HISTORY sessions are stored)
  6. Premium must fit MAX_TRADE_COST

Returns the candidate with the tightest spread (ties -> nearer strike).
"""
import datetime as dt
import json
import logging
import os
from dataclasses import dataclass

from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionSnapshotRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetStatus, ContractType
from alpaca.trading.requests import GetOptionContractsRequest

import config
from scanner import Signal
from utils import now_et

log = logging.getLogger("options_chain")


@dataclass
class ContractPick:
    option_symbol: str
    underlying: str
    otype: str              # "call" | "put"
    strike: float
    expiry: dt.date
    bid: float
    ask: float
    spread: float
    open_interest: int
    iv: float | None
    iv_rank: float | None

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def cost(self) -> float:
        """Worst-case fill cost for one contract."""
        return self.ask * 100 * config.MAX_CONTRACTS


class ChainFetcher:
    def __init__(self, trading: TradingClient, option_data: OptionHistoricalDataClient):
        self.trading = trading
        self.data = option_data
        self._iv_history = self._load_iv_history()

    # ---------------- IV rank ----------------

    def _load_iv_history(self) -> dict:
        if os.path.exists(config.IV_HISTORY_FILE):
            try:
                with open(config.IV_HISTORY_FILE) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                log.warning("iv_history.json unreadable, starting fresh")
        return {}

    def _save_iv_history(self):
        with open(config.IV_HISTORY_FILE, "w") as f:
            json.dump(self._iv_history, f, indent=1)

    def _record_iv(self, underlying: str, iv: float):
        """Persist one ATM-ish IV reading per underlying per session."""
        today = now_et().date().isoformat()
        series = self._iv_history.setdefault(underlying, {})
        if today not in series:  # first reading of the day wins
            series[today] = round(iv, 4)
            # Keep ~1 year of sessions
            for stale in sorted(series)[:-252]:
                del series[stale]
            self._save_iv_history()

    def iv_rank(self, underlying: str, current_iv: float) -> float | None:
        """(current - 52wk low) / (52wk high - low) * 100. None until warmed up."""
        series = self._iv_history.get(underlying, {})
        if len(series) < config.MIN_IV_HISTORY:
            return None
        ivs = list(series.values())
        lo, hi = min(ivs), max(ivs)
        if hi <= lo:
            return None
        return (current_iv - lo) / (hi - lo) * 100

    # ---------------- chain fetch ----------------

    def _contracts(self, signal: Signal) -> list:
        today = now_et().date()
        ctype = ContractType.CALL if signal.direction == "call" else ContractType.PUT
        # Wide strike net (+/-5%); exact OTM picking happens below.
        req = GetOptionContractsRequest(
            underlying_symbols=[signal.symbol],
            status=AssetStatus.ACTIVE,
            type=ctype,
            expiration_date_gte=today + dt.timedelta(days=config.MIN_DTE),
            expiration_date_lte=today + dt.timedelta(days=config.MAX_DTE),
            strike_price_gte=str(round(signal.spot * 0.95, 2)),
            strike_price_lte=str(round(signal.spot * 1.05, 2)),
            limit=300,
        )
        resp = self.trading.get_option_contracts(req)
        return list(resp.option_contracts or [])

    @staticmethod
    def _otm_strikes(contracts: list, signal: Signal) -> dict:
        """Map (expiry, strike) -> nearness rank for strikes inside the OTM band.

        The band is OTM_MIN_PCT..OTM_MAX_PCT percent of spot, so it lands on
        comparable contracts whether the underlying trades at $200 or $700.
        Rank 1 = nearest to the money within the band (used for tie-breaks
        and for recording the IV history).
        """
        if signal.direction == "call":
            lo = signal.spot * (1 + config.OTM_MIN_PCT / 100)
            hi = signal.spot * (1 + config.OTM_MAX_PCT / 100)
        else:
            lo = signal.spot * (1 - config.OTM_MAX_PCT / 100)
            hi = signal.spot * (1 - config.OTM_MIN_PCT / 100)
        by_expiry: dict[dt.date, list[float]] = {}
        for c in contracts:
            strike = float(c.strike_price)
            if lo <= strike <= hi:
                by_expiry.setdefault(c.expiration_date, []).append(strike)
        targets: dict[tuple[dt.date, float], int] = {}
        for expiry, strikes in by_expiry.items():
            strikes.sort(key=lambda s: abs(s - signal.spot))
            for rank, strike in enumerate(strikes, start=1):
                targets[(expiry, strike)] = rank
        return targets

    def find_contract(self, signal: Signal) -> ContractPick | None:
        """Best tradable contract for a signal, or None with the reason logged."""
        try:
            contracts = self._contracts(signal)
        except Exception as exc:
            log.warning("%s: contract fetch failed: %s", signal.symbol, exc)
            return None
        if not contracts:
            log.info("%s: no contracts in 1-7 DTE window", signal.symbol)
            return None

        targets = self._otm_strikes(contracts, signal)
        candidates = {
            c.symbol: c for c in contracts
            if (c.expiration_date, float(c.strike_price)) in targets
        }
        if not candidates:
            log.info("%s: no strikes %.1f-%.1f%% OTM found", signal.symbol,
                     config.OTM_MIN_PCT, config.OTM_MAX_PCT)
            return None

        # Open interest filter (from contract metadata)
        viable = {
            sym: c for sym, c in candidates.items()
            if int(c.open_interest or 0) > config.MIN_OPEN_INTEREST
        }
        if not viable:
            log.info("%s: all candidates under %d OI", signal.symbol, config.MIN_OPEN_INTEREST)
            return None

        try:
            snaps = self.data.get_option_snapshot(
                OptionSnapshotRequest(symbol_or_symbols=list(viable))
            )
        except Exception as exc:
            log.warning("%s: snapshot fetch failed: %s", signal.symbol, exc)
            return None

        picks: list[ContractPick] = []
        rejected = {"no quote": 0, "wide spread": 0, "over budget": 0, "IV rank": 0}
        for sym, c in viable.items():
            snap = snaps.get(sym)
            if not snap or not snap.latest_quote:
                rejected["no quote"] += 1
                continue
            bid = float(snap.latest_quote.bid_price or 0)
            ask = float(snap.latest_quote.ask_price or 0)
            if bid <= 0 or ask <= 0:
                rejected["no quote"] += 1
                continue
            spread = ask - bid
            mid = (bid + ask) / 2
            # Flat dollar caps strangle expensive options ($0.15 on a $9 option
            # is a fine price) — allow the larger of the flat cap and % of mid.
            if spread > max(config.MAX_SPREAD, mid * config.MAX_SPREAD_PCT / 100):
                rejected["wide spread"] += 1
                continue
            iv = float(snap.implied_volatility) if snap.implied_volatility else None
            pick = ContractPick(
                option_symbol=sym,
                underlying=signal.symbol,
                otype=signal.direction,
                strike=float(c.strike_price),
                expiry=c.expiration_date,
                bid=bid,
                ask=ask,
                spread=spread,
                open_interest=int(c.open_interest or 0),
                iv=iv,
                iv_rank=None,
            )
            # IV rank — recorded BEFORE the budget check so the warm-up history
            # accumulates even on days when nothing fits the budget.
            if iv is not None:
                if targets[(c.expiration_date, pick.strike)] == 1:
                    self._record_iv(signal.symbol, iv)
                rank = self.iv_rank(signal.symbol, iv)
                pick.iv_rank = rank
                if rank is None:
                    log.warning("%s: IV history warming up (%d/%d sessions) — rank check skipped",
                                signal.symbol,
                                len(self._iv_history.get(signal.symbol, {})),
                                config.MIN_IV_HISTORY)
                elif rank > config.MAX_IV_RANK:
                    rejected["IV rank"] += 1
                    log.info("%s: IV rank %.0f > %.0f — premium too rich", sym, rank,
                             config.MAX_IV_RANK)
                    continue

            if pick.cost > config.MAX_TRADE_COST:
                rejected["over budget"] += 1
                log.debug("%s: $%.2f exceeds $%.0f budget", sym, pick.cost, config.MAX_TRADE_COST)
                continue

            picks.append(pick)

        if not picks:
            log.info("%s: 0/%d contracts passed — %s", signal.symbol, len(viable),
                     ", ".join(f"{n} {why}" for why, n in rejected.items() if n))
            return None

        # Tightest spread first; prefer the closer strike on ties.
        picks.sort(key=lambda p: (p.spread, targets[(p.expiry, p.strike)]))
        best = picks[0]
        log.info("PICK %s — strike %.1f exp %s bid/ask %.2f/%.2f OI %d IV rank %s",
                 best.option_symbol, best.strike, best.expiry, best.bid, best.ask,
                 best.open_interest,
                 f"{best.iv_rank:.0f}" if best.iv_rank is not None else "n/a")
        return best
