"""governor.py — risk governor + kill switches. Protects capital first.

Enforced BEFORE every entry, regardless of signal quality:
  - per-trade risk cap: defined max loss <= RISK_PCT_PER_TRADE % of equity (skip,
    never resize, if nothing fits);
  - buying-power / intraday-margin check (incl. spread margin on the upgrade path);
  - daily max-loss kill: halt new entries once the day is down DAILY_MAX_LOSS_PCT;
  - trailing drawdown kill: halt the whole sleeve at TRAILING_DD_PCT off peak
    (persists across restarts until manually reset);
  - caps on trades/day, concurrent positions, and intraday round-trips;
  - cash_account_mode: one full-capital round-trip/day on settled funds, no leverage.

State is persisted as JSON so restarts and the dashboard see the same picture.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass

from iobot import config
from iobot.clock import now_et

log = logging.getLogger("governor")


@dataclass
class GovState:
    date: str = ""
    day_start_equity: float = 0.0
    peak_equity: float = 0.0
    day_realized_pnl: float = 0.0
    trades_today: int = 0
    round_trips_today: int = 0
    sleeve_halted: bool = False        # trailing-DD kill (sticky)
    halt_reason: str = ""


class RiskGovernor:
    def __init__(self, state_file: str = config.GOVERNOR_STATE_FILE):
        self.state_file = state_file
        self.state = self._load()

    # ---------------- persistence ----------------

    def _load(self) -> GovState:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file) as f:
                    return GovState(**json.load(f))
            except (json.JSONDecodeError, OSError, TypeError):
                log.warning("governor state unreadable — starting fresh")
        return GovState()

    def _save(self):
        with open(self.state_file, "w") as f:
            json.dump(asdict(self.state), f, indent=1)

    # ---------------- daily roll / equity tracking ----------------

    def begin_day(self, equity: float):
        """Roll counters at the first call of a new trading day; track peak."""
        today = now_et().date().isoformat()
        if self.state.date != today:
            self.state.date = today
            self.state.day_start_equity = equity
            self.state.day_realized_pnl = 0.0
            self.state.trades_today = 0
            self.state.round_trips_today = 0
            log.info("new trading day %s — counters reset (start equity $%.0f)",
                     today, equity)
        if equity > self.state.peak_equity:
            self.state.peak_equity = equity
        self._save()

    def update_equity(self, equity: float):
        if equity > self.state.peak_equity:
            self.state.peak_equity = equity
        # Sticky trailing-DD kill on the whole sleeve.
        if not self.state.sleeve_halted and self.state.peak_equity > 0:
            dd = (self.state.peak_equity - equity) / self.state.peak_equity * 100
            if dd >= config.TRAILING_DD_PCT:
                self.state.sleeve_halted = True
                self.state.halt_reason = (f"trailing DD {dd:.1f}% >= "
                                          f"{config.TRAILING_DD_PCT:.0f}% off peak")
                log.critical("SLEEVE HALTED — %s", self.state.halt_reason)
        self._save()

    # ---------------- gates ----------------

    def _daily_loss_hit(self, equity: float) -> bool:
        if self.state.day_start_equity <= 0:
            return False
        dd = (self.state.day_start_equity - equity) / self.state.day_start_equity * 100
        return dd >= config.DAILY_MAX_LOSS_PCT

    def can_open(self, equity: float, open_count: int) -> tuple[bool, str]:
        """Count/kill-switch gates that don't depend on the specific contract."""
        if self.state.sleeve_halted:
            return False, f"sleeve halted ({self.state.halt_reason})"
        if self._daily_loss_hit(equity):
            return False, f"daily max-loss kill (-{config.DAILY_MAX_LOSS_PCT:.0f}%) reached"
        if open_count >= config.MAX_CONCURRENT:
            return False, f"max concurrent positions ({config.MAX_CONCURRENT}) open"
        if self.state.trades_today >= config.MAX_TRADES_PER_DAY:
            return False, f"max trades/day ({config.MAX_TRADES_PER_DAY}) reached"
        if self.state.round_trips_today >= config.MAX_ROUND_TRIPS_PER_DAY:
            return False, f"max round-trips/day ({config.MAX_ROUND_TRIPS_PER_DAY}) reached"
        if config.CASH_ACCOUNT_MODE and self.state.trades_today >= 1:
            return False, "cash account: one full-capital round-trip/day already used"
        return True, "ok"

    def approve_risk(self, max_loss: float, required_bp: float,
                     account, cap_equity: float | None = None) -> tuple[bool, str]:
        """Per-trade defined-risk cap + buying-power/settled-funds check.

        max_loss      = defined worst-case dollars (premium, or spread net debit)
        required_bp   = capital the broker needs reserved (premium, or debit+margin)
        cap_equity    = equity base for the per-trade % cap (the SLEEVE equity; falls
                        back to the account equity when not supplied)
        Returns (ok, reason). On failure the engine SKIPS — never resizes up.
        """
        base = account.equity if cap_equity is None else cap_equity
        cap = config.RISK_PCT_PER_TRADE / 100 * base
        if max_loss > cap:
            return False, (f"max loss ${max_loss:.0f} > per-trade cap ${cap:.0f} "
                           f"({config.RISK_PCT_PER_TRADE:.0f}% of equity) — skip")
        if config.CASH_ACCOUNT_MODE:
            if required_bp > account.cash:
                return False, (f"cash account: need ${required_bp:.0f} settled, "
                               f"have ${account.cash:.0f}")
            return True, "ok"
        bp = account.options_buying_power or account.buying_power
        if required_bp > bp:
            return False, f"insufficient buying power (need ${required_bp:.0f}, have ${bp:.0f})"
        return True, "ok"

    # ---------------- bookkeeping ----------------

    def record_entry(self):
        self.state.trades_today += 1
        self._save()

    def record_exit(self, pnl: float):
        self.state.day_realized_pnl += pnl
        self.state.round_trips_today += 1
        self._save()

    def reset_halt(self):
        """Manual trailing-DD reset (operator action)."""
        self.state.sleeve_halted = False
        self.state.halt_reason = ""
        self._save()
        log.warning("sleeve halt manually reset")

    # ---------------- dashboard ----------------

    def snapshot(self, equity: float) -> dict:
        cap = config.RISK_PCT_PER_TRADE / 100 * equity
        day_dd = ((self.state.day_start_equity - equity) / self.state.day_start_equity * 100
                  if self.state.day_start_equity > 0 else 0.0)
        peak_dd = ((self.state.peak_equity - equity) / self.state.peak_equity * 100
                   if self.state.peak_equity > 0 else 0.0)
        return {
            "equity": equity,
            "day_start_equity": self.state.day_start_equity,
            "peak_equity": self.state.peak_equity,
            "day_realized_pnl": self.state.day_realized_pnl,
            "day_drawdown_pct": day_dd,
            "daily_kill_at_pct": config.DAILY_MAX_LOSS_PCT,
            "dist_to_daily_kill_pct": config.DAILY_MAX_LOSS_PCT - day_dd,
            "peak_drawdown_pct": peak_dd,
            "trailing_kill_at_pct": config.TRAILING_DD_PCT,
            "dist_to_trailing_kill_pct": config.TRAILING_DD_PCT - peak_dd,
            "trades_today": self.state.trades_today,
            "max_trades_day": config.MAX_TRADES_PER_DAY,
            "round_trips_today": self.state.round_trips_today,
            "max_round_trips": config.MAX_ROUND_TRIPS_PER_DAY,
            "per_trade_risk_cap": cap,
            "per_trade_risk_pct": config.RISK_PCT_PER_TRADE,
            "sleeve_halted": self.state.sleeve_halted,
            "halt_reason": self.state.halt_reason,
            "cash_account_mode": config.CASH_ACCOUNT_MODE,
        }
