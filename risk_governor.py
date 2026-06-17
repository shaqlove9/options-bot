"""risk_governor.py — independent risk layer for the equity sleeve.

Separate from the options `risk_manager.py` (left untouched). Two responsibilities:

  1. KILL SWITCHES — active in paper NOW (pure downside protection, mirroring the
     trend sleeve's circuit breaker; they essentially never fire in normal trading
     so they don't starve the meta-model's data):
       - daily max loss: flatten + halt for the day at -EQ_GOV_DAILY_MAX_LOSS_PCT
         of start-of-day equity
       - trailing drawdown: halt if equity falls EQ_GOV_TRAILING_DD_PCT from its peak
       - hard caps on concurrent positions and entries/day
     State is persisted (peak equity, halt flags) and TAGGED with the account id, so
     pointing at a different account can never carry a stale peak and false-trip —
     the same guard we added to the trend executor.

  2. VOL-TARGETED SIZING — built and SHADOW-logged but GATED behind
     EQ_GOV_SIZING_ACTIVE. Until the gate, `target_notional` returns the flat
     EQ_NOTIONAL_PER_TRADE so the in-flight forward-test stays homogeneous; the
     would-be vol-targeted size is always returned alongside for shadow logging.
"""
import datetime as dt
import json
import logging
import os

import config
from utils import now_et

log = logging.getLogger("risk_governor")


class EquityRiskGovernor:
    def __init__(self, account_id: str | None = None, state_file: str | None = None):
        self.state_file = state_file or config.EQ_GOV_STATE_FILE
        self.account_id = account_id
        self.state = self._load()

    # ---------------- persistence ----------------

    def _default_state(self) -> dict:
        return {"account": self.account_id, "peak_equity": 0.0, "day": None,
                "day_start_equity": 0.0, "halted_day": False, "halted_dd": False,
                "halt_reason": None, "trades_today": 0}

    def _load(self) -> dict:
        try:
            with open(self.state_file) as f:
                s = json.load(f)
        except (OSError, ValueError):
            return self._default_state()
        # Account-tag guard: a peak recorded against another account is meaningless
        # here and would false-trip the drawdown breaker — start fresh.
        if self.account_id and s.get("account") not in (None, self.account_id):
            log.warning("governor account changed (%s -> %s) — resetting peak/halt",
                        s.get("account"), self.account_id)
            return self._default_state()
        if self.account_id and s.get("account") is None and float(s.get("peak_equity", 0)) > 0:
            log.warning("governor state untagged with a stale peak — resetting for %s",
                        self.account_id)
            return self._default_state()
        s.setdefault("account", self.account_id)
        return s

    def _save(self):
        try:
            with open(self.state_file, "w") as f:
                json.dump(self.state, f, indent=2, default=str)
        except OSError:
            log.exception("could not persist governor state")

    # ---------------- daily roll + equity tracking ----------------

    def on_equity(self, equity: float) -> list[str]:
        """Call each cycle with current account equity. Rolls the day, updates the
        peak, and evaluates the kill switches. Returns triggered events."""
        events: list[str] = []
        today = now_et().date().isoformat()
        s = self.state
        if s.get("day") != today:
            s.update(day=today, day_start_equity=equity, halted_day=False,
                     trades_today=0)
            log.info("governor: new session %s, start equity $%.2f", today, equity)
        s["peak_equity"] = max(float(s.get("peak_equity", 0.0)), equity)

        if config.EQ_GOV_ENABLED:
            # daily max-loss kill
            start = s.get("day_start_equity") or equity
            if start > 0 and not s["halted_day"]:
                day_loss_pct = (start - equity) / start * 100
                if day_loss_pct >= config.EQ_GOV_DAILY_MAX_LOSS_PCT:
                    s["halted_day"] = True
                    s["halt_reason"] = f"daily loss {day_loss_pct:.2f}% >= {config.EQ_GOV_DAILY_MAX_LOSS_PCT:.1f}%"
                    events.append("halted_day")
                    log.error("GOVERNOR daily kill: %s — flatten + halt for the day", s["halt_reason"])
            # trailing drawdown kill
            peak = s["peak_equity"]
            if peak > 0 and not s["halted_dd"]:
                dd_pct = (peak - equity) / peak * 100
                if dd_pct >= config.EQ_GOV_TRAILING_DD_PCT:
                    s["halted_dd"] = True
                    s["halt_reason"] = f"trailing DD {dd_pct:.1f}% >= {config.EQ_GOV_TRAILING_DD_PCT:.0f}% (peak ${peak:,.0f})"
                    events.append("halted_dd")
                    log.error("GOVERNOR drawdown kill: %s — flatten + halt", s["halt_reason"])
        self._save()
        return events

    @property
    def halted(self) -> bool:
        return bool(self.state.get("halted_day") or self.state.get("halted_dd"))

    def reset_halt(self, which: str = "all"):
        s = self.state
        if which in ("all", "day"):
            s["halted_day"] = False
        if which in ("all", "dd"):
            s["halted_dd"] = False
        s["halt_reason"] = None
        self._save()

    # ---------------- entry gate ----------------

    def can_enter(self, open_positions: int) -> tuple[bool, str]:
        if not config.EQ_GOV_ENABLED:
            return True, "ok"
        s = self.state
        if s.get("halted_dd"):
            return False, f"governor halted (drawdown): {s.get('halt_reason')}"
        if s.get("halted_day"):
            return False, f"governor halted (daily loss): {s.get('halt_reason')}"
        if open_positions >= config.EQ_GOV_MAX_CONCURRENT:
            return False, f"governor max concurrent positions ({config.EQ_GOV_MAX_CONCURRENT})"
        if s.get("trades_today", 0) >= config.EQ_GOV_MAX_TRADES_DAY:
            return False, f"governor daily trade cap ({config.EQ_GOV_MAX_TRADES_DAY})"
        return True, "ok"

    def record_entry(self):
        self.state["trades_today"] = self.state.get("trades_today", 0) + 1
        self._save()

    # ---------------- sizing ----------------

    def _vol_target_notional(self, equity: float, stop_frac: float,
                             win_rate: float | None, payoff: float | None) -> float:
        """Dollar notional so the trade risks ~risk_frac of equity at its stop.
        Fractional-Kelly caps the risk fraction once win-rate/payoff are measured."""
        risk_frac = config.EQ_GOV_RISK_FRAC
        if win_rate is not None and payoff and payoff > 0:
            kelly = max(0.0, win_rate - (1 - win_rate) / payoff)
            risk_frac = min(risk_frac, config.EQ_GOV_KELLY_FRAC * kelly)
        if stop_frac <= 0:
            return 0.0
        notional = (risk_frac * equity) / stop_frac
        return float(min(notional, config.EQ_GOV_MAX_NOTIONAL_FRAC * equity))

    def target_notional(self, equity: float, stop_frac: float,
                        win_rate: float | None = None,
                        payoff: float | None = None) -> tuple[float, float]:
        """Returns (live_notional, shadow_vol_target_notional).
        live = flat EQ_NOTIONAL_PER_TRADE unless EQ_GOV_SIZING_ACTIVE; shadow is
        always the vol-targeted figure for logging/comparison."""
        shadow = self._vol_target_notional(equity, stop_frac, win_rate, payoff)
        live = shadow if config.EQ_GOV_SIZING_ACTIVE else config.EQ_NOTIONAL_PER_TRADE
        return live, shadow

    # ---------------- dashboard ----------------

    def summary(self, equity: float | None = None) -> dict:
        s = dict(self.state)
        if equity is not None:
            start = s.get("day_start_equity") or 0
            peak = s.get("peak_equity") or 0
            s["day_loss_pct"] = round((start - equity) / start * 100, 2) if start else None
            s["dd_pct"] = round((peak - equity) / peak * 100, 2) if peak else None
            s["dist_to_daily_kill_pct"] = (
                round(config.EQ_GOV_DAILY_MAX_LOSS_PCT - s["day_loss_pct"], 2)
                if s.get("day_loss_pct") is not None else None)
            s["dist_to_dd_kill_pct"] = (
                round(config.EQ_GOV_TRAILING_DD_PCT - s["dd_pct"], 2)
                if s.get("dd_pct") is not None else None)
        s["halted"] = self.halted
        s["sizing_active"] = config.EQ_GOV_SIZING_ACTIVE
        return s
