"""spread_governor.py — independent risk layer for the defined-risk spread sleeve.

Mirrors risk_governor.EquityRiskGovernor (kill switches active in paper, account-id
tagged state so a stale peak can't false-trip), plus a buying-power / margin check
specific to spreads: Alpaca holds a universal spread maintenance margin (~$1,000) on
top of the debit, so we verify buying power covers debit + SPREAD_MARGIN_PER_SPREAD
before every submit.

Per-trade max-loss capping lives in spread_chain.select_spread (the structure decides
risk); the governor enforces account-level limits: daily-loss kill, trailing-drawdown
kill, and hard caps on concurrent spreads + entries/day.
"""
import json
import logging

import config
from utils import now_et

log = logging.getLogger("spread_governor")


class SpreadRiskGovernor:
    def __init__(self, account_id: str | None = None, state_file: str | None = None):
        self.state_file = state_file or config.SPREAD_STATE_FILE
        self.account_id = account_id
        self.state = self._load()

    def _default(self) -> dict:
        return {"account": self.account_id, "peak_equity": 0.0, "day": None,
                "day_start_equity": 0.0, "halted_day": False, "halted_dd": False,
                "halt_reason": None, "trades_today": 0}

    def _load(self) -> dict:
        try:
            with open(self.state_file) as f:
                s = json.load(f)
        except (OSError, ValueError):
            return self._default()
        if self.account_id and s.get("account") not in (None, self.account_id):
            log.warning("spread governor account changed (%s -> %s) — resetting peak/halt",
                        s.get("account"), self.account_id)
            return self._default()
        if self.account_id and s.get("account") is None and float(s.get("peak_equity", 0)) > 0:
            log.warning("spread governor state untagged with stale peak — resetting for %s",
                        self.account_id)
            return self._default()
        s.setdefault("account", self.account_id)
        return s

    def _save(self):
        try:
            with open(self.state_file, "w") as f:
                json.dump(self.state, f, indent=2, default=str)
        except OSError:
            log.exception("could not persist spread governor state")

    # ---------------- equity tracking + kills ----------------

    def on_equity(self, equity: float) -> list[str]:
        events: list[str] = []
        today = now_et().date().isoformat()
        s = self.state
        if s.get("day") != today:
            s.update(day=today, day_start_equity=equity, halted_day=False, trades_today=0)
            log.info("spread governor: new session %s, start equity $%.2f", today, equity)
        s["peak_equity"] = max(float(s.get("peak_equity", 0.0)), equity)

        start = s.get("day_start_equity") or equity
        if start > 0 and not s["halted_day"]:
            day_loss = (start - equity) / start * 100
            if day_loss >= config.SPREAD_DAILY_MAX_LOSS_PCT:
                s["halted_day"] = True
                s["halt_reason"] = f"daily loss {day_loss:.2f}% >= {config.SPREAD_DAILY_MAX_LOSS_PCT:.1f}%"
                events.append("halted_day")
                log.error("SPREAD GOVERNOR daily kill: %s", s["halt_reason"])
        peak = s["peak_equity"]
        if peak > 0 and not s["halted_dd"]:
            dd = (peak - equity) / peak * 100
            if dd >= config.SPREAD_TRAILING_DD_PCT:
                s["halted_dd"] = True
                s["halt_reason"] = f"trailing DD {dd:.1f}% >= {config.SPREAD_TRAILING_DD_PCT:.0f}% (peak ${peak:,.0f})"
                events.append("halted_dd")
                log.error("SPREAD GOVERNOR drawdown kill: %s", s["halt_reason"])
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

    # ---------------- entry gates ----------------

    def can_enter(self, open_spreads: int) -> tuple[bool, str]:
        s = self.state
        if s.get("halted_dd"):
            return False, f"halted (drawdown): {s.get('halt_reason')}"
        if s.get("halted_day"):
            return False, f"halted (daily loss): {s.get('halt_reason')}"
        if open_spreads >= config.SPREAD_MAX_CONCURRENT:
            return False, f"max concurrent spreads ({config.SPREAD_MAX_CONCURRENT})"
        if s.get("trades_today", 0) >= config.SPREAD_MAX_TRADES_DAY:
            return False, f"daily trade cap ({config.SPREAD_MAX_TRADES_DAY})"
        return True, "ok"

    def buying_power_ok(self, buying_power: float, max_loss: float) -> tuple[bool, str]:
        """Spreads hold a universal maintenance margin on top of the debit."""
        need = max_loss + config.SPREAD_MARGIN_PER_SPREAD
        if buying_power < need:
            return False, (f"insufficient buying power ${buying_power:,.0f} < "
                           f"${need:,.0f} (debit ${max_loss:,.0f} + margin "
                           f"${config.SPREAD_MARGIN_PER_SPREAD:,.0f})")
        return True, "ok"

    def record_entry(self):
        self.state["trades_today"] = self.state.get("trades_today", 0) + 1
        self._save()

    # ---------------- dashboard ----------------

    def summary(self, equity: float | None = None) -> dict:
        s = dict(self.state)
        if equity is not None:
            start = s.get("day_start_equity") or 0
            peak = s.get("peak_equity") or 0
            s["day_loss_pct"] = round((start - equity) / start * 100, 2) if start else None
            s["dd_pct"] = round((peak - equity) / peak * 100, 2) if peak else None
            s["dist_to_daily_kill_pct"] = (
                round(config.SPREAD_DAILY_MAX_LOSS_PCT - s["day_loss_pct"], 2)
                if s.get("day_loss_pct") is not None else None)
            s["dist_to_dd_kill_pct"] = (
                round(config.SPREAD_TRAILING_DD_PCT - s["dd_pct"], 2)
                if s.get("dd_pct") is not None else None)
        s["halted"] = self.halted
        return s
