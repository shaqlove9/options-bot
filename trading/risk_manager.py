"""risk_manager.py — enforces daily loss limit, position limits, pauses.

Rules:
  - Max 3 open positions
  - Max $50 premium per trade (enforced again here as a backstop)
  - Daily P&L <= -$75  -> halt for the day
  - 2 consecutive losses -> pause new entries for 30 minutes
"""
import datetime as dt
import logging
from dataclasses import dataclass, field

import config
import data.trade_store as trade_store
from utils import now_et

log = logging.getLogger("risk")


@dataclass
class RiskState:
    day: dt.date
    daily_pnl: float = 0.0
    consecutive_losses: int = 0
    halted: bool = False
    paused_until: dt.datetime | None = None
    trades_closed: int = 0
    wins: int = 0
    losses: int = 0
    events: list[str] = field(default_factory=list)


class RiskManager:
    def __init__(self):
        self.state = RiskState(day=now_et().date())

    def _roll_day(self):
        today = now_et().date()
        if today != self.state.day:
            log.info("New session %s — resetting daily risk state", today)
            self.state = RiskState(day=today)

    # ---------------- crash recovery ----------------

    def restore_from_log(self):
        """Rebuild today's P&L from trades.csv so the daily loss limit and
        pause logic survive a mid-day restart. Without this, a crash at -$60
        would reset the limit and allow another -$75 of losses."""
        pnls = trade_store.today_pnls()
        if not pnls:
            return

        s = self.state
        s.daily_pnl = sum(pnls)
        s.trades_closed = len(pnls)
        s.wins = sum(1 for p in pnls if p > 0)
        s.losses = s.trades_closed - s.wins
        trailing_losses = 0
        for p in reversed(pnls):
            if p < 0:
                trailing_losses += 1
            else:
                break
        if config.LIVE_MODE and trailing_losses >= config.CONSEC_LOSS_PAUSE:
            s.paused_until = now_et() + dt.timedelta(minutes=config.PAUSE_MINUTES)
            s.consecutive_losses = 0
        else:
            s.consecutive_losses = trailing_losses
        if config.LIVE_MODE and s.daily_pnl <= -config.MAX_DAILY_LOSS:
            s.halted = True
        log.info("Restored today's state from trades.csv: $%+.2f over %d trades "
                 "(%dW/%dL)%s", s.daily_pnl, s.trades_closed, s.wins, s.losses,
                 " — HALTED" if s.halted else "")

    # ---------------- entry gate ----------------

    def can_enter(self, open_positions: int, trade_cost: float) -> tuple[bool, str]:
        """Single gate every prospective entry must pass."""
        self._roll_day()
        s = self.state
        # All loss-based blocks (halt, pause, worst-case check) are live-only —
        # paper mode trades through them to maximise learning data.
        if config.LIVE_MODE:
            if s.halted:
                return False, "daily loss limit hit — halted for the day"
            if s.paused_until and now_et() < s.paused_until:
                remaining = (s.paused_until - now_et()).total_seconds() / 60
                return False, f"paused after consecutive losses ({remaining:.0f} min left)"
            if s.daily_pnl - trade_cost * config.STOP_LOSS_PCT / 100 <= -config.MAX_DAILY_LOSS:
                return False, "trade's worst-case loss would breach the daily limit"
        if open_positions >= config.MAX_OPEN_POSITIONS:
            return False, f"max open positions ({config.MAX_OPEN_POSITIONS}) reached"
        if trade_cost > config.MAX_TRADE_COST:
            return False, f"trade cost ${trade_cost:.2f} exceeds ${config.MAX_TRADE_COST:.0f} cap"
        if config.MAX_DAILY_TRADES is not None and s.trades_closed >= config.MAX_DAILY_TRADES:
            return False, f"daily trade cap ({config.MAX_DAILY_TRADES}) reached"
        return True, "ok"

    # ---------------- exit accounting ----------------

    def record_exit(self, pnl: float) -> list[str]:
        """Update state after a closed trade. Returns triggered events
        ("paused", "halted") so the caller can fire alerts."""
        self._roll_day()
        s = self.state
        s.daily_pnl += pnl
        s.trades_closed += 1
        events: list[str] = []

        if pnl < 0:
            s.losses += 1
            s.consecutive_losses += 1
            if (config.LIVE_MODE and s.consecutive_losses >= config.CONSEC_LOSS_PAUSE
                    and not s.halted):
                s.paused_until = now_et() + dt.timedelta(minutes=config.PAUSE_MINUTES)
                s.consecutive_losses = 0
                events.append("paused")
                log.warning("%d consecutive losses — pausing entries until %s",
                            config.CONSEC_LOSS_PAUSE, s.paused_until.strftime("%H:%M"))
        else:
            s.wins += 1
            s.consecutive_losses = 0

        if config.LIVE_MODE and s.daily_pnl <= -config.MAX_DAILY_LOSS and not s.halted:
            s.halted = True
            events.append("halted")
            log.error("DAILY LOSS LIMIT HIT (%.2f) — trading halted for the day", s.daily_pnl)

        log.info("Day P&L: $%+.2f (%d trades, %dW/%dL)",
                 s.daily_pnl, s.trades_closed, s.wins, s.losses)
        return events

    # ---------------- summary ----------------

    def summary(self) -> dict:
        s = self.state
        return {
            "date": s.day.isoformat(),
            "pnl": s.daily_pnl,
            "trades": s.trades_closed,
            "wins": s.wins,
            "losses": s.losses,
            "win_rate": (s.wins / s.trades_closed * 100) if s.trades_closed else 0.0,
            "halted": s.halted,
        }
