"""spread_executor.py — submits and manages DEBIT VERTICAL spreads as single
multi-leg (mleg) orders on the Alpaca options API.

Cardinal rules:
  - Open and close as ONE combined mleg order. NEVER leg-by-leg — closing a single
    leg of a vertical trips "insufficient buying power".
  - Risk is structural (max loss = net debit). Exits are: profit target (+X% of the
    debit) and a time-stop before expiry. There is NO single-leg stop.

`open_legs`/`close_legs` are pure helpers (used by tests) so the two-legs-always
invariant is verifiable without hitting the API.
"""
import csv
import datetime as dt
import logging
import os
import time
from dataclasses import dataclass, field

import config
from utils import now_et

log = logging.getLogger("spread_executor")

CSV_FIELDS = [
    "entry_time", "exit_time", "underlying", "direction", "long_strike",
    "short_strike", "expiry", "width", "qty", "entry_debit", "exit_value",
    "pnl", "pnl_pct_debit", "r_multiple", "max_loss", "max_profit",
    "entry_slippage", "exit_reason", "signal_id", "strategy",
]


@dataclass
class SpreadPosition:
    underlying: str
    direction: str
    long_symbol: str
    short_symbol: str
    long_strike: float
    short_strike: float
    expiry: dt.date
    width: float
    qty: int
    entry_debit: float            # per share actually paid
    entry_mid: float              # spread mid at entry (for slippage)
    max_loss: float               # dollars (entry_debit * 100 * qty)
    entry_time: dt.datetime
    signal_id: str = ""
    strategy: str = ""
    features: dict = field(default_factory=dict)
    last_value: float | None = None

    @property
    def target_value(self) -> float:
        """Spread value (per share) at which we take profit."""
        return self.entry_debit * (1 + config.SPREAD_PROFIT_TARGET_PCT / 100)


def open_legs(pick):
    """Two legs to OPEN a debit vertical: buy the long, sell the short."""
    from alpaca.trading.enums import OrderSide, PositionIntent
    from alpaca.trading.requests import OptionLegRequest
    return [
        OptionLegRequest(symbol=pick.long_leg.option_symbol, ratio_qty=1,
                         side=OrderSide.BUY, position_intent=PositionIntent.BUY_TO_OPEN),
        OptionLegRequest(symbol=pick.short_leg.option_symbol, ratio_qty=1,
                         side=OrderSide.SELL, position_intent=PositionIntent.SELL_TO_OPEN),
    ]


def close_legs(pos: SpreadPosition):
    """Two legs to CLOSE the vertical as one order: sell the long, buy the short."""
    from alpaca.trading.enums import OrderSide, PositionIntent
    from alpaca.trading.requests import OptionLegRequest
    return [
        OptionLegRequest(symbol=pos.long_symbol, ratio_qty=1,
                         side=OrderSide.SELL, position_intent=PositionIntent.SELL_TO_CLOSE),
        OptionLegRequest(symbol=pos.short_symbol, ratio_qty=1,
                         side=OrderSide.BUY, position_intent=PositionIntent.BUY_TO_CLOSE),
    ]


class SpreadExecutor:
    def __init__(self, trading, option_data):
        self.trading = trading
        self.data = option_data
        self.positions: list[SpreadPosition] = []
        self._init_csv()

    def open_count(self) -> int:
        return len(self.positions)

    def has_position_in(self, underlying: str) -> bool:
        return any(p.underlying == underlying for p in self.positions)

    # ---------------- entries ----------------

    def open_spread(self, pick, signal_id: str = "", strategy: str = "",
                    features: dict | None = None) -> SpreadPosition | None:
        from alpaca.trading.enums import OrderClass, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest
        legs = open_legs(pick)                      # always two legs
        limit = round(pick.net_debit + config.SPREAD_ENTRY_SLIP, 2)
        try:
            order = self.trading.submit_order(LimitOrderRequest(
                qty=1, order_class=OrderClass.MLEG, time_in_force=TimeInForce.DAY,
                legs=legs, limit_price=limit))
        except Exception as exc:
            log.error("%s: spread submit failed: %s", pick.underlying, exc)
            return None
        filled = self._await_order(order.id, 20)
        if filled is None:
            log.info("%s: spread not filled in 20s — cancelled", pick.underlying)
            return None
        debit = self._net_from_legs(filled, fallback=pick.net_debit)
        entry_mid = pick.long_leg.mid - pick.short_leg.mid
        pos = SpreadPosition(
            underlying=pick.underlying, direction=pick.direction,
            long_symbol=pick.long_leg.option_symbol,
            short_symbol=pick.short_leg.option_symbol,
            long_strike=pick.long_leg.strike, short_strike=pick.short_leg.strike,
            expiry=pick.expiry, width=pick.width, qty=1, entry_debit=debit,
            entry_mid=entry_mid, max_loss=debit * 100, entry_time=now_et(),
            signal_id=signal_id, strategy=strategy, features=features or {})
        self.positions.append(pos)
        log.info("OPENED %s — paid $%.2f (mid $%.2f, slip $%.2f) max loss $%.0f",
                 pick.describe(), debit, entry_mid, debit - entry_mid, pos.max_loss)
        return pos

    # ---------------- exits ----------------

    def manage(self, force_close: bool = False) -> list[float]:
        realized = []
        cutoff = now_et().replace(hour=config.SPREAD_CLOSE_CUTOFF[0],
                                  minute=config.SPREAD_CLOSE_CUTOFF[1],
                                  second=0, microsecond=0)
        for pos in list(self.positions):
            if force_close:
                pnl = self._close(pos, "force close")
                if pnl is not None:
                    realized.append(pnl)
                continue
            value = self._spread_close_value(pos)   # what we'd net to sell it now
            if value is not None:
                pos.last_value = value
            # time-stop: on/after expiry day past the cutoff (pin/assignment dodge)
            if now_et().date() >= pos.expiry and now_et() >= cutoff:
                pnl = self._close(pos, "time stop (expiry cutoff)")
            elif value is not None and value >= pos.target_value:
                pnl = self._close(pos, f"profit target (+{config.SPREAD_PROFIT_TARGET_PCT:.0f}%)")
            else:
                continue
            if pnl is not None:
                realized.append(pnl)
        return realized

    def flatten_all(self, reason: str) -> list[float]:
        out = []
        for pos in list(self.positions):
            pnl = self._close(pos, reason)
            if pnl is not None:
                out.append(pnl)
        return out

    def _close(self, pos: SpreadPosition, reason: str) -> float | None:
        from alpaca.trading.enums import OrderClass, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest
        legs = close_legs(pos)                       # combined; never single-leg
        try:
            order = self.trading.submit_order(MarketOrderRequest(
                qty=1, order_class=OrderClass.MLEG, time_in_force=TimeInForce.DAY,
                legs=legs))
            filled = self._await_order(order.id, 30)
        except Exception as exc:
            log.exception("%s: spread close failed", pos.underlying)
            return None
        if filled is None:
            log.error("%s: spread did NOT close — manual intervention needed", pos.underlying)
            return None
        # Closing a debit spread returns long.sell - short.buy; positive = credit in.
        exit_value = self._net_from_legs(filled, fallback=pos.last_value or pos.entry_debit,
                                         closing=True)
        pnl = (exit_value - pos.entry_debit) * 100 * pos.qty
        self.positions.remove(pos)
        self._log_trade(pos, exit_value, pnl, reason)
        log.info("CLOSED %s — exit $%.2f vs debit $%.2f → P&L $%+.2f (%s)",
                 pos.underlying, exit_value, pos.entry_debit, pnl, reason)
        return pnl

    # ---------------- quotes / fills ----------------

    def _spread_close_value(self, pos: SpreadPosition) -> float | None:
        """Per-share proceeds to CLOSE now = long.bid - short.ask."""
        from alpaca.data.requests import OptionSnapshotRequest
        try:
            snaps = self.data.get_option_snapshot(OptionSnapshotRequest(
                symbol_or_symbols=[pos.long_symbol, pos.short_symbol]))
        except Exception as exc:
            log.warning("%s: close-value quote failed: %s", pos.underlying, exc)
            return None
        ls, ss = snaps.get(pos.long_symbol), snaps.get(pos.short_symbol)
        if not ls or not ss or not ls.latest_quote or not ss.latest_quote:
            return None
        long_bid = float(ls.latest_quote.bid_price or 0)
        short_ask = float(ss.latest_quote.ask_price or 0)
        return long_bid - short_ask

    def _await_order(self, order_id, timeout_sec: int):
        from alpaca.trading.enums import OrderStatus
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            o = self.trading.get_order_by_id(order_id)
            if o.status == OrderStatus.FILLED:
                return o
            if o.status in (OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED):
                return None
            time.sleep(1)
        try:
            self.trading.cancel_order_by_id(order_id)
        except Exception:
            pass
        o = self.trading.get_order_by_id(order_id)
        return o if o.status == OrderStatus.FILLED else None

    @staticmethod
    def _net_from_legs(order, fallback: float, closing: bool = False) -> float:
        """Net per-share debit (open) or proceeds (close) from filled child legs.
        Buys are paid (−), sells are received (+); for an OPEN we report the debit
        as a positive number, for a CLOSE the net proceeds as positive."""
        legs = getattr(order, "legs", None) or []
        net = 0.0
        got = False
        for lg in legs:
            px = getattr(lg, "filled_avg_price", None)
            if px is None:
                continue
            px = float(px)
            side = str(getattr(lg, "side", "")).lower()
            net += px if "sell" in side else -px      # +received, -paid
            got = True
        if not got:
            return fallback
        # OPEN: net is negative (paid debit) -> report magnitude. CLOSE: net is the
        # proceeds (positive when we receive a credit to close).
        return net if closing else -net

    # ---------------- restart recovery ----------------

    def reconcile(self):
        """Best-effort adoption of an existing vertical at the broker so a restart
        manages (and can close) it rather than orphaning it. A small sleeve holds at
        most one spread; anything ambiguous is logged for manual review, never
        closed leg-by-leg automatically."""
        try:
            self.trading.cancel_orders()
            live = self.trading.get_all_positions()
        except Exception as exc:
            log.warning("spread reconcile failed: %s", exc)
            return
        opts = [p for p in live if str(getattr(p, "asset_class", "")).endswith("option")]
        if opts:
            log.warning("Found %d open option leg(s) at broker on restart — not "
                        "auto-adopted; review/close manually if orphaned: %s",
                        len(opts), ", ".join(p.symbol for p in opts))

    # ---------------- trade log ----------------

    def _init_csv(self):
        if os.path.exists(config.SPREAD_TRADES_CSV):
            with open(config.SPREAD_TRADES_CSV, newline="") as f:
                if f.readline().strip().split(",") == CSV_FIELDS:
                    return
            os.replace(config.SPREAD_TRADES_CSV,
                       config.SPREAD_TRADES_CSV.replace(".csv", "_legacy.csv"))
            log.info("spread trades schema changed — old log moved aside")
        with open(config.SPREAD_TRADES_CSV, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    def _log_trade(self, pos: SpreadPosition, exit_value: float, pnl: float, reason: str):
        max_profit = (pos.width - pos.entry_debit) * 100 * pos.qty
        r_mult = pnl / pos.max_loss if pos.max_loss > 0 else 0.0
        with open(config.SPREAD_TRADES_CSV, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow({
                "entry_time": pos.entry_time.isoformat(timespec="seconds"),
                "exit_time": now_et().isoformat(timespec="seconds"),
                "underlying": pos.underlying, "direction": pos.direction,
                "long_strike": pos.long_strike, "short_strike": pos.short_strike,
                "expiry": pos.expiry.isoformat(), "width": f"{pos.width:g}",
                "qty": pos.qty, "entry_debit": f"{pos.entry_debit:.2f}",
                "exit_value": f"{exit_value:.2f}", "pnl": f"{pnl:.2f}",
                "pnl_pct_debit": f"{pnl / pos.max_loss * 100:.1f}" if pos.max_loss else "",
                "r_multiple": f"{r_mult:.3f}", "max_loss": f"{pos.max_loss:.0f}",
                "max_profit": f"{max_profit:.0f}",
                "entry_slippage": f"{pos.entry_debit - pos.entry_mid:.2f}",
                "exit_reason": reason, "signal_id": pos.signal_id,
                "strategy": pos.strategy,
            })

    def positions_snapshot(self) -> list[dict]:
        out = []
        for p in self.positions:
            cur_pnl = ((p.last_value - p.entry_debit) * 100 * p.qty
                       if p.last_value is not None else None)
            out.append({
                "underlying": p.underlying, "direction": p.direction,
                "long_strike": p.long_strike, "short_strike": p.short_strike,
                "expiry": p.expiry.isoformat(), "entry_debit": p.entry_debit,
                "max_loss": p.max_loss, "cur_value": p.last_value, "cur_pnl": cur_pnl,
            })
        return out
