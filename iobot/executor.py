"""executor.py — submits and manages positions; logs every closed trade.

Single long leg (primary):
  - entry: marketable limit BUY (ask + slip).
  - stop: a LEVEL on the UNDERLYING. When the underlying crosses it we sell the
    option immediately with a market order, whatever the option is worth.
  - target: underlying level at TARGET_R x the risk distance.
  - time-stop: flat by FORCE_CLOSE; never held into expiry/overnight.

Debit vertical (upgrade): opened/closed as ONE mleg order (never leg-by-leg —
closing a single leg trips "insufficient buying power"). Risk is structural, so
there is NO single-leg stop; exits are a profit target on spread value + time-stop.

`single_exit_levels()` and the mleg `*_legs()` builders are pure for unit tests.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import time
from dataclasses import asdict, dataclass, field

from iobot import chain, config, journal, notify
from iobot.clock import now_et, past_force_close

log = logging.getLogger("executor")


# ---------------- positions ----------------

@dataclass
class SingleLegPosition:
    underlying: str
    direction: str
    option_symbol: str
    strike: float
    expiry: dt.date
    qty: int
    entry_fill: float            # premium per share paid
    entry_mid: float
    entry_underlying: float      # spot at entry (stop/target reference)
    stop_level: float
    target_level: float
    max_loss: float
    entry_time: dt.datetime
    signal_id: str = ""
    features: dict = field(default_factory=dict)
    last_value: float | None = None       # current option mid (dashboard)
    structure: str = "single"


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
    entry_debit: float
    entry_mid: float
    max_loss: float
    entry_underlying: float
    entry_time: dt.datetime
    signal_id: str = ""
    features: dict = field(default_factory=dict)
    last_value: float | None = None
    structure: str = "spread"

    @property
    def target_value(self) -> float:
        return self.entry_debit * (1 + config.SPREAD_PROFIT_TARGET_PCT / 100)


# ---------------- pure helpers (unit-tested) ----------------

def single_exit_levels(direction: str, entry_underlying: float) -> tuple[float, float]:
    """(stop_level, target_level) on the underlying for a single long leg."""
    risk = entry_underlying * config.STOP_UNDERLYING_PCT / 100
    if direction == "call":
        return entry_underlying - risk, entry_underlying + config.TARGET_R * risk
    return entry_underlying + risk, entry_underlying - config.TARGET_R * risk


def should_carry_overnight(expiry: dt.date, entry_date: dt.date, today: dt.date) -> bool:
    """PURE: may a position be held past the EOD time-stop? Carry only if overnight is
    enabled, it hasn't been held MAX_HOLD_DAYS, and it isn't within
    OVERNIGHT_MIN_DTE_KEEP sessions of expiry (avoid expiry/gamma risk)."""
    if not config.ALLOW_OVERNIGHT:
        return False
    if (today - entry_date).days >= config.MAX_HOLD_DAYS:
        return False
    if (expiry - today).days <= config.OVERNIGHT_MIN_DTE_KEEP:
        return False
    return True


def single_stop_hit(direction: str, price: float, stop_level: float) -> bool:
    return price <= stop_level if direction == "call" else price >= stop_level


def single_target_hit(direction: str, price: float, target_level: float) -> bool:
    return price >= target_level if direction == "call" else price <= target_level


def open_legs(pick):
    from alpaca.trading.enums import OrderSide, PositionIntent
    from alpaca.trading.requests import OptionLegRequest
    return [
        OptionLegRequest(symbol=pick.long_leg.option_symbol, ratio_qty=1,
                         side=OrderSide.BUY, position_intent=PositionIntent.BUY_TO_OPEN),
        OptionLegRequest(symbol=pick.short_leg.option_symbol, ratio_qty=1,
                         side=OrderSide.SELL, position_intent=PositionIntent.SELL_TO_OPEN),
    ]


def close_legs(pos: SpreadPosition):
    from alpaca.trading.enums import OrderSide, PositionIntent
    from alpaca.trading.requests import OptionLegRequest
    return [
        OptionLegRequest(symbol=pos.long_symbol, ratio_qty=1,
                         side=OrderSide.SELL, position_intent=PositionIntent.SELL_TO_CLOSE),
        OptionLegRequest(symbol=pos.short_symbol, ratio_qty=1,
                         side=OrderSide.BUY, position_intent=PositionIntent.BUY_TO_CLOSE),
    ]


# ---------------- executor ----------------

class Executor:
    def __init__(self, trading, stock_data, option_data, conn):
        self.trading = trading
        self.stock = stock_data
        self.option = option_data
        self.conn = conn
        self.positions: list = []

    def open_count(self) -> int:
        return len(self.positions)

    def has_position_in(self, underlying: str) -> bool:
        return any(p.underlying == underlying for p in self.positions)

    # ---------------- entries ----------------

    def open_single(self, pick, signal, features: dict | None = None
                    ) -> SingleLegPosition | None:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest
        qty = chain.position_qty(pick.ask)
        if qty <= 0:
            log.info("%s: contract $%.2f unaffordable under $%.0f cap — skip",
                     pick.underlying, pick.ask, config.POSITION_MAX_DOLLARS)
            return None
        limit = round(pick.ask + config.ENTRY_SLIP, 2)
        try:
            order = self.trading.submit_order(LimitOrderRequest(
                symbol=pick.option_symbol, qty=qty, side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY, limit_price=limit))
        except Exception as exc:
            log.error("%s: single submit failed: %s", pick.underlying, exc)
            return None
        filled = self._await(order.id, config.ORDER_FILL_TIMEOUT)
        if filled is None:
            log.info("%s: single not filled in %ds — cancelled", pick.underlying,
                     config.ORDER_FILL_TIMEOUT)
            return None
        fill = float(getattr(filled, "filled_avg_price", None) or pick.ask)
        if signal.stop_level is not None and signal.target_level is not None:
            stop_level, target_level = signal.stop_level, signal.target_level  # e.g. range scalp
        else:
            stop_level, target_level = single_exit_levels(signal.direction, signal.spot)
        pos = SingleLegPosition(
            underlying=pick.underlying, direction=signal.direction,
            option_symbol=pick.option_symbol, strike=pick.strike, expiry=pick.expiry,
            qty=qty, entry_fill=fill, entry_mid=pick.mid,
            entry_underlying=signal.spot, stop_level=stop_level, target_level=target_level,
            max_loss=fill * 100 * qty, entry_time=now_et(),
            signal_id=signal.signal_id, features=features or {})
        self.positions.append(pos)
        self._persist(pos)
        notify.entry(pos, signal.reason())
        log.info("OPENED %s @ $%.2f (mid $%.2f) stop u/l %.2f target u/l %.2f maxloss $%.0f",
                 pick.describe(), fill, pick.mid, stop_level, target_level, pos.max_loss)
        return pos

    def open_spread(self, pick, signal, features: dict | None = None
                    ) -> SpreadPosition | None:
        from alpaca.trading.enums import OrderClass, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest
        legs = open_legs(pick)
        limit = round(pick.net_debit + config.ENTRY_SLIP, 2)
        try:
            order = self.trading.submit_order(LimitOrderRequest(
                qty=config.QTY, order_class=OrderClass.MLEG, time_in_force=TimeInForce.DAY,
                legs=legs, limit_price=limit))
        except Exception as exc:
            log.error("%s: spread submit failed: %s", pick.underlying, exc)
            return None
        filled = self._await(order.id, config.ORDER_FILL_TIMEOUT)
        if filled is None:
            log.info("%s: spread not filled — cancelled", pick.underlying)
            return None
        debit = self._net_from_legs(filled, fallback=pick.net_debit)
        entry_mid = pick.long_leg.mid - pick.short_leg.mid
        pos = SpreadPosition(
            underlying=pick.underlying, direction=pick.direction,
            long_symbol=pick.long_leg.option_symbol,
            short_symbol=pick.short_leg.option_symbol,
            long_strike=pick.long_leg.strike, short_strike=pick.short_leg.strike,
            expiry=pick.expiry, width=pick.width, qty=config.QTY, entry_debit=debit,
            entry_mid=entry_mid, max_loss=debit * 100 * config.QTY,
            entry_underlying=signal.spot, entry_time=now_et(),
            signal_id=signal.signal_id, features=features or {})
        self.positions.append(pos)
        self._persist(pos)
        notify.entry(pos, signal.reason())
        log.info("OPENED %s — paid $%.2f maxloss $%.0f", pick.describe(), debit, pos.max_loss)
        return pos

    # ---------------- management ----------------

    def manage(self, force_close: bool = False) -> list[float]:
        """Check every open position once; close on stop/target/time-stop.
        Returns realized P&L of any closes (engine forwards to the governor).

        Overnight-aware: at the EOD time-stop a position is force-flattened ONLY if it
        is unsafe to carry (near expiry / past the hold horizon). Otherwise its normal
        stop/target check still runs (a broken thesis exits), and an intact position is
        carried overnight."""
        realized: list[float] = []
        eod = force_close or past_force_close(now_et())
        today = now_et().date()
        for pos in list(self.positions):
            if eod and not should_carry_overnight(pos.expiry, pos.entry_time.date(), today):
                pnl = self._close(pos, self._eod_reason(pos, today))
                if pnl is not None:
                    realized.append(pnl)
                continue
            # Intraday management — also runs at EOD for carried positions so a
            # last-bar stop/target still exits before we hold overnight.
            if isinstance(pos, SingleLegPosition):
                pnl = self._manage_single(pos)
            else:
                pnl = self._manage_spread(pos)
            if pnl is not None:
                realized.append(pnl)
        return realized

    @staticmethod
    def _eod_reason(pos, today: dt.date) -> str:
        if (today - pos.entry_time.date()).days >= config.MAX_HOLD_DAYS:
            return "time stop (max hold days)"
        if (pos.expiry - today).days <= config.OVERNIGHT_MIN_DTE_KEEP:
            return "time stop (near expiry)"
        return "time stop (EOD flat)"

    def _manage_single(self, pos: SingleLegPosition) -> float | None:
        px = self._underlying_price(pos.underlying)
        pos.last_value = self._option_mid(pos.option_symbol)
        if px is None:
            return None
        if single_stop_hit(pos.direction, px, pos.stop_level):
            return self._close(pos, "stop (underlying)")
        if single_target_hit(pos.direction, px, pos.target_level):
            return self._close(pos, "profit target (underlying)")
        return None

    def _manage_spread(self, pos: SpreadPosition) -> float | None:
        value = self._spread_close_value(pos)
        if value is not None:
            pos.last_value = value
            if value >= pos.target_value:
                return self._close(pos, f"profit target (+{config.SPREAD_PROFIT_TARGET_PCT:.0f}%)")
        return None

    def flatten_all(self, reason: str) -> list[float]:
        return [p for p in (self._close(pos, reason) for pos in list(self.positions))
                if p is not None]

    # ---------------- closing ----------------

    def _close(self, pos, reason: str) -> float | None:
        if isinstance(pos, SingleLegPosition):
            return self._close_single(pos, reason)
        return self._close_spread(pos, reason)

    def _close_single(self, pos: SingleLegPosition, reason: str) -> float | None:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest
        try:
            order = self.trading.submit_order(MarketOrderRequest(
                symbol=pos.option_symbol, qty=pos.qty, side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY))
            filled = self._await(order.id, 30)
        except Exception:
            log.exception("%s: single close failed", pos.underlying)
            return None
        if filled is None:
            log.error("%s: single did NOT close — manual review", pos.underlying)
            return None
        exit_fill = float(getattr(filled, "filled_avg_price", None) or pos.last_value or 0.0)
        pnl = (exit_fill - pos.entry_fill) * 100 * pos.qty
        self.positions.remove(pos)
        self._unpersist(pos)
        self._log(pos, exit_fill, pnl, reason)
        log.info("CLOSED %s @ $%.2f vs $%.2f → P&L $%+.2f (%s)",
                 pos.option_symbol, exit_fill, pos.entry_fill, pnl, reason)
        return pnl

    def _close_spread(self, pos: SpreadPosition, reason: str) -> float | None:
        from alpaca.trading.enums import OrderClass, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest
        legs = close_legs(pos)
        try:
            order = self.trading.submit_order(MarketOrderRequest(
                qty=pos.qty, order_class=OrderClass.MLEG, time_in_force=TimeInForce.DAY,
                legs=legs))
            filled = self._await(order.id, 30)
        except Exception:
            log.exception("%s: spread close failed", pos.underlying)
            return None
        if filled is None:
            log.error("%s: spread did NOT close — manual review", pos.underlying)
            return None
        exit_value = self._net_from_legs(filled, fallback=pos.last_value or pos.entry_debit,
                                         closing=True)
        pnl = (exit_value - pos.entry_debit) * 100 * pos.qty
        self.positions.remove(pos)
        self._unpersist(pos)
        self._log(pos, exit_value, pnl, reason)
        log.info("CLOSED %s exit $%.2f vs debit $%.2f → P&L $%+.2f (%s)",
                 pos.underlying, exit_value, pos.entry_debit, pnl, reason)
        return pnl

    # ---------------- trade log ----------------

    def _log(self, pos, exit_fill: float, pnl: float, reason: str):
        r_mult = pnl / pos.max_loss if pos.max_loss > 0 else 0.0
        exit_underlying = self._underlying_price(pos.underlying)
        single = isinstance(pos, SingleLegPosition)
        journal.log_trade(self.conn, {
            "signal_id": pos.signal_id,
            "structure": pos.structure,
            "symbol": pos.underlying,
            "direction": pos.direction,
            "entry_time": pos.entry_time.isoformat(timespec="seconds"),
            "exit_time": now_et().isoformat(timespec="seconds"),
            "entry_underlying": pos.entry_underlying,
            "exit_underlying": exit_underlying,
            "contract": pos.option_symbol if single else pos.long_symbol,
            "strike": pos.strike if single else pos.long_strike,
            "short_strike": None if single else pos.short_strike,
            "expiry": pos.expiry.isoformat(),
            "qty": pos.qty,
            "entry_fill": pos.entry_fill if single else pos.entry_debit,
            "exit_fill": exit_fill,
            "entry_mid": pos.entry_mid,
            "max_loss": pos.max_loss,
            "pnl": pnl,
            "r_multiple": r_mult,
            "entry_slippage": (pos.entry_fill - pos.entry_mid) if single
                              else (pos.entry_debit - pos.entry_mid),
            "exit_reason": reason,
        })
        notify.exit(pos, exit_fill, pnl, reason)

    # ---------------- quotes / fills ----------------

    def _underlying_price(self, symbol: str) -> float | None:
        from alpaca.data.requests import StockLatestTradeRequest
        try:
            t = self.stock.get_stock_latest_trade(
                StockLatestTradeRequest(symbol_or_symbols=symbol))
            return float(t[symbol].price)
        except Exception as exc:
            log.warning("%s: underlying price failed: %s", symbol, exc)
            return None

    def _option_mid(self, symbol: str) -> float | None:
        from alpaca.data.requests import OptionSnapshotRequest
        try:
            snaps = self.option.get_option_snapshot(
                OptionSnapshotRequest(symbol_or_symbols=[symbol]))
            q = snaps.get(symbol)
            if q and q.latest_quote:
                return (float(q.latest_quote.bid_price or 0) +
                        float(q.latest_quote.ask_price or 0)) / 2
        except Exception:
            pass
        return None

    def _spread_close_value(self, pos: SpreadPosition) -> float | None:
        from alpaca.data.requests import OptionSnapshotRequest
        try:
            snaps = self.option.get_option_snapshot(OptionSnapshotRequest(
                symbol_or_symbols=[pos.long_symbol, pos.short_symbol]))
        except Exception as exc:
            log.warning("%s: spread value quote failed: %s", pos.underlying, exc)
            return None
        ls, ss = snaps.get(pos.long_symbol), snaps.get(pos.short_symbol)
        if not ls or not ss or not ls.latest_quote or not ss.latest_quote:
            return None
        return float(ls.latest_quote.bid_price or 0) - float(ss.latest_quote.ask_price or 0)

    def _await(self, order_id, timeout_sec: int):
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
        legs = getattr(order, "legs", None) or []
        net, got = 0.0, False
        for lg in legs:
            px = getattr(lg, "filled_avg_price", None)
            if px is None:
                continue
            side = str(getattr(lg, "side", "")).lower()
            net += float(px) if "sell" in side else -float(px)
            got = True
        if not got:
            return fallback
        return net if closing else -net

    # ---------------- persistence (overnight carry survives restarts) ----------------

    @staticmethod
    def _pos_key(pos) -> str:
        return pos.option_symbol if isinstance(pos, SingleLegPosition) else pos.long_symbol

    @staticmethod
    def _serialize(pos) -> dict:
        d = asdict(pos)
        d["entry_time"] = pos.entry_time.isoformat()
        d["expiry"] = pos.expiry.isoformat()
        return d

    @staticmethod
    def _deserialize(structure: str, d: dict):
        d = dict(d)
        d["entry_time"] = dt.datetime.fromisoformat(d["entry_time"])
        d["expiry"] = dt.date.fromisoformat(d["expiry"])
        cls = SingleLegPosition if structure == "single" else SpreadPosition
        return cls(**d)

    def _persist(self, pos):
        try:
            self.conn.execute(
                "INSERT OR REPLACE INTO positions(key, structure, opened_at, expiry, "
                "data_json) VALUES (?,?,?,?,?)",
                (self._pos_key(pos), pos.structure, pos.entry_time.isoformat(),
                 pos.expiry.isoformat(), json.dumps(self._serialize(pos))))
            self.conn.commit()
        except Exception:
            log.exception("%s: persist position failed", getattr(pos, "underlying", "?"))

    def _unpersist(self, pos):
        try:
            self.conn.execute("DELETE FROM positions WHERE key=?", (self._pos_key(pos),))
            self.conn.commit()
        except Exception:
            log.exception("%s: unpersist position failed", getattr(pos, "underlying", "?"))

    def _rehydrate(self, held: set[str]):
        """Reload persisted positions, keeping only those still held at the broker;
        delete rows for any that closed while the bot was down."""
        try:
            rows = self.conn.execute(
                "SELECT structure, data_json FROM positions").fetchall()
        except Exception:
            log.exception("rehydrate query failed")
            return
        recovered = 0
        for row in rows:
            try:
                pos = self._deserialize(row["structure"], json.loads(row["data_json"]))
            except Exception:
                log.exception("could not deserialize a persisted position — dropping row")
                continue
            if isinstance(pos, SingleLegPosition):
                legs_ok = pos.option_symbol in held
            else:
                legs_ok = pos.long_symbol in held and pos.short_symbol in held
            if legs_ok:
                self.positions.append(pos)
                recovered += 1
            else:
                self._unpersist(pos)
        if recovered:
            log.warning("rehydrated %d open position(s) from store on restart", recovered)

    def reconcile(self):
        """On restart, cancel stale orders, rehydrate persisted positions that the
        broker still holds, and warn about any broker legs we don't track."""
        try:
            self.trading.cancel_orders()
            live = self.trading.get_all_positions()
        except Exception as exc:
            log.warning("reconcile failed: %s", exc)
            return
        opts = [p for p in live if str(getattr(p, "asset_class", "")).endswith("option")]
        held = {p.symbol for p in opts}
        self._rehydrate(held)
        accounted: set[str] = set()
        for pos in self.positions:
            if isinstance(pos, SingleLegPosition):
                accounted.add(pos.option_symbol)
            else:
                accounted.update({pos.long_symbol, pos.short_symbol})
        orphans = held - accounted
        if orphans:
            log.warning("Found %d broker option leg(s) with no tracked position — "
                        "review/close manually if orphaned: %s", len(orphans),
                        ", ".join(sorted(orphans)))

    def positions_snapshot(self) -> list[dict]:
        out = []
        for p in self.positions:
            single = isinstance(p, SingleLegPosition)
            if single:
                cur = ((p.last_value - p.entry_fill) * 100 * p.qty
                       if p.last_value is not None else None)
            else:
                cur = ((p.last_value - p.entry_debit) * 100 * p.qty
                       if p.last_value is not None else None)
            out.append({
                "structure": p.structure, "underlying": p.underlying,
                "direction": p.direction, "strike": p.strike if single else p.long_strike,
                "short_strike": None if single else p.short_strike,
                "expiry": p.expiry.isoformat(),
                "entry": p.entry_fill if single else p.entry_debit,
                "max_loss": p.max_loss, "cur_value": p.last_value, "cur_pnl": cur,
                "stop_level": p.stop_level if single else None,
                "target_level": p.target_level if single else None,
            })
        return out
