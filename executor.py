"""executor.py — places and manages option orders via the Alpaca options API.

Entries: marketable limit at the ask (cancelled if unfilled after 20s).
Exits:   +40% take profit, -30% stop loss, hard flatten at 3:45 PM ET.
Every closed trade is appended via trade_store.
"""
import datetime as dt
import logging
import re
import time
from dataclasses import dataclass

from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionLatestQuoteRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, OrderSide, OrderStatus, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

import alerts
import config
import trade_store
from options_chain import ContractPick
from utils import now_et

log = logging.getLogger("executor")


@dataclass
class Position:
    option_symbol: str
    underlying: str
    otype: str
    strike: float
    expiry: dt.date
    qty: int
    entry_price: float
    entry_time: dt.datetime
    entry_reason: str
    features: dict          # signal features at entry (learner training data)
    win_prob: float | None  # model's P(win) at entry, if a model existed
    last_bid: float | None = None       # latest seen quote (for the dashboard)
    last_pnl_pct: float | None = None
    peak_pct: float = 0.0               # best gain seen — drives the trailing stop
    tp_pct: float = 0.0                 # per-strategy take profit (set at entry)


@dataclass
class PendingOrder:
    order_id: str
    pick: ContractPick | None   # for entries; None for exits
    side: str                   # "buy" | "sell"
    entry_reason: str
    features: dict
    win_prob: float | None
    tp_pct: float
    submit_time: float          # time.monotonic()
    timeout_sec: int
    position: Position | None = None   # for exits
    exit_reason: str = ""
    is_market: bool = False
    cancel_sent: bool = False


# OCC option symbol, e.g. SPY260612C00600000
_OCC_RE = re.compile(r"^([A-Z]+)(\d{6})([CP])(\d{8})$")


def parse_occ(symbol: str) -> tuple[str, dt.date, str, float] | None:
    """-> (underlying, expiry, 'call'|'put', strike) or None."""
    m = _OCC_RE.match(symbol)
    if not m:
        return None
    underlying, ymd, cp, strike = m.groups()
    expiry = dt.datetime.strptime(ymd, "%y%m%d").date()
    return underlying, expiry, "call" if cp == "C" else "put", int(strike) / 1000


class Executor:
    def __init__(self, trading: TradingClient, option_data: OptionHistoricalDataClient,
                 risk):
        self.trading = trading
        self.data = option_data
        self.risk = risk
        self.positions: dict[str, Position] = {}
        self._underlying_symbols: set[str] = set()    # O(1) lookup for has_position_in
        self._pending_entries: dict[str, PendingOrder] = {}   # order_id -> PendingOrder
        self._pending_exits: dict[str, PendingOrder] = {}   # option_symbol -> PendingOrder
        self._failed_exits: dict[str, int] = {}       # symbol -> consecutive failure count
        self._failed_exit_last_alert: dict[str, float] = {}  # symbol -> monotonic time of last alert
        self._quote_failures: dict[str, int] = {}    # symbol -> consecutive quote failure count
        trade_store.init()

    # ---------------- crash recovery ----------------

    def reconcile(self) -> int:
        """Adopt option positions that already exist at Alpaca (e.g. after a
        crash/restart) so they get managed instead of orphaned overnight.
        Also cancels any stray open orders from the previous run."""
        try:
            self.trading.cancel_orders()
        except Exception as exc:
            log.warning("Could not cancel stray orders: %s", exc)
        try:
            live = self.trading.get_all_positions()
        except Exception as exc:
            log.error("Position reconciliation failed: %s", exc)
            return 0

        adopted = 0
        for p in live:
            if getattr(p, "asset_class", None) != AssetClass.US_OPTION:
                log.warning("Non-option position %s exists in this account — "
                            "this bot will NOT manage it", p.symbol)
                continue
            if p.symbol in self.positions:
                continue
            parsed = parse_occ(p.symbol)
            if not parsed:
                log.error("Cannot parse option symbol %s — close it manually!",
                          p.symbol)
                alerts.error(f"Unmanageable position {p.symbol} — close manually!")
                continue
            underlying, expiry, otype, strike = parsed
            self.positions[p.symbol] = Position(
                option_symbol=p.symbol, underlying=underlying, otype=otype,
                strike=strike, expiry=expiry, qty=int(float(p.qty)),
                entry_price=float(p.avg_entry_price), entry_time=now_et(),
                entry_reason="adopted at restart (reconciled with broker)",
                features={}, win_prob=None,
            )
            self._underlying_symbols.add(underlying)
            adopted += 1
            log.warning("ADOPTED existing position %s x%s @ $%.2f — now managed",
                        p.symbol, p.qty, float(p.avg_entry_price))
        if adopted:
            alerts.error(f"Restart recovery: adopted {adopted} existing "
                         f"position(s) — TP/SL/time-stop now active on them.")
        return adopted

    # ---------------- entries ----------------

    def open_position(self, pick: ContractPick, entry_reason: str,
                      features: dict | None = None,
                      win_prob: float | None = None,
                      tp_pct: float | None = None) -> bool:
        """Submit a limit buy order (non-blocking). Returns True if submitted.
        The fill is collected later via check_pending_entries()."""
        qty = config.MAX_CONTRACTS
        try:
            order = self.trading.submit_order(LimitOrderRequest(
                symbol=pick.option_symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                limit_price=round(pick.ask, 2),
            ))
        except Exception as exc:
            log.error("%s: entry order submit failed: %s", pick.option_symbol, exc)
            return False
        self._pending_entries[order.id] = PendingOrder(
            order_id=order.id, pick=pick, side="buy",
            entry_reason=entry_reason,
            features=features or {},
            win_prob=win_prob,
            tp_pct=tp_pct if tp_pct is not None else config.TAKE_PROFIT_PCT,
            submit_time=time.monotonic(),
            timeout_sec=config.ENTRY_FILL_TIMEOUT_SEC,
        )
        log.info("%s: entry order submitted @ $%.2f limit",
                 pick.option_symbol, pick.ask)
        return True

    def check_pending_entries(self) -> list[Position]:
        """Poll each pending entry once. Returns list of newly filled positions."""
        filled: list[Position] = []
        for oid in list(self._pending_entries):
            pending = self._pending_entries[oid]
            try:
                order = self.trading.get_order_by_id(oid)
            except Exception as exc:
                log.warning("Entry order %s status check failed: %s", oid, exc)
                continue

            if order.status == OrderStatus.FILLED:
                del self._pending_entries[oid]
                fill = float(order.filled_avg_price)
                pick = pending.pick
                pos = Position(
                    option_symbol=pick.option_symbol,
                    underlying=pick.underlying,
                    otype=pick.otype,
                    strike=pick.strike,
                    expiry=pick.expiry,
                    qty=config.MAX_CONTRACTS,
                    entry_price=fill,
                    entry_time=now_et(),
                    entry_reason=pending.entry_reason,
                    features=pending.features,
                    win_prob=pending.win_prob,
                    tp_pct=pending.tp_pct,
                )
                self.positions[pick.option_symbol] = pos
                self._underlying_symbols.add(pick.underlying)
                log.info("FILLED %s x%d @ $%.2f ($%.0f)",
                         pos.option_symbol, pos.qty, fill,
                         fill * 100 * pos.qty)
                alerts.entry(pick, pos.qty, fill, pending.entry_reason)
                filled.append(pos)
                continue

            terminal = order.status in (OrderStatus.CANCELED,
                                        OrderStatus.REJECTED,
                                        OrderStatus.EXPIRED,
                                        OrderStatus.SUSPENDED)
            elapsed = time.monotonic() - pending.submit_time
            timed_out = elapsed >= pending.timeout_sec and not pending.cancel_sent

            if timed_out and not terminal:
                pending.cancel_sent = True
                log.info("%s: entry not filled in %ds — cancelling",
                         pending.pick.option_symbol, pending.timeout_sec)
                try:
                    self.trading.cancel_order_by_id(oid)
                except Exception:
                    pass
                continue

            if terminal:
                del self._pending_entries[oid]
                log.info("%s: entry order %s (status: %s)",
                         pending.pick.option_symbol,
                         "cancelled" if pending.cancel_sent else "rejected",
                         order.status)
        return filled

    # ---------------- exits ----------------

    _FAILED_EXIT_MAX_RETRIES = 3
    _FAILED_EXIT_ALERT_INTERVAL = 60  # seconds between repeated alerts

    def manage_positions(self, force_close: bool = False) -> list[float]:
        """Check every open position against TP/SL/time stop. Returns the list
        of realized P&Ls from this pass (fed to the risk manager by main).
        Non-blocking: submits exit orders and polls pending ones."""
        realized: list[float] = []

        # Collect fills from previously submitted exit orders
        realized.extend(self.check_pending_exits())

        # Retry previously failed exits before normal management
        for sym in list(self._failed_exits):
            if sym not in self.positions:
                self._failed_exits.pop(sym, None)
                self._failed_exit_last_alert.pop(sym, None)
                continue
            if sym in self._pending_exits:
                continue  # already has a pending exit order
            count = self._failed_exits[sym]
            if count < self._FAILED_EXIT_MAX_RETRIES:
                log.info("%s: retrying failed exit (attempt %d)", sym, count + 1)
                self._initiate_close(self.positions[sym],
                                     "retry after failed exit", market=True)
            else:
                # Stop retrying to prevent order spam, but keep alerting
                now = time.monotonic()
                last = self._failed_exit_last_alert.get(sym, 0)
                if now - last >= self._FAILED_EXIT_ALERT_INTERVAL:
                    self._failed_exit_last_alert[sym] = now
                    log.error("%s: exit failed %d times — MANUAL CLOSE REQUIRED",
                              sym, count)
                    alerts.error(f"{sym}: exit failed {count} times — "
                                 f"close manually!")

        for sym in list(self.positions):
            if sym in self._pending_exits:
                continue  # exit already in flight
            pos = self.positions[sym]
            if force_close:
                self._initiate_close(pos, "time stop 3:45 PM ET", market=True)
                continue

            quote = self._latest_quote(sym)
            if quote is None:
                fails = self._quote_failures.get(sym, 0) + 1
                self._quote_failures[sym] = fails
                if fails >= 5 and fails % 5 == 0:
                    log.error("%s: %d consecutive quote failures — position unmanaged",
                              sym, fails)
                    alerts.error(f"{sym}: {fails} consecutive quote failures — "
                                 f"TP/SL not being checked!")
                continue
            self._quote_failures.pop(sym, None)
            bid, _ask = quote
            if bid <= 0 or pos.entry_price <= 0:
                continue
            change_pct = (bid - pos.entry_price) / pos.entry_price * 100
            pos.last_bid, pos.last_pnl_pct = bid, change_pct
            pos.peak_pct = max(pos.peak_pct, change_pct)

            tp = pos.tp_pct or config.TAKE_PROFIT_PCT
            if change_pct >= tp:
                reason = f"take profit ({change_pct:+.0f}%)"
            elif change_pct <= -config.STOP_LOSS_PCT:
                reason = f"stop loss ({change_pct:+.0f}%)"
            elif (pos.peak_pct >= config.TRAIL_TRIGGER_PCT
                  and pos.peak_pct - change_pct >= config.TRAIL_GIVEBACK_PCT):
                reason = (f"trailing stop (peaked {pos.peak_pct:+.0f}%, "
                          f"now {change_pct:+.0f}%)")
            else:
                continue
            self._initiate_close(pos, reason)
        return realized

    def flatten_all(self, reason: str) -> list[float]:
        """Close everything at market — used for halts and the 3:45 time stop.
        Stays blocking (polls in a tight loop) since this is a safety operation."""
        # Submit all exits
        for sym in list(self.positions):
            if sym not in self._pending_exits:
                self._initiate_close(self.positions[sym], reason, market=True)

        # Poll until all pending exits resolve or 60s hard cap
        deadline = time.monotonic() + 60
        realized: list[float] = []
        while self._pending_exits and time.monotonic() < deadline:
            realized.extend(self.check_pending_exits())
            if self._pending_exits:
                time.sleep(1)
        # Drain any stragglers
        realized.extend(self.check_pending_exits())
        if self._pending_exits:
            log.error("flatten_all: %d exits still pending after 60s hard cap",
                      len(self._pending_exits))
            for sym in list(self._pending_exits):
                del self._pending_exits[sym]
        return realized

    def _finalize_exit(self, pos: Position, fill: float, reason: str) -> float:
        """Bookkeeping after a confirmed fill on an exit order. Returns P&L."""
        pnl = (fill - pos.entry_price) * 100 * pos.qty
        pnl_pct = (fill - pos.entry_price) / pos.entry_price * 100
        trade_store.append_trade(
            entry_time=pos.entry_time, underlying=pos.underlying,
            option_symbol=pos.option_symbol, otype=pos.otype,
            strike=pos.strike, expiry=pos.expiry, qty=pos.qty,
            entry_price=pos.entry_price, exit_price=fill,
            pnl=pnl, pnl_pct=pnl_pct,
            entry_reason=pos.entry_reason, exit_reason=reason,
            features=pos.features, win_prob=pos.win_prob,
        )
        del self.positions[pos.option_symbol]
        self._underlying_symbols = {p.underlying for p in self.positions.values()}
        self._failed_exits.pop(pos.option_symbol, None)
        self._failed_exit_last_alert.pop(pos.option_symbol, None)
        self._quote_failures.pop(pos.option_symbol, None)
        alerts.trade_exit(pos, fill, pnl, reason)
        log.info("CLOSED %s @ $%.2f — P&L $%+.2f (%s)", pos.option_symbol, fill, pnl, reason)
        return pnl

    def _initiate_close(self, pos: Position, reason: str,
                        market: bool = False) -> bool:
        """Submit a sell order without blocking. Returns True if order submitted."""
        sym = pos.option_symbol
        if sym in self._pending_exits:
            return False  # already has a pending exit
        try:
            if market:
                req = MarketOrderRequest(symbol=sym, qty=pos.qty,
                                         side=OrderSide.SELL,
                                         time_in_force=TimeInForce.DAY)
                is_market = True
            else:
                quote = self._latest_quote(sym)
                bid = quote[0] if quote else 0
                if bid <= 0:
                    req = MarketOrderRequest(symbol=sym, qty=pos.qty,
                                             side=OrderSide.SELL,
                                             time_in_force=TimeInForce.DAY)
                    is_market = True
                else:
                    req = LimitOrderRequest(symbol=sym, qty=pos.qty,
                                            side=OrderSide.SELL,
                                            time_in_force=TimeInForce.DAY,
                                            limit_price=round(bid, 2))
                    is_market = False
            order = self.trading.submit_order(req)
            self._pending_exits[sym] = PendingOrder(
                order_id=order.id, pick=None, side="sell",
                entry_reason=pos.entry_reason, features=pos.features,
                win_prob=pos.win_prob, tp_pct=pos.tp_pct,
                submit_time=time.monotonic(), timeout_sec=30,
                position=pos, exit_reason=reason, is_market=is_market,
            )
            log.info("%s: exit order submitted (%s) — %s",
                     sym, "market" if is_market else "limit", reason)
            return True
        except Exception as exc:
            count = self._failed_exits.get(sym, 0) + 1
            self._failed_exits[sym] = count
            log.exception("%s: exit submit failed (attempt %d)", sym, count)
            alerts.error(f"Exit submit error on {sym} (attempt {count}): {exc}")
            return False

    def check_pending_exits(self) -> list[float]:
        """Poll each pending exit once. Returns list of realized P&Ls."""
        realized: list[float] = []
        for sym in list(self._pending_exits):
            pending = self._pending_exits[sym]
            try:
                order = self.trading.get_order_by_id(pending.order_id)
            except Exception as exc:
                log.warning("%s: exit order status check failed: %s", sym, exc)
                continue

            if order.status == OrderStatus.FILLED:
                fill = float(order.filled_avg_price)
                del self._pending_exits[sym]
                pnl = self._finalize_exit(pending.position, fill,
                                          pending.exit_reason)
                realized.append(pnl)
                continue

            terminal = order.status in (OrderStatus.CANCELED,
                                        OrderStatus.REJECTED,
                                        OrderStatus.EXPIRED,
                                        OrderStatus.SUSPENDED)
            elapsed = time.monotonic() - pending.submit_time
            timed_out = elapsed >= pending.timeout_sec and not pending.cancel_sent

            if timed_out and not terminal:
                # Cancel timed-out limit order; next poll will see terminal status
                pending.cancel_sent = True
                try:
                    self.trading.cancel_order_by_id(pending.order_id)
                except Exception:
                    pass
                continue

            if terminal:
                del self._pending_exits[sym]
                if pending.is_market:
                    # Market order rejected/cancelled — record as failed exit
                    count = self._failed_exits.get(sym, 0) + 1
                    self._failed_exits[sym] = count
                    log.error("%s: market exit failed (status %s, attempt %d)",
                              sym, order.status, count)
                    alerts.error(f"Market exit failed on {sym} "
                                 f"(attempt {count}) — will retry")
                    continue
                # Limit order didn't fill — check if position still at broker
                try:
                    broker_pos = self.trading.get_open_position(sym)
                except Exception:
                    broker_pos = None
                if broker_pos is None or int(float(broker_pos.qty)) <= 0:
                    # Position gone — order likely filled in race window
                    log.info("%s: position gone at broker after exit cancel",
                             sym)
                    if order.status == OrderStatus.FILLED:
                        fill = float(order.filled_avg_price)
                    else:
                        fill = pending.position.last_bid or pending.position.entry_price
                        log.warning("%s: using estimated fill $%.2f",
                                    sym, fill)
                    pnl = self._finalize_exit(pending.position, fill,
                                              pending.exit_reason)
                    realized.append(pnl)
                else:
                    # Still held — escalate to market
                    log.warning("%s: exit limit not filled — retrying at market",
                                sym)
                    self._initiate_close(pending.position,
                                         pending.exit_reason, market=True)
        return realized

    def _latest_quote(self, symbol: str) -> tuple[float, float] | None:
        try:
            quotes = self.data.get_option_latest_quote(
                OptionLatestQuoteRequest(symbol_or_symbols=symbol))
            q = quotes[symbol]
            return float(q.bid_price or 0), float(q.ask_price or 0)
        except Exception as exc:
            log.warning("%s: quote fetch failed: %s", symbol, exc)
            return None

    # ---------------- helpers ----------------

    def has_position_in(self, underlying: str) -> bool:
        return underlying in self._underlying_symbols

    def has_pending_entry_for(self, underlying: str) -> bool:
        """True if there's already a pending entry order for this underlying."""
        return any(p.pick.underlying == underlying
                   for p in self._pending_entries.values())

    def pending_count(self) -> int:
        """Number of pending entry orders (for risk limit checks)."""
        return len(self._pending_entries)

    def open_count(self) -> int:
        return len(self.positions)

    def positions_snapshot(self) -> list[dict]:
        """JSON-safe view of open positions for the dashboard."""
        return [{
            "symbol": p.option_symbol,
            "underlying": p.underlying,
            "type": p.otype,
            "strike": p.strike,
            "expiry": p.expiry.isoformat(),
            "qty": p.qty,
            "entry_price": p.entry_price,
            "entry_time": p.entry_time.isoformat(timespec="seconds"),
            "last_bid": p.last_bid,
            "pnl_pct": p.last_pnl_pct,
        } for p in self.positions.values()]

    def pending_entries_snapshot(self) -> list[dict]:
        """JSON-safe view of pending entry orders for the dashboard."""
        return [{
            "symbol": p.pick.option_symbol,
            "underlying": p.pick.underlying,
            "type": p.pick.otype,
            "ask": p.pick.ask,
            "reason": p.entry_reason,
            "elapsed_sec": round(time.monotonic() - p.submit_time),
        } for p in self._pending_entries.values()]

