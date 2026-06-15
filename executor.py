"""executor.py — places and manages option orders via the Alpaca options API.

Entries: marketable limit at the ask (cancelled if unfilled after 20s).
Exits:   +40% take profit, -30% stop loss, hard flatten at 3:45 PM ET.
Every closed trade is appended to trades.csv.
"""
import csv
import datetime as dt
import logging
import os
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
from options_chain import ContractPick
from utils import now_et

log = logging.getLogger("executor")

CSV_FIELDS = [
    "entry_time", "exit_time", "ticker", "option_symbol", "type", "strike",
    "expiry", "qty", "entry_price", "exit_price", "pnl", "pnl_pct",
    "entry_reason", "exit_reason",
    # Signal features at entry — training data for learner.py
    "strategy", "momentum_pct", "day_change_pct", "rsi", "rel_volume",
    "vwap_dist_pct", "iv", "spread", "dte", "minutes_since_open", "win_prob",
]


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
        self._init_csv()

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
                      tp_pct: float | None = None) -> Position | None:
        qty = config.MAX_CONTRACTS
        order = self.trading.submit_order(LimitOrderRequest(
            symbol=pick.option_symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            limit_price=round(pick.ask, 2),   # marketable limit — caps slippage
        ))
        fill = self._await_fill(order.id, config.ENTRY_FILL_TIMEOUT_SEC)
        if fill is None:
            log.info("%s: entry not filled in %ds — cancelled",
                     pick.option_symbol, config.ENTRY_FILL_TIMEOUT_SEC)
            return None

        pos = Position(
            option_symbol=pick.option_symbol,
            underlying=pick.underlying,
            otype=pick.otype,
            strike=pick.strike,
            expiry=pick.expiry,
            qty=qty,
            entry_price=fill,
            entry_time=now_et(),
            entry_reason=entry_reason,
            features=features or {},
            win_prob=win_prob,
            tp_pct=tp_pct if tp_pct is not None else config.TAKE_PROFIT_PCT,
        )
        self.positions[pick.option_symbol] = pos
        log.info("FILLED %s x%d @ $%.2f ($%.0f)", pos.option_symbol, qty, fill, fill * 100 * qty)
        alerts.entry(pick, qty, fill, entry_reason)
        return pos

    def _await_fill(self, order_id, timeout_sec: int) -> float | None:
        """Poll for a fill; cancel and return None on timeout."""
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            order = self.trading.get_order_by_id(order_id)
            if order.status == OrderStatus.FILLED:
                return float(order.filled_avg_price)
            if order.status in (OrderStatus.CANCELED, OrderStatus.REJECTED,
                                OrderStatus.EXPIRED):
                return None
            time.sleep(1)
        try:
            self.trading.cancel_order_by_id(order_id)
        except Exception:
            pass  # may have filled in the race — caught on the next status check
        order = self.trading.get_order_by_id(order_id)
        if order.status == OrderStatus.FILLED:
            return float(order.filled_avg_price)
        return None

    # ---------------- exits ----------------

    def manage_positions(self, force_close: bool = False) -> list[float]:
        """Check every open position against TP/SL/time stop. Returns the list
        of realized P&Ls from this pass (fed to the risk manager by main)."""
        realized: list[float] = []
        for sym in list(self.positions):
            pos = self.positions[sym]
            if force_close:
                pnl = self._close(pos, "time stop 3:45 PM ET", market=True)
                if pnl is not None:
                    realized.append(pnl)
                continue

            quote = self._latest_quote(sym)
            if quote is None:
                continue
            bid, _ask = quote
            if bid <= 0 or pos.entry_price <= 0:
                continue
            change_pct = (bid - pos.entry_price) / pos.entry_price * 100
            pos.last_bid, pos.last_pnl_pct = bid, change_pct
            pos.peak_pct = max(pos.peak_pct, change_pct)

            tp = pos.tp_pct or config.TAKE_PROFIT_PCT
            if change_pct >= tp:
                pnl = self._close(pos, f"take profit ({change_pct:+.0f}%)")
            elif change_pct <= -config.STOP_LOSS_PCT:
                pnl = self._close(pos, f"stop loss ({change_pct:+.0f}%)")
            elif (pos.peak_pct >= config.TRAIL_TRIGGER_PCT
                  and pos.peak_pct - change_pct >= config.TRAIL_GIVEBACK_PCT):
                # Lock in gains: peaked past the trigger, gave back too much.
                pnl = self._close(pos, f"trailing stop (peaked {pos.peak_pct:+.0f}%, "
                                       f"now {change_pct:+.0f}%)")
            else:
                continue
            if pnl is not None:
                realized.append(pnl)
        return realized

    def flatten_all(self, reason: str) -> list[float]:
        """Close everything at market — used for halts and the 3:45 time stop."""
        realized = []
        for sym in list(self.positions):
            pnl = self._close(self.positions[sym], reason, market=True)
            if pnl is not None:
                realized.append(pnl)
        return realized

    def _close(self, pos: Position, reason: str, market: bool = False) -> float | None:
        try:
            if market:
                req = MarketOrderRequest(symbol=pos.option_symbol, qty=pos.qty,
                                         side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
            else:
                quote = self._latest_quote(pos.option_symbol)
                bid = quote[0] if quote else 0
                if bid <= 0:
                    req = MarketOrderRequest(symbol=pos.option_symbol, qty=pos.qty,
                                             side=OrderSide.SELL,
                                             time_in_force=TimeInForce.DAY)
                else:
                    req = LimitOrderRequest(symbol=pos.option_symbol, qty=pos.qty,
                                            side=OrderSide.SELL,
                                            time_in_force=TimeInForce.DAY,
                                            limit_price=round(bid, 2))
            order = self.trading.submit_order(req)
            fill = self._await_fill(order.id, 30)
            if fill is None:
                log.warning("%s: exit limit not filled — retrying at market",
                            pos.option_symbol)
                order = self.trading.submit_order(MarketOrderRequest(
                    symbol=pos.option_symbol, qty=pos.qty, side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY))
                fill = self._await_fill(order.id, 30)
            if fill is None:
                log.error("%s: COULD NOT EXIT — manual intervention needed",
                          pos.option_symbol)
                alerts.error(f"Failed to exit {pos.option_symbol} — close manually!")
                return None
        except Exception as exc:
            log.exception("%s: exit failed", pos.option_symbol)
            alerts.error(f"Exit error on {pos.option_symbol}: {exc}")
            return None

        pnl = (fill - pos.entry_price) * 100 * pos.qty
        pnl_pct = (fill - pos.entry_price) / pos.entry_price * 100
        del self.positions[pos.option_symbol]
        self._log_trade(pos, fill, pnl, pnl_pct, reason)
        alerts.exit(pos, fill, pnl, reason)
        log.info("CLOSED %s @ $%.2f — P&L $%+.2f (%s)", pos.option_symbol, fill, pnl, reason)
        return pnl

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
        return any(p.underlying == underlying for p in self.positions.values())

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

    # ---------------- trade log ----------------

    def _init_csv(self):
        if os.path.exists(config.TRADES_CSV):
            with open(config.TRADES_CSV, newline="") as f:
                header = f.readline().strip().split(",")
            if header == CSV_FIELDS:
                return
            # Schema changed (feature columns added) — keep the old log aside.
            legacy = config.TRADES_CSV.replace(".csv", "_legacy.csv")
            os.replace(config.TRADES_CSV, legacy)
            log.info("trades.csv schema changed — old log moved to %s", legacy)
        with open(config.TRADES_CSV, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    def _log_trade(self, pos: Position, exit_price: float, pnl: float,
                   pnl_pct: float, exit_reason: str):
        with open(config.TRADES_CSV, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow({
                "entry_time": pos.entry_time.isoformat(timespec="seconds"),
                "exit_time": now_et().isoformat(timespec="seconds"),
                "ticker": pos.underlying,
                "option_symbol": pos.option_symbol,
                "type": pos.otype,
                "strike": pos.strike,
                "expiry": pos.expiry.isoformat(),
                "qty": pos.qty,
                "entry_price": f"{pos.entry_price:.2f}",
                "exit_price": f"{exit_price:.2f}",
                "pnl": f"{pnl:.2f}",
                "pnl_pct": f"{pnl_pct:.1f}",
                "entry_reason": pos.entry_reason,
                "exit_reason": exit_reason,
                **{k: pos.features.get(k, "") for k in
                   ("strategy", "momentum_pct", "day_change_pct", "rsi",
                    "rel_volume", "vwap_dist_pct", "iv", "spread", "dte",
                    "minutes_since_open")},
                "win_prob": f"{pos.win_prob:.3f}" if pos.win_prob is not None else "",
            })
