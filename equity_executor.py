"""equity_executor.py — places and manages SHARE orders via the Alpaca stock
API. The equity sleeve: same momentum signal as the options bot, but it buys
the underlying shares (call -> long, put -> short) instead of an option.

Mirrors the public surface main.py uses on Executor (reconcile, manage_positions,
flatten_all, has_position_in, open_count, positions_snapshot) plus
open_position_equity for entries. Exits are % moves of the SHARE price
(EQ_TAKE_PROFIT_PCT / EQ_STOP_LOSS_PCT / trailing), flat by 3:45 ET.

Trades are appended to config.TRADES_CSV (trades_equity.csv in equity mode), a
separate file from the options log.
"""
import csv
import datetime as dt
import logging
import os
import time
from dataclasses import dataclass

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, OrderSide, OrderStatus, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

import alerts
import config
from scanner import Signal
from utils import now_et

log = logging.getLogger("equity_executor")

CSV_FIELDS = [
    "entry_time", "exit_time", "ticker", "side", "qty", "entry_price",
    "exit_price", "pnl", "pnl_pct", "r_multiple", "entry_reason", "exit_reason",
    "signal_id",
    "strategy", "momentum_pct", "day_change_pct", "rsi", "rel_volume",
    "vwap_dist_pct", "minutes_since_open",
]


@dataclass
class EqPosition:
    ticker: str
    direction: str          # "call" = long shares, "put" = short shares
    qty: int                # always positive; direction carries the sign
    entry_price: float
    entry_time: dt.datetime
    entry_reason: str
    features: dict
    tp_pct: float
    signal_id: str = ""          # joins this trade's outcome to its captured features
    last_price: float | None = None
    last_pnl_pct: float | None = None
    peak_pct: float = 0.0

    @property
    def is_long(self) -> bool:
        return self.direction == "call"


class EquityExecutor:
    def __init__(self, trading: TradingClient, stock_data: StockHistoricalDataClient,
                 risk):
        self.trading = trading
        self.data = stock_data
        self.risk = risk
        self.positions: dict[str, EqPosition] = {}
        self._init_csv()

    # ---------------- crash recovery ----------------

    def reconcile(self) -> int:
        """Adopt existing SHARE positions at Alpaca so a restart manages them
        instead of orphaning them; cancel stray orders."""
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
            if getattr(p, "asset_class", None) != AssetClass.US_EQUITY:
                log.warning("Non-equity position %s exists — equity sleeve will "
                            "NOT manage it", p.symbol)
                continue
            if p.symbol in self.positions:
                continue
            qty = int(float(p.qty))                    # negative if short at broker
            self.positions[p.symbol] = EqPosition(
                ticker=p.symbol,
                direction="call" if qty >= 0 else "put",
                qty=abs(qty),
                entry_price=float(p.avg_entry_price),
                entry_time=now_et(),
                entry_reason="adopted at restart (reconciled with broker)",
                features={}, tp_pct=config.EQ_TAKE_PROFIT_PCT,
            )
            adopted += 1
            log.warning("ADOPTED equity position %s x%s @ $%.2f — now managed",
                        p.symbol, qty, float(p.avg_entry_price))
        if adopted:
            alerts.error(f"Restart recovery: adopted {adopted} share position(s) "
                         f"— stops now active on them.")
        return adopted

    # ---------------- entries ----------------

    def open_position_equity(self, signal: Signal, entry_reason: str,
                             features: dict | None = None,
                             tp_pct: float | None = None,
                             signal_id: str = "",
                             notional: float | None = None) -> EqPosition | None:
        if signal.direction == "put" and not config.EQ_ALLOW_SHORT:
            log.info("%s: short disabled (EQ_ALLOW_SHORT=False) — skipping put signal",
                     signal.symbol)
            return None

        quote = self._latest_quote(signal.symbol)
        if quote is None:
            return None
        bid, ask = quote
        long_ = signal.direction == "call"
        # Marketable limit: cross the spread but cap slippage.
        ref = ask if long_ else bid
        if ref <= 0:
            ref = signal.spot
        notional = config.EQ_NOTIONAL_PER_TRADE if notional is None else notional
        qty = int(notional // ref)
        if qty < 1:
            log.info("%s: share price $%.2f too high for $%.0f notional",
                     signal.symbol, ref, notional)
            return None

        side = OrderSide.BUY if long_ else OrderSide.SELL
        order = self.trading.submit_order(LimitOrderRequest(
            symbol=signal.symbol, qty=qty, side=side,
            time_in_force=TimeInForce.DAY, limit_price=round(ref, 2)))
        fill = self._await_fill(order.id, config.ENTRY_FILL_TIMEOUT_SEC)
        if fill is None:
            log.info("%s: equity entry not filled in %ds — cancelled",
                     signal.symbol, config.ENTRY_FILL_TIMEOUT_SEC)
            return None

        pos = EqPosition(
            ticker=signal.symbol, direction=signal.direction, qty=qty,
            entry_price=fill, entry_time=now_et(), entry_reason=entry_reason,
            features=features or {},
            tp_pct=tp_pct if tp_pct is not None else config.EQ_TAKE_PROFIT_PCT,
            signal_id=signal_id,
        )
        self.positions[signal.symbol] = pos
        log.info("FILLED %s %s x%d @ $%.2f ($%.0f notional)", signal.symbol,
                 "LONG" if long_ else "SHORT", qty, fill, fill * qty)
        return pos

    def _await_fill(self, order_id, timeout_sec: int) -> float | None:
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
            pass
        order = self.trading.get_order_by_id(order_id)
        if order.status == OrderStatus.FILLED:
            return float(order.filled_avg_price)
        return None

    # ---------------- exits ----------------

    def manage_positions(self, force_close: bool = False) -> list[float]:
        realized: list[float] = []
        for sym in list(self.positions):
            pos = self.positions[sym]
            if force_close:
                pnl = self._close(pos, "time stop 3:45 PM ET")
                if pnl is not None:
                    realized.append(pnl)
                continue

            price = self._latest_price(sym)
            if price is None or price <= 0 or pos.entry_price <= 0:
                continue
            # Favourable move %: up for longs, down for shorts.
            move = ((price - pos.entry_price) if pos.is_long
                    else (pos.entry_price - price)) / pos.entry_price * 100
            pos.last_price, pos.last_pnl_pct = price, move
            pos.peak_pct = max(pos.peak_pct, move)

            tp = pos.tp_pct or config.EQ_TAKE_PROFIT_PCT
            if move >= tp:
                pnl = self._close(pos, f"take profit ({move:+.2f}%)")
            elif move <= -config.EQ_STOP_LOSS_PCT:
                pnl = self._close(pos, f"stop loss ({move:+.2f}%)")
            elif (pos.peak_pct >= config.EQ_TRAIL_TRIGGER_PCT
                  and pos.peak_pct - move >= config.EQ_TRAIL_GIVEBACK_PCT):
                pnl = self._close(pos, f"trailing stop (peaked {pos.peak_pct:+.2f}%, "
                                       f"now {move:+.2f}%)")
            else:
                continue
            if pnl is not None:
                realized.append(pnl)
        return realized

    def flatten_all(self, reason: str) -> list[float]:
        realized = []
        for sym in list(self.positions):
            pnl = self._close(self.positions[sym], reason)
            if pnl is not None:
                realized.append(pnl)
        return realized

    def _close(self, pos: EqPosition, reason: str) -> float | None:
        # Closing a long = SELL; closing a short = BUY back.
        side = OrderSide.SELL if pos.is_long else OrderSide.BUY
        try:
            order = self.trading.submit_order(MarketOrderRequest(
                symbol=pos.ticker, qty=pos.qty, side=side,
                time_in_force=TimeInForce.DAY))
            fill = self._await_fill(order.id, 30)
            if fill is None:
                log.error("%s: COULD NOT EXIT — manual intervention needed", pos.ticker)
                alerts.error(f"Failed to exit {pos.ticker} shares — close manually!")
                return None
        except Exception as exc:
            log.exception("%s: exit failed", pos.ticker)
            alerts.error(f"Exit error on {pos.ticker}: {exc}")
            return None

        gross = (fill - pos.entry_price) if pos.is_long else (pos.entry_price - fill)
        pnl = gross * pos.qty
        pnl_pct = gross / pos.entry_price * 100
        del self.positions[pos.ticker]
        self._log_trade(pos, fill, pnl, pnl_pct, reason)
        log.info("CLOSED %s @ $%.2f — P&L $%+.2f (%s)", pos.ticker, fill, pnl, reason)
        return pnl

    # ---------------- quotes ----------------

    def _latest_quote(self, symbol: str) -> tuple[float, float] | None:
        try:
            q = self.data.get_stock_latest_quote(
                StockLatestQuoteRequest(symbol_or_symbols=symbol))[symbol]
            return float(q.bid_price or 0), float(q.ask_price or 0)
        except Exception as exc:
            log.warning("%s: quote fetch failed: %s", symbol, exc)
            return None

    def _latest_price(self, symbol: str) -> float | None:
        q = self._latest_quote(symbol)
        if q is None:
            return None
        bid, ask = q
        if bid > 0 and ask > 0:
            return (bid + ask) / 2          # mid
        return bid or ask or None

    # ---------------- helpers ----------------

    def has_position_in(self, underlying: str) -> bool:
        return underlying in self.positions

    def open_count(self) -> int:
        return len(self.positions)

    def positions_snapshot(self) -> list[dict]:
        return [{
            "symbol": p.ticker,
            "underlying": p.ticker,
            "type": "long" if p.is_long else "short",
            "strike": None,
            "expiry": "",
            "qty": p.qty,
            "entry_price": p.entry_price,
            "entry_time": p.entry_time.isoformat(timespec="seconds"),
            "last_bid": p.last_price,
            "pnl_pct": p.last_pnl_pct,
        } for p in self.positions.values()]

    # ---------------- trade log ----------------

    def _init_csv(self):
        if os.path.exists(config.TRADES_CSV):
            with open(config.TRADES_CSV, newline="") as f:
                header = f.readline().strip().split(",")
            if header == CSV_FIELDS:
                return
            legacy = config.TRADES_CSV.replace(".csv", "_legacy.csv")
            os.replace(config.TRADES_CSV, legacy)
            log.info("equity trades schema changed — old log moved to %s", legacy)
        with open(config.TRADES_CSV, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    def _log_trade(self, pos: EqPosition, exit_price: float, pnl: float,
                   pnl_pct: float, exit_reason: str):
        with open(config.TRADES_CSV, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow({
                "entry_time": pos.entry_time.isoformat(timespec="seconds"),
                "exit_time": now_et().isoformat(timespec="seconds"),
                "ticker": pos.ticker,
                "side": "long" if pos.is_long else "short",
                "qty": pos.qty,
                "entry_price": f"{pos.entry_price:.2f}",
                "exit_price": f"{exit_price:.2f}",
                "pnl": f"{pnl:.2f}",
                "pnl_pct": f"{pnl_pct:.2f}",
                # R-multiple = realized move in units of the stop distance, for sizing.
                "r_multiple": f"{pnl_pct / config.EQ_STOP_LOSS_PCT:.3f}"
                              if config.EQ_STOP_LOSS_PCT else "",
                "entry_reason": pos.entry_reason,
                "exit_reason": exit_reason,
                "signal_id": pos.signal_id,
                **{k: pos.features.get(k, "") for k in
                   ("strategy", "momentum_pct", "day_change_pct", "rsi",
                    "rel_volume", "vwap_dist_pct", "minutes_since_open")},
            })
