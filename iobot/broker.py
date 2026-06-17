"""broker.py — Alpaca paper clients + account helpers.

PAPER-ONLY GUARD: clients are always constructed against the paper endpoint
(`paper=True`). If `config.GO_LIVE` is set but live execution is not allowed
(it never is in this build), we refuse to start — the flag records intent only.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.trading.client import TradingClient

from iobot import config

log = logging.getLogger("broker")


class LiveExecutionDisabled(RuntimeError):
    """Raised if something tries to run against live money in this paper-only build."""


@dataclass
class Clients:
    trading: TradingClient
    stock_data: StockHistoricalDataClient
    option_data: OptionHistoricalDataClient


def build_clients() -> Clients:
    if config.GO_LIVE and not config.ALLOW_LIVE_EXECUTION:
        raise LiveExecutionDisabled(
            "IOBOT_GO_LIVE is set but this build ships no live-execution code. "
            "Stay on paper until the validation gate passes and a reviewed "
            "live path is added.")
    if not config.ALPACA_API_KEY or not config.ALPACA_SECRET_KEY:
        raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY not set in .env")
    trading = TradingClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY, paper=True)
    return Clients(
        trading=trading,
        stock_data=StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY),
        option_data=OptionHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY),
    )


@dataclass
class AccountSnapshot:
    equity: float
    cash: float
    buying_power: float
    options_buying_power: float
    options_level: int
    is_paper: bool


def account_snapshot(trading: TradingClient) -> AccountSnapshot:
    a = trading.get_account()
    return AccountSnapshot(
        equity=float(a.equity or 0),
        cash=float(a.cash or 0),
        buying_power=float(a.buying_power or 0),
        options_buying_power=float(getattr(a, "options_buying_power", 0) or 0),
        options_level=int(getattr(a, "options_approved_level", 0) or 0),
        is_paper=True,
    )


def market_is_open(trading: TradingClient) -> bool:
    """Authoritative open/closed (handles holidays/half-days)."""
    try:
        return bool(trading.get_clock().is_open)
    except Exception as exc:  # be conservative: if we can't tell, don't trade
        log.warning("clock check failed (%s) — treating market as CLOSED", exc)
        return False
