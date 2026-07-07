"""main.py — orchestrator. Scans every 30s, enters on signals, manages exits.

Run:  python main.py          (paper mode — default)
      LIVE_MODE=true in .env  (real money — read the README first)
"""
import datetime as dt
import json
import logging
import math
import os
import sys
import time

import requests

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.trading.client import TradingClient

import ai_analyst
from data_rest import (RestStockBarProvider, RestOptionQuoteProvider,
                       RestOptionSnapshotProvider, RestOrderEventProvider)
from data_hybrid import HybridStockBarProvider, HybridOptionQuoteProvider
from data_stream_orders import StreamOrderEventProvider
from stream_threads import StockBarStreamThread, OptionQuoteStreamThread
from stream_trading import TradingStreamThread
import alerts
import config
import earnings
from executor import Executor
from learner import Learner, extract_features
from options_chain import ChainFetcher
from risk_manager import RiskManager
from scanner import Scanner
from utils import in_entry_window, is_market_day, now_et, past_force_close, retry

log = logging.getLogger("main")


def build_clients():
    if not config.ALPACA_API_KEY or not config.ALPACA_SECRET_KEY:
        sys.exit("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY — copy .env.example to .env")
    trading = TradingClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY,
                            paper=config.PAPER)
    stock_data = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
    option_data = OptionHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
    return trading, stock_data, option_data


def write_status(executor, risk, learner, market_open: bool,
                  running: bool = True, streams: dict | None = None):
    """Heartbeat for the dashboard (app.py). Written atomically each cycle."""
    s = risk.summary()
    status = {
        "updated": now_et().isoformat(timespec="seconds"),
        "running": running,
        "mode": "LIVE" if config.LIVE_MODE else "PAPER",
        "market_open": market_open,
        "daily_pnl": s["pnl"],
        "trades": s["trades"],
        "wins": s["wins"],
        "losses": s["losses"],
        "win_rate": s["win_rate"],
        "halted": s["halted"],
        "paused_until": (risk.state.paused_until.isoformat(timespec="seconds")
                         if risk.state.paused_until else None),
        "open_positions": executor.positions_snapshot(),
        "learner": {
            "trained_on": learner.trained_on,
            "auc": None if math.isnan(learner.auc) else round(learner.auc, 3),
            "gating": learner.gating,
        },
    }
    if streams:
        status["streams"] = {
            name: obj.is_connected() for name, obj in streams.items()
        }
    tmp = config.STATUS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(status, f, indent=1)
    os.replace(tmp, config.STATUS_FILE)


@retry(max_attempts=3, delay=10.0, backoff=1.0,
       exceptions=(requests.exceptions.ConnectionError, OSError))
def get_clock_with_retry(trading):
    """Retry get_clock on transient network errors before giving up."""
    return trading.get_clock()


def stop_requested() -> bool:
    return os.path.exists(config.STOP_FLAG_FILE)


def sleep_responsive(seconds: float):
    """Sleep in small chunks so a dashboard Stop press is noticed within ~2s
    instead of after a full market-closed 60s nap."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and not stop_requested():
        time.sleep(min(2, deadline - time.monotonic()))


def main():
    mode = "LIVE" if config.LIVE_MODE else "PAPER"
    log.info("Starting options bot — %s MODE — universe %s", mode, config.UNIVERSE)
    if config.LIVE_MODE:
        log.warning("*** LIVE MODE — REAL MONEY ***")

    trading, stock_data, option_data = build_clients()
    risk = RiskManager()

    rest_bar_provider = RestStockBarProvider(stock_data)
    rest_quote_provider = RestOptionQuoteProvider(option_data)
    snapshot_provider = RestOptionSnapshotProvider(option_data)
    rest_order_provider = RestOrderEventProvider(trading)

    # Start stream threads (daemon — auto-cleanup on exit)
    trading_stream = TradingStreamThread()
    stock_stream = StockBarStreamThread()
    option_stream = OptionQuoteStreamThread()
    trading_stream.start()
    stock_stream.start()
    option_stream.start()
    streams = {"trading": trading_stream, "stock_bars": stock_stream,
               "option_quotes": option_stream}

    # Hybrid providers: stream-first, REST fallback
    bar_provider = HybridStockBarProvider(stock_stream, rest_bar_provider)
    quote_provider = HybridOptionQuoteProvider(option_stream, rest_quote_provider)
    order_provider = StreamOrderEventProvider(trading_stream, rest_order_provider)

    scanner = Scanner(bar_provider)
    chain = ChainFetcher(trading, snapshot_provider)
    executor = Executor(trading, quote_provider, order_provider, risk)
    learner = Learner()   # trains from trades.csv once enough history exists

    # Crash recovery: adopt any positions left at the broker, clear stray
    # orders, and rebuild today's P&L so the daily loss limit holds.
    executor.reconcile()
    risk.restore_from_log()

    cooldowns: dict[str, dt.datetime] = {}   # underlying -> last entry time
    summary_sent_for: dt.date | None = None
    briefing_sent_for: dt.date | None = None
    next_scan = 0.0                          # monotonic deadline for next scan

    # NOTE: stale stop flags are cleared by the dashboard's Start button, not
    # here — clearing at boot would swallow a Stop pressed during startup.

    while True:
        try:
            now = now_et()
            order_provider.consume_stream_events()

            # --- graceful stop from the dashboard ---
            if stop_requested():
                log.info("Stop requested from dashboard — flattening and shutting down")
                for pnl in executor.flatten_all("dashboard stop"):
                    risk.record_exit(pnl)
                os.remove(config.STOP_FLAG_FILE)
                trading_stream.stop()
                stock_stream.stop()
                option_stream.stop()
                write_status(executor, risk, learner, False, running=False)
                break

            if not is_market_day(now) or not get_clock_with_retry(trading).is_open:
                # Pre-market AI news briefing, once per market day (9:00-9:30 ET)
                if (is_market_day(now) and briefing_sent_for != now.date()
                        and dt.time(9, 0) <= now.time() < dt.time(9, 30)):
                    briefing_sent_for = now.date()
                    ai_analyst.morning_briefing()
                write_status(executor, risk, learner, market_open=False,
                             streams=streams)
                sleep_responsive(60)
                continue

            # --- 3:45 PM hard time stop: flatten, summarize, idle ---
            if past_force_close(now):
                if executor.open_count():
                    for pnl in executor.flatten_all("time stop 3:45 PM ET"):
                        risk.record_exit(pnl)
                if summary_sent_for != now.date() and risk.state.trades_closed:
                    alerts.daily_summary(risk.summary())
                    summary_sent_for = now.date()
                    learner.maybe_retrain()   # learn from today's trades
                    ai_analyst.daily_report() # plain-English AI recap
                write_status(executor, risk, learner, market_open=True,
                             streams=streams)
                sleep_responsive(60)
                continue

            # --- manage open positions every cycle ---
            for pnl in executor.manage_positions():
                for event in risk.record_exit(pnl):
                    if event == "paused":
                        alerts.pause(config.PAUSE_MINUTES)
                    elif event == "halted":
                        alerts.halt(risk.state.daily_pnl)
                        for p in executor.flatten_all("daily loss halt"):
                            risk.record_exit(p)

            # --- collect fills from pending entry orders ---
            for pos in executor.check_pending_entries():
                cooldowns[pos.underlying] = now_et()

            # --- look for new entries (every SCAN_INTERVAL_SEC; exits are
            #     checked more often when positions are open) ---
            if (in_entry_window(now) and not risk.state.halted
                    and time.monotonic() >= next_scan):
                next_scan = time.monotonic() + config.SCAN_INTERVAL_SEC
                for signal in scanner.scan():
                    last = cooldowns.get(signal.symbol)
                    if last and (now - last).total_seconds() < config.SYMBOL_COOLDOWN_MIN * 60:
                        continue
                    if executor.has_position_in(signal.symbol):
                        continue
                    if executor.has_pending_entry_for(signal.symbol):
                        continue
                    if earnings.blocks(signal.symbol):   # IV-crush protection
                        continue

                    pick = chain.find_contract(signal)
                    if pick is None:
                        continue

                    ok, why = risk.can_enter(
                        executor.open_count() + executor.pending_count(),
                        pick.cost)
                    if not ok:
                        log.info("Entry blocked: %s", why)
                        continue

                    # ML filter: block setups that resemble past losers
                    features = extract_features(signal, pick)
                    allowed, win_prob = learner.allows(features)
                    if not allowed:
                        log.info("ENTRY BLOCKED by model: %s P(win)=%.2f < %.2f",
                                 pick.option_symbol, win_prob,
                                 config.ML_WIN_PROB_THRESHOLD)
                        continue
                    if win_prob is not None:
                        log.info("Model P(win)=%.2f for %s", win_prob, pick.option_symbol)

                    tp = (config.RUNNER_TAKE_PROFIT_PCT if signal.strategy == "runner"
                          else config.TAKE_PROFIT_PCT)
                    executor.open_position(pick, signal.reason(), features,
                                           win_prob, tp_pct=tp)

            write_status(executor, risk, learner, market_open=True,
                         streams=streams)
            # Tight loop while holding positions or pending orders (fast
            # TP/SL/trailing checks + fill polling); relaxed cadence when flat.
            has_activity = (executor.open_count() or executor.pending_count())
            sleep_responsive(config.MANAGE_INTERVAL_SEC if has_activity
                             else config.SCAN_INTERVAL_SEC)

        except KeyboardInterrupt:
            log.info("Interrupted — flattening open positions before exit")
            for pnl in executor.flatten_all("manual shutdown"):
                risk.record_exit(pnl)
            trading_stream.stop()
            stock_stream.stop()
            option_stream.stop()
            write_status(executor, risk, learner, False, running=False)
            break
        except Exception as exc:
            log.exception("Main loop error")
            alerts.error(f"Main loop error: {exc}")
            try:
                write_status(executor, risk, learner, market_open=False,
                             streams=streams)
            except Exception:
                pass  # don't let status write failure mask the original error
            time.sleep(config.SCAN_INTERVAL_SEC)


if __name__ == "__main__":
    main()
