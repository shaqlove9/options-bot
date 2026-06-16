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
import alerts
import config
import earnings
from equity_executor import EquityExecutor
from executor import Executor
from learner import Learner, extract_features
from options_chain import ChainFetcher
from risk_manager import RiskManager
from scanner import Scanner
from utils import in_entry_window, is_market_day, now_et, past_force_close

log = logging.getLogger("main")


def build_clients():
    if not config.ALPACA_API_KEY or not config.ALPACA_SECRET_KEY:
        sys.exit("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY — copy .env.example to .env")
    trading = TradingClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY,
                            paper=config.PAPER)
    stock_data = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
    option_data = OptionHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
    return trading, stock_data, option_data


def write_status(executor, risk, learner, market_open: bool, running: bool = True):
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
    tmp = config.STATUS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(status, f, indent=1)
    os.replace(tmp, config.STATUS_FILE)


def get_clock_with_retry(trading, retries: int = 3, delay: float = 10.0):
    """Retry get_clock on transient network errors before giving up."""
    for attempt in range(retries):
        try:
            return trading.get_clock()
        except requests.exceptions.ConnectionError:
            if attempt == retries - 1:
                raise
            log.warning("Network error fetching clock (attempt %d/%d) — retrying in %.0fs",
                        attempt + 1, retries, delay)
            time.sleep(delay)


def equity_features(signal, now: dt.datetime) -> dict:
    """Signal features logged with each equity trade (mirrors the options
    feature columns the learner uses, minus the option-specific ones)."""
    open_et = now.replace(hour=9, minute=30, second=0, microsecond=0)
    return {
        "strategy": signal.strategy,
        "momentum_pct": f"{signal.momentum_pct:.3f}",
        "day_change_pct": f"{signal.day_change_pct:.3f}",
        "rsi": f"{signal.rsi:.2f}",
        "rel_volume": f"{signal.rel_volume:.2f}",
        "vwap_dist_pct": f"{signal.vwap_dist_pct:.3f}",
        "minutes_since_open": int((now - open_et).total_seconds() // 60),
    }


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
    scanner = Scanner(stock_data)
    learner = Learner()   # trains from trades.csv once enough history exists
    equity_mode = config.INSTRUMENT == "equity"
    if equity_mode:
        chain = None
        executor = EquityExecutor(trading, stock_data, risk)
        log.info("INSTRUMENT=equity — trading SHARES (momentum sleeve)")
    else:
        chain = ChainFetcher(trading, option_data)
        executor = Executor(trading, option_data, risk)

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

            # --- graceful stop from the dashboard ---
            if stop_requested():
                log.info("Stop requested from dashboard — flattening and shutting down")
                for pnl in executor.flatten_all("dashboard stop"):
                    risk.record_exit(pnl)
                os.remove(config.STOP_FLAG_FILE)
                write_status(executor, risk, learner, False, running=False)
                break

            if not is_market_day(now) or not get_clock_with_retry(trading).is_open:
                # Pre-market AI news briefing, once per market day (9:00-9:30 ET)
                if (is_market_day(now) and briefing_sent_for != now.date()
                        and dt.time(9, 0) <= now.time() < dt.time(9, 30)):
                    briefing_sent_for = now.date()
                    ai_analyst.morning_briefing()
                write_status(executor, risk, learner, market_open=False)
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
                    if not equity_mode:       # learner/AI report are options-schema
                        learner.maybe_retrain()   # learn from today's trades
                        ai_analyst.daily_report() # plain-English AI recap
                write_status(executor, risk, learner, market_open=True)
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
                    if earnings.blocks(signal.symbol):   # IV-crush protection
                        continue

                    # --- equity sleeve: trade the shares, no option chain ---
                    if equity_mode:
                        risk_amt = config.EQ_NOTIONAL_PER_TRADE * config.EQ_STOP_LOSS_PCT / 100
                        if risk_amt > config.EQ_MAX_TRADE_RISK:
                            log.warning("equity per-trade risk $%.0f exceeds "
                                        "EQ_MAX_TRADE_RISK $%.0f — skipping",
                                        risk_amt, config.EQ_MAX_TRADE_RISK)
                            continue
                        ok, why = risk.can_enter(executor.open_count(), risk_amt)
                        if not ok:
                            log.info("Entry blocked: %s", why)
                            continue
                        if executor.open_position_equity(
                                signal, signal.reason(),
                                equity_features(signal, now),
                                tp_pct=config.EQ_TAKE_PROFIT_PCT):
                            cooldowns[signal.symbol] = now
                        continue

                    pick = chain.find_contract(signal)
                    if pick is None:
                        continue

                    ok, why = risk.can_enter(executor.open_count(), pick.cost)
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
                    if executor.open_position(pick, signal.reason(), features,
                                              win_prob, tp_pct=tp):
                        cooldowns[signal.symbol] = now

            write_status(executor, risk, learner, market_open=True)
            # Tight loop while holding positions (fast TP/SL/trailing checks);
            # relaxed cadence when flat.
            sleep_responsive(config.MANAGE_INTERVAL_SEC if executor.open_count()
                             else config.SCAN_INTERVAL_SEC)

        except KeyboardInterrupt:
            log.info("Interrupted — flattening open positions before exit")
            for pnl in executor.flatten_all("manual shutdown"):
                risk.record_exit(pnl)
            write_status(executor, risk, learner, False, running=False)
            break
        except Exception as exc:
            log.exception("Main loop error")
            alerts.error(f"Main loop error: {exc}")
            time.sleep(config.SCAN_INTERVAL_SEC)


if __name__ == "__main__":
    main()
