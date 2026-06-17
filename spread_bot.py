"""spread_bot.py — orchestrator for the defined-risk debit-spread sleeve.

Its OWN paper service (refuses LIVE_MODE; no live-money path here). Each cycle:
scan SPY/QQQ for a directional signal, build a defined-risk vertical, run it past the
liquidity filter + risk governor + buying-power check, submit as ONE mleg order, and
manage exits (profit target / time-stop) — closing combined, never leg-by-leg. Every
rejected signal is logged with its reason; every closed trade is logged for the gate.

Run:
  python spread_bot.py --dry-run   # read-only: build candidate spreads vs the live
                                    # chain and print selection + filters (no submit)
  python spread_bot.py             # paper trading loop (needs .env.spread keys)
"""
import argparse
import csv
import datetime as dt
import json
import logging
import os
import sys
import time
import uuid

import alerts
import config
import feature_store
import spread_chain
from scanner import Scanner, Signal
from spread_executor import SpreadExecutor
from spread_governor import SpreadRiskGovernor
from utils import in_entry_window, is_market_day, now_et, past_force_close

log = logging.getLogger("spread_bot")


def build_clients():
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.trading.client import TradingClient
    if not config.ALPACA_API_KEY or not config.ALPACA_SECRET_KEY:
        sys.exit("Missing ALPACA keys — wire .env.spread (a SEPARATE L3 paper account).")
    trading = TradingClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY, paper=True)
    stock_data = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
    option_data = OptionHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
    return trading, stock_data, option_data


def log_reject(symbol: str, direction: str, reason: str):
    new = not os.path.exists(config.SPREAD_REJECTS_CSV)
    with open(config.SPREAD_REJECTS_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts", "symbol", "direction", "reason"])
        w.writerow([now_et().isoformat(timespec="seconds"), symbol, direction, reason])
    log.info("REJECT %s %s — %s", symbol, direction, reason)


def write_status(executor, gov, equity, market_open, running=True):
    status = {
        "updated": now_et().isoformat(timespec="seconds"),
        "running": running, "market_open": market_open, "mode": "PAPER",
        "equity": equity, "open_spreads": executor.positions_snapshot(),
        "governor": gov.summary(equity) if gov else {},
    }
    tmp = config.SPREAD_STATUS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(status, f, indent=1, default=str)
    os.replace(tmp, config.SPREAD_STATUS_FILE)


def _spot(stock_data, symbol: str) -> float | None:
    from alpaca.data.requests import StockLatestQuoteRequest
    try:
        q = stock_data.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=symbol))[symbol]
        bid, ask = float(q.bid_price or 0), float(q.ask_price or 0)
        return (bid + ask) / 2 if bid and ask else (bid or ask or None)
    except Exception as exc:
        log.warning("%s: spot fetch failed: %s", symbol, exc)
        return None


def dry_run():
    """Read-only: build a candidate spread for each whitelisted name against the
    live chain and print the selection + reject reasons. Needs only read access."""
    trading, stock_data, option_data = build_clients()
    try:
        equity = float(trading.get_account().equity)
    except Exception:
        equity = 10000.0
    print("=" * 64)
    print(f"SPREAD SLEEVE — DRY RUN (no orders). Equity ${equity:,.0f}, "
          f"risk cap ${config.SPREAD_MAX_RISK_PCT/100*equity:,.0f}/trade")
    print("=" * 64)
    for symbol in config.SPREAD_UNIVERSE:
        spot = _spot(stock_data, symbol)
        if spot is None:
            print(f"{symbol}: no spot — skipped")
            continue
        # Synthesize a bullish signal purely to exercise the chain/selection path.
        sig = Signal(symbol=symbol, direction="call", strategy="dryrun",
                     momentum_pct=0.0, day_change_pct=0.0, rsi=60.0, rel_volume=1.5,
                     spot=spot, vwap_dist_pct=0.0, time=now_et())
        pick, reason = spread_chain.build_spread(sig, trading, option_data, equity)
        if pick:
            print(f"\n{symbol} (spot ${spot:.2f}):  {pick.describe()}")
            for leg, tag in ((pick.long_leg, "long "), (pick.short_leg, "short")):
                ok, why = leg.liquid()
                print(f"   {tag} {leg.option_symbol}  bid/ask {leg.bid:.2f}/{leg.ask:.2f}"
                      f"  OI {leg.open_interest}  liquidity: {why}")
            print(f"   max loss ${pick.max_loss:.0f} (<= cap ✓)  reward:risk {pick.reward_risk:.2f}")
        else:
            print(f"\n{symbol} (spot ${spot:.2f}):  NO SPREAD — {reason}")
    print("\n" + "=" * 64)
    print("Dry run complete — selection + filters exercised, nothing submitted.")


def main_loop():
    if config.LIVE_MODE:
        log.error("LIVE_MODE set — spread_bot is PAPER-ONLY and will not run live.")
        sys.exit(3)
    trading, stock_data, option_data = build_clients()
    try:
        acct_id = str(trading.get_account().id)
    except Exception:
        acct_id = None
    scanner = Scanner(stock_data)
    executor = SpreadExecutor(trading, option_data)
    gov = SpreadRiskGovernor(account_id=acct_id)
    feature_store.init(config.SPREAD_META_DB)
    executor.reconcile()
    cooldowns: dict[str, dt.datetime] = {}
    next_scan = 0.0
    log.info("Spread sleeve started — PAPER — universe %s", config.SPREAD_UNIVERSE)

    while True:
        try:
            now = now_et()
            try:
                clock = trading.get_clock()
                market_open = bool(clock.is_open)
            except Exception:
                market_open = is_market_day(now)
            try:
                equity = float(trading.get_account().equity)
            except Exception:
                equity = gov.state.get("peak_equity", 0.0)

            if not market_open:
                write_status(executor, gov, equity, market_open=False)
                time.sleep(60)
                continue

            # time-stop: flatten by the cutoff
            if past_force_close(now) or now >= now.replace(
                    hour=config.SPREAD_CLOSE_CUTOFF[0], minute=config.SPREAD_CLOSE_CUTOFF[1],
                    second=0, microsecond=0):
                for pnl in executor.flatten_all("time stop / cutoff"):
                    pass
                write_status(executor, gov, equity, market_open=True)
                time.sleep(60)
                continue

            executor.manage()                       # profit-target / time exits
            for ev in gov.on_equity(equity):        # kill switches
                for _ in executor.flatten_all(f"governor {ev}"):
                    pass
                alerts.error(f"Spread governor {ev}: {gov.state.get('halt_reason')}")

            if (in_entry_window(now) and not gov.halted
                    and time.monotonic() >= next_scan):
                next_scan = time.monotonic() + config.SCAN_INTERVAL_SEC
                for signal in scanner.scan():
                    if signal.symbol not in config.SPREAD_UNIVERSE:
                        continue
                    last = cooldowns.get(signal.symbol)
                    if last and (now - last).total_seconds() < config.SYMBOL_COOLDOWN_MIN * 60:
                        continue
                    if executor.has_position_in(signal.symbol):
                        continue

                    ok, why = gov.can_enter(executor.open_count())
                    if not ok:
                        log_reject(signal.symbol, signal.direction, f"governor: {why}")
                        continue
                    pick, reason = spread_chain.build_spread(signal, trading, option_data, equity)
                    if pick is None:
                        log_reject(signal.symbol, signal.direction, reason)
                        continue
                    bp_ok, bp_why = gov.buying_power_ok(
                        float(trading.get_account().buying_power), pick.max_loss)
                    if not bp_ok:
                        log_reject(signal.symbol, signal.direction, bp_why)
                        continue

                    # capture pre-entry context (meta-layer compatible)
                    signal_id = uuid.uuid4().hex
                    feats = {"strategy": signal.strategy, "direction": signal.direction,
                             "rsi": round(signal.rsi, 2), "momentum_pct": round(signal.momentum_pct, 3),
                             "rel_volume": round(signal.rel_volume, 2),
                             "net_debit": round(pick.net_debit, 2), "width": pick.width,
                             "max_loss": pick.max_loss, "reward_risk": round(pick.reward_risk, 3)}
                    feature_store.record_signal(signal_id, signal.time, signal.symbol,
                                                signal.direction, signal.strategy, feats,
                                                path=config.SPREAD_META_DB)
                    if executor.open_spread(pick, signal_id, signal.strategy, feats):
                        gov.record_entry()
                        cooldowns[signal.symbol] = now

            write_status(executor, gov, equity, market_open=True)
            time.sleep(config.SCAN_INTERVAL_SEC if executor.open_count() == 0 else 10)
        except KeyboardInterrupt:
            log.info("Interrupted — leaving positions (defined risk) and exiting")
            write_status(executor, gov, 0.0, market_open=False, running=False)
            break
        except Exception as exc:
            log.exception("spread loop error")
            alerts.error(f"Spread loop error: {exc}")
            time.sleep(config.SCAN_INTERVAL_SEC)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="read-only: build candidate spreads vs the live chain, no orders")
    args = ap.parse_args()
    if args.dry_run:
        dry_run()
    else:
        main_loop()


if __name__ == "__main__":
    main()
