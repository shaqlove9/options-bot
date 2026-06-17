"""trend_executor.py — disciplined PAPER executor for the trend sleeve.

Wraps the validated trend signal (trend_backtest.latest_target_weights — the SAME
code the backtest used, so live can't drift from what was validated) in a risk
engine + execution layer. On each run it rebalances Alpaca PAPER positions toward
the target weights, logs the plan, and enforces hard guardrails. Meant to be run
on a monthly cadence by cron (trend is a monthly-rebalance strategy, not a loop).

DISCIPLINE / SAFETY (the whole point — a bot executes these flawlessly, a human doesn't):
  - PAPER ONLY. Refuses to run if config.LIVE_MODE is set. There is NO live path
    wired here; going live is a deliberate, gated, manual change (see
    TREND_GOLIVE_CHECKLIST.md). The executor will not place a live order.
  - DRY-RUN by default: computes + logs the intended rebalance WITHOUT sending
    orders. Pass --execute to actually submit PAPER orders.
  - DRAWDOWN CIRCUIT BREAKER: tracks peak equity; if the account falls --max-dd
    below its peak, it HALTS — flattens to cash and refuses new risk until you
    manually clear the halt (--reset-halt). This is what stops a bad streak from
    becoming a blow-up; it cannot be overridden by the signal.
  - Every rebalance is logged (trend_rebalances.csv), status_trend.json is
    refreshed for the dashboard, and a Discord summary is sent.

Usage
-----
  python trend_executor.py                     # dry-run: show the rebalance plan
  python trend_executor.py --execute           # submit the paper orders
  python trend_executor.py --mode long-only --gross 1.0 --max-dd 0.30
  python trend_executor.py --reset-halt         # clear a tripped circuit breaker
"""
import argparse
import datetime as dt
import json
import logging
import os
import sys

import pandas as pd

import config
import trend_backtest as tb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
log = logging.getLogger("trend_exec")

_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(_DIR, "trend_state.json")
REBAL_CSV = os.path.join(_DIR, "trend_rebalances.csv")
STATUS_FILE = os.path.join(_DIR, "status_trend.json")
MIN_ORDER_USD = 5.0          # don't bother rebalancing dust


def load_state() -> dict:
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"peak_equity": 0.0, "halted": False, "halt_reason": None,
                "last_rebalance": None}


def save_state(s: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(s, f, indent=2, default=str)


def alert(title: str, desc: str, color=None):
    try:
        import alerts
        alerts._send(title, desc, color if color is not None else alerts.BLUE)
    except Exception as exc:                       # alerts must never break the run
        log.warning("alert failed: %s", exc)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lookback", type=int, default=12)
    ap.add_argument("--mode", choices=["long-short", "long-only"], default="long-only")
    ap.add_argument("--gross", type=float, default=1.0, help="gross leverage (sum|w|)")
    ap.add_argument("--vol-window", type=int, default=60)
    ap.add_argument("--max-dd", type=float, default=0.30,
                    help="circuit breaker: halt+flatten if equity falls this far from peak")
    ap.add_argument("--history-days", type=int, default=900)
    ap.add_argument("--execute", action="store_true",
                    help="actually submit PAPER orders (default: dry-run plan only)")
    ap.add_argument("--reset-halt", action="store_true", help="clear a tripped circuit breaker")
    args = ap.parse_args()

    # ---- hard paper guard -------------------------------------------------
    if getattr(config, "LIVE_MODE", False):
        log.error("LIVE_MODE is set — trend_executor is PAPER-ONLY and will not run live.")
        sys.exit(3)

    state = load_state()
    if args.reset_halt:
        state.update(halted=False, halt_reason=None)
        save_state(state)
        log.info("Circuit breaker cleared. Next run will resume normal rebalancing.")
        return

    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.trading.client import TradingClient
    sdata = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
    trading = TradingClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY, paper=True)

    acct = trading.get_account()
    equity = float(acct.equity)
    log.info("PAPER account equity $%.2f (status=%s)", equity, acct.status)

    # ---- signal: target weights (same code path as the backtest) ----------
    start = dt.date.today() - dt.timedelta(days=args.history_days)
    closes = tb.fetch_closes(sdata, tb.UNIVERSE, start)
    if closes.empty or len(closes) < args.vol_window + 5:
        log.error("insufficient price history; aborting.")
        sys.exit(2)
    target_w = tb.latest_target_weights(closes, lookback=args.lookback, mode=args.mode,
                                        gross=args.gross, vol_window=args.vol_window)
    prices = closes.iloc[-1]

    # ---- drawdown circuit breaker -----------------------------------------
    peak = max(float(state.get("peak_equity", 0.0)), equity)
    dd = (peak - equity) / peak if peak > 0 else 0.0
    if not state.get("halted") and dd >= args.max_dd:
        state["halted"] = True
        state["halt_reason"] = f"drawdown {dd:.1%} >= max {args.max_dd:.0%} (peak ${peak:,.0f})"
        log.warning("CIRCUIT BREAKER TRIPPED: %s — flattening to cash.", state["halt_reason"])
        alert("🛑 Trend sleeve HALTED — drawdown breaker", state["halt_reason"], color=0xE74C3C)
    halted = state.get("halted", False)
    if halted:
        target_w = target_w * 0.0                  # flat: exit everything

    # ---- current positions -> rebalance plan ------------------------------
    try:
        positions = {p.symbol: float(p.qty) for p in trading.get_all_positions()}
    except Exception as exc:
        log.error("could not read positions: %s", exc)
        positions = {}

    plan = []
    for sym in sorted(set(target_w.index) | set(positions)):
        px = float(prices.get(sym, 0.0))
        if px <= 0:
            continue
        tgt_qty = (equity * float(target_w.get(sym, 0.0))) / px
        cur_qty = positions.get(sym, 0.0)
        delta = tgt_qty - cur_qty
        if abs(delta) * px < MIN_ORDER_USD:
            continue
        plan.append({"symbol": sym, "side": "buy" if delta > 0 else "sell",
                     "qty": round(abs(delta), 4), "price": round(px, 2),
                     "usd": round(abs(delta) * px, 2),
                     "target_w": round(float(target_w.get(sym, 0.0)), 4)})

    # ---- report + (optionally) execute ------------------------------------
    print("=" * 60)
    print(f"TREND REBALANCE PLAN  {'[HALTED→FLAT]' if halted else ''}"
          f"  {'EXECUTE' if args.execute else 'DRY-RUN'}")
    print(f"equity ${equity:,.2f} | peak ${peak:,.0f} | dd {dd:.1%} | "
          f"{args.mode} {args.lookback}mo gross {args.gross:.1f}x")
    print("-" * 60)
    if not plan:
        print("Already at target — no orders.")
    for o in plan:
        print(f"  {o['side'].upper():4} {o['qty']:>9.4f} {o['symbol']:5} "
              f"@ ${o['price']:>8.2f}  (${o['usd']:>9.2f}, w={o['target_w']:+.3f})")
    print("=" * 60)

    submitted, errors = 0, []
    if args.execute and plan:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        for o in plan:
            try:
                trading.submit_order(MarketOrderRequest(
                    symbol=o["symbol"], qty=o["qty"], time_in_force=TimeInForce.DAY,
                    side=OrderSide.BUY if o["side"] == "buy" else OrderSide.SELL))
                submitted += 1
            except Exception as exc:
                errors.append(f"{o['symbol']}: {exc}")
                log.error("order failed %s: %s", o["symbol"], exc)

    # ---- persist state / log / alert --------------------------------------
    state["peak_equity"] = peak
    state["last_rebalance"] = dt.datetime.now().isoformat(timespec="seconds")
    save_state(state)

    row = {"ts": state["last_rebalance"], "equity": round(equity, 2),
           "peak": round(peak, 2), "dd_pct": round(dd * 100, 2), "halted": halted,
           "mode": args.mode, "gross": args.gross, "n_orders": len(plan),
           "executed": submitted, "dry_run": not args.execute,
           "targets": {o["symbol"]: o["target_w"] for o in plan}}
    pd.DataFrame([row]).to_csv(REBAL_CSV, mode="a", header=not os.path.exists(REBAL_CSV), index=False)
    with open(STATUS_FILE, "w") as f:
        json.dump({**row, "errors": errors}, f, indent=2, default=str)

    n_long = int((target_w > 0).sum())
    desc = (f"equity ${equity:,.0f} | dd {dd:.1%} | {n_long} longs | "
            f"{len(plan)} orders {'submitted' if args.execute else '(dry-run)'}")
    if errors:
        desc += f"\n⚠️ {len(errors)} order error(s)"
    alert("📈 Trend rebalance" + (" [HALTED]" if halted else ""), desc,
          color=(0xE67E22 if halted else None))
    log.info("done — %s", desc)


if __name__ == "__main__":
    main()
