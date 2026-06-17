"""trend_forward.py — live-paper vs backtest scoreboard for the trend sleeve.

Reads the executor's logged equity curve (trend_rebalances.csv), turns it into
realized period returns, re-runs the trend backtest to get the expected baseline,
and prints them side by side with a PRE-COMMITTED PASS/FAIL/INSUFFICIENT verdict
tied to TREND_GOLIVE_CHECKLIST.md. This is the tool that decides paper→live — set
the thresholds before reading the results, not after.

Read-only/paper-only: never trades, never touches the executor's state.

Usage
-----
  python trend_forward.py                 # scoreboard + verdict
  python trend_forward.py --notify        # Discord-alert when the verdict changes
  python trend_forward.py --json out.json
Exit code: 0 PASS, 1 FAIL, 2 INSUFFICIENT (wire into cron if wanted).
"""
import argparse
import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd

import config
import trend_backtest as tb

_DIR = os.path.dirname(os.path.abspath(__file__))
REBAL_CSV = os.path.join(_DIR, "trend_rebalances.csv")
STATUS_FILE = os.path.join(_DIR, "status_trend.json")
NOTIFY_STATE = os.path.join(_DIR, "trend_forward_state.json")

# Pre-committed gate (mirror the checklist). Tighten only, never loosen to rescue.
DECISION = {
    "min_rebalances": 6,         # ≥6 monthly rebalances before judging
    "min_ann_return": 0.0,       # realized annualised return must be net positive
    "max_dd_mult": 1.3,          # realized max DD ≤ 1.3× the backtest's
    "min_sharpe_frac": 0.0,      # realized Sharpe ≥ this × backtest Sharpe (0 = just >0 handled below)
}


def load_realized() -> pd.DataFrame:
    if not os.path.exists(REBAL_CSV):
        return pd.DataFrame()
    df = pd.read_csv(REBAL_CSV)
    df["ts"] = pd.to_datetime(df["ts"])
    # only rows from real (executed) rebalances reflect held-position P&L
    if "dry_run" in df.columns:
        df = df[~df["dry_run"].astype(str).str.lower().isin(["true", "1"])]
    return df.dropna(subset=["equity"]).sort_values("ts").reset_index(drop=True)


def realized_stats(df: pd.DataFrame) -> dict:
    if len(df) < 2:
        return {"n": len(df)}
    eq = df["equity"].to_numpy(float)
    rets = eq[1:] / eq[:-1] - 1.0
    n_periods = len(rets)
    years = max((df["ts"].iloc[-1] - df["ts"].iloc[0]).days / 365.25, 1e-9)
    ann_return = (eq[-1] / eq[0]) ** (1 / years) - 1 if eq[0] > 0 else float("nan")
    vol = np.std(rets, ddof=1) * np.sqrt(12) if n_periods > 1 else float("nan")
    sharpe = (np.mean(rets) / np.std(rets, ddof=1) * np.sqrt(12)
              if n_periods > 1 and np.std(rets, ddof=1) > 0 else float("nan"))
    peak = np.maximum.accumulate(eq)
    max_dd = float((eq / peak - 1).min())
    return {"n": n_periods + 1, "rebalances": n_periods, "period":
            f"{df['ts'].iloc[0].date()} → {df['ts'].iloc[-1].date()}",
            "ann_return": float(ann_return), "vol": float(vol), "sharpe": float(sharpe),
            "max_dd": max_dd, "cur_equity": float(eq[-1]),
            "total_return": float(eq[-1] / eq[0] - 1)}


def backtest_baseline(years: float) -> dict:
    from alpaca.data.historical.stock import StockHistoricalDataClient
    client = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
    start = dt.date.today() - dt.timedelta(days=int(max(years, 3) * 365) + 400)
    closes = tb.fetch_closes(client, tb.UNIVERSE, start)
    m = tb.backtest(closes, lookback=12, mode="long-only", gross=1.0,
                    vol_window=60, cost_bps=5.0)
    s = tb.stats(m)
    return {"ann_return": s["cagr"], "vol": s["vol"], "sharpe": s["sharpe"],
            "max_dd": s["max_dd"]}


def evaluate(live: dict, bt: dict, status: dict) -> tuple[str, list[str]]:
    if live.get("rebalances", 0) < DECISION["min_rebalances"]:
        return ("INSUFFICIENT",
                [f"only {live.get('rebalances', 0)} executed rebalances; need "
                 f"{DECISION['min_rebalances']}"])
    reasons, ok = [], True

    def chk(cond, msg):
        nonlocal ok
        ok = ok and cond
        reasons.append(f"  {'✓' if cond else '✗'} {msg}")

    chk(live["ann_return"] >= DECISION["min_ann_return"],
        f"ann return {live['ann_return']*100:+.1f}% ≥ {DECISION['min_ann_return']*100:.0f}%")
    dd_floor = bt["max_dd"] * DECISION["max_dd_mult"]
    chk(live["max_dd"] >= dd_floor,
        f"max DD {live['max_dd']*100:.1f}% ≥ floor {dd_floor*100:.1f}% "
        f"(1.3× backtest {bt['max_dd']*100:.1f}%)")
    chk(not np.isnan(live["sharpe"]) and live["sharpe"] > 0,
        f"realized Sharpe {live['sharpe']:.2f} > 0 (backtest {bt['sharpe']:.2f})")
    chk(not status.get("halted", False), "circuit breaker not tripped")
    chk(not status.get("errors"), "no execution errors")
    return ("PASS" if ok else "FAIL"), reasons


def render(live, bt, status, verdict, reasons):
    print("=" * 60)
    print("TREND SLEEVE — LIVE (paper) vs BACKTEST")
    print("=" * 60)
    if live.get("rebalances", 0) < 1:
        print("No executed rebalances logged yet in trend_rebalances.csv")
        print("(run `trend_executor.py --execute` monthly on an ISOLATED paper account)")
    else:
        print(f"Live: {live['rebalances']} rebalances  ({live['period']})  "
              f"equity ${live['cur_equity']:,.0f}")
        print(f"{'':18}{'LIVE':>14}{'BACKTEST':>14}")
        print(f"{'Ann return':18}{live['ann_return']*100:>13.1f}%{bt['ann_return']*100:>13.1f}%")
        print(f"{'Sharpe':18}{live['sharpe']:>14.2f}{bt['sharpe']:>14.2f}")
        print(f"{'Max drawdown':18}{live['max_dd']*100:>13.1f}%{bt['max_dd']*100:>13.1f}%")
        if status.get("halted"):
            print(f"  ⚠️ HALTED: {status.get('halt_reason')}")
    print("=" * 60)
    print(f"DECISION  →  {verdict}")
    for r in reasons:
        print(r)
    print("=" * 60)


def maybe_notify(verdict, live, reasons):
    try:
        with open(NOTIFY_STATE) as f:
            last = json.load(f).get("verdict")
    except (OSError, ValueError):
        last = None
    try:
        with open(NOTIFY_STATE, "w") as f:
            json.dump({"verdict": verdict, "updated": dt.datetime.now().isoformat()}, f)
    except OSError:
        pass
    if verdict == last or verdict not in ("PASS", "FAIL"):
        return
    try:
        import alerts
        color = alerts.GREEN if verdict == "PASS" else alerts.RED
        alerts._send(f"📈 Trend forward test → {verdict}",
                     f"(was {last or 'n/a'})\n" + "\n".join(reasons), color)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--notify", action="store_true")
    ap.add_argument("--json", metavar="PATH")
    args = ap.parse_args()

    live_df = load_realized()
    live = realized_stats(live_df)
    try:
        with open(STATUS_FILE) as f:
            status = json.load(f)
    except (OSError, ValueError):
        status = {}

    years = max(live.get("rebalances", 0) / 12, 0.1)
    try:
        bt = backtest_baseline(years)
    except Exception as exc:
        print(f"[warn] backtest baseline failed: {exc}", file=sys.stderr)
        bt = {"ann_return": float("nan"), "vol": float("nan"),
              "sharpe": float("nan"), "max_dd": float("nan")}

    verdict, reasons = evaluate(live, bt, status)
    render(live, bt, status, verdict, reasons)
    if args.notify:
        maybe_notify(verdict, live, reasons)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"generated": dt.datetime.now().isoformat(timespec="seconds"),
                       "verdict": verdict, "live": live, "backtest": bt,
                       "decision_rule": DECISION}, f, indent=2, default=str)
    sys.exit({"PASS": 0, "FAIL": 1, "INSUFFICIENT": 2}[verdict])


if __name__ == "__main__":
    main()
