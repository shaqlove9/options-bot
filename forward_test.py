"""forward_test.py — live-vs-backtest tracking harness for the EQUITY sleeve.

Why this exists
---------------
The equity (shares) backtest looks profitable and cost-robust *in sample*. The
only thing that turns that into a real edge is an out-of-sample forward run in
paper. This script is the scoreboard for that run: it reads the live paper
trades the equity bot logs (config.TRADES_CSV, normally trades_equity.csv),
re-runs the SAME backtest cost model (backtest.simulate_equity) over the SAME
calendar window, and prints them side by side so live results are directly
comparable to the backtest — exactly what the strategy review flagged as the
gate before trusting the numbers.

It is READ-ONLY and PAPER-ONLY: it never submits an order, never touches the
running bot, never edits state files. Safe to run on the prod VM at any time.

Comparability notes (read once)
-------------------------------
  - The backtest sizes each position at EQ_CAPITAL*EQ_LEVERAGE/MAX_OPEN_POSITIONS
    (~$1,333) while the live bot sizes at config.EQ_NOTIONAL_PER_TRADE ($4,000).
    Raw-dollar P&L is therefore NOT apples-to-apples. So every headline metric
    here is rate-based or normalized to % of notional (pnl_pct), which IS
    comparable. Dollar totals are shown for context only and labelled as such.
  - Sharpe is computed identically on both sides: daily summed pnl_pct treated
    as a return on equal capital units, annualized by sqrt(252). This is
    notional-independent, so live and backtest Sharpe are directly comparable
    (it differs slightly from backtest.report_equity's $-based Sharpe — that's
    fine; what matters is the two sides here use one method).
  - Backtest fills already include the modeled friction (penny spread +
    slippage). Live fills are whatever the paper broker actually gave. If live
    underperforms the backtest, friction/latency is the usual culprit.

Usage
-----
  python forward_test.py                 # live stats + backtest over same window
  python forward_test.py --days 30       # cap the backtest window to 30 days
  python forward_test.py --no-backtest   # live stats only (no Alpaca call)
  python forward_test.py --csv path.csv  # a different trade log
  python forward_test.py --json out.json # also dump the comparison as JSON

The exit status is 0 if the live run PASSES the decision rule, 1 if it FAILS,
2 if there is not yet enough data to decide — so you can wire it into a cron
check if you want.
"""
import argparse
import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd

import config

# This harness is for the EQUITY sleeve specifically, so it points at the equity
# trade log regardless of the INSTRUMENT env var (config.TRADES_CSV falls back to
# the options log when run outside the equity environment).
EQUITY_TRADES_CSV = os.path.join(os.path.dirname(os.path.abspath(config.__file__)),
                                 "trades_equity.csv")

# ---------------------------------------------------------------------------
# PRE-COMMITTED decision rule. Set these BEFORE you read the results so you do
# not rationalize a losing run later. The harness is just enforcing the promise
# you made to yourself. Tune to taste, but tune them now, not after.
# ---------------------------------------------------------------------------
DECISION = {
    "min_trades": 40,          # don't judge on noise; need a real sample first
    "min_avg_pnl_pct": 0.0,    # live avg trade must be net positive (% of notional)
    "min_win_rate": 0.0,       # set >0 if your edge is win-rate driven
    "min_sharpe": 0.5,         # live annualized Sharpe floor
    # live may underperform the backtest, but not by more than this much of it:
    "max_avg_pnl_pct_shortfall_frac": 0.50,   # live avg ≥ 50% of backtest avg
    "max_win_rate_shortfall_pts": 10.0,       # live win% ≥ backtest win% − 10pts
}

# Common capital unit for the notional-independent Sharpe (value is irrelevant;
# it cancels — we only need both sides to use the same one).
_SHARPE_UNIT = 1.0


def _backtest_notional() -> float:
    """Per-position notional the equity backtest sizes at, for the $ caveat note."""
    import backtest as bt
    return bt.EQ_CAPITAL * bt.EQ_LEVERAGE / config.MAX_OPEN_POSITIONS


# ---------------------------------------------------------------------------
# loading / normalization
# ---------------------------------------------------------------------------

def _norm_side(v: str) -> str:
    """Map either log's direction vocabulary to long/short."""
    v = str(v).strip().lower()
    if v in ("call", "long", "buy"):
        return "long"
    if v in ("put", "short", "sell"):
        return "short"
    return v


def load_live_trades(path: str) -> pd.DataFrame:
    """Read the live equity trade log into a normalized frame. Returns an empty
    frame (with the right columns) if the log has only a header / is missing."""
    cols = ["entry_time", "exit_time", "side", "pnl", "pnl_pct",
            "exit_reason", "strategy", "ticker"]
    try:
        df = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=cols)
    if df.empty:
        return pd.DataFrame(columns=cols)
    if "pnl_pct" not in df.columns or "side" not in df.columns:
        raise ValueError(
            f"{path} is not an equity trade log (missing expected columns). "
            f"Point --csv at trades_equity.csv.")
    for c in ("entry_time", "exit_time"):
        df[c] = pd.to_datetime(df[c], errors="coerce", utc=False)
    for c in ("pnl", "pnl_pct"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df["side"] = df.get("side", "").map(_norm_side)
    df["strategy"] = df.get("strategy", "").fillna("").replace("", "scalp")
    return df.dropna(subset=["exit_time", "pnl", "pnl_pct"]).reset_index(drop=True)


def backtest_trades_df(days: int, style: str) -> pd.DataFrame:
    """Run the equity backtest over `days` and return a frame with the same
    columns as the live frame. Imports backtest lazily so --no-backtest never
    needs Alpaca creds/network."""
    import backtest as bt
    from alpaca.data.historical import StockHistoricalDataClient

    client = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
    all_bars = {}
    for sym in config.UNIVERSE:
        if sym not in bt.ASSUMED_IV:
            continue
        bars = bt.fetch_bars(client, sym, days)
        if not bars.empty:
            all_bars[sym] = bars
    trades = bt.simulate_equity(all_bars, style=style)
    if not trades:
        return pd.DataFrame(columns=["entry_time", "exit_time", "side", "pnl",
                                     "pnl_pct", "exit_reason", "strategy", "ticker"])
    df = pd.DataFrame([vars(t) for t in trades])
    df = df.rename(columns={"direction": "side", "symbol": "ticker"})
    df["side"] = df["side"].map(_norm_side)
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    return df


def clip_to_window(bt_df: pd.DataFrame, live_df: pd.DataFrame) -> pd.DataFrame:
    """Restrict the backtest to the calendar span actually covered live, so the
    comparison is over the same days (once there are live trades to bound it)."""
    if live_df.empty or bt_df.empty:
        return bt_df
    lo = live_df["entry_time"].min().date()
    hi = live_df["exit_time"].max().date()
    d = bt_df["exit_time"].dt.date
    return bt_df[(d >= lo) & (d <= hi)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# stats — identical computation for both sides
# ---------------------------------------------------------------------------

def compute_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n": 0}
    wins = df[df.pnl > 0]
    losses = df[df.pnl <= 0]
    gross_win = wins.pnl.sum()
    gross_loss = abs(losses.pnl.sum())

    # Notional-independent daily return = sum of per-trade % of notional / 100.
    daily_ret = (df.groupby(df.exit_time.dt.date).pnl_pct.sum() / 100.0)
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
              if len(daily_ret) > 1 and daily_ret.std() > 0 else float("nan"))
    equity = _SHARPE_UNIT + daily_ret.cumsum()
    dd = (equity - equity.cummax())
    max_dd_pct = dd.min() * 100  # already a fraction of the unit -> percent

    per_strat = {}
    for s, g in df.groupby("strategy"):
        per_strat[s] = {"n": len(g), "win_rate": (g.pnl > 0).mean() * 100,
                        "avg_pnl_pct": g.pnl_pct.mean(), "pnl": g.pnl.sum()}

    return {
        "n": len(df),
        "period": f"{df.entry_time.min().date()} → {df.exit_time.max().date()}",
        "win_rate": len(wins) / len(df) * 100,
        "wins": len(wins), "losses": len(losses),
        "avg_pnl_pct": df.pnl_pct.mean(),
        "avg_win_pct": wins.pnl_pct.mean() if len(wins) else float("nan"),
        "avg_loss_pct": losses.pnl_pct.mean() if len(losses) else float("nan"),
        "total_pnl_usd": df.pnl.sum(),
        "avg_pnl_usd": df.pnl.mean(),
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "sharpe": sharpe,
        "max_dd_pct": max_dd_pct,
        "exits": df.exit_reason.value_counts().to_dict(),
        "per_strategy": per_strat,
    }


# ---------------------------------------------------------------------------
# decision rule
# ---------------------------------------------------------------------------

def evaluate(live: dict, bt: dict | None) -> tuple[str, list[str]]:
    """Return ('PASS'|'FAIL'|'INSUFFICIENT', [reasons...])."""
    if live.get("n", 0) < DECISION["min_trades"]:
        return ("INSUFFICIENT",
                [f"only {live.get('n', 0)} live trades; need "
                 f"{DECISION['min_trades']} before deciding"])

    reasons, ok = [], True

    def check(cond, msg):
        nonlocal ok
        mark = "✓" if cond else "✗"
        if not cond:
            ok = False
        reasons.append(f"  {mark} {msg}")

    check(live["avg_pnl_pct"] >= DECISION["min_avg_pnl_pct"],
          f"avg trade {live['avg_pnl_pct']:+.3f}% ≥ floor "
          f"{DECISION['min_avg_pnl_pct']:+.3f}%")
    check(live["win_rate"] >= DECISION["min_win_rate"],
          f"win rate {live['win_rate']:.1f}% ≥ floor {DECISION['min_win_rate']:.1f}%")
    check(not np.isnan(live["sharpe"]) and live["sharpe"] >= DECISION["min_sharpe"],
          f"Sharpe {live['sharpe']:.2f} ≥ floor {DECISION['min_sharpe']:.2f}")

    if bt and bt.get("n", 0) > 0:
        floor = bt["avg_pnl_pct"] * DECISION["max_avg_pnl_pct_shortfall_frac"]
        # if backtest avg is positive, live must reach a fraction of it;
        # if backtest avg is negative, this guard is vacuous (skip).
        if bt["avg_pnl_pct"] > 0:
            check(live["avg_pnl_pct"] >= floor,
                  f"avg trade {live['avg_pnl_pct']:+.3f}% ≥ {floor:+.3f}% "
                  f"({DECISION['max_avg_pnl_pct_shortfall_frac']:.0%} of backtest "
                  f"{bt['avg_pnl_pct']:+.3f}%)")
        wr_floor = bt["win_rate"] - DECISION["max_win_rate_shortfall_pts"]
        check(live["win_rate"] >= wr_floor,
              f"win rate {live['win_rate']:.1f}% ≥ {wr_floor:.1f}% "
              f"(backtest {bt['win_rate']:.1f}% − "
              f"{DECISION['max_win_rate_shortfall_pts']:.0f}pts)")

    return ("PASS" if ok else "FAIL"), reasons


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def _fmt(v, spec="{:.2f}"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if v == float("inf"):
        return "∞"
    return spec.format(v)


def _row(label, lv, bv):
    return f"{label:<22}{lv:>16}{bv:>16}"


def render(live: dict, bt: dict | None, verdict: str, reasons: list[str],
           csv_path: str = EQUITY_TRADES_CSV):
    print("=" * 56)
    print("EQUITY SLEEVE — LIVE (paper) vs BACKTEST")
    print("=" * 56)
    if live.get("n", 0) == 0:
        print("No live trades logged yet in", csv_path)
        print("(The equity paper bot hasn't closed any positions.)")
    has_bt = bool(bt and bt.get("n", 0) > 0)
    has_live = bool(live.get("n", 0))
    if has_live:
        print(f"Live window:     {live['period']}")
    if has_bt:
        span = "  (clipped to the live span)" if has_live else "  (no live trades yet — baseline only)"
        print(f"Backtest window: {bt['period']}{span}")
    print(_row("", "LIVE" if has_live else "—", "BACKTEST" if has_bt else "—"))
    print("-" * 54)
    if has_live or has_bt:
        # Pull a metric from whichever side has data; "—" for an empty side, so
        # the backtest baseline is visible even before the first live trade closes.
        def cell(d, key, spec="{:.2f}"):
            return _fmt(d.get(key), spec) if d.get("n", 0) else "—"

        lv, b = (live if has_live else {}), (bt if has_bt else {})
        print(_row("Trades", lv.get("n", "—"), b.get("n", "—")))
        print(_row("Win rate %", cell(lv, "win_rate", "{:.1f}"), cell(b, "win_rate", "{:.1f}")))
        print(_row("Avg trade % notional", cell(lv, "avg_pnl_pct", "{:+.3f}"), cell(b, "avg_pnl_pct", "{:+.3f}")))
        print(_row("Avg win %", cell(lv, "avg_win_pct", "{:+.3f}"), cell(b, "avg_win_pct", "{:+.3f}")))
        print(_row("Avg loss %", cell(lv, "avg_loss_pct", "{:+.3f}"), cell(b, "avg_loss_pct", "{:+.3f}")))
        print(_row("Profit factor", cell(lv, "profit_factor"), cell(b, "profit_factor")))
        print(_row("Sharpe (ann.)", cell(lv, "sharpe"), cell(b, "sharpe")))
        print(_row("Max drawdown %", cell(lv, "max_dd_pct"), cell(b, "max_dd_pct")))
        print(_row("Total P&L $ (ctx*)", cell(lv, "total_pnl_usd", "{:+.2f}"), cell(b, "total_pnl_usd", "{:+.2f}")))
        print("-" * 54)
        if has_bt:
            print("* $ totals use different position sizing live ($%.0f/pos) vs the"
                  % config.EQ_NOTIONAL_PER_TRADE)
            print("  backtest (~$%.0f/pos) — compare the %% rows, not the $ row."
                  % _backtest_notional())
        else:
            print("* compare the % rows, not the $ row (sizing differs).")
        if live.get("exits"):
            print("Live exits:    ", live["exits"])
        if live.get("per_strategy"):
            print("Live by strategy:")
            for s, g in live["per_strategy"].items():
                print(f"  {s:8s} {g['n']:>3} trades  {g['win_rate']:5.1f}% win  "
                      f"{g['avg_pnl_pct']:+.3f}% avg")
    print("=" * 56)
    print(f"DECISION RULE  →  {verdict}")
    for r in reasons:
        print(r)
    if verdict == "INSUFFICIENT":
        print("  (keep the paper run going; re-run this when more trades close)")
    print("=" * 56)


# ---------------------------------------------------------------------------
# optional Discord notification (for the daily cron check)
# ---------------------------------------------------------------------------

# Remembers the last verdict so a daily cron alerts only when the verdict
# *changes* into a decisive state — silent through weeks of INSUFFICIENT, one
# ping the day it becomes PASS/FAIL, and again on any later PASS↔FAIL flip.
NOTIFY_STATE = os.path.join(os.path.dirname(EQUITY_TRADES_CSV),
                            "forward_test_state.json")


def maybe_notify(verdict: str, live: dict, bt: dict | None, reasons: list[str]):
    """Send a Discord alert iff the verdict changed and is now decisive."""
    try:
        with open(NOTIFY_STATE) as f:
            last = json.load(f).get("verdict")
    except (OSError, ValueError):
        last = None

    changed = verdict != last
    # persist current verdict regardless, so the next run compares correctly
    try:
        with open(NOTIFY_STATE, "w") as f:
            json.dump({"verdict": verdict,
                       "updated": dt.datetime.now().isoformat(timespec="seconds")}, f)
    except OSError as exc:
        print(f"[warn] could not write {NOTIFY_STATE}: {exc}", file=sys.stderr)

    # Only ping on a change INTO a decisive verdict (skip None→INSUFFICIENT etc.)
    if not (changed and verdict in ("PASS", "FAIL")):
        return

    import alerts
    color = {"PASS": alerts.GREEN, "FAIL": alerts.RED}[verdict]
    btline = (f"\nBacktest baseline: {bt['avg_pnl_pct']:+.3f}%/trade, "
              f"{bt['win_rate']:.1f}% win, Sharpe {bt['sharpe']:.2f}"
              if bt and bt.get("n") else "")
    desc = (f"**{verdict}** — equity sleeve forward test "
            f"(was {last or 'n/a'})\n"
            f"Live: {live.get('n', 0)} trades, "
            f"{live.get('avg_pnl_pct', float('nan')):+.3f}%/trade, "
            f"{live.get('win_rate', float('nan')):.1f}% win, "
            f"Sharpe {live.get('sharpe', float('nan')):.2f}{btline}\n\n"
            + "\n".join(reasons))
    alerts._send(f"📊 Forward test → {verdict}", desc, color)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=60,
                    help="max backtest window in days (clipped to the live span)")
    ap.add_argument("--style", choices=["momentum", "reversion"], default="momentum",
                    help="signal style; must match how the live sleeve is running")
    ap.add_argument("--csv", default=EQUITY_TRADES_CSV, help="live trade log path")
    ap.add_argument("--no-backtest", action="store_true",
                    help="live stats only; skip the Alpaca backtest")
    ap.add_argument("--json", metavar="PATH", help="also write the comparison as JSON")
    ap.add_argument("--notify", action="store_true",
                    help="Discord-alert when the verdict changes into PASS/FAIL "
                         "(for the daily cron check)")
    args = ap.parse_args()

    live_df = load_live_trades(args.csv)
    live_stats = compute_stats(live_df)

    bt_stats = None
    if not args.no_backtest:
        try:
            bt_df = clip_to_window(backtest_trades_df(args.days, args.style), live_df)
            bt_stats = compute_stats(bt_df)
        except Exception as exc:                       # creds/network/etc.
            print(f"[warn] backtest skipped: {exc}\n", file=sys.stderr)

    verdict, reasons = evaluate(live_stats, bt_stats)
    render(live_stats, bt_stats, verdict, reasons, csv_path=args.csv)

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"generated": dt.datetime.now().isoformat(timespec="seconds"),
                       "verdict": verdict, "live": live_stats, "backtest": bt_stats,
                       "decision_rule": DECISION}, f, indent=2, default=str)
        print(f"\nWrote {args.json}")

    if args.notify:
        maybe_notify(verdict, live_stats, bt_stats, reasons)

    sys.exit({"PASS": 0, "FAIL": 1, "INSUFFICIENT": 2}[verdict])


if __name__ == "__main__":
    main()
