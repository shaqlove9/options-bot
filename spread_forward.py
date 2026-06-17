"""spread_forward.py — paper validation scoreboard + keep/kill gate for the spread sleeve.

Reads the closed spread trades and reports realized stats in R-multiples (R = P&L / max
loss, where max loss = the net debit). The gate is pre-committed and strict, mirroring the
equity/trend sleeves:

    KEEP only if  >= SPREAD_GATE_MIN_TRADES (40) closed trades
              AND expected R per trade, NET of modeled bid/ask + fees, is positive
              AND the more recent half is also positive (not a stale early edge).

Otherwise INSUFFICIENT (< gate) or FAIL. No edge is assumed — the data decides.

Run:  python spread_forward.py [--notify] [--json PATH]
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd

import config


def load_trades() -> pd.DataFrame:
    import os
    if not os.path.exists(config.SPREAD_TRADES_CSV):
        return pd.DataFrame()
    df = pd.read_csv(config.SPREAD_TRADES_CSV)
    if df.empty:
        return df
    for c in ("pnl", "max_loss", "r_multiple", "entry_debit"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce")
    return df.dropna(subset=["pnl", "max_loss"]).sort_values("exit_time").reset_index(drop=True)


def r_net(df: pd.DataFrame) -> np.ndarray:
    """R-multiple net of a modeled per-trade fee (2 legs in + 2 out)."""
    fee = 4 * config.SPREAD_FEE_PER_CONTRACT          # dollars, round-trip both legs
    pnl_net = df["pnl"].to_numpy(float) - fee
    ml = df["max_loss"].to_numpy(float)
    return np.where(ml > 0, pnl_net / ml, 0.0)


def evaluate(df: pd.DataFrame) -> tuple[str, dict, list[str]]:
    n = len(df)
    if n == 0:
        return "INSUFFICIENT", {"n": 0}, [f"0 closed trades; need {config.SPREAD_GATE_MIN_TRADES}"]
    r = r_net(df)
    exp_R = float(r.mean())
    win_rate = float((df["pnl"] > 0).mean() * 100)
    gross_win = float(df.loc[df["pnl"] > 0, "pnl"].sum())
    gross_loss = float(-df.loc[df["pnl"] < 0, "pnl"].sum())
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    half = n // 2
    recent_R = float(r[half:].mean()) if n >= 2 else exp_R
    m = {"n": n, "expected_R_net": round(exp_R, 4), "recent_half_R": round(recent_R, 4),
         "win_rate": round(win_rate, 1), "profit_factor": round(pf, 2),
         "total_pnl": round(float(df["pnl"].sum()), 2)}

    if n < config.SPREAD_GATE_MIN_TRADES:
        return "INSUFFICIENT", m, [f"{n} closed trades; need {config.SPREAD_GATE_MIN_TRADES}"]
    reasons = []
    if exp_R <= 0:
        reasons.append(f"expected R net {exp_R:+.3f} not positive")
    if recent_R <= 0:
        reasons.append(f"recent-half R {recent_R:+.3f} not positive")
    return ("PASS" if not reasons else "FAIL"), m, reasons or ["meets all gate criteria"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--notify", action="store_true")
    ap.add_argument("--json", metavar="PATH")
    args = ap.parse_args()

    df = load_trades()
    verdict, m, reasons = evaluate(df)
    print("=" * 60)
    print("SPREAD SLEEVE — paper validation scoreboard")
    print("=" * 60)
    if m.get("n"):
        print(f"Closed trades:    {m['n']}")
        if "expected_R_net" in m:
            print(f"Expected R (net): {m['expected_R_net']:+.3f}  "
                  f"(recent half {m['recent_half_R']:+.3f})")
            print(f"Win rate:         {m['win_rate']:.0f}%   profit factor {m['profit_factor']}")
            print(f"Total P&L:        ${m['total_pnl']:+,.2f}")
    else:
        print("No closed spread trades yet.")
    print("-" * 60)
    print(f"DECISION → {verdict}")
    for r in reasons:
        print(f"  - {r}")
    print("=" * 60)

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"verdict": verdict, "metrics": m, "reasons": reasons}, f,
                      indent=2, default=str)
    if args.notify:
        try:
            import alerts
            alerts.error(f"Spread sleeve verdict: {verdict} ({m.get('n', 0)} trades) — "
                         f"{'; '.join(reasons)}")
        except Exception:
            pass
    return {"PASS": 0, "FAIL": 1, "INSUFFICIENT": 2}[verdict]


if __name__ == "__main__":
    sys.exit(main())
