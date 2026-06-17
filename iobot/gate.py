"""gate.py — paper-first validation gate + live-vs-modeled scoring.

The sleeve stays paper until: at least GATE_MIN_TRADES closed trades AND a positive
expected R *net of modeled bid/ask + fees*, with the recent half also non-negative
(a guard against a verdict resting on stale early luck). Until then the verdict is
INSUFFICIENT and there is no path to live — `broker.py` enforces paper regardless.
"""
from __future__ import annotations

import logging

from iobot import config, journal

log = logging.getLogger("gate")


def modeled_friction_r(max_loss: float, structure: str) -> float:
    """Conservative round-trip friction (fees + half-spread slip) as an R fraction.
    Charged on top of realized fills so the gate can't be gamed by optimistic
    paper fills. 1 leg for single, 2 for a vertical; open + close = x2."""
    legs = 2 if structure == "spread" else 1
    per_leg = config.MODELED_FEE_PER_CONTRACT + config.MODELED_SLIP_PER_LEG * 100
    cost = per_leg * legs * config.QTY * 2
    return cost / max_loss if max_loss and max_loss > 0 else 0.0


def evaluate(conn) -> dict:
    df = journal.closed_trades_df(conn)
    df = df[df["pnl"].notna()] if not df.empty else df
    n = len(df)
    base = {
        "n_trades": n, "min_trades": config.GATE_MIN_TRADES,
        "win_rate": None, "gross_expected_r": None, "net_expected_r": None,
        "recent_half_net_r": None, "verdict": "INSUFFICIENT", "passes": False,
    }
    if n == 0:
        base["detail"] = "no closed trades yet"
        return base

    net_r = df.apply(
        lambda row: float(row["r_multiple"] or 0.0)
        - modeled_friction_r(float(row["max_loss"] or 0.0), str(row["structure"])),
        axis=1)
    win_rate = float((df["pnl"] > 0).mean())
    gross = float(df["r_multiple"].astype(float).mean())
    net = float(net_r.mean())
    recent = float(net_r.iloc[n // 2:].mean())

    base.update(win_rate=win_rate, gross_expected_r=gross, net_expected_r=net,
                recent_half_net_r=recent)
    if n < config.GATE_MIN_TRADES:
        base["detail"] = f"{n}/{config.GATE_MIN_TRADES} trades — accruing"
        return base

    passes = net > 0 and recent >= 0
    base["verdict"] = "KEEP" if passes else "KILL"
    base["passes"] = passes
    base["detail"] = (f"net E[R]={net:+.3f}, recent half={recent:+.3f} → "
                      f"{base['verdict']}")
    log.info("GATE: %s", base["detail"])
    return base
