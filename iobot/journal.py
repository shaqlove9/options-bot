"""journal.py — trade log, rejected-signal log, shadow log, and read queries.

The label (1 if the trade hit its target before its stop, else 0) is the ONLY
outcome attached to a signal; it is written here at close time and joined back to
the Phase-1 feature row by signal_id for training. Features never see it.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime

import pandas as pd

from iobot.clock import now_et

log = logging.getLogger("journal")


def log_reject(conn, symbol: str, direction: str, stage: str, reason: str):
    """Record a signal we declined to trade, with the failing reason."""
    conn.execute(
        "INSERT INTO rejects(time, symbol, direction, stage, reason) VALUES (?,?,?,?,?)",
        (now_et().isoformat(timespec="seconds"), symbol, direction, stage, reason))
    conn.commit()
    log.info("REJECT %s %s [%s] — %s", symbol, direction, stage, reason)


def label_from_reason(exit_reason: str) -> int:
    """1 only when the profit target was reached before the stop."""
    return 1 if exit_reason.lower().startswith("profit target") else 0


def log_trade(conn, trade: dict) -> int:
    """Insert a closed trade. `trade` carries entry/exit context; label is derived
    from exit_reason. Returns the new row id."""
    trade = dict(trade)
    trade["label"] = label_from_reason(trade.get("exit_reason", ""))
    cols = ["signal_id", "structure", "symbol", "direction", "entry_time", "exit_time",
            "entry_underlying", "exit_underlying", "contract", "strike", "short_strike",
            "expiry", "qty", "entry_fill", "exit_fill", "entry_mid", "max_loss", "pnl",
            "r_multiple", "entry_slippage", "exit_reason", "label"]
    placeholders = ",".join("?" for _ in cols)
    cur = conn.execute(
        f"INSERT INTO trades({','.join(cols)}) VALUES ({placeholders})",
        [trade.get(c) for c in cols])
    conn.commit()
    log.info("LOGGED trade #%d %s %s pnl $%+.2f R %.2f label %d (%s)",
             cur.lastrowid, trade.get("symbol"), trade.get("direction"),
             trade.get("pnl", 0.0), trade.get("r_multiple", 0.0), trade["label"],
             trade.get("exit_reason"))
    return cur.lastrowid


def log_shadow(conn, signal_id: str, proba: float | None, would_skip: bool,
               size_mult: float, actual_action: str):
    conn.execute(
        "INSERT INTO shadow(time, signal_id, proba, would_skip, size_mult, "
        "actual_action) VALUES (?,?,?,?,?,?)",
        (now_et().isoformat(timespec="seconds"), signal_id, proba,
         1 if would_skip else 0, size_mult, actual_action))
    conn.commit()


def recent_streak(conn) -> tuple[int, int]:
    """(win_streak, loss_streak) — consecutive wins/losses at the tail of history."""
    rows = conn.execute(
        "SELECT pnl FROM trades WHERE pnl IS NOT NULL ORDER BY exit_time DESC LIMIT 50"
    ).fetchall()
    wins = losses = 0
    for r in rows:
        if r["pnl"] > 0 and losses == 0:
            wins += 1
        elif r["pnl"] <= 0 and wins == 0:
            losses += 1
        else:
            break
    return wins, losses


def minutes_since_last_exit(conn, symbol: str, direction: str) -> float | None:
    """Minutes since the most recently CLOSED trade in this symbol+direction, or None
    if there has never been one. Used by the post-close re-entry cooldown."""
    row = conn.execute(
        "SELECT exit_time FROM trades WHERE symbol=? AND direction=? "
        "AND exit_time IS NOT NULL ORDER BY exit_time DESC LIMIT 1",
        (symbol, direction)).fetchone()
    if not row or not row["exit_time"]:
        return None
    try:
        last = datetime.fromisoformat(row["exit_time"])
    except ValueError:
        return None
    return (now_et() - last).total_seconds() / 60.0


def realized_pnl_total(conn) -> float:
    """Cumulative realized P&L of every closed trade — the sleeve's running result.
    Added to SLEEVE_CAPITAL to get sleeve equity (the governor's risk base)."""
    row = conn.execute(
        "SELECT COALESCE(SUM(pnl), 0.0) AS s FROM trades WHERE pnl IS NOT NULL"
    ).fetchone()
    return float(row["s"] or 0.0)


def closed_trades_df(conn) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT * FROM trades ORDER BY exit_time", conn)


def labeled_training_data(conn) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join captured features to labeled trades (time-ordered by entry).
    Returns (X, info): feature matrix and an aligned info frame with columns
    [label, r, max_loss, structure] for net-R scoring."""
    rows = conn.execute(
        "SELECT t.entry_time AS entry_time, t.label AS label, t.r_multiple AS r, "
        "t.max_loss AS max_loss, t.structure AS structure, f.features_json AS fj "
        "FROM trades t JOIN features f ON f.signal_id = t.signal_id "
        "WHERE t.label IS NOT NULL AND f.features_json IS NOT NULL "
        "ORDER BY t.entry_time").fetchall()
    if not rows:
        return pd.DataFrame(), pd.DataFrame(columns=["label", "r", "max_loss", "structure"])
    X = pd.DataFrame([json.loads(r["fj"]) for r in rows]).fillna(0.0)
    info = pd.DataFrame({
        "label": [int(r["label"]) for r in rows],
        "r": [float(r["r"] or 0.0) for r in rows],
        "max_loss": [float(r["max_loss"] or 0.0) for r in rows],
        "structure": [str(r["structure"]) for r in rows],
    })
    return X, info
