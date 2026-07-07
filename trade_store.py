"""trade_store.py — single module that owns all trade persistence.

SQLite backend with WAL mode for concurrent reads during bot writes.
Auto-migrates existing trades.csv data on first run.

To swap for Postgres or another backend, change this file only — all
callers use the same public API.
"""
import csv
import datetime as dt
import logging
import os
import sqlite3

import pandas as pd

import config
from utils import now_et

log = logging.getLogger("trade_store")

# Feature columns used by the ML learner (single source of truth).
FEATURE_NUMERIC = ["momentum_pct", "day_change_pct", "rsi", "rel_volume",
                   "vwap_dist_pct", "iv", "spread", "dte", "minutes_since_open"]
FEATURE_COLS = FEATURE_NUMERIC + ["ticker", "type", "strategy"]

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS trades (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_time    TEXT NOT NULL,
    exit_time     TEXT NOT NULL,
    ticker        TEXT NOT NULL,
    option_symbol TEXT NOT NULL,
    type          TEXT NOT NULL,
    strike        REAL NOT NULL,
    expiry        TEXT NOT NULL,
    qty           INTEGER NOT NULL,
    entry_price   REAL NOT NULL,
    exit_price    REAL NOT NULL,
    pnl           REAL NOT NULL,
    pnl_pct       REAL NOT NULL,
    entry_reason  TEXT NOT NULL,
    exit_reason   TEXT NOT NULL,
    strategy      TEXT,
    momentum_pct  REAL,
    day_change_pct REAL,
    rsi           REAL,
    rel_volume    REAL,
    vwap_dist_pct REAL,
    iv            REAL,
    spread        REAL,
    dte           INTEGER,
    minutes_since_open INTEGER,
    win_prob      REAL
)
"""

# Columns for CSV export (preserves compatibility with dashboard download).
_CSV_COLUMNS = [
    "entry_time", "exit_time", "ticker", "option_symbol", "type", "strike",
    "expiry", "qty", "entry_price", "exit_price", "pnl", "pnl_pct",
    "entry_reason", "exit_reason", "strategy", "momentum_pct", "day_change_pct",
    "rsi", "rel_volume", "vwap_dist_pct", "iv", "spread", "dte",
    "minutes_since_open", "win_prob",
]

_conn: sqlite3.Connection | None = None


def _get_conn() -> sqlite3.Connection:
    """Lazy singleton connection with WAL mode enabled."""
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(config.TRADES_DB, check_same_thread=False)
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA foreign_keys=ON")
        _conn.row_factory = sqlite3.Row
    return _conn


def init():
    """Create the trades table if needed and migrate CSV data if present."""
    conn = _get_conn()
    conn.execute(_CREATE_TABLE)
    conn.commit()
    _migrate_csv(conn)


def _migrate_csv(conn: sqlite3.Connection):
    """One-time import of trades.csv into SQLite. Renames CSV after success."""
    if not os.path.exists(config.TRADES_CSV):
        return
    # Skip if DB already has data (migration already happened).
    count = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    if count > 0:
        return
    try:
        with open(config.TRADES_CSV, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return
            rows = list(reader)
        if not rows:
            return
        imported = 0
        for row in rows:
            _insert_row(conn, row)
            imported += 1
        conn.commit()
        # Rename CSV so migration doesn't re-run.
        migrated = config.TRADES_CSV.replace(".csv", "_migrated.csv")
        os.rename(config.TRADES_CSV, migrated)
        log.info("Migrated %d trades from CSV to SQLite (old file: %s)",
                 imported, migrated)
    except Exception:
        conn.rollback()
        log.exception("CSV migration failed — CSV left in place, will retry next start")


def _insert_row(conn: sqlite3.Connection, row: dict):
    """Insert one trade row from a dict (used by migration and append)."""
    def _float(val, default=None):
        if val is None or val == "":
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    def _int(val, default=None):
        if val is None or val == "":
            return default
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return default

    conn.execute(
        """INSERT INTO trades (
            entry_time, exit_time, ticker, option_symbol, type, strike, expiry,
            qty, entry_price, exit_price, pnl, pnl_pct, entry_reason, exit_reason,
            strategy, momentum_pct, day_change_pct, rsi, rel_volume,
            vwap_dist_pct, iv, spread, dte, minutes_since_open, win_prob
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            row.get("entry_time", ""),
            row.get("exit_time", ""),
            row.get("ticker", ""),
            row.get("option_symbol", ""),
            row.get("type", ""),
            _float(row.get("strike"), 0),
            row.get("expiry", ""),
            _int(row.get("qty"), 1),
            _float(row.get("entry_price"), 0),
            _float(row.get("exit_price"), 0),
            _float(row.get("pnl"), 0),
            _float(row.get("pnl_pct"), 0),
            row.get("entry_reason", ""),
            row.get("exit_reason", ""),
            row.get("strategy"),
            _float(row.get("momentum_pct")),
            _float(row.get("day_change_pct")),
            _float(row.get("rsi")),
            _float(row.get("rel_volume")),
            _float(row.get("vwap_dist_pct")),
            _float(row.get("iv")),
            _float(row.get("spread")),
            _int(row.get("dte")),
            _int(row.get("minutes_since_open")),
            _float(row.get("win_prob")),
        ),
    )


# ---- Public API (same signatures as the CSV version) ----


def append_trade(*, entry_time: dt.datetime, underlying: str,
                 option_symbol: str, otype: str, strike: float,
                 expiry: dt.date, qty: int, entry_price: float,
                 exit_price: float, pnl: float, pnl_pct: float,
                 entry_reason: str, exit_reason: str,
                 features: dict, win_prob: float | None):
    """Append one closed trade to the database."""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO trades (
            entry_time, exit_time, ticker, option_symbol, type, strike, expiry,
            qty, entry_price, exit_price, pnl, pnl_pct, entry_reason, exit_reason,
            strategy, momentum_pct, day_change_pct, rsi, rel_volume,
            vwap_dist_pct, iv, spread, dte, minutes_since_open, win_prob
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            entry_time.isoformat(timespec="seconds"),
            now_et().isoformat(timespec="seconds"),
            underlying,
            option_symbol,
            otype,
            strike,
            expiry.isoformat(),
            qty,
            round(entry_price, 2),
            round(exit_price, 2),
            round(pnl, 2),
            round(pnl_pct, 1),
            entry_reason,
            exit_reason,
            features.get("strategy"),
            features.get("momentum_pct"),
            features.get("day_change_pct"),
            features.get("rsi"),
            features.get("rel_volume"),
            features.get("vwap_dist_pct"),
            features.get("iv"),
            features.get("spread"),
            features.get("dte"),
            features.get("minutes_since_open"),
            win_prob,
        ),
    )
    conn.commit()


def today_pnls() -> list[float]:
    """Chronological list of today's trade P&Ls (for risk_manager crash recovery)."""
    conn = _get_conn()
    today = now_et().date().isoformat()
    rows = conn.execute(
        "SELECT pnl FROM trades WHERE exit_time >= ? ORDER BY id",
        (today,),
    ).fetchall()
    return [float(r["pnl"]) for r in rows]


def training_data() -> tuple[pd.DataFrame, pd.Series] | None:
    """Feature matrix + win labels for ML training. None if insufficient data."""
    conn = _get_conn()
    query = f"SELECT {', '.join(FEATURE_COLS)}, pnl FROM trades WHERE momentum_pct IS NOT NULL AND rsi IS NOT NULL AND pnl IS NOT NULL"
    df = pd.read_sql_query(query, conn)
    if df.empty:
        return None
    X = df[FEATURE_COLS].copy()
    X[FEATURE_NUMERIC] = X[FEATURE_NUMERIC].apply(pd.to_numeric, errors="coerce")
    y = (pd.to_numeric(df["pnl"], errors="coerce") > 0).astype(int)
    return X, y


def today_trades_csv() -> tuple[pd.DataFrame, float]:
    """Today's closed trades as a DataFrame + total P&L (for AI analyst)."""
    conn = _get_conn()
    today = now_et().date().isoformat()
    df = pd.read_sql_query(
        f"SELECT * FROM trades WHERE exit_time >= ?",
        conn, params=(today,),
    )
    if df.empty:
        return df, 0.0
    pnl = pd.to_numeric(df["pnl"], errors="coerce").sum()
    return df, float(pnl)


def all_trades() -> pd.DataFrame:
    """All closed trades as a DataFrame (for dashboard history tab)."""
    conn = _get_conn()
    df = pd.read_sql_query("SELECT * FROM trades ORDER BY id", conn)
    if df.empty:
        return df
    df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce")
    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce")
    return df.dropna(subset=["exit_time", "pnl"])


def trade_count() -> int:
    """Total number of closed trades in the database."""
    conn = _get_conn()
    return conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
