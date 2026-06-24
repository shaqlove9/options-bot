"""store.py — single SQLite store (signals, features, trades, rejects, shadow).

Append-only by intent: feature rows and trades are inserted once and only updated
to attach a post-hoc label. One file keeps the dashboard, gate, and meta layer in
sync without a server.
"""
from __future__ import annotations

import sqlite3

from iobot import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    signal_id   TEXT PRIMARY KEY,
    captured_at TEXT,
    symbol      TEXT,
    direction   TEXT,
    source      TEXT,
    spot        REAL
);
CREATE TABLE IF NOT EXISTS features (
    signal_id      TEXT PRIMARY KEY,
    captured_at_ts REAL,   -- epoch seconds when the signal fired
    max_asof_ts    REAL,   -- newest input timestamp used (must be <= captured_at_ts)
    features_json  TEXT
);
CREATE TABLE IF NOT EXISTS trades (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id     TEXT,
    structure     TEXT,    -- "single" | "spread"
    symbol        TEXT,
    direction     TEXT,
    entry_time    TEXT,
    exit_time     TEXT,
    entry_underlying REAL,
    exit_underlying  REAL,
    contract      TEXT,
    strike        REAL,
    short_strike  REAL,
    expiry        TEXT,
    qty           INTEGER,
    entry_fill    REAL,
    exit_fill     REAL,
    entry_mid     REAL,
    max_loss      REAL,
    pnl           REAL,
    r_multiple    REAL,
    entry_slippage REAL,
    exit_reason   TEXT,
    label         INTEGER  -- 1 if target hit before stop, 0 otherwise; NULL until set
);
CREATE TABLE IF NOT EXISTS rejects (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    time      TEXT,
    symbol    TEXT,
    direction TEXT,
    stage     TEXT,
    reason    TEXT
);
CREATE TABLE IF NOT EXISTS shadow (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    time         TEXT,
    signal_id    TEXT,
    proba        REAL,
    would_skip   INTEGER,
    size_mult    REAL,
    actual_action TEXT
);
-- Open positions, persisted so an overnight carry survives a restart (the executor
-- rehydrates these and reconciles them against the live broker positions). A row is
-- written on open and deleted on close.
CREATE TABLE IF NOT EXISTS positions (
    key        TEXT PRIMARY KEY,   -- option_symbol (single) / long_symbol (spread)
    structure  TEXT,               -- "single" | "spread"
    opened_at  TEXT,
    expiry     TEXT,
    data_json  TEXT                -- full serialized position
);
"""


def connect(db_file: str | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_file or config.DB_FILE)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn
