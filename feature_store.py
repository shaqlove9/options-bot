"""feature_store.py — append-only SQLite store for the equity meta-labeling layer.

One row per *signal* (the moment the primary momentum signal fired, BEFORE entry),
keyed by a `signal_id` that is later carried onto the closed-trade log so features
can be joined to outcomes. A second table records the model/governor SHADOW
decision for that signal (what it WOULD have done) without changing live behaviour.

Design:
  - Append-only: we never UPDATE or DELETE rows here; outcomes live in the trade
    CSV and are joined by signal_id at training time.
  - Features are stored as a JSON blob so the schema can evolve without migrations;
    `read_signals()` explodes them back into columns for training/dashboard.
  - Short-lived connections + WAL so the bot (writer) and the dashboard / trainer
    (readers) never block each other.
  - NEVER store post-entry information here — callers pass only the pre-entry
    snapshot from equity_features.snapshot().
"""
import datetime as dt
import json
import logging
import os
import sqlite3

import pandas as pd

import config

log = logging.getLogger("feature_store")


def _connect(path: str | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(path or config.META_DB, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init(path: str | None = None) -> None:
    """Create tables if absent. Idempotent — safe to call on every startup."""
    with _connect(path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                signal_id  TEXT PRIMARY KEY,
                ts         TEXT NOT NULL,
                symbol     TEXT,
                direction  TEXT,
                strategy   TEXT,
                features   TEXT NOT NULL
            )""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS shadow (
                signal_id  TEXT PRIMARY KEY,
                ts         TEXT NOT NULL,
                would_take INTEGER,
                win_prob   REAL,
                size_mult  REAL,
                gov_size   REAL,
                note       TEXT
            )""")
        conn.commit()


def record_signal(signal_id: str, ts: dt.datetime, symbol: str, direction: str,
                  strategy: str, features: dict, path: str | None = None) -> None:
    """Persist the pre-entry feature snapshot. INSERT OR IGNORE keeps it append-only
    and idempotent if the same signal_id is ever offered twice."""
    try:
        with _connect(path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO signals "
                "(signal_id, ts, symbol, direction, strategy, features) "
                "VALUES (?,?,?,?,?,?)",
                (signal_id, ts.isoformat(), symbol, direction, strategy,
                 json.dumps(features, default=str)))
            conn.commit()
    except sqlite3.Error:
        log.exception("record_signal failed for %s (continuing — never block a trade)",
                      signal_id)


def record_shadow(signal_id: str, ts: dt.datetime, decision: dict,
                  path: str | None = None) -> None:
    """Persist what the model + governor WOULD have done for this signal."""
    try:
        with _connect(path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO shadow "
                "(signal_id, ts, would_take, win_prob, size_mult, gov_size, note) "
                "VALUES (?,?,?,?,?,?,?)",
                (signal_id, ts.isoformat(),
                 int(decision.get("would_take", 1)),
                 decision.get("win_prob"),
                 decision.get("size_mult"),
                 decision.get("gov_size"),
                 decision.get("note", "")))
            conn.commit()
    except sqlite3.Error:
        log.exception("record_shadow failed for %s (continuing)", signal_id)


def read_signals(path: str | None = None) -> pd.DataFrame:
    """All captured signals with the JSON features exploded into columns.
    Returns an empty frame if the store doesn't exist yet."""
    db = path or config.META_DB
    if not os.path.exists(db):
        return pd.DataFrame()
    with _connect(path) as conn:
        df = pd.read_sql_query("SELECT * FROM signals", conn)
    if df.empty:
        return df
    feats = pd.json_normalize(df["features"].apply(json.loads))
    feats.index = df.index
    out = pd.concat([df.drop(columns=["features"]), feats], axis=1)
    # The base table (symbol/direction/strategy) and the features JSON can both
    # carry direction/strategy — keep the first (authoritative base) column.
    out = out.loc[:, ~out.columns.duplicated()]
    out["ts"] = pd.to_datetime(out["ts"])
    return out.sort_values("ts").reset_index(drop=True)


def read_shadow(path: str | None = None) -> pd.DataFrame:
    db = path or config.META_DB
    if not os.path.exists(db):
        return pd.DataFrame()
    with _connect(path) as conn:
        df = pd.read_sql_query("SELECT * FROM shadow", conn)
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"])
    return df


def count_signals(path: str | None = None) -> int:
    db = path or config.META_DB
    if not os.path.exists(db):
        return 0
    with _connect(path) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0])
