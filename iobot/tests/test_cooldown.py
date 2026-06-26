"""Post-close same-symbol+direction re-entry cooldown (journal helper)."""
from __future__ import annotations

from iobot import journal
from iobot.clock import now_et


def _log_close(conn, symbol, direction, minutes_ago):
    """Insert a minimal closed trade whose exit_time is `minutes_ago` in the past."""
    exit_time = (now_et()
                 .replace(microsecond=0)
                 .timestamp()) - minutes_ago * 60.0
    import datetime as dt
    from iobot.clock import ET
    iso = dt.datetime.fromtimestamp(exit_time, ET).isoformat()
    journal.log_trade(conn, {
        "signal_id": f"{symbol}-{direction}-{minutes_ago}", "structure": "single",
        "symbol": symbol, "direction": direction, "entry_time": iso, "exit_time": iso,
        "qty": 1, "pnl": 10.0, "r_multiple": 0.1, "max_loss": 100.0,
        "exit_reason": "profit target (underlying)",
    })


def test_no_prior_trade_returns_none(conn):
    assert journal.minutes_since_last_exit(conn, "AAPL", "put") is None


def test_recent_close_is_within_cooldown(conn):
    _log_close(conn, "AAPL", "put", minutes_ago=12)
    mins = journal.minutes_since_last_exit(conn, "AAPL", "put")
    assert mins is not None and 10 <= mins < 15        # ~12m → would block a 30m cooldown


def test_old_close_is_past_cooldown(conn):
    _log_close(conn, "AAPL", "put", minutes_ago=90)
    mins = journal.minutes_since_last_exit(conn, "AAPL", "put")
    assert mins is not None and mins >= 60             # ~90m → 30m cooldown has expired


def test_direction_is_scoped(conn):
    """A closed PUT must not trigger the cooldown for a fresh CALL on the same name."""
    _log_close(conn, "AAPL", "put", minutes_ago=5)
    assert journal.minutes_since_last_exit(conn, "AAPL", "call") is None
    assert journal.minutes_since_last_exit(conn, "AAPL", "put") is not None


def test_symbol_is_scoped(conn):
    _log_close(conn, "AAPL", "put", minutes_ago=5)
    assert journal.minutes_since_last_exit(conn, "PLTR", "put") is None
