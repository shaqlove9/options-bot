"""Validation gate: hard-fails until enough trades + positive net E[R]."""
from __future__ import annotations

from iobot import config, gate


def _insert_trade(conn, r, pnl, structure="single", max_loss=400.0):
    conn.execute(
        "INSERT INTO trades(signal_id, structure, symbol, direction, entry_time, "
        "exit_time, max_loss, pnl, r_multiple, exit_reason, label) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        ("s", structure, "SPY", "call", "2026-06-17T11:00", "2026-06-17T11:30",
         max_loss, pnl, r, "profit target" if pnl > 0 else "stop", 1 if pnl > 0 else 0))
    conn.commit()


def test_insufficient_when_few_trades(conn, monkeypatch):
    monkeypatch.setattr(config, "GATE_MIN_TRADES", 40)
    _insert_trade(conn, 1.0, 100)
    res = gate.evaluate(conn)
    assert res["verdict"] == "INSUFFICIENT" and not res["passes"]


def test_keep_when_positive_net_r(conn, monkeypatch):
    monkeypatch.setattr(config, "GATE_MIN_TRADES", 6)
    for _ in range(8):
        _insert_trade(conn, 0.8, 320)        # solid winners, friction can't flip them
    res = gate.evaluate(conn)
    assert res["verdict"] == "KEEP" and res["passes"]
    assert res["net_expected_r"] > 0


def test_kill_when_negative_net_r(conn, monkeypatch):
    monkeypatch.setattr(config, "GATE_MIN_TRADES", 6)
    for _ in range(8):
        _insert_trade(conn, -0.5, -200)
    res = gate.evaluate(conn)
    assert res["verdict"] == "KILL" and not res["passes"]


def test_no_trades_is_insufficient(conn):
    res = gate.evaluate(conn)
    assert res["verdict"] == "INSUFFICIENT" and res["n_trades"] == 0
