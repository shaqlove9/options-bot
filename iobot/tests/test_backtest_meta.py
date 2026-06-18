"""backtest meta-frame: label mapping, sequence-feature fill, and net-R wiring.

Pure (no network) — exercises _meta_frame on synthetic trades so the contract the
walk-forward depends on (entry-time ordering, profit-target labels, streak/trades_today
fill, net-R = backtest r_multiple) can't silently drift."""
from __future__ import annotations

import datetime as dt

from iobot import backtest
from iobot.clock import ET


def _trade(minute: int, reason: str, pnl: float, r: float, day=1) -> backtest.Trade:
    t = dt.datetime(2026, 1, day, 10, minute, tzinfo=ET)
    return backtest.Trade(
        symbol="SPY", direction="call", entry_time=t, exit_time=t + dt.timedelta(minutes=30),
        entry_underlying=500.0, exit_underlying=501.0, strike=500.0, iv=0.2,
        entry_opt=2.0, exit_opt=2.5, pnl=pnl, r_multiple=r, exit_reason=reason)


def test_meta_frame_labels_order_and_sequence_features():
    # deliberately out of entry-time order to prove sorting
    trades = [
        _trade(40, "profit target (underlying)", +100.0, 0.5),   # win
        _trade(0, "stop (underlying)", -50.0, -0.3),             # loss (earliest)
        _trade(20, "profit target (underlying)", +80.0, 0.4),    # win
    ]
    X, y, net_r, ordered = backtest._meta_frame(trades)

    # sorted by entry_time
    assert [t.entry_time.minute for t in ordered] == [0, 20, 40]
    # label = 1 iff profit target hit first (journal.label_from_reason)
    assert list(y) == [0, 1, 1]
    # net_r is the backtest r_multiple verbatim (already net of cost), in order
    assert list(net_r) == [-0.3, 0.4, 0.5]
    # streaks reflect outcomes BEFORE each trade: loss, then win, then win-streak 1
    assert [t.features["win_streak"] for t in ordered] == [0.0, 0.0, 1.0]
    assert [t.features["loss_streak"] for t in ordered] == [0.0, 1.0, 0.0]
    # all same day -> trades_today increments 0,1,2
    assert [t.features["trades_today"] for t in ordered] == [0.0, 1.0, 2.0]
    # X has one row per trade and the sequence features are columns
    assert len(X) == 3
    assert {"win_streak", "loss_streak", "trades_today"} <= set(X.columns)


def test_meta_frame_trades_today_resets_per_day():
    trades = [
        _trade(0, "stop (x)", -10.0, -0.1, day=1),
        _trade(30, "stop (x)", -10.0, -0.1, day=1),
        _trade(0, "stop (x)", -10.0, -0.1, day=2),
    ]
    _, _, _, ordered = backtest._meta_frame(trades)
    assert [t.features["trades_today"] for t in ordered] == [0.0, 1.0, 0.0]
