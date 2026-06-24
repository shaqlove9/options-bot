"""Tests for the senior-trader pivot: confluence signal, $1k sleeve risk cap,
overnight-carry decision, and position persistence/rehydration."""
from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import numpy as np
import pandas as pd

from iobot import chain, config, executor, features, governor, store, strategies
from iobot.clock import ET
from iobot.executor import SingleLegPosition, should_carry_overnight


# ---------------- confluence signal ----------------

def _session_starts(day: dt.date) -> list[dt.datetime]:
    base = dt.datetime(day.year, day.month, day.day, 9, 30, tzinfo=ET)
    return [base + dt.timedelta(minutes=15 * i) for i in range(26)]


def _intraday_uptrend():
    prior, today = dt.date(2026, 6, 16), dt.date(2026, 6, 17)
    idx = _session_starts(prior) + _session_starts(today)
    n = len(idx)
    closes = 100 + np.arange(n) * 0.15
    opens = np.concatenate([[closes[0]], closes[:-1]])
    highs = np.maximum(opens, closes) + 0.03
    lows = np.minimum(opens, closes) - 0.03
    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes,
         "volume": np.full(n, 200_000.0), "vwap": closes},
        index=pd.DatetimeIndex(idx))
    now = idx[-1] + dt.timedelta(minutes=15)
    return df, today, now


def _daily_uptrend(today: dt.date) -> pd.DataFrame:
    days = pd.bdate_range(end=today - dt.timedelta(days=1), periods=30, tz=ET)
    closes = 90 + np.arange(30) * 0.5
    return pd.DataFrame(
        {"open": closes, "high": closes + 0.5, "low": closes - 0.5, "close": closes,
         "volume": np.full(30, 5e6)}, index=days)


def test_confluence_fires_long_on_clean_uptrend(monkeypatch):
    monkeypatch.setattr(config, "REL_VOLUME_MIN", 0.0)   # volume always confirms
    monkeypatch.setattr(config, "CONFLUENCE_MIN", 4)
    intraday, today, now = _intraday_uptrend()
    daily = _daily_uptrend(today)
    sig = strategies.confluence("SPY", intraday, daily, now)
    assert sig is not None
    assert sig.direction == "call"
    assert sig.conf_score >= config.CONFLUENCE_MIN
    # ATR-based exits: stop below, target above spot for a call.
    assert sig.stop_level < sig.spot < sig.target_level
    # sub-scores propagate into the captured feature row.
    row = features.build(sig, features.RegimeContext(daily={}))
    assert row.features["conf_score"] == sig.conf_score
    assert row.features["conf_trend"] == sig.conf_trend


def test_confluence_abstains_when_factors_disagree(monkeypatch):
    monkeypatch.setattr(config, "CONFLUENCE_MIN", 5)     # demand near-perfect alignment
    intraday, today, now = _intraday_uptrend()
    # no daily -> regime factor neutral, so the score can't reach 5
    assert strategies.confluence("SPY", intraday, None, now) is None


# ---------------- sleeve-equity risk cap ----------------

def test_per_trade_cap_uses_sleeve_equity(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RISK_PCT_PER_TRADE", 30.0)
    gov = governor.RiskGovernor(state_file=str(tmp_path / "gov.json"))
    big_acct = SimpleNamespace(equity=100_000.0, cash=100_000.0,
                               buying_power=100_000.0, options_buying_power=100_000.0)
    # cap is 30% of the $1,000 sleeve = $300, NOT 30% of the big paper account.
    ok, _ = gov.approve_risk(250.0, 250.0, big_acct, cap_equity=1000.0)
    assert ok
    ok, why = gov.approve_risk(350.0, 350.0, big_acct, cap_equity=1000.0)
    assert not ok and "cap" in why
    # without a sleeve cap it falls back to the (huge) account equity.
    ok, _ = gov.approve_risk(350.0, 350.0, big_acct)
    assert ok


# ---------------- overnight carry decision ----------------

def test_should_carry_overnight(monkeypatch):
    monkeypatch.setattr(config, "ALLOW_OVERNIGHT", True)
    monkeypatch.setattr(config, "OVERNIGHT_MIN_DTE_KEEP", 1)
    monkeypatch.setattr(config, "MAX_HOLD_DAYS", 5)
    today = dt.date(2026, 6, 17)
    # plenty of DTE, freshly opened -> carry
    assert should_carry_overnight(today + dt.timedelta(days=5), today, today)
    # expires within the keep window -> flatten
    assert not should_carry_overnight(today + dt.timedelta(days=1), today, today)
    # held the max number of days -> flatten
    assert not should_carry_overnight(today + dt.timedelta(days=5),
                                      today - dt.timedelta(days=5), today)
    # overnight disabled -> never carry
    monkeypatch.setattr(config, "ALLOW_OVERNIGHT", False)
    assert not should_carry_overnight(today + dt.timedelta(days=5), today, today)


# ---------------- position persistence / rehydrate ----------------

def _pos() -> SingleLegPosition:
    return SingleLegPosition(
        underlying="F", direction="call", option_symbol="F260619C00012000",
        strike=12.0, expiry=dt.date(2026, 6, 19), qty=2, entry_fill=1.5,
        entry_mid=1.45, entry_underlying=12.0, stop_level=11.8, target_level=12.4,
        max_loss=300.0, entry_time=dt.datetime(2026, 6, 17, 11, 0, tzinfo=ET),
        signal_id="F-x", features={"conf_score": 4.0})


def test_persist_and_rehydrate_round_trip(conn):
    ex = executor.Executor(None, None, None, conn)
    pos = _pos()
    ex._persist(pos)
    # a fresh executor rehydrates the position the broker still holds.
    ex2 = executor.Executor(None, None, None, conn)
    ex2._rehydrate({pos.option_symbol})
    assert len(ex2.positions) == 1
    got = ex2.positions[0]
    assert got.option_symbol == pos.option_symbol
    assert got.qty == 2
    assert got.expiry == pos.expiry
    assert isinstance(got.entry_time, dt.datetime)
    assert got.features["conf_score"] == 4.0


def test_rehydrate_drops_position_not_held_at_broker(conn):
    ex = executor.Executor(None, None, None, conn)
    ex._persist(_pos())
    ex2 = executor.Executor(None, None, None, conn)
    ex2._rehydrate(set())     # broker no longer holds it (closed while down)
    assert ex2.positions == []
    # the stale row was deleted.
    n = conn.execute("SELECT COUNT(*) AS c FROM positions").fetchone()["c"]
    assert n == 0
