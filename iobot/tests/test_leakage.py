"""Phase-1 capture must never let post-entry / future-timestamped data into a row."""
from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from iobot import config, features
from iobot.clock import ET
from iobot.tests.conftest import make_signal


@pytest.fixture(autouse=True)
def _pinned_universe(monkeypatch):
    """`features.build` only reads regime frames for symbols in config.UNIVERSE, so the
    frames below must belong to it. Pinned because the live .env universe changes: when
    SPY/QQQ were dropped from it these frames stopped being read at all, which silently
    voided the leakage assertion rather than failing loudly."""
    monkeypatch.setattr(config, "UNIVERSE", ["SPY", "QQQ"])


def _daily(last_date: dt.date) -> pd.DataFrame:
    idx = pd.date_range(end=pd.Timestamp(last_date), periods=60, freq="D", tz="UTC")
    return pd.DataFrame({"close": range(60)}, index=idx)


def test_assert_no_leakage_raises_on_future_input():
    with pytest.raises(features.LeakageError):
        features.assert_no_leakage(captured_at_ts=1000.0, max_asof_ts=5000.0)


def test_assert_no_leakage_allows_past_input():
    features.assert_no_leakage(captured_at_ts=5000.0, max_asof_ts=4000.0)  # no raise


def test_capture_rejects_future_dated_feature(conn):
    """A regime frame dated AFTER the signal instant must be caught by capture."""
    sig = make_signal(when=dt.datetime(2026, 6, 17, 11, 0, tzinfo=ET))
    future = sig.time.date() + dt.timedelta(days=10)
    ctx = features.RegimeContext(daily={"SPY": _daily(future), "QQQ": _daily(future)})
    row = features.build(sig, ctx)
    assert row.max_asof_ts > row.captured_at_ts        # build surfaced the future ts
    with pytest.raises(features.LeakageError):
        features.capture(conn, sig, row)
    # nothing was written
    assert conn.execute("SELECT COUNT(*) FROM features").fetchone()[0] == 0


def test_capture_writes_clean_row(conn):
    sig = make_signal(when=dt.datetime(2026, 6, 17, 11, 0, tzinfo=ET))
    past = sig.time.date() - dt.timedelta(days=1)
    ctx = features.RegimeContext(daily={"SPY": _daily(past), "QQQ": _daily(past)})
    row = features.build(sig, ctx)
    features.capture(conn, sig, row)
    assert conn.execute("SELECT COUNT(*) FROM features").fetchone()[0] == 1
    assert "minute_of_day" in row.features and "vix" in row.features
