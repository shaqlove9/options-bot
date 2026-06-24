"""Webhook receiver: auth/dedup/enrichment, and that an enqueued signal flows
through the SAME Engine._handle_signal pipeline (governor + meta + selection +
risk) as scanner signals — with every guard intact. No network, no broker."""
from __future__ import annotations

import datetime as dt
import queue as _queue
from types import SimpleNamespace

import pandas as pd
import pytest

from iobot import config, engine, features, webhook
from iobot.clock import ET, now_et
from iobot.signals import Signal


@pytest.fixture(autouse=True)
def _auth(monkeypatch):
    monkeypatch.setattr(config, "WEBHOOK_SECRET", "s3cret")
    monkeypatch.setattr(config, "WEBHOOK_IP_ALLOWLIST", [])


# --------------------------------------------------------------------------- #
# auth / validation
# --------------------------------------------------------------------------- #

def test_secret_rejection():
    d = webhook.Deduper(60)
    p, s, _ = webhook.validate_payload(
        {"secret": "nope", "symbol": "SPY", "direction": "call"}, "1.2.3.4", deduper=d)
    assert p is None and s == 401
    p, s, _ = webhook.validate_payload(
        {"symbol": "SPY", "direction": "call"}, "1.2.3.4", deduper=d)
    assert p is None and s == 401
    p, s, _ = webhook.validate_payload(
        {"secret": "s3cret", "symbol": "spy", "direction": "call"}, "1.2.3.4", deduper=d)
    assert p is not None and s == 200 and p["symbol"] == "SPY"   # upper-cased


def test_ip_allowlist(monkeypatch):
    monkeypatch.setattr(config, "WEBHOOK_IP_ALLOWLIST", ["9.9.9.9"])
    d = webhook.Deduper(60)
    ok = {"secret": "s3cret", "symbol": "SPY", "direction": "call"}
    assert webhook.validate_payload(ok, "1.2.3.4", deduper=d)[1] == 403
    assert webhook.validate_payload(ok, "9.9.9.9", deduper=d)[0] is not None


def test_bad_direction():
    d = webhook.Deduper(60)
    p, s, _ = webhook.validate_payload(
        {"secret": "s3cret", "symbol": "SPY", "direction": "long"}, "1.2.3.4", deduper=d)
    assert p is None and s == 400


# --------------------------------------------------------------------------- #
# dedup
# --------------------------------------------------------------------------- #

def test_dedup():
    d = webhook.Deduper(60)
    base = {"secret": "s3cret", "symbol": "SPY", "direction": "call", "id": "abc"}
    assert webhook.validate_payload(dict(base), "1.2.3.4", deduper=d)[0] is not None
    p, s, m = webhook.validate_payload(dict(base), "1.2.3.4", deduper=d)
    assert p is None and s == 200 and "duplicate" in m
    assert webhook.validate_payload({**base, "id": "def"}, "1.2.3.4", deduper=d)[0] is not None


def test_deduper_ttl():
    d = webhook.Deduper(ttl_sec=10)
    assert d.seen_recently("x", now=100.0) is False
    assert d.seen_recently("x", now=105.0) is True       # within TTL -> dup
    assert d.seen_recently("x", now=120.0) is False      # expired
    assert d.seen_recently(None) is False                # missing id never dedups


# --------------------------------------------------------------------------- #
# enrichment: payload -> Signal
# --------------------------------------------------------------------------- #

class _FailSrc:
    def _intraday(self, symbol):
        return None
    def _daily(self, symbol):
        return None


def test_enrich_advisory_on_data_failure():
    sig = webhook.enrich_signal(
        {"symbol": "QQQ", "direction": "put", "price": 500.0}, _FailSrc())
    assert isinstance(sig, Signal)
    assert (sig.symbol, sig.direction) == ("QQQ", "put")
    assert sig.advisory is True and sig.source == "tradingview"
    assert sig.spot == 500.0                              # price seeds spot
    assert sig.rsi == 0.0 and sig.atr_pct == 0.0          # features default to 0


def test_enrich_live():
    today = now_et().date()
    idx = pd.DatetimeIndex(
        [dt.datetime.combine(today, dt.time(10, 0)) + dt.timedelta(minutes=15 * i)
         for i in range(20)]).tz_localize(ET)
    closes = [100, 101, 102, 101, 103, 104, 103, 105, 106, 105,
              107, 108, 107, 109, 110, 109, 111, 112, 111, 113]
    df = pd.DataFrame({
        "open": [c - 0.5 for c in closes],
        "high": [c + 1 for c in closes],
        "low":  [c - 1 for c in closes],
        "close": [float(c) for c in closes],
        "volume": [1000] * 20,
        "vwap": [105.0] * 20,
    }, index=idx)

    class _GoodSrc:
        def _intraday(self, symbol):
            return df
        def _daily(self, symbol):
            return None        # rel_volume -> 0, but no exception -> not advisory

    sig = webhook.enrich_signal(
        {"symbol": "SPY", "direction": "call", "price": None}, _GoodSrc())
    assert sig.advisory is False
    assert sig.spot == 113.0                              # last close
    assert sig.rsi != 0.0 and sig.atr_pct > 0.0
    assert abs(sig.vwap_dist_pct - (113.0 - 105.0) / 105.0 * 100) < 1e-6


# --------------------------------------------------------------------------- #
# engine pipeline: an enqueued signal flows through _handle_signal w/ guards
# --------------------------------------------------------------------------- #

class _Gov:
    def __init__(self):
        self.state = SimpleNamespace(trades_today=0)
        self.entries = 0
        self.can = (True, "")
        self.appr = (True, "")
    def can_open(self, equity, open_count):
        return self.can
    def approve_risk(self, max_loss, required_bp, acct):
        return self.appr
    def record_entry(self):
        self.entries += 1


class _Exec:
    def __init__(self):
        self.opened = []
        self._pos = set()
    def has_position_in(self, symbol):
        return symbol in self._pos
    def open_count(self):
        return len(self._pos)
    def open_single(self, pick, sig, feats):
        self.opened.append((pick, sig, feats))
        self._pos.add(sig.symbol)
        return object()
    def open_spread(self, pick, sig, feats):
        return self.open_single(pick, sig, feats)


class _Meta:
    def __init__(self, would_skip=False, active=False):
        self._ws, self._act, self.calls = would_skip, active, 0
    def decision(self, feats):
        self.calls += 1
        return (0.2 if self._ws else 0.9, self._ws, 1.0)
    def active(self):
        return self._act


def _make_engine(monkeypatch, conn, meta_obj=None, gov=None, ex=None):
    monkeypatch.setattr("iobot.chain.build_contract",
                        lambda sig, trading, option_data: (SimpleNamespace(max_loss=40.0), "ok"))
    eng = engine.Engine.__new__(engine.Engine)
    eng.conn = conn
    eng.gov = gov or _Gov()
    eng.exec = ex or _Exec()
    eng.meta = meta_obj or _Meta()
    eng.clients = SimpleNamespace(trading=None, option_data=None, stock_data=None)
    eng.webhook_queue = _queue.Queue()
    return eng


def _ctx():
    return features.RegimeContext(daily={})


def _acct():
    return SimpleNamespace(equity=10_000.0, cash=10_000.0,
                           buying_power=20_000.0, options_buying_power=10_000.0)


def _mk(symbol="SPY", direction="call", advisory=False):
    s = Signal(symbol=symbol, direction=direction, spot=500.0, momentum_pct=0.6,
               rsi=70.0, rel_volume=1.5, vwap_dist_pct=0.3, atr_pct=0.8, time=now_et())
    s.source = "tradingview"
    s.advisory = advisory
    return s


def test_drain_webhook_is_fifo(conn, monkeypatch):
    eng = _make_engine(monkeypatch, conn)
    eng.webhook_queue.put(_mk())
    eng.webhook_queue.put(_mk(symbol="QQQ"))
    assert [s.symbol for s in eng._drain_webhook()] == ["SPY", "QQQ"]
    assert eng._drain_webhook() == []


def test_queued_signal_enters(conn, monkeypatch):
    eng = _make_engine(monkeypatch, conn)
    eng.webhook_queue.put(_mk())
    (sig,) = eng._drain_webhook()
    eng._handle_signal(sig, _acct(), _ctx())
    assert len(eng.exec.opened) == 1 and eng.gov.entries == 1


def test_governor_blocks(conn, monkeypatch):
    gov = _Gov(); gov.can = (False, "max trades/day")
    eng = _make_engine(monkeypatch, conn, gov=gov)
    eng._handle_signal(_mk(), _acct(), _ctx())
    assert eng.exec.opened == [] and eng.gov.entries == 0


def test_existing_position_blocks(conn, monkeypatch):
    ex = _Exec(); ex._pos.add("SPY")
    eng = _make_engine(monkeypatch, conn, ex=ex)
    eng._handle_signal(_mk(), _acct(), _ctx())
    assert len(ex.opened) == 0


def test_risk_cap_blocks(conn, monkeypatch):
    gov = _Gov(); gov.appr = (False, "exceeds per-trade risk")
    eng = _make_engine(monkeypatch, conn, gov=gov)
    eng._handle_signal(_mk(), _acct(), _ctx())
    assert eng.exec.opened == []


def test_no_contract_blocks(conn, monkeypatch):
    eng = _make_engine(monkeypatch, conn)
    monkeypatch.setattr("iobot.chain.build_contract",
                        lambda sig, trading, option_data: (None, "no liquid contract"))
    eng._handle_signal(_mk(), _acct(), _ctx())
    assert eng.exec.opened == []


def test_meta_vetoes_live_signal_when_active(conn, monkeypatch):
    eng = _make_engine(monkeypatch, conn, meta_obj=_Meta(would_skip=True, active=True))
    eng._handle_signal(_mk(), _acct(), _ctx())
    assert eng.exec.opened == [] and eng.meta.calls == 1


def test_advisory_bypasses_meta_veto(conn, monkeypatch):
    eng = _make_engine(monkeypatch, conn, meta_obj=_Meta(would_skip=True, active=True))
    eng._handle_signal(_mk(advisory=True), _acct(), _ctx())
    assert len(eng.exec.opened) == 1            # advisory entered despite would-skip
