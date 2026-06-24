"""Discord notification embed builders (pure) + disabled no-op."""
from __future__ import annotations

import datetime as dt

from iobot import config, notify
from iobot.clock import ET
from iobot.executor import SingleLegPosition


def _pos(qty=2, entry=1.5, max_loss=300.0) -> SingleLegPosition:
    return SingleLegPosition(
        underlying="F", direction="call", option_symbol="F260619C00012000",
        strike=12.0, expiry=dt.date(2026, 6, 19), qty=qty, entry_fill=entry,
        entry_mid=entry - 0.05, entry_underlying=12.0, stop_level=11.8,
        target_level=12.4, max_loss=max_loss,
        entry_time=dt.datetime(2026, 6, 24, 10, 0, tzinfo=ET), signal_id="F-x")


def test_entry_embed_shape():
    e = notify.entry_embed(_pos(), reason="+0.5% 15m, RSI 70")
    assert "Opened F CALL" in e["title"]
    assert e["color"] == notify.GREEN
    names = {f["name"] for f in e["fields"]}
    assert {"Contract", "Entry", "Max loss", "Stop / Target (u/l)", "Signal"} <= names
    # contract line reflects the dollar-sized qty
    contract = next(f["value"] for f in e["fields"] if f["name"] == "Contract")
    assert contract.startswith("2 × 12")


def test_exit_embed_win_vs_loss():
    win = notify.exit_embed(_pos(), exit_fill=2.7, pnl=120.0, reason="profit target")
    assert win["color"] == notify.GREEN and win["title"].startswith("✅")
    assert any(f["value"] == "$+120.00" for f in win["fields"])
    assert any(f["value"] == "+0.40R" for f in win["fields"])  # 120/300

    loss = notify.exit_embed(_pos(), exit_fill=0.9, pnl=-120.0, reason="stop (underlying)")
    assert loss["color"] == notify.RED and loss["title"].startswith("❌")


def test_halt_embed():
    e = notify.halt_embed("trailing DD 40.0% >= 40% off peak")
    assert "halted" in e["title"].lower() and e["color"] == notify.RED
    assert "trailing DD" in e["description"]


def test_post_is_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(config, "DISCORD_WEBHOOK_URL", "")
    # must not raise or spawn work when no webhook is configured
    assert notify._post({"title": "x"}) is None
