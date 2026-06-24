"""notify.py — optional Discord notifications for entries, exits, and halts.

Fire-and-forget: each notification is POSTed from a daemon thread with a short
timeout and all errors are swallowed, so a slow/down Discord never blocks or
crashes the trading loop. Disabled (no-op) when DISCORD_WEBHOOK_URL is unset.

The embed builders (`*_embed`) are pure and unit-tested; `entry/exit/halt` wrap
them with the fire-and-forget POST.
"""
from __future__ import annotations

import logging
import threading

from iobot import config
from iobot.clock import now_et

log = logging.getLogger("notify")

GREEN = 0x2ECC71   # winning / opened
RED = 0xE74C3C     # losing / halted
AMBER = 0xF1C40F   # break-even-ish


def _post(embed: dict):
    url = config.DISCORD_WEBHOOK_URL
    if not url:
        return  # disabled

    def _run():
        try:
            import requests
            requests.post(url, json={"embeds": [embed]}, timeout=5)
        except Exception as exc:           # never let Discord break trading
            log.debug("discord notify failed: %s", exc)

    threading.Thread(target=_run, daemon=True).start()


def _f(name: str, value: str, inline: bool = True) -> dict:
    return {"name": name, "value": value, "inline": inline}


def _held(entry_time) -> str:
    secs = max(0, int((now_et() - entry_time).total_seconds()))
    h, m = divmod(secs // 60, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m"


def _is_single(pos) -> bool:
    return getattr(pos, "structure", "single") == "single"


# ---------------- pure embed builders ----------------

def entry_embed(pos, reason: str = "") -> dict:
    arrow = "📈" if pos.direction == "call" else "📉"
    if _is_single(pos):
        contract = f"{pos.qty} × {pos.strike:g} {pos.direction.upper()} exp {pos.expiry}"
        entry = f"${pos.entry_fill:.2f}"
        fields = [
            _f("Contract", contract, inline=False),
            _f("Entry", entry),
            _f("Max loss", f"${pos.max_loss:.0f}"),
            _f("Stop / Target (u/l)", f"{pos.stop_level:.2f} / {pos.target_level:.2f}"),
        ]
    else:
        contract = (f"{pos.qty} × {pos.long_strike:g}/{pos.short_strike:g} "
                    f"{pos.direction.upper()} spread exp {pos.expiry}")
        fields = [
            _f("Contract", contract, inline=False),
            _f("Debit", f"${pos.entry_debit:.2f}"),
            _f("Max loss", f"${pos.max_loss:.0f}"),
        ]
    if reason:
        fields.append(_f("Signal", reason, inline=False))
    return {
        "title": f"{arrow} Opened {pos.underlying} {pos.direction.upper()}",
        "color": GREEN,
        "fields": fields,
        "footer": {"text": f"iobot · paper · {now_et():%H:%M:%S ET}"},
    }


def exit_embed(pos, exit_fill: float, pnl: float, reason: str) -> dict:
    win = pnl > 0
    r = pnl / pos.max_loss if getattr(pos, "max_loss", 0) else 0.0
    icon = "✅" if win else "❌"
    entry_px = pos.entry_fill if _is_single(pos) else pos.entry_debit
    return {
        "title": f"{icon} Closed {pos.underlying} {pos.direction.upper()}",
        "color": GREEN if win else RED,
        "fields": [
            _f("P&L", f"${pnl:+.2f}"),
            _f("R multiple", f"{r:+.2f}R"),
            _f("Held", _held(pos.entry_time)),
            _f("Entry → Exit", f"${entry_px:.2f} → ${exit_fill:.2f}", inline=False),
            _f("Reason", reason, inline=False),
        ],
        "footer": {"text": f"iobot · paper · {now_et():%H:%M:%S ET}"},
    }


def halt_embed(reason: str) -> dict:
    return {
        "title": "⛔ Sleeve halted — trading stopped",
        "color": RED,
        "description": reason,
        "footer": {"text": f"iobot · paper · {now_et():%H:%M:%S ET}"},
    }


# ---------------- fire-and-forget senders ----------------

def entry(pos, reason: str = ""):
    _post(entry_embed(pos, reason))


def exit(pos, exit_fill: float, pnl: float, reason: str):
    _post(exit_embed(pos, exit_fill, pnl, reason))


def halt(reason: str):
    _post(halt_embed(reason))
