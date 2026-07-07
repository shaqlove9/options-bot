"""alerts — webhook notifications for trade events.

Supports Discord and Slack backends. Set ALERT_BACKEND in .env to "discord",
"slack", or "auto" (default — picks whichever webhook URL is configured;
Discord wins if both are set). If neither URL is set, alerts are silently
skipped. Alerts never crash the trading loop.

Public API (unchanged regardless of backend):
    entry(), trade_exit(), halt(), pause(), daily_summary(), error(), ai_report()
"""
import logging

import requests

import config
from alerts import discord, slack

log = logging.getLogger("alerts")

# Re-export colors so consumers can reference them if needed
GREEN, RED, ORANGE, BLUE, PURPLE = (
    discord.GREEN, discord.RED, discord.ORANGE, discord.BLUE, discord.PURPLE
)


def _get_backend():
    """Resolve which backend module to use. Returns module or None."""
    backend = config.ALERT_BACKEND
    if backend == "discord":
        return discord if discord.is_configured() else None
    if backend == "slack":
        return slack if slack.is_configured() else None
    # auto: prefer whichever is configured; Discord wins if both set
    if discord.is_configured():
        return discord
    if slack.is_configured():
        return slack
    return None


def _send(title: str, description: str, color: int):
    backend = _get_backend()
    if backend is None:
        log.debug("No alert webhook configured — skipping: %s", title)
        return
    try:
        backend.send(title, description, color)
    except requests.RequestException as exc:
        # Alerts must never crash the trading loop.
        log.warning("%s alert failed: %s", backend.__name__.split(".")[-1].capitalize(), exc)


def entry(pick, qty: int, fill_price: float, reason: str):
    mode = "LIVE" if config.LIVE_MODE else "PAPER"
    _send(
        f"🟢 [{mode}] ENTRY {pick.underlying} {pick.otype.upper()}",
        f"**{pick.option_symbol}**\n"
        f"Strike {pick.strike} | Exp {pick.expiry} | Qty {qty}\n"
        f"Fill: ${fill_price:.2f} (${fill_price * 100 * qty:.0f})\n"
        f"Reason: {reason}",
        BLUE,
    )


def trade_exit(pos, exit_price: float, pnl: float, reason: str):
    mode = "LIVE" if config.LIVE_MODE else "PAPER"
    color = GREEN if pnl >= 0 else RED
    pct = (exit_price - pos.entry_price) / pos.entry_price * 100 if pos.entry_price else 0
    _send(
        f"{'✅' if pnl >= 0 else '❌'} [{mode}] EXIT {pos.underlying} {pos.otype.upper()}",
        f"**{pos.option_symbol}**\n"
        f"Entry ${pos.entry_price:.2f} → Exit ${exit_price:.2f} ({pct:+.0f}%)\n"
        f"P&L: **${pnl:+.2f}**\n"
        f"Reason: {reason}",
        color,
    )


def halt(daily_pnl: float):
    _send(
        "🛑 TRADING HALTED — daily loss limit",
        f"Daily P&L: **${daily_pnl:+.2f}** (limit -${config.MAX_DAILY_LOSS:.0f})\n"
        "No more trades today. Open positions were flattened.",
        RED,
    )


def pause(minutes: int):
    _send(
        "⏸️ Trading paused",
        f"{config.CONSEC_LOSS_PAUSE} consecutive losses — pausing new entries "
        f"for {minutes} minutes.",
        ORANGE,
    )


def daily_summary(summary: dict):
    _send(
        f"📊 Daily summary — {summary['date']}",
        f"P&L: **${summary['pnl']:+.2f}**\n"
        f"Trades: {summary['trades']} ({summary['wins']}W / {summary['losses']}L, "
        f"{summary['win_rate']:.0f}% win rate)\n"
        f"Halted: {'yes' if summary['halted'] else 'no'}",
        GREEN if summary["pnl"] >= 0 else RED,
    )


def error(message: str):
    _send("⚠️ Bot error", message[:1900], ORANGE)


def ai_report(title: str, text: str):
    """Post a long AI-written report, split to fit webhook limits."""
    backend = _get_backend()
    limit = backend.max_chunk_size() if backend else 3800
    chunks = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut < 1000:
            cut = limit
        chunks.append(text[:cut])
        text = text[cut:]
    chunks.append(text)
    for i, chunk in enumerate(chunks):
        _send(title if i == 0 else f"{title} (cont.)", chunk, PURPLE)
