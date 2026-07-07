"""alerts.py — Webhook alerts: entries, exits, halts, daily summary.

Supports Discord and Slack. Set ALERT_BACKEND in .env to "discord", "slack",
or "auto" (default — picks whichever webhook URL is configured). If neither
URL is set, alerts are silently skipped. Alerts never crash the trading loop.
"""
import json
import logging

import requests

import config

log = logging.getLogger("alerts")

# Discord embed colors
GREEN, RED, ORANGE, BLUE, PURPLE = 0x2ECC71, 0xE74C3C, 0xE67E22, 0x3498DB, 0x9B59B6

# Slack color strings (hex without 0x prefix)
_SLACK_COLORS = {
    GREEN: "#2ecc71", RED: "#e74c3c", ORANGE: "#e67e22",
    BLUE: "#3498db", PURPLE: "#9b59b6",
}


def _backend() -> str:
    """Resolve which backend to use. Returns "discord", "slack", or "none"."""
    backend = config.ALERT_BACKEND
    if backend == "discord":
        return "discord" if config.DISCORD_WEBHOOK_URL else "none"
    if backend == "slack":
        return "slack" if config.SLACK_WEBHOOK_URL else "none"
    # auto: prefer whichever is configured; Discord wins if both are set
    if config.DISCORD_WEBHOOK_URL:
        return "discord"
    if config.SLACK_WEBHOOK_URL:
        return "slack"
    return "none"


def _send_discord(title: str, description: str, color: int):
    payload = {"embeds": [{"title": title, "description": description, "color": color}]}
    resp = requests.post(config.DISCORD_WEBHOOK_URL, json=payload, timeout=10)
    resp.raise_for_status()


def _send_slack(title: str, description: str, color: int):
    # Convert markdown bold **text** to Slack bold *text*
    text = description.replace("**", "*")
    slack_color = _SLACK_COLORS.get(color, "#3498db")
    payload = {
        "attachments": [{
            "color": slack_color,
            "title": title,
            "text": text,
            "mrkdwn_in": ["text"],
        }]
    }
    resp = requests.post(config.SLACK_WEBHOOK_URL,
                         data=json.dumps(payload),
                         headers={"Content-Type": "application/json"},
                         timeout=10)
    resp.raise_for_status()


def _send(title: str, description: str, color: int):
    backend = _backend()
    if backend == "none":
        log.debug("No alert webhook configured — skipping: %s", title)
        return
    try:
        if backend == "discord":
            _send_discord(title, description, color)
        elif backend == "slack":
            _send_slack(title, description, color)
    except requests.RequestException as exc:
        # Alerts must never crash the trading loop.
        log.warning("%s alert failed: %s", backend.capitalize(), exc)


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
    """Post a long AI-written report, split to fit webhook limits.
    Discord embeds cap at ~4096 chars; Slack attachments at ~7600."""
    limit = 3800 if _backend() == "discord" else 7500
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
