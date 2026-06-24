"""alerts.py — Discord webhook alerts: entries, exits, halts, daily summary."""
import logging

import requests

import config

log = logging.getLogger("alerts")

GREEN, RED, ORANGE, BLUE = 0x2ECC71, 0xE74C3C, 0xE67E22, 0x3498DB


def _send(title: str, description: str, color: int):
    if not config.DISCORD_WEBHOOK_URL:
        log.debug("No DISCORD_WEBHOOK_URL set — skipping alert: %s", title)
        return
    payload = {"embeds": [{"title": title, "description": description, "color": color}]}
    try:
        resp = requests.post(config.DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as exc:
        # Alerts must never crash the trading loop.
        log.warning("Discord alert failed: %s", exc)


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


def exit(pos, exit_price: float, pnl: float, reason: str):
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


def webhook_received(sig):
    """A TradingView alert passed auth/validation and was enqueued."""
    mode = "LIVE" if config.LIVE_MODE else "PAPER"
    extra = ("\n⚠️ _advisory only — live data unavailable, ML gating skipped_"
             if getattr(sig, "advisory", False) else "")
    _send(
        f"📡 [{mode}] Webhook alert — {sig.symbol} {sig.direction.upper()}",
        f"Strategy **{sig.strategy}** from TradingView. Queued for the entry "
        f"pipeline (risk limits + ML gate still apply).{extra}",
        BLUE,
    )


def webhook_trade(pick, reason: str):
    """A TradingView-sourced alert made it all the way through to a filled trade."""
    mode = "LIVE" if config.LIVE_MODE else "PAPER"
    _send(
        f"📡✅ [{mode}] Webhook → TRADE {pick.underlying} {pick.otype.upper()}",
        f"**{pick.option_symbol}** opened from a TradingView alert.\n"
        f"Reason: {reason}",
        PURPLE,
    )


PURPLE = 0x9B59B6


def ai_report(title: str, text: str):
    """Post a long AI-written report, split to fit Discord's embed limit."""
    chunks = []
    while len(text) > 3800:
        cut = text.rfind("\n", 0, 3800)   # split on a line break when possible
        if cut < 1000:
            cut = 3800
        chunks.append(text[:cut])
        text = text[cut:]
    chunks.append(text)
    for i, chunk in enumerate(chunks):
        _send(title if i == 0 else f"{title} (cont.)", chunk, PURPLE)
