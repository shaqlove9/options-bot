"""Discord webhook backend for alerts."""
import logging

import requests

import config

log = logging.getLogger("alerts.discord")

# Embed color codes
GREEN, RED, ORANGE, BLUE, PURPLE = 0x2ECC71, 0xE74C3C, 0xE67E22, 0x3498DB, 0x9B59B6


def is_configured() -> bool:
    return bool(config.DISCORD_WEBHOOK_URL)


def send(title: str, description: str, color: int):
    """Send a Discord embed via webhook. Raises on HTTP errors."""
    payload = {"embeds": [{"title": title, "description": description, "color": color}]}
    resp = requests.post(config.DISCORD_WEBHOOK_URL, json=payload, timeout=10)
    resp.raise_for_status()


def max_chunk_size() -> int:
    """Max text length per embed (Discord caps at ~4096)."""
    return 3800
