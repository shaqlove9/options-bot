"""Slack webhook backend for alerts."""
import json
import logging

import requests

import config

log = logging.getLogger("alerts.slack")

# Map Discord integer colors to Slack hex strings
_COLORS = {
    0x2ECC71: "#2ecc71",  # green
    0xE74C3C: "#e74c3c",  # red
    0xE67E22: "#e67e22",  # orange
    0x3498DB: "#3498db",  # blue
    0x9B59B6: "#9b59b6",  # purple
}


def is_configured() -> bool:
    return bool(config.SLACK_WEBHOOK_URL)


def send(title: str, description: str, color: int):
    """Send a Slack attachment via webhook. Raises on HTTP errors."""
    # Convert markdown bold **text** to Slack bold *text*
    text = description.replace("**", "*")
    slack_color = _COLORS.get(color, "#3498db")
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


def max_chunk_size() -> int:
    """Max text length per attachment (Slack caps at ~7600)."""
    return 7500
