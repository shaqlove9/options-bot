#!/usr/bin/env bash
# setup-cloudflare-tunnel.sh — finish wiring the iobot webhook to a public HTTPS
# URL via Cloudflare Tunnel. Run this AFTER the three manual prerequisites:
#
#   1. A (free) Cloudflare account.
#   2. A domain added to Cloudflare (nameservers pointed at Cloudflare). No domain?
#      Buy one (~$10/yr) — Cloudflare Registrar works and lands it in your account.
#   3. cloudflared logged in to that account:
#          cloudflared tunnel login
#      (opens a URL — authorize your domain in the browser; writes ~/.cloudflared/cert.pem)
#
# Then run:  bash deploy/setup-cloudflare-tunnel.sh webhook.yourdomain.com
#
# It creates the tunnel, writes ~/.cloudflared/config.yml, routes DNS, and installs
# a systemd service so the tunnel runs on boot. Idempotent-ish: safe to re-run.
set -euo pipefail

HOSTNAME="${1:-}"
TUNNEL_NAME="${2:-iobot-webhook}"
CFDIR="/home/ubuntu/.cloudflared"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -z "$HOSTNAME" ]; then
  echo "usage: bash deploy/setup-cloudflare-tunnel.sh <hostname> [tunnel-name]"
  echo "example: bash deploy/setup-cloudflare-tunnel.sh webhook.yourdomain.com"
  exit 1
fi
if [ ! -f "$CFDIR/cert.pem" ]; then
  echo "ERROR: $CFDIR/cert.pem not found — run 'cloudflared tunnel login' first."
  exit 1
fi

echo ">> Creating tunnel '$TUNNEL_NAME' (skipped if it already exists)..."
cloudflared tunnel create "$TUNNEL_NAME" 2>/dev/null || echo "   (tunnel already exists — reusing)"
UUID="$(cloudflared tunnel list --output json | python3 -c \
  "import sys,json; print(next(t['id'] for t in json.load(sys.stdin) if t['name']=='$TUNNEL_NAME'))")"
echo "   tunnel UUID: $UUID"

echo ">> Writing $CFDIR/config.yml ..."
sed -e "s/TUNNEL_UUID/$UUID/g" -e "s/HOSTNAME/$HOSTNAME/g" \
    "$REPO_DIR/deploy/cloudflared-config.yml.template" > "$CFDIR/config.yml"

echo ">> Routing DNS: $HOSTNAME -> $TUNNEL_NAME ..."
cloudflared tunnel route dns "$TUNNEL_NAME" "$HOSTNAME" || echo "   (DNS route may already exist — continuing)"

echo ">> Installing systemd service ..."
sudo cp "$REPO_DIR/deploy/cloudflared-iobot.service" /etc/systemd/system/cloudflared-iobot.service
sudo systemctl daemon-reload
sudo systemctl enable --now cloudflared-iobot.service
sleep 2
sudo systemctl is-active cloudflared-iobot.service

cat <<DONE

DONE ✅  Your webhook is now reachable at:
    https://$HOSTNAME/tv-webhook   (health: https://$HOSTNAME/health)

In TradingView, set the alert webhook URL to the line above and use this JSON body
(your secret is in ~/options-bot/.env as IOBOT_WEBHOOK_SECRET):
    {"secret":"<IOBOT_WEBHOOK_SECRET>","symbol":"{{ticker}}","direction":"call"}

Note: the receiver stays bound to 127.0.0.1 (IOBOT_WEBHOOK_HOST=127.0.0.1); only
the tunnel reaches it, so nothing is exposed except through Cloudflare's HTTPS edge.
DONE
