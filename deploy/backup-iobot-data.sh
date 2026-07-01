#!/usr/bin/env bash
# Daily off-box backup of iobot_data/ AND /home/ubuntu/pumpbot -> orphan branch
# `iobot-data-backup` of the (private) options-bot repo. Installed via
# deploy/iobot-backup.{service,timer}.
#
# DBs are stored as FULL SQL TEXT DUMPS (not binary .db files): taken from a
# consistent point-in-time snapshot (sqlite backup API, safe while the bots are
# writing), they restore the whole DB, and as text they diff to small deltas so
# the repo doesn't bloat. Restore:
#   python -c "import sqlite3; sqlite3.connect('iobot.db').executescript(open('iobot.sql').read())"
#
# pumpbot has no GitHub remote, so its full git history rides along as a bundle:
#   git clone pumpbot/pumpbot-code.bundle pumpbot
#
# Secrets are NEVER backed up: .env files are copied with credential lines stripped.
set -euo pipefail

SRC=/home/ubuntu/options-bot/iobot_data
ENV_FILE=/home/ubuntu/options-bot/.env
PSRC=/home/ubuntu/pumpbot
DEST=/home/ubuntu/iobot-backup
PY=/home/ubuntu/options-bot/.venv/bin/python

cd "$DEST"

dump_db() {  # dump_db <src.db> <dest.sql> — consistent snapshot -> full SQL text dump
    "$PY" - "$1" "$2" <<'EOF'
import sqlite3, sys
src_path, dump_path = sys.argv[1], sys.argv[2]
src = sqlite3.connect(f"file:{src_path}?mode=ro", uri=True)
snap = sqlite3.connect(":memory:")
src.backup(snap)          # point-in-time copy, safe under concurrent writes
src.close()
with open(dump_path, "w") as f:
    for line in snap.iterdump():
        f.write(line + "\n")
snap.close()
EOF
}

# ── iobot ──────────────────────────────────────────────────────────────────────
dump_db "$SRC/iobot.db" "$DEST/iobot.sql"

cp "$SRC/governor_state.json" "$SRC/status.json" "$DEST/" 2>/dev/null || true
cp "$SRC/iobot.log" "$DEST/" 2>/dev/null || true
mkdir -p "$DEST/models"
cp -r "$SRC/models/." "$DEST/models/" 2>/dev/null || true

# Config tuning is valuable; secrets are not. Strip anything credential-shaped.
grep -viE '^[A-Z0-9_]*(KEY|SECRET|TOKEN|PASSWORD|WEBHOOK_URL)[A-Z0-9_]*=' \
    "$ENV_FILE" > "$DEST/env.sanitized"

# ── pumpbot (added 2026-07-01) ─────────────────────────────────────────────────
mkdir -p "$DEST/pumpbot"
dump_db "$PSRC/pumpbot.db" "$DEST/pumpbot/pumpbot.sql"
cp "$PSRC/watchlist.json" "$PSRC/config.yaml" "$DEST/pumpbot/" 2>/dev/null || true
# Whole-line credential strip: pumpbot's Helius key lives INSIDE the SOLANA_* URLs
# (?api-key=...), so match anywhere in the line, not just the var name.
grep -viE 'key|secret|token|password|webhook' "$PSRC/.env" \
    > "$DEST/pumpbot/env.sanitized" || true
# Full code history (pumpbot has no GitHub remote of its own).
git -C "$PSRC" bundle create "$DEST/pumpbot/pumpbot-code.bundle" --all --quiet

git add -A
if git diff --cached --quiet; then
    echo "backup: no changes since last run"
    exit 0
fi
git commit -q -m "iobot_data backup $(date -u +'%Y-%m-%d %H:%M UTC')"
git push -q -u origin iobot-data-backup
echo "backup: pushed $(date -u +'%Y-%m-%d %H:%M UTC')"
