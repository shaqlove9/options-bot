#!/usr/bin/env bash
# Daily off-box backup of iobot_data/ -> orphan branch `iobot-data-backup` of the
# (private) options-bot repo. Installed via deploy/iobot-backup.{service,timer}.
#
# The DB is stored as a FULL SQL TEXT DUMP (not the binary .db): it is taken from
# a consistent point-in-time snapshot (sqlite backup API, safe while the bot is
# writing), it restores the whole DB, and as text it diffs to small deltas so the
# repo doesn't bloat. Restore:
#   .venv/bin/python -c "import sqlite3; sqlite3.connect('iobot.db').executescript(open('iobot.sql').read())"
#
# Secrets are NEVER backed up: .env is copied with key/secret/webhook lines stripped.
set -euo pipefail

SRC=/home/ubuntu/options-bot/iobot_data
ENV_FILE=/home/ubuntu/options-bot/.env
DEST=/home/ubuntu/iobot-backup
PY=/home/ubuntu/options-bot/.venv/bin/python

cd "$DEST"

# Consistent snapshot -> full SQL text dump.
"$PY" - "$SRC/iobot.db" "$DEST/iobot.sql" <<'EOF'
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

cp "$SRC/governor_state.json" "$SRC/status.json" "$DEST/" 2>/dev/null || true
cp "$SRC/iobot.log" "$DEST/" 2>/dev/null || true
mkdir -p "$DEST/models"
cp -r "$SRC/models/." "$DEST/models/" 2>/dev/null || true

# Config tuning is valuable; secrets are not. Strip anything credential-shaped.
grep -viE '^[A-Z0-9_]*(KEY|SECRET|TOKEN|PASSWORD|WEBHOOK_URL)[A-Z0-9_]*=' \
    "$ENV_FILE" > "$DEST/env.sanitized"

git add -A
if git diff --cached --quiet; then
    echo "backup: no changes since last run"
    exit 0
fi
git commit -q -m "iobot_data backup $(date -u +'%Y-%m-%d %H:%M UTC')"
git push -q -u origin iobot-data-backup
echo "backup: pushed $(date -u +'%Y-%m-%d %H:%M UTC')"
