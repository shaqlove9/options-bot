# iobot_data backup branch

Automated daily backup of `/home/ubuntu/options-bot/iobot_data/` from the prod
VM (orphan branch — shares no history with the code). Pushed by
`deploy/backup-iobot-data.sh` via the `iobot-backup.timer` systemd timer
(daily 21:15 UTC, after US market close).

| file | what |
|---|---|
| `iobot.sql` | FULL SQL text dump of `iobot.db` (trades, features, signals, rejects, shadow, positions) from a consistent point-in-time snapshot |
| `governor_state.json` | sleeve equity / peak / halt state |
| `status.json` | last engine heartbeat |
| `iobot.log` | rotating prod log (latest 2MB window) |
| `models/` | trained meta-model artifacts, if any |
| `env.sanitized` | `.env` with all key/secret/token/webhook lines stripped — config tuning only, NO credentials |

## Restore

```bash
cd options-bot
git fetch origin iobot-data-backup && git worktree add /tmp/restore origin/iobot-data-backup
mkdir -p iobot_data
.venv/bin/python -c "import sqlite3; sqlite3.connect('iobot_data/iobot.db').executescript(open('/tmp/restore/iobot.sql').read())"
cp /tmp/restore/governor_state.json iobot_data/
cp -r /tmp/restore/models iobot_data/
# then recreate .env from env.sanitized + fresh Alpaca/Discord/Anthropic credentials
```
