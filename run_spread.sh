#!/usr/bin/env bash
# Runner for the defined-risk spread sleeve on its ISOLATED L3 paper account.
# Loads .env.spread so its keys win over .env (python-dotenv won't override env vars),
# and routes logs to spread.log so they never mix with the equity/options bot.
#
#   ./run_spread.sh --dry-run   # read-only: candidate spreads vs the live chain (safe)
#   ./run_spread.sh score       # print the validation scoreboard (+ Discord notify)
#   ./run_spread.sh             # foreground paper loop (normally systemd runs this)
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -f .env.spread ]]; then
  echo "Missing .env.spread — copy .env.spread.example and fill a SEPARATE L3 paper account's keys." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1091
source .env.spread
set +a
export BOT_LOG_FILE="$PWD/spread.log"

if [[ "${1:-}" == "score" ]]; then
  exec .venv/bin/python spread_forward.py --notify
fi
exec .venv/bin/python spread_bot.py "$@"
