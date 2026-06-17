#!/usr/bin/env bash
# Launch the EQUITY-sleeve paper instance from .env.equity.
# Loads .env.equity into the environment (these win over .env, since python-dotenv
# does not override existing env vars), then runs main.py in equity mode.
#
#   cp .env.equity.example .env.equity   # fill in a SEPARATE paper account's keys
#   ./run_equity.sh
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -f .env.equity ]]; then
  echo "Missing .env.equity — copy .env.equity.example and fill in a separate paper account's keys." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1091
source .env.equity
set +a

if [[ "${INSTRUMENT:-}" != "equity" ]]; then
  echo "Refusing to start: INSTRUMENT is '${INSTRUMENT:-unset}', expected 'equity'." >&2
  exit 1
fi

echo "Starting EQUITY sleeve (paper) — status: ${STATUS_FILE:-status.json}"
exec .venv/bin/python main.py
