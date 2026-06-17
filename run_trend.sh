#!/usr/bin/env bash
# Monthly trend rebalance on an ISOLATED paper account.
#
# The trend sleeve MUST use its own Alpaca paper account, separate from the
# options/equity sleeves — they share symbols (SPY/QQQ/IWM) and would collide.
# .env.trend holds that account's keys; loaded here so they win over .env
# (python-dotenv does not override existing env vars).
#
#   cp .env.trend.example .env.trend     # fill in a SEPARATE paper account's keys
#   ./run_trend.sh                       # dry-run plan (safe)
#   ./run_trend.sh --execute             # submit the monthly paper rebalance
#
# Then it runs the forward scoreboard (--notify) so the verdict tracks itself.
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -f .env.trend ]]; then
  echo "Missing .env.trend — copy .env.trend.example and fill in a SEPARATE paper account's keys." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1091
source .env.trend
set +a

LOG="trend.log"
{
  echo "================ $(date -u +'%Y-%m-%d %H:%M:%S UTC') ================"
  .venv/bin/python trend_executor.py "$@"
  .venv/bin/python trend_forward.py --notify
} >> "$LOG" 2>&1
echo "trend rebalance + scoreboard done — see trend.log"
