#!/usr/bin/env bash
# Daily forward-test check for the equity sleeve (driven by cron).
# Runs the live-vs-backtest scoreboard, appends a timestamped block to
# forward_test.log, refreshes status_forward.json for the dashboard, and lets
# forward_test.py --notify fire a Discord alert the day the verdict turns
# decisive (PASS/FAIL). Read-only/paper-only: never touches the running bots.
set -u
cd "$(dirname "$0")" || exit 1

LOG="forward_test.log"
{
  echo "================ $(date -u +'%Y-%m-%d %H:%M:%S UTC') ================"
  ./.venv/bin/python forward_test.py --days 60 --notify --json status_forward.json
  echo "(exit $?)"
  echo
} >> "$LOG" 2>&1
