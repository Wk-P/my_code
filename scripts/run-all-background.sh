#!/usr/bin/env bash
# run-all-background.sh — Start run_all_sequential.sh in background.
# Logs to run-all-log.log, PID saved to run-all.pid.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$ROOT_DIR/run-all.pid"
LOG_FILE="$ROOT_DIR/run-all-log.log"

if [[ -f "$PID_FILE" ]]; then
    EXISTING=$(cat "$PID_FILE")
    if kill -0 "$EXISTING" 2>/dev/null; then
        echo "Already running (PID $EXISTING). Use run-all-stop.sh to stop first."
        exit 1
    fi
fi

nohup bash "$ROOT_DIR/scripts/run_all_sequential.sh" "$@" > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

echo "Started PID: $(cat "$PID_FILE")"
echo "Log:         $LOG_FILE"
echo ""
echo "Follow log:  tail -f run-all-log.log"
echo "Stop:        bash run-all-stop.sh"
