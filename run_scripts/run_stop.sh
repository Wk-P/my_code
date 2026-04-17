#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/run_background.pid"

if [[ ! -f "$PID_FILE" ]]; then
    echo "No PID file found. Nothing to stop."
    exit 0
fi

PID="$(cat "$PID_FILE")"

if kill -0 "$PID" 2>/dev/null; then
    kill "$PID"
    echo "Stopped (PID $PID)."
    rm -f "$PID_FILE"
else
    echo "Process $PID is not running. Cleaning up stale PID file."
    rm -f "$PID_FILE"
fi
