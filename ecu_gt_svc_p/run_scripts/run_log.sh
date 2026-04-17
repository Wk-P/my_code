#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/run_background.log"

if [[ ! -f "$LOG_FILE" ]]; then
    echo "No log file found: $LOG_FILE"
    exit 1
fi

exec tail -f "$LOG_FILE"
