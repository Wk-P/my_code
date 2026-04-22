#!/usr/bin/env bash
# Run the full training pipeline in the background via nohup.
# PID is written to run_scripts/run_background.pid
# Logs go to run_scripts/run_background.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_FILE="$SCRIPT_DIR/run_background.log"
PID_FILE="$SCRIPT_DIR/run_background.pid"

# Check if a previous run is still active
if [[ -f "$PID_FILE" ]]; then
    OLD_PID="$(cat "$PID_FILE")"
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Error: a previous run is still active (PID $OLD_PID)."
        echo "  Stop it with:  kill $OLD_PID"
        echo "  Or check logs: tail -f $LOG_FILE"
        exit 1
    else
        echo "Stale PID file found (PID $OLD_PID is no longer running). Overwriting."
        rm -f "$PID_FILE"
    fi
fi

# Launch
nohup bash -c "source '$SCRIPT_DIR/all_run_shell_script.sh' \"\$@\"" -- "$@" \
    > "$LOG_FILE" 2>&1 &
BG_PID=$!
echo "$BG_PID" > "$PID_FILE"

echo "Training started in background."
echo "  PID      : $BG_PID  (saved to $PID_FILE)"
echo "  Log      : $LOG_FILE"
echo ""
echo "Useful commands:"
echo "  tail -f $LOG_FILE          # follow live output"
echo "  kill \$(cat $PID_FILE)      # stop the run"
echo "  cat $PID_FILE              # check PID"
