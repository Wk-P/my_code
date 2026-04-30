#!/usr/bin/env bash
# run-all-stop.sh — Stop run_all_sequential.sh and any running project background process.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$ROOT_DIR/run-all.pid"
PROJECTS=(ecu_eq_svc_p ecu_gt_svc_p ecu_lt_svc_p)

kill_pid() {
    local pid="$1" label="$2"
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" && echo "  killed $label (PID $pid)"
    else
        echo "  $label (PID $pid) already stopped"
    fi
}

echo "Stopping run_all_sequential ..."
if [[ -f "$PID_FILE" ]]; then
    kill_pid "$(cat "$PID_FILE")" "run_all_sequential"
    rm -f "$PID_FILE"
else
    echo "  run-all.pid not found, skipping"
fi

echo ""
echo "Stopping project background processes ..."
for proj in "${PROJECTS[@]}"; do
    proj_pid="$ROOT_DIR/$proj/run_scripts/run_background.pid"
    [[ -f "$proj_pid" ]] || continue
    kill_pid "$(cat "$proj_pid")" "$proj"
done

echo ""
echo "Done."
