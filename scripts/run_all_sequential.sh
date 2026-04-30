#!/usr/bin/env bash
# run_all_sequential.sh — Launch every ecu_* project via its run_background.sh,
# one at a time. Each project starts only after the previous one finishes.
#
# Usage:
#   bash run_all_sequential.sh [-- <extra args forwarded to each run_background.sh>]
#
# Example (quick smoke test):
#   bash run_all_sequential.sh -- --total-timesteps 1000

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECTS=(ecu_eq_svc_p ecu_gt_svc_p ecu_lt_svc_p)
PASS_ARGS=("$@")

POLL_INTERVAL=30   # seconds between PID liveness checks

info()    { echo ""; echo "================================================================"; echo "  $*"; echo "================================================================"; }
log()     { echo "  [$(date '+%H:%M:%S')] $*"; }
elapsed() { printf '%dm%ds' $(( ($1)/60 )) $(( ($1)%60 )); }

GRAND_START=$(date +%s)

for proj in "${PROJECTS[@]}"; do
    proj_dir="$ROOT_DIR/$proj"
    run_bg="$proj_dir/run_scripts/run_background.sh"
    pid_file="$proj_dir/run_scripts/run_background.pid"
    log_file="$proj_dir/run_scripts/run_background.log"

    [[ -d "$proj_dir" ]]  || { echo "WARNING: $proj not found, skipping."; continue; }
    [[ -f "$run_bg"    ]]  || { echo "WARNING: $run_bg not found, skipping."; continue; }

    info "Starting: $proj"
    log "Launching $run_bg ..."

    bash "$run_bg" "${PASS_ARGS[@]+"${PASS_ARGS[@]}"}"

    # Wait for the background process to appear in the PID file
    for _ in {1..10}; do
        [[ -f "$pid_file" ]] && break
        sleep 1
    done

    if [[ ! -f "$pid_file" ]]; then
        echo "ERROR: PID file not created at $pid_file — cannot wait. Aborting." >&2
        exit 1
    fi

    BG_PID=$(cat "$pid_file")
    log "Background PID: $BG_PID  |  Log: $log_file"
    log "Waiting for $proj to finish ..."

    T_START=$(date +%s)
    while kill -0 "$BG_PID" 2>/dev/null; do
        sleep "$POLL_INTERVAL"
        ELAPSED=$(( $(date +%s) - T_START ))
        log "$proj still running ... $(elapsed $ELAPSED) elapsed"
    done

    ELAPSED=$(( $(date +%s) - T_START ))
    log "$proj finished in $(elapsed $ELAPSED)."
done

GRAND_ELAPSED=$(( $(date +%s) - GRAND_START ))
info "All projects done in $(elapsed $GRAND_ELAPSED)."
