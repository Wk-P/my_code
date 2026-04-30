#!/usr/bin/env bash
# clean_all_results.sh — Wipe all experiment outputs across every ecu_* project.
#
# Deletes:
#   • problem_*/results/   — timestamped run dirs, ilp_cache.json, model zips
#   • run_scripts/results_combined/combined_results.png
#   • run_scripts/results_combined/problem_*/summary.csv
#   • run_scripts/results_combined/problem_*/ilp_cache.json
#   • run_scripts/results_combined/ilp_cache.json
#   • run_scripts/results_combined/problem_*/<timestamped dirs>
#   • run_scripts/run_background.{log,pid}
#
# Keeps:
#   • Directory skeletons (results/, results_combined/problem_*/)
#   • Source files (combined.py, all_run_shell_script.sh, etc.)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECTS=(ecu_eq_svc_p ecu_gt_svc_p ecu_lt_svc_p)

TOTAL_DELETED=0

rel() { echo "${1#"$ROOT_DIR"/}"; }   # relative path for display

rm_file() {
    [[ -f "$1" ]] || return 0
    echo "  rm  $(rel "$1")"
    rm -f "$1"
    TOTAL_DELETED=$(( TOTAL_DELETED + 1 ))
}

rm_dir_contents() {
    # Delete entries inside $1 whose names match the shell glob $2
    local dir="$1" glob="$2"
    [[ -d "$dir" ]] || return 0
    local count=0
    while IFS= read -r -d '' entry; do
        echo "  rm  $(rel "$entry")"
        rm -rf "$entry"
        count=$(( count + 1 ))
    done < <(find "$dir" -maxdepth 1 -name "$glob" -print0 2>/dev/null)
    TOTAL_DELETED=$(( TOTAL_DELETED + count ))
}

for proj in "${PROJECTS[@]}"; do
    proj_dir="$ROOT_DIR/$proj"
    [[ -d "$proj_dir" ]] || { echo "WARNING: $proj_dir not found, skipping."; continue; }

    echo ""
    echo "[$proj]"

    # ── Every problem*/results/ (problem2_ilp, problem3_ppo, problem_dqn, …) ──
    for results_dir in "$proj_dir"/problem*/results; do
        [[ -d "$results_dir" ]] || continue
        rm_dir_contents "$results_dir" "[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_*"
        rm_dir_contents "$results_dir" "result_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_*"
        rm_file         "$results_dir/ilp_cache.json"
        rm_dir_contents "$results_dir" "*.zip"
    done

    # ── run_scripts/results_combined/ ─────────────────────────────────────────
    combined_dir="$proj_dir/run_scripts/results_combined"
    if [[ -d "$combined_dir" ]]; then
        rm_file "$combined_dir/combined_results.png"
        rm_file "$combined_dir/ilp_cache.json"
        for sub_dir in "$combined_dir"/problem*/; do
            [[ -d "$sub_dir" ]] || continue
            rm_file         "$sub_dir/summary.csv"
            rm_file         "$sub_dir/ilp_cache.json"
            rm_dir_contents "$sub_dir" "result_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_*"
        done
    fi

    # ── run_scripts background log / pid ──────────────────────────────────────
    rm_file "$proj_dir/run_scripts/run_background.log"
    rm_file "$proj_dir/run_scripts/run_background.pid"
done

echo ""
echo "Done. Total items removed: $TOTAL_DELETED"
