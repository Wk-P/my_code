#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$SCRIPT_DIR/.venv/bin/python"

# ── argument parsing ──────────────────────────────────────────────────────────
NUM_SEEDS=""
EPISODES=""
EXP_GROUPS=()

usage() {
    echo "Usage: $0 --sum-seeds <n> --eps <m> --exp <group...>"
    echo "  --sum-seeds <n>        number of seeds to randomly sample from [0, 50]"
    echo "  --eps <m>              target episodes per model per seed"
    echo "  --exp <group...>       groups to run: lt, gt, eq, or all (must be alone)"
    echo "                         examples: --exp all"
    echo "                                   --exp lt eq"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sum-seeds) NUM_SEEDS="$2"; shift 2 ;;
        --eps)       EPISODES="$2";  shift 2 ;;
        --exp)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                EXP_GROUPS+=("$1")
                shift
            done
            ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

[[ -z "$NUM_SEEDS" ]]        && { echo "Error: --sum-seeds is required"; usage; }
[[ -z "$EPISODES" ]]         && { echo "Error: --eps is required";       usage; }
[[ ${#EXP_GROUPS[@]} -eq 0 ]] && { echo "Error: --exp is required";      usage; }

if ! [[ "$NUM_SEEDS" =~ ^[1-9][0-9]*$ ]] || (( NUM_SEEDS > 51 )); then
    echo "Error: --sum-seeds must be an integer in [1, 51]"
    exit 1
fi
if ! [[ "$EPISODES" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --eps must be a positive integer"
    exit 1
fi

# validate --exp values and "all" exclusivity
for g in "${EXP_GROUPS[@]}"; do
    case "$g" in
        lt|gt|eq|all) ;;
        *) echo "Error: unknown group '$g' (valid: lt, gt, eq, all)"; exit 1 ;;
    esac
done

for g in "${EXP_GROUPS[@]}"; do
    if [[ "$g" == "all" && ${#EXP_GROUPS[@]} -gt 1 ]]; then
        echo "Error: 'all' must appear alone (cannot be combined with other groups)"
        exit 1
    fi
done

# resolve "all" → "lt gt eq"
if [[ "${EXP_GROUPS[*]}" == "all" ]]; then
    EXP_GROUPS=(lt gt eq)
fi

# build --group argument for run_experiment.py
GROUP_ARG="${EXP_GROUPS[*]}"   # e.g. "lt eq" or "lt gt eq"

# ── sample seeds from [0, 50] ─────────────────────────────────────────────────
SEEDS=($($PYTHON -c "
import random
seeds = random.sample(range(51), $NUM_SEEDS)
print(*seeds)
"))

echo "============================================================"
echo "  run_all.sh"
echo "  seeds ($NUM_SEEDS): ${SEEDS[*]}"
echo "  episodes:           $EPISODES"
echo "  groups:             ${EXP_GROUPS[*]}"
echo "  python:             $PYTHON"
echo "============================================================"
echo ""

# ── launch via run_experiment.py ──────────────────────────────────────────────
nohup $PYTHON "$SCRIPT_DIR/run_experiment.py" \
    --group ${EXP_GROUPS[@]} \
    --seed "${SEEDS[@]}" \
    --episodes "$EPISODES" \
    --name ecu_exp \
    > /tmp/run_all_launcher.log 2>&1 &

LAUNCHER_PID=$!
echo "Launcher PID=$LAUNCHER_PID  (EQ / LT / GT will spawn as subprocesses)"
echo ""

# wait briefly for subprocesses to appear, then show their PIDs
sleep 3
echo "Active experiment processes:"
ps aux | grep run_experiment | grep -v grep | awk '{printf "  PID=%-8s %s %s %s %s %s %s\n", $2, $11, $13, $14, $15, $16, $17}' || true
echo ""
echo "Monitor logs:"
echo "  tail -f /tmp/run_all_launcher.log"
for g in "${EXP_GROUPS[@]}"; do
    echo "  tail -f reports/ecu_exp_*/$(echo "$g" | tr '[:lower:]' '[:upper:]')/group.log"
done
