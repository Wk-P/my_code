#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$SCRIPT_DIR/.venv/bin/python"

# ── argument parsing ──────────────────────────────────────────────────────────
NUM_SEEDS=""
EPISODES=""

usage() {
    echo "Usage: $0 --sum-seeds <n> --eps <m>"
    echo "  --sum-seeds <n>   number of seeds to randomly sample from [0, 50]"
    echo "  --eps <m>         target episodes per model per seed"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sum-seeds) NUM_SEEDS="$2"; shift 2 ;;
        --eps)       EPISODES="$2";  shift 2 ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

[[ -z "$NUM_SEEDS" ]] && { echo "Error: --sum-seeds is required"; usage; }
[[ -z "$EPISODES" ]]  && { echo "Error: --eps is required";       usage; }

if ! [[ "$NUM_SEEDS" =~ ^[1-9][0-9]*$ ]] || (( NUM_SEEDS > 51 )); then
    echo "Error: --sum-seeds must be an integer in [1, 51]"
    exit 1
fi
if ! [[ "$EPISODES" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --eps must be a positive integer"
    exit 1
fi

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
echo "  python:             $PYTHON"
echo "============================================================"
echo ""

# ── launch via run_experiment.py --group all (EQ/LT/GT parallel) ─────────────
nohup $PYTHON "$SCRIPT_DIR/run_experiment.py" \
    --group all \
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
echo "  tail -f reports/ecu_exp_*/EQ/group.log"
echo "  tail -f reports/ecu_exp_*/LT/group.log"
echo "  tail -f reports/ecu_exp_*/GT/group.log"
