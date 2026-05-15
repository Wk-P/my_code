#!/bin/bash
set -e

SEEDS=(1 2)
EXPERIMENT_GROUPS=(eq lt gt)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

echo "Starting $((${#SEEDS[@]} * ${#EXPERIMENT_GROUPS[@]})) parallel experiments (${#EXPERIMENT_GROUPS[@]} groups × ${#SEEDS[@]} seeds) ..."
echo "Logs → $LOG_DIR"
echo ""

for GROUP in "${EXPERIMENT_GROUPS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        nohup python -u "$SCRIPT_DIR/run_experiment.py" --seed "$SEED" --group "$GROUP" \
            > "$LOG_DIR/${GROUP}_seed${SEED}.log" 2>&1 &
        echo "  [${GROUP} seed=${SEED}] PID=$!"
    done
done

echo ""
echo "All processes launched. Monitor with:"
echo "  tail -f $LOG_DIR/eq_seed1.log"
echo "  grep 'All done\|train.*ep=' $LOG_DIR/*.log"
