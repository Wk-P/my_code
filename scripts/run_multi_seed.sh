#!/usr/bin/env bash
# 多种子重复实验：每轮用不同种子完整训练+评估，结束后汇总平均值。
#
# Usage:
#   bash scripts/run_multi_seed.sh                          # 默认 3 个种子
#   bash scripts/run_multi_seed.sh --seeds "42 123 456 789 1024"
#   bash scripts/run_multi_seed.sh --seeds "42 123" --total-timesteps 50000
#
# 结果归档到: logs/multi_seed_results/seed_<N>/<group>/<problem>.csv
# 汇总报告在: logs/multi_seed_results/aggregate_summary.csv

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE_DIR="$ROOT_DIR/logs/multi_seed_results"
LOG_FILE="$ARCHIVE_DIR/run_multi_seed.log"

# ── 默认参数 ──────────────────────────────────────────────────────────────────
SEEDS=(42 123 456)
EXTRA_ARGS=()

# ── 解析参数 ──────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seeds)
            IFS=' ' read -r -a SEEDS <<< "$2"
            shift 2
            ;;
        --total-timesteps)
            EXTRA_ARGS+=(--total-timesteps "$2")
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

ENV_GROUPS=(ecu_eq_svc_p ecu_gt_svc_p ecu_lt_svc_p)
PROBLEM_DIRS=(problem3_ppo problem4_ppo_mask problem5_ppo_lagrangian problem6_ppo_opt problem_dqn problem_ddqn)

ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG_FILE"; }

mkdir -p "$ARCHIVE_DIR"
> "$LOG_FILE"

log "=== 多种子实验启动 ==="
log "种子列表: ${SEEDS[*]}"
log "额外参数: ${EXTRA_ARGS[*]:-（无）}"
log "归档目录: $ARCHIVE_DIR"

GLOBAL_START=$(date +%s)
TOTAL_RUNS=${#SEEDS[@]}
RUN_IDX=0

for seed in "${SEEDS[@]}"; do
    RUN_IDX=$(( RUN_IDX + 1 ))
    log ""
    log "━━━ 第 ${RUN_IDX}/${TOTAL_RUNS} 轮  SEED=${seed} ━━━"
    RUN_START=$(date +%s)

    # ── 1. 清理上一轮结果 ────────────────────────────────────────────────────
    log "[seed=${seed}] 清理上轮结果 ..."
    bash "$ROOT_DIR/scripts/clean_all_results.sh" >> "$LOG_FILE" 2>&1

    # ── 2. 设置种子并启动并行训练 ────────────────────────────────────────────
    log "[seed=${seed}] 启动并行训练 ..."
    export TRAIN_SEED=$seed
    bash "$ROOT_DIR/scripts/run_all_parallel.sh" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

    # ── 3. 等待所有组完成（轮询 PID 文件消失）────────────────────────────────
    sleep 5   # 确保 PID 文件已写入
    WAIT_SECS=0
    while [[ -f "$ROOT_DIR/pids/run_all_parallel.pid" ]]; do
        sleep 60
        WAIT_SECS=$(( WAIT_SECS + 60 ))
        MINS=$(( WAIT_SECS / 60 ))
        log "[seed=${seed}] 等待完成 ... ${MINS}m"
    done

    RUN_ELAPSED=$(( $(date +%s) - RUN_START ))
    RUN_MINS=$(( RUN_ELAPSED / 60 ))
    RUN_SECS=$(( RUN_ELAPSED % 60 ))
    log "[seed=${seed}] 训练完成，耗时 ${RUN_MINS}m${RUN_SECS}s"

    # ── 4. 归档 summary.csv ──────────────────────────────────────────────────
    log "[seed=${seed}] 归档结果 ..."
    for group in "${ENV_GROUPS[@]}"; do
        RUN_GROUP_DIR="$ARCHIVE_DIR/seed_${seed}/${group}"
        mkdir -p "$RUN_GROUP_DIR"
        for problem in "${PROBLEM_DIRS[@]}"; do
            SRC="$ROOT_DIR/$group/run_scripts/results_combined/${problem}/summary.csv"
            if [[ -f "$SRC" ]]; then
                cp "$SRC" "$RUN_GROUP_DIR/${problem}.csv"
                log "  archived: seed_${seed}/${group}/${problem}.csv"
            else
                log "  WARNING: missing $SRC"
            fi
        done
    done
done

TOTAL_ELAPSED=$(( $(date +%s) - GLOBAL_START ))
TOTAL_MINS=$(( TOTAL_ELAPSED / 60 ))
TOTAL_SECS=$(( TOTAL_ELAPSED % 60 ))
log ""
log "=== 全部 ${TOTAL_RUNS} 轮完成，总耗时 ${TOTAL_MINS}m${TOTAL_SECS}s ==="
log "开始汇总 ..."

# ── 5. 汇总平均值 ─────────────────────────────────────────────────────────────
PYTHON_PATH="$ROOT_DIR/.venv/bin/python"
"$PYTHON_PATH" "$ROOT_DIR/scripts/aggregate_results.py" \
    --archive-dir "$ARCHIVE_DIR" \
    --seeds "${SEEDS[@]}" \
    2>&1 | tee -a "$LOG_FILE"

log "=== 完成 ==="
log "汇总报告: $ARCHIVE_DIR/aggregate_summary.csv"
