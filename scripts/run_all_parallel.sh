#!/usr/bin/env bash
# 并行启动三个实验组，各自写入独立日志，全部完成后汇报结果。
# 首次调用时自动进入后台，立即返回终端控制权。

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_GROUPS=(ecu_eq_svc_p ecu_gt_svc_p ecu_lt_svc_p)
PID_FILE="$ROOT_DIR/pids/run_all_parallel.pid"
LOG_FILE="$ROOT_DIR/logs/run_all_parallel.log"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" >> "$LOG_FILE"; echo "[$(ts)] $*"; }

# ── 前台段：防重复检查 → 清日志 → 后台启动 → 退出 ─────────────────────────
if [[ "${_PARALLEL_BG:-0}" != "1" ]]; then
    if [[ -f "$PID_FILE" ]]; then
        OLD_PID="$(cat "$PID_FILE")"
        if kill -0 "$OLD_PID" 2>/dev/null; then
            echo "Error: already running (PID $OLD_PID). 先停止："
            echo "  bash scripts/stop_all_parallel.sh"
            exit 1
        else
            rm -f "$PID_FILE"
        fi
    fi

    > "$LOG_FILE"
    nohup env _PARALLEL_BG=1 bash "$0" "$@" >> "$LOG_FILE" 2>&1 &
    BG_PID=$!
    echo "$BG_PID" > "$PID_FILE"
    echo "实验已在后台启动 (PID $BG_PID)"
    echo "  进度日志: tail -f $LOG_FILE"
    echo "  各组日志: bash scripts/log_all_parallel.sh"
    echo "  一键终止: bash scripts/stop_all_parallel.sh"
    exit 0
fi

# ── 后台段：正式执行 ──────────────────────────────────────────────────────────
log "=== 并行实验启动 ==="
log "实验组: ${ENV_GROUPS[*]}"

declare -A GROUP_PIDS

for group in "${ENV_GROUPS[@]}"; do
    SCRIPT="$ROOT_DIR/$group/run_scripts/run_background.sh"
    if [[ ! -f "$SCRIPT" ]]; then
        log "ERROR: $SCRIPT 不存在，跳过 $group"
        continue
    fi
    log "[$group] 启动中 ..."
    bash "$SCRIPT" "$@" >> "$LOG_FILE" 2>&1
    BG_PID="$(cat "$ROOT_DIR/$group/run_scripts/run_background.pid")"
    GROUP_PIDS[$group]=$BG_PID
    log "[$group] PID=$BG_PID  日志: $ROOT_DIR/$group/run_scripts/run_background.log"
done

log "--- 全部已启动，等待完成 ---"
log "实时日志查看: tail -f $ROOT_DIR/ecu_{eq,gt,lt}_svc_p/run_scripts/run_background.log"

GLOBAL_START=$(date +%s)
declare -A START_TIMES
for group in "${!GROUP_PIDS[@]}"; do
    START_TIMES[$group]=$GLOBAL_START
done

ALL_DONE=0
while [[ $ALL_DONE -eq 0 ]]; do
    sleep 60
    ALL_DONE=1
    for group in "${!GROUP_PIDS[@]}"; do
        PID=${GROUP_PIDS[$group]}
        if kill -0 "$PID" 2>/dev/null; then
            ELAPSED=$(( $(date +%s) - START_TIMES[$group] ))
            MINS=$(( ELAPSED / 60 ))
            SECS=$(( ELAPSED % 60 ))
            log "[$group] 运行中 ... ${MINS}m${SECS}s"
            ALL_DONE=0
        fi
    done
done

log ""
log "=== 全部完成 ==="
TOTAL_ELAPSED=$(( $(date +%s) - GLOBAL_START ))
TOTAL_MINS=$(( TOTAL_ELAPSED / 60 ))
TOTAL_SECS=$(( TOTAL_ELAPSED % 60 ))

for group in "${!GROUP_PIDS[@]}"; do
    GROUP_LOG="$ROOT_DIR/$group/run_scripts/run_background.log"
    RESULT=$(grep -E "^\[timer\] main finished" "$GROUP_LOG" 2>/dev/null | tail -1 || echo "（未找到计时信息）")
    log "[$group] $RESULT"
done

log "总耗时: ${TOTAL_MINS}m${TOTAL_SECS}s"
log "=== 结束 ==="

rm -f "$PID_FILE"
