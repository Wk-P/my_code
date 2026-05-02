#!/usr/bin/env bash
# 终止所有并行实验组及顶层调度进程。

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_GROUPS=(ecu_eq_svc_p ecu_gt_svc_p ecu_lt_svc_p)
PARALLEL_PID_FILE="$ROOT_DIR/pids/run_all_parallel.pid"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

# 终止顶层调度进程
if [[ -f "$PARALLEL_PID_FILE" ]]; then
    PID="$(cat "$PARALLEL_PID_FILE")"
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID"
        echo "[$(ts)] 调度进程已终止 (PID $PID)"
    else
        echo "[$(ts)] 调度进程已不存在 (PID $PID)"
    fi
    rm -f "$PARALLEL_PID_FILE"
else
    echo "[$(ts)] 未找到调度 PID 文件，跳过"
fi

# 终止各组后台进程
for group in "${ENV_GROUPS[@]}"; do
    PID_FILE="$ROOT_DIR/$group/run_scripts/run_background.pid"
    if [[ ! -f "$PID_FILE" ]]; then
        echo "[$(ts)] [$group] 无 PID 文件，跳过"
        continue
    fi
    PID="$(cat "$PID_FILE")"
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID"
        echo "[$(ts)] [$group] 已终止 (PID $PID)"
    else
        echo "[$(ts)] [$group] 进程已不存在 (PID $PID)"
    fi
    rm -f "$PID_FILE"
done

echo "[$(ts)] 全部终止完成"
