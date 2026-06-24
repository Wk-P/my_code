#!/usr/bin/env bash
# 同时跟踪三个实验组的实时日志。

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

tail -f -n 100 \
    "$ROOT_DIR/ecu_eq_svc_p/run_scripts/run_background.log" \
    "$ROOT_DIR/ecu_gt_svc_p/run_scripts/run_background.log" \
    "$ROOT_DIR/ecu_lt_svc_p/run_scripts/run_background.log"
