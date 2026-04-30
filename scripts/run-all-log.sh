#!/usr/bin/env bash
# run-all-log.sh — Tail the accumulated run-all log.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tail -f "$ROOT_DIR/run-all-log.log"
