#!/usr/bin/env bash
# 前端改动后跑这个：build 静态资源 + 重启面板服务（后端已开 --reload，会自动生效，重启是保险）。
set -euo pipefail
cd "$(dirname "$0")/frontend"
npm run build
systemctl --user restart my-code-panel.service
echo "deployed."
