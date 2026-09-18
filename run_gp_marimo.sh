#!/bin/bash
# 🌊 启动 GP 漂移诊断器 Marimo 控制台
# 用法: ./run_gp_marimo.sh              → v1 编辑器
#       ./run_gp_marimo.sh v2           → v2 编辑器（贝叶斯模型比较 + 谱截断）
#       ./run_gp_marimo.sh --run        → v1 只读 App 模式
#       ./run_gp_marimo.sh --run v2     → v2 只读 App 模式
MODE=edit
TARGET=gp_diagnoistoc_marimo.py
for arg in "$@"; do
    if [ "$arg" = "--run" ]; then
        MODE=run
    elif [ "$arg" = "v2" ]; then
        TARGET=gp_diagnoistoc_v2_marimo.py
    fi
done
python3 -m marimo "$MODE" "$TARGET"
