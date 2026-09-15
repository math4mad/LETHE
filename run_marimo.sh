#!/bin/bash
# 🎹 启动 Marimo 认知控制台（编辑模式）
# 用法: ./run_marimo.sh          → 编辑器打开
#       ./run_marimo.sh --run    → 只读 App 模式
if [ "$1" = "--run" ]; then
    python3 -m marimo run cognitive_engine_marimo.py
else
    python3 -m marimo edit cognitive_engine_marimo.py
fi
