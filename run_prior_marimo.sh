#!/bin/bash
# ⏳ 启动 时间轴的指针 · 先验指针 Marimo 控制台
# 用法: ./run_prior_marimo.sh        → 编辑器
#       ./run_prior_marimo.sh --run  → 只读 App 模式
MODE=edit
[ "$1" = "--run" ] && MODE=run
python3 -m marimo "$MODE" prior_pointer_marimo.py
