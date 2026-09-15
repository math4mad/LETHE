#!/bin/bash

# 检查是否传入了参数
if [ -z "$1" ]; then
    echo " 用法: ./run_engine.sh <要观察的词汇>"
    echo " 示例: ./run_engine.sh 塑胶凳"
    echo " 提示: 如果不带参数运行，将执行默认测试用例"
    echo "----------------------------------------"
    python3 cognitive_engine.py
else
    python3 cognitive_engine.py "$1"
fi