#!/usr/bin/env bash
# 用法: collect-kaggle.sh <kernel-slug> <dest-dir> —— 完赛则收日志/产物入匣并退0, 在跑退1(队列自鸣重试)
set -u; export PATH=/opt/miniconda3/envs/default/bin:$PATH
SLUG=$1; DEST=$2
ST=$(kaggle kernels status "$SLUG" 2>/dev/null | grep -o 'KernelWorkerStatus\.[A-Z]*')
case "$ST" in
  *COMPLETE*|*ERROR*|*CANCEL*)
    mkdir -p "$DEST"
    kaggle kernels output "$SLUG" -p "$DEST" >/dev/null 2>&1
    echo "收讫 $SLUG ($ST) → $DEST"; exit 0;;
  *) echo "在途 $SLUG ($ST)"; exit 1;;
esac
