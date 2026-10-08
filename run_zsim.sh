#!/bin/bash
# zsim —— 义素 MaxSim 探针（Zero 整数定点第二实现）
# 用法: ./run_zsim.sh <词>
#       ./run_zsim.sh            # 无参：列出概念空间与词库
# 需 zero 编译器，见 zsim/README.md 的安装说明。

set -eu
here=$(cd "$(dirname "$0")" && pwd)
cd "$here/zsim"

ZERO=${ZERO:-}
if [ -z "$ZERO" ]; then
    for c in "$here/.bin/zero" "$HOME/Programming/code-2026/zero-sandbox/.bin/zero" "$(command -v zero 2>/dev/null || true)"; do
        if [ -n "$c" ] && [ -x "$c" ]; then ZERO=$c; break; fi
    done
fi
if [ -z "$ZERO" ]; then
    echo "run_zsim: 未找到 zero 编译器。安装见 zsim/README.md" >&2
    exit 1
fi

if [ "$#" -eq 0 ]; then
    exec "$ZERO" run
fi
exec "$ZERO" run -- "$@"
