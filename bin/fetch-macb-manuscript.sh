#!/usr/bin/env bash
# 捞 MacB 上 2024 石子案原稿候选清单 (只列名不取件)
set -u
L="/Users/mac/Programming/code-2026/Concept-Space-Sphere/external/macb-manuscript-list.txt"
ssh -o BatchMode=yes -o ConnectTimeout=8 macb 'find ~/Documents ~/Desktop ~/Programming -maxdepth 5 \( -iname "*.qmd" -o -iname "*.pptx" -o -iname "*.docx" -o -iname "*.md" \) 2>/dev/null | grep -iE "石子|函数|向量|辅导|对应|math"' > "$L" 2>/dev/null
test -s "$L" && { echo "捞到 $(wc -l < "$L" | tr -d " ") 条候选"; exit 0; }
echo "MacB 未归或无命中"; exit 1
