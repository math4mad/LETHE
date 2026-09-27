#!/usr/bin/env bash
# Semantic Scholar 文献探针 · 四题落档 (429 则由队列退避重试)
set -u
OUT=/Users/mac/Programming/code-2026/Concept-Space-Sphere/corpus/ladder05/analysis/s2_litprobe.md
Q=("solitary pretend play preschool duration" "event scripts young children daily routine" "role play theory of mind preschool Taylor Carlson" "sensitive period language brain development Hensch")
{ echo "# S2 文献探针 · $(date '+%F %T')"
for i in 0 1 2 3; do
  echo; echo "## 题${i}: ${Q[$i]}"
  curl -s -m 15 --get "https://api.semanticscholar.org/graph/v1/paper/search" \
  sleep 12
done; } >> "$OUT" 2>&1
grep -q "⚠" "$OUT" && { echo "仍挡门"; exit 1; } || echo "S2-OK"
