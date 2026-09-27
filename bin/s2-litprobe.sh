#!/usr/bin/env bash
# Semantic Scholar 文献探针 · 四题落档 (429 则由队列退避重试)
set -u
OUT=/Users/mac/Programming/code-2026/Concept-Space-Sphere/corpus/ladder05/analysis/s2_litprobe.md
Q=("solitary pretend play preschool duration" "event scripts young children daily routine" "role play theory of mind preschool Taylor Carlson" "sensitive period language brain development Hensch")
{ echo "# S2 文献探针 · $(date '+%F %T')"
for i in 0 1 2 3; do
  echo; echo "## 题${i}: ${Q[$i]}"
  curl -s -m 15 --get "https://api.semanticscholar.org/graph/v1/paper/search" \
    --data-urlencode "query=${Q[$i]}" -d limit=5 -d "fields=title,year,citationCount,externalIds" | python3 -c '
import json,sys
d=json.load(sys.stdin)
if "data" not in d: sys.exit(f"  ⚠ {str(d)[:80]}")
for p in d["data"]:
    doi=(p.get("externalIds") or {}).get("DOI","")
    print(f"  {p.get(chr(121)+chr(101)+chr(97)+chr(114),chr(8212))} 引{p.get(chr(99)+chr(105)+chr(116)+chr(97)+chr(116)+chr(105)+chr(111)+chr(110)+chr(67)+chr(111)+chr(117)+chr(110)+chr(116),0):5d}  {p.get(chr(116)+chr(105)+chr(116)+chr(108)+chr(101),\"\")[:88]}")
    if doi: print("       DOI:",doi)
'
  sleep 12
done; } >> "$OUT" 2>&1
grep -q "⚠" "$OUT" && { echo "仍挡门"; exit 1; } || echo "S2-OK"
