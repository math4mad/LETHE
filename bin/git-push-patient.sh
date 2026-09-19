#!/usr/bin/env bash
# 依「GitHub 通讯障碍重试律」推仓: 3 次快试 → 10/20/40 分钟各一试; 败则挂账。
set -u
D="${1:?repo dir}"
cd "$D" || exit 1
LOGP="$D/.pending-push"; touch "$LOGP"
export GIT_TERMINAL_PROMPT=0
try() { if git push -q origin main 2>/tmp/gpp.err; then echo "$(date '+%F %T') PUSHED ok" | tee -a "$LOGP"; rm -f "$LOGP"; return 0; else echo "$(date '+%F %T') fail: $(tail -1 /tmp/gpp.err | cut -c1-90)" >> "$LOGP"; return 1; fi }
for i in 1 2 3; do try && exit 0; sleep 20; done
for m in 10 20 40; do sleep $((m*60)); try && exit 0; done
echo "$(date '+%F %T') 六试皆败 — 收兵挂账, 下次会话开场先补推" | tee -a "$LOGP"; exit 1
