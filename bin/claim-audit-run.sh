#!/usr/bin/env bash
# claim-audit-run.sh — 周检封装：跑「抽查三类」(claim-audit.py)，并把 append-only 日志补提交。
# 设计：
#   ① 只对 bin/claim-audit.log 一个路径做 commit（--only），不动他人暂存区，防并发会话互相吞账；
#   ② 无改动则跳过；③ push 失败不阻断，挂「GitHub 通讯障碍重试律」患者推后台，收兵挂账由该脚本自理。
# launchd: com.concept-space.claim-audit（周日 09:00）
set -u
REPO="/Users/mac/Programming/code-2026/Concept-Space-Sphere"
cd "$REPO" || exit 1
LOG="bin/claim-audit.log"

/usr/bin/python3 bin/claim-audit.py --n 2 "$@"
rc=$?

if [ -n "$(git status --porcelain -- "$LOG")" ]; then
  git add -- "$LOG"
  git -c user.name=math4mad -c user.email=math4mad@users.noreply.github.com \
      commit -q --only -m "chore(claim-audit): 周检 $(date '+%F') 自动追账" -- "$LOG" || true
  if ! GIT_TERMINAL_PROMPT=0 git push -q origin main 2>/dev/null; then
    nohup "$REPO/bin/git-push-patient.sh" "$REPO" >/dev/null 2>&1 &
  fi
fi
exit $rc
