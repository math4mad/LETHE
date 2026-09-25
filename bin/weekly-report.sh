#!/usr/bin/env bash
# bin/weekly-report.sh — 园周报: 每周日 21:30 汇总近 7 日两园账, 写进飞书「园周报」表 + 简报进袋
# 园律: 看板可镜像, 账本不搬家 —— 本表只读 git 仓与队列产物, 不新增事实源。
# 用法: weekly-report.sh [天数, 默认7] [--dry]
set -uo pipefail
export LC_ALL=C LANG=C
DAYS="${1:-7}"; DRY="${2:-}"
CS="$HOME/Programming/code-2026/Concept-Space-Sphere"
CH="$HOME/Programming/code-2026/chora"
AT="$HOME/Programming/code-2026/cora-atlas"
source ~/.config/pocket/feishu.env
PY=/opt/miniconda3/envs/default/bin/python
timeout_sh() { local t=$1; shift; "$@" & local p=$!; ( sleep "$t"; kill -9 $p 2>/dev/null ) & local w=$!; wait $p 2>/dev/null; local rc=$?; kill -9 $w 2>/dev/null; wait $w 2>/dev/null; return $rc; }
WEEK=$(date +%G-W%V)
SINCE="$DAYS days ago"
STAMP=$(date '+%F %T')

# --- 基线 commit (7 日前最后一个) ---
base() { b=$(git -C "$1" rev-list -1 --before="@$(($(date +%s) - DAYS*86400))" HEAD 2>/dev/null); [ -n "$b" ] || b=$(git -C "$1" rev-list --max-parents=0 HEAD | tail -1); echo "$b"; }
AB=$(base "$AT"); CB=$(base "$CH")

# --- ① 新碑与律 (LEDGER/GLOSSARY 新增行) ---
LEDGER_NEW=""; GLOSS_NEW=""
LEDGER_NEW=$(git -C "$AT" log --since="$SINCE" -p --format= -- LEDGER.md 2>/dev/null | grep '^+|' | grep -v '^+++' | awk -F'|' '{k=$2; sub(/^ +/,"",k); sub(/ +$/,"",k); if(k!="" && k!="件" && k!="---" && k!="案" && k!="石" && k!="#") print "· ["k"] "substr($0, index($0,"| "k" |")+length(k)+5, 80)}' | awk '!x[$0]++' | head -14)
GLOSS_NEW=$(git -C "$AT" log --since="$SINCE" -p --format= -- GLOSSARY.md 2>/dev/null | grep '^+[^+]' | awk -F'｜' '{gsub(/^ *\*?\*?/,"",$1); if(length($1)>2) print "· "substr($1,1,60)}' | awk '!x[$0]++' | head -10)
[ -n "$GLOSS_NEW" ] && LEDGER_NEW="$LEDGER_NEW
$GLOSS_NEW"
[ -z "$LEDGER_NEW" ] && LEDGER_NEW="(本周无新碑行)"

# --- ② 读数与战果 (近7日 report_*.json / 队列收货) ---
READS=$(git -C "$CH" log --since="$SINCE" --name-only --format= -- '*report_*.json' 2>/dev/null | LC_ALL=C sort -u | head -12 | while read -r f; do
  [ -f "$CH/$f" ] || continue
  s=$($PY -c "
import json,sys
try:
  r=json.load(open(sys.argv[1])); m=r.get('_meta',{})
  print(m.get('_selfcheck') or m.get('stage') or '')
except Exception: print('')" "$CH/$f" 2>/dev/null)
  echo "· $(basename "$(dirname "$f")")/$(basename "$f") ${s:+[$s]}"
done)
COLLECTS=$(ls -t "$CS/.queue/done"/*.json 2>/dev/null | while read -r j; do
  m=$(stat -f %m "$j"); now=$(date +%s); [ $(( (now-m)/86400 )) -lt "$DAYS" ] || continue
  $PY -c "import json;d=json.load(open('$j'));print('· ☑',d.get('id'),'—',(d.get('note') or '')[:40])'" 2>/dev/null
done | head -12)
READS="$READS
$COLLECTS"
[ -z "$(echo "$READS" | tr -d '[:space:]')" ] && READS="(近${DAYS}日无新读数)"

# --- ③ 哨账与通道 ---
JOBS=$(python3 "$CS/bin/queue.py" list 2>/dev/null | grep -c "^◻" || true)
DEAD=$(ls "$CS/.queue/dead"/*.json 2>/dev/null | wc -l | tr -d ' ')
PUSH=""
for r in "$CH" "$AT"; do
  n=$(git -C "$r" rev-list HEAD..origin/main --count 2>/dev/null || echo "?")
  [ "$n" = "0" ] || PUSH="$PUSH· $(basename "$r") 未推 $n 笔 "
done
PEND=$(ls "$CS/.pending-push"* 2>/dev/null | wc -l | tr -d ' ')
CHAN="· 在队哨 $JOBS · 死信 $DEAD · pending-push 档 $PEND"
[ -n "$PUSH" ] && CHAN="$CHAN
· 补推挂账: $PUSH" || CHAN="$CHAN
· 两园 HEAD==origin ✓"
REM=$(timeout_sh 100 "$CS/bin/remind.sh" read 2>/dev/null || true)
MS=$(printf '%s' "$REM" | grep -c "^◻\[MS\]" || true)
AP=$(printf '%s' "$REM" | grep -c "^◻ " || true)
CHAN="$CHAN
· 待办: Apple $AP · MS镜像 $MS (勾=批复)"

# --- ④ 下周候办 (在队 note + 近7日碑行里的"候") ---
TODO=$(python3 "$CS/bin/queue.py" list 2>/dev/null | grep "^◻" | sed 's/◻ 在队 /· /;s/ *tries=[0-9\/]* *next=[0-9:]*//' | head -8)
HOU=$(echo "$LEDGER_NEW" | grep -o "候[^ )】]*" | sort -u | head -8 | sed 's/^/· /')
TODO="$TODO
$HOU"
[ -z "$(echo "$TODO" | tr -d '[:space:]')" ] && TODO="(空)"

# --- 组卡 (变量走 env, 免 heredoc 展开与引号注入) ---
export W_STAMP="$STAMP" W_NUM="$WEEK" W_RANGE="近 ${DAYS} 日 (截至 $STAMP)" \
  W_LED="$LEDGER_NEW" W_RED="$READS" W_CHAN="$CHAN" W_TODO="$TODO"
CARD=$($PY - <<'PYEOF'
import json, os
g = os.environ.get
def clean(x):
    x = x or ""
    try:
        return x.encode("utf-8", "ignore").decode("utf-8", "ignore")
    except Exception:
        return x
print(json.dumps({
 "周号": clean(g("W_NUM")), "区间": clean(g("W_RANGE")),
 "新碑与律": clean(g("W_LED"))[:4000], "读数与战果": clean(g("W_RED"))[:4000],
 "哨账与通道": clean(g("W_CHAN"))[:2000], "下周候办": clean(g("W_TODO"))[:2000],
 "完稿戳": clean(g("W_STAMP"))}, ensure_ascii=False))
PYEOF
)

if [ "$DRY" = "--dry" ]; then echo "$CARD" | $PY -m json.tool; exit 0; fi

# --- 上表 (lark-cli 用户身份) + 进袋 (webhook 简报) ---
source ~/.config/pocket/feishu.env 2>/dev/null
TID="${GARDEN_WEEK_TABLE:-tblariSvec9c8GTo}"
lark-cli api POST "/open-apis/bitable/v1/apps/$GARDEN_BASE/tables/$TID/records" \
  --data "{\"fields\":$CARD}" --jq '{code,msg}' 2>&1 | tail -2
BRIEF="🗞 园周报 $WEEK (近${DAYS}日)
$(echo "$LEDGER_NEW" | head -4)
$(echo "$READS" | grep -v '^$' | head -3)
$CHAN"
"$CS/bin/pocket.sh" "$(echo "$BRIEF" | head -c 1200)"
echo "$(date '+%F %T') weekly-report done"
