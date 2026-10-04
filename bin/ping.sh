#!/usr/bin/env bash
# bin/ping.sh — 园 ⇄ 主人 iPhone 的 iMessage 道 (双向: 能发, 也能读回)
#
# 三条道分工 (一账一道):
#   bin/remind.sh : 园里的日程与决策点 → iCloud 提醒事项 (可勾选 = 批复)
#   bin/bark.sh   : 「agent 正在等你」的即时一响 → Bark / APNs (只出不进, 可 ttl 自焚)
#   bin/ping.sh   : iMessage —— 唯一**能回信**的一条 (发出去, 主人的回复可由 read/new 拾回)
#
# 前置 (一次即可):
#   · 信息.app 已登录 iMessage (设置里不再转圈)
#   · 给 Pi/终端开「完全磁盘访问」并重启宿主 app —— 读 chat.db 需要 (见 skill diagnose-imessage-delivery)
#
# 用法:
#   bin/ping.sh send "正文"      # 发一条, 随后立即回读 chat.db 报告投递状态
#   bin/ping.sh read [N]         # 看最近 N 条对方来讯 (默认 10)
#   bin/ping.sh new              # 只看上次读过之后新增的来讯 (水位线)
#   bin/ping.sh status           # 体检: chat.db 可读? iMessage 在? 收件人是谁?
#
# 收件人: $PING_TO, 否则取 ~/.zshrc 里的 PING_TO="..." (不进仓)
# 水位线: ${PING_STATE:-$HOME/.local/state/lola-ping.json}
set -euo pipefail

CHATDB="$HOME/Library/Messages/chat.db"
STATE="${PING_STATE:-$HOME/.local/state/lola-ping.json}"
SLEEP_VERIFY="${PING_SLEEP:-4}"

die() { echo "[ping] ✗ $*" >&2; exit 1; }

ping_to() {
  if [[ -n "${PING_TO:-}" ]]; then printf '%s' "$PING_TO"; return 0; fi
  local f="${PING_RC:-$HOME/.zshrc}"
  [[ -r "$f" ]] || return 0
  grep -o 'PING_TO="[^"]*"' "$f" 2>/dev/null | tail -1 | sed 's/.*="//;s/"$//' || true
}

need_db() {
  [[ -r "$CHATDB" ]] || die "读不到 $CHATDB —— 请给宿主 app 开「完全磁盘访问」并 ⌘Q 重启它 (skill: diagnose-imessage-delivery)"
  sqlite3 "file:$CHATDB?mode=ro" "select 1;" >/dev/null 2>&1 \
    || die "chat.db 打不开 (权限或锁) —— 检查完全磁盘访问是否对**当前进程**生效"
}

sq() { sqlite3 -noheader -list -separator $'\t' "file:$CHATDB?mode=ro" "$1"; }

# ---------------------------------------------------------------- send
cmd_send() {
  need_db
  local to body; to="$(ping_to)"
  [[ -n "$to" ]] || die '未设收件人: 请 export PING_TO=<句柄> 或写进 ~/.zshrc 的 PING_TO="..."'
  body="${1:-}"
  [[ -n "$body" ]] || die '用法: bin/ping.sh send "正文"'
  local stamp; stamp="$(date '+%Y-%m-%d %H:%M:%S')"

  osascript >/dev/null <<APPLESCRIPT
tell application "Messages"
  set theService to first service whose service type is iMessage
  set theBuddy to buddy "$to" of theService
  send "$body" to theBuddy
end tell
APPLESCRIPT
  echo "[ping] → $to  已交给「信息」app  ($stamp)"

  sleep "$SLEEP_VERIFY"
  local row
  row="$(sq "
    select m.is_sent, m.is_delivered, m.error,
           coalesce(nullif(datetime(m.date_delivered/1000000000+978307200,'unixepoch','localtime'),'2001-01-01 08:00:00'),'——'),
           (select count(*) from message e
              join chat_message_join j2 on j2.message_id=e.ROWID
              join chat c2 on c2.ROWID=j2.chat_id
             where c2.chat_identifier like '%'||'$to'||'%'
               and e.is_from_me=0 and abs(e.date - m.date) < 5000000000)
    from message m
    join chat_message_join j on j.message_id=m.ROWID
    join chat c on c.ROWID=j.chat_id
    where c.chat_identifier like '%'||'$to'||'%' and m.is_from_me=1
    order by m.date desc limit 1;")"

  # 等回声落库再推进水位线, 免得回声日后冒充来讯
  local waited=0
  while (( waited < 12 )); do
    local n
    n="$(sq "select count(*) from message m
              join chat_message_join j on j.message_id=m.ROWID
              join chat c on c.ROWID=j.chat_id
             where c.chat_identifier like '%'||'$to'||'%' and m.is_from_me=0
               and m.date > (strftime('%s','now') - 978307200 - 30) * 1000000000;") "
    (( n > 0 )) && break
    sleep 1; waited=$((waited+1))
  done

  IFS=$'\t' read -r sent dlv err dat echo_n <<<"$row"
  echo "[ping]   sent=$sent  delivered=$dlv  error=$err  投递时刻=$dat  回声=$echo_n"
  if [[ "$dlv" == "1" ]]; then
    echo "[ping]   ✓ Apple 已投递 (回声 $echo_n —— 0 表示尚未见回执, 可稍后用 read 复核)"
  else
    echo "[ping]   ⚠ 尚无投递回执: 句柄可能未被 Apple 认, 或号码句柄还在激活中 (可稍后复核)"
  fi
  _save_seen
}

# ---------------------------------------------------------------- read
_watermark() {
  [[ -r "$STATE" ]] || { echo 0; return; }
  python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("last_rowid",0))' "$STATE" 2>/dev/null || echo 0
}
_save_seen() {
  need_db
  local to; to="$(ping_to)"; [[ -n "$to" ]] || return 0
  local mx
  mx="$(sq "select coalesce(max(m.ROWID),0) from message m
              join chat_message_join j on j.message_id=m.ROWID
              join chat c on c.ROWID=j.chat_id
             where c.chat_identifier like '%'||'$to'||'%' and m.is_from_me=0;")"
  mkdir -p "$(dirname "$STATE")"
  python3 - "$STATE" "$mx" "$to" <<'PY'
import json,sys,datetime
p, mx, to = sys.argv[1], int(sys.argv[2] or 0), sys.argv[3]
json.dump({"last_rowid":mx,"handle":to,"updated":datetime.datetime.now().astimezone().isoformat(timespec="seconds")},
          open(p,"w"), ensure_ascii=False, indent=2)
PY
}

_show() { # $1 = extra SQL predicate
  need_db
  local to n; to="$(ping_to)"; n="${2:-10}"
  [[ -n "$to" ]] || die '未设收件人 ($PING_TO)'
  local out
  out="$(sq "
    select datetime(m.date/1000000000+978307200,'unixepoch','localtime'),
           m.ROWID,
           coalesce(nullif(replace(replace(m.text,char(10),' '),char(13),''),
                           ''),
                    '(正文在 attributedBody, 未解析)')
    from message m
    join chat_message_join j on j.message_id=m.ROWID
    join chat c on c.ROWID=j.chat_id
    where c.chat_identifier like '%'||'$to'||'%' and m.is_from_me=0
      -- 滤掉「发给自己」的回声副本。回声的形状: reply_to_guid 指向发出件, 同文, 且**早于**该发出件
      -- (退回环到得比本机写入发出记录更早)。主人真回信时, iCloud 同步来的发出件与来讯**同刻**,
      -- 不满足严格早于, 故不会被误滤 —— 这是区分「自言自语」与「主人回了一个字」的关键。
      and (m.reply_to_guid is null
           or not exists (select 1 from message o
                           where o.guid = m.reply_to_guid and o.is_from_me = 1
                             and coalesce(o.text,'') = coalesce(m.text,'')
                             and m.date < o.date))
      $1
    order by m.date desc limit $n;")"
  if [[ -z "$out" ]]; then echo "[ping] (无来讯)"; return 0; fi
  while IFS=$'\t' read -r t rid txt; do
    printf '[%s] #%s  %s\n' "$t" "$rid" "$txt"
  done <<<"$out"
}

cmd_read() { _show "" "${1:-10}"; _save_seen; }
cmd_new()  { local w; w="$(_watermark)"; _show "and m.ROWID > $w" "${1:-20}"; _save_seen; }

# ---------------------------------------------------------------- status
cmd_status() {
  local to; to="$(ping_to)"
  echo "收件人 (PING_TO): ${to:-<未设>}"
  echo "水位线文件       : $STATE  (last_rowid=$(_watermark))"
  printf 'chat.db          : '
  if [[ -r "$CHATDB" ]] && sqlite3 "file:$CHATDB?mode=ro" "select 1;" >/dev/null 2>&1; then
    echo "可读 ✓"
  else
    echo "读不到 ✗ (需「完全磁盘访问」+ 重启宿主 app)"
    return 1
  fi
  printf 'iMessage 服务     : '
  osascript -e 'tell application "Messages" to get id of (first service whose service type is iMessage)' 2>/dev/null \
    || { echo "取不到 ✗ (信息.app 未登录?)"; return 1; }
  [[ -n "$to" ]] || return 0
  sq "select '  本会话 iMessage: 发出 ' ||
             (select count(*) from message m join chat_message_join j on j.message_id=m.ROWID join chat c on c.ROWID=j.chat_id
               where c.chat_identifier like '%'||'$to'||'%' and m.is_from_me=1) || ' 条 / 真来讯 ' ||
             (select count(*) from message m join chat_message_join j on j.message_id=m.ROWID join chat c on c.ROWID=j.chat_id
               where c.chat_identifier like '%'||'$to'||'%' and m.is_from_me=0
                 and (m.reply_to_guid is null
                      or not exists (select 1 from message o where o.guid=m.reply_to_guid and o.is_from_me=1
                                       and coalesce(o.text,'')=coalesce(m.text,'') and m.date < o.date))) || ' 条' ||
             ' (回声副本 ' ||
             (select count(*) from message m join chat_message_join j on j.message_id=m.ROWID join chat c on c.ROWID=j.chat_id
               where c.chat_identifier like '%'||'$to'||'%' and m.is_from_me=0
                 and exists (select 1 from message o where o.guid=m.reply_to_guid and o.is_from_me=1
                              and coalesce(o.text,'')=coalesce(m.text,'') and m.date < o.date)) || ' 条, 已滤)';"
  sq "select '  最近投递: ' || coalesce(nullif(datetime(m.date_delivered/1000000000+978307200,'unixepoch','localtime'),'2001-01-01 08:00:00'),'(无回执)') ||
             '  delivered=' || m.is_delivered || ' error=' || m.error
      from message m join chat_message_join j on j.message_id=m.ROWID join chat c on c.ROWID=j.chat_id
      where c.chat_identifier like '%'||'$to'||'%' and m.is_from_me=1 and m.date_delivered > 0
      order by m.date desc limit 1;"
}

case "${1:-}" in
  send)   shift; cmd_send "$@" ;;
  read)   shift; cmd_read "$@" ;;
  new|recv) shift; cmd_new "$@" ;;
  status) cmd_status ;;
  ""|-h|--help|help) sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//' ;;
  *) die "未知子命令: $1  (send | read | new | status)" ;;
esac
