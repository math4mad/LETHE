#!/usr/bin/env bash
# bin/remind.sh — 园区 ⇄ Apple 提醒事项 联动器 (the owner's reminder app is now a park instrument)
#
# 协议 (the follow protocol, three moves):
#   1. add    : Lola/agent 挂决策点或日程 → 主人 iPhone 经 iCloud 同步收到
#   2. read   : 会话开场或主人说「读提醒」→ 列出未决与已办
#   3. 语义约定: 主人【勾掉】=准奏/点菜; 在【备注(body)追加文字】=回复指令;
#               完成时间即批复时间戳 (close date is the ratification stamp)
#
# 用法:
#   bin/remind.sh add  "标题" "正文" ["2026-09-19 09:00"]     # 挂一条
#   bin/remind.sh read                                         # 未决+今日已办
#   bin/remind.sh done   "标题片段"                             # agent 侧销账(实验完成后勾掉)
#   bin/remind.sh purge  "标题片段"                             # 删除(慎用;销账优先用 done)
set -euo pipefail
LIST="Concept-Space"

osa() { osascript -e "tell application \"Reminders\" to $1" ; }

case "${1:-}" in
add)
    TITLE="${2:?标题}"; BODY="${3:-}"; DUE="${4:-}"
    OFFSET=""
    if [ -n "$DUE" ]; then
        # epoch-seconds delta — locale-proof (AppleScript date literals die in zh locale)
        OFFSET=$(python3 -c "import sys,time,datetime;d=sys.argv[1];t=time.mktime(datetime.datetime.strptime(d,'%Y-%m-%d %H:%M').timetuple());print(max(60,int(t-time.time())))" "$DUE")
    fi
    osascript <<OSA
tell application "Reminders"
    if not (exists list "$LIST") then make new list with properties {name:"$LIST"}
    tell list "$LIST"
        set r to make new reminder with properties {name:"$TITLE", body:"$BODY"}
        if "$OFFSET" is not "" then set due date of r to (current date) + ("$OFFSET" as integer)
    end tell
end tell
OSA
    echo "⏰ 已挂: $TITLE"
    ;;
read)
    osascript <<OSA
tell application "Reminders"
    set out to "── 未决 open ──\\n"
    tell list "$LIST" to repeat with r in (reminders whose completed is false)
        set nm to name of r
        set bd to body of r
        set dd to due date of r
        set out to out & "◻ " & nm & (my tail(bd)) & my dued(dd) & "\\n"
    end repeat
    set out to out & "── 已办 done (批复) ──\\n"
    tell list "$LIST" to repeat with r in (reminders whose completed is true)
        set cg to "·"
        try
            set cg to modification date of r as string
        end try
        set out to out & "☑ " & name of r & "  ✎" & (body of r) & "  ⌚" & cg & "\\n"
    end repeat
    return out
end tell
on tail(s)
    if s is "" then return ""
    return "  ⟪" & (text 1 thru (my min2(120, length of s)) of s) & "⟫"
end tail
on dued(d)
    if d is missing value then return ""
    return "  ⏱" & (d as string)
end dued
on min2(a, b)
    if a < b then return a
    return b
end min2
OSA
    ;;
done)
    PAT="${2:?标题片段}"
    osascript -e "tell application \"Reminders\" to tell list \"$LIST\" to set completed of (first reminder whose name contains \"$PAT\") to true" && echo "☑ 已销账: $PAT"
    ;;
purge)
    PAT="${2:?标题片段}"
    osascript -e "tell application \"Reminders\" to tell list \"$LIST\" to delete (every reminder whose name contains \"$PAT\")" && echo "✖ 已删: $PAT"
    ;;
*)
    sed -n '2,12p' "$0"; exit 1;;
esac
