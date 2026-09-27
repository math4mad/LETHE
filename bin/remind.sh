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
#   bin/remind.sh add  "标题" "正文" ["2026-09-19 09:00"]   # 挂一条 (默认单道: iCloud Concept-Space)
#   MIRROR=1 bin/remind.sh add ...                          # 显式加挂 MS To Do 镜像 (主人 0928 一账一道律: 默认停)
#   bin/remind.sh read                                       # 未决+近日已办 (主道为准, 镜像并集只作参考)
#   bin/remind.sh done   "标题片段"                           # agent 侧销账 (两侧都勾, 幂等)
#   bin/remind.sh purge  "标题片段"                           # 删除(慎用;销账优先用 done)
#
# 铁律 (0928 验尸定):
#   · 认账必认 account+名 —— 本机曾有 iCloud 与 Exchange 两只同名 "Concept-Space",
#     `tell list "Concept-Space"` 按位解析被夺舍, 条目双挂。一律 first list whose
#     {name is LIST and name of container of it is "iCloud"}。
#   · 同题 open 在则 add 跳 (防双写; 需更新备注请手披或用新题)。
#   · 镜像律 (0924 接通, 0928 主人裁定默认关): MS To Do 借道 Mac 原生 Exchange 落「任务」列表。
#   · 日期只走偏移量, 禁 AppleScript 日期字面量 (zh locale 灾)。读卷必走批量取属性。
set -euo pipefail
LIST="Concept-Space"
TODO_LIST="任务"

MAIN_PTR='first list whose (name is "'"$LIST"'" and name of container of it is "iCloud")'

# 镜像器: 对 Exchange 容器下的 TODO_LIST 动刀; 任何失败只吐 TODO-SKIP, 不断主道
todo() { # $1 = AppleScript body (在 tell L 上下文内执行)
    osascript <<OSA 2>/dev/null || echo "TODO-SKIP"
tell application "Reminders"
    try
        set L to first list whose name is "$TODO_LIST"
        if name of container of L is not "Exchange" then error
    on error
        return "TODO-SKIP"
    end try
    tell L
        $1
    end tell
end tell
OSA
}

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
    set L to $MAIN_PTR
    if not (exists L) then
        tell application "Reminders" to make new list with properties {name:"$LIST"}
        set L to $MAIN_PTR
    end if
    tell L
        if (count of (reminders whose name is "$TITLE" and completed is false)) > 0 then
            return "DUP-SKIP"
        end if
        set r to make new reminder with properties {name:"$TITLE", body:"$BODY"}
        if "$OFFSET" is not "" then set due date of r to (current date) + ("$OFFSET" as integer)
    end tell
    return "OK"
end tell
OSA
    ;;
read)
    ( osascript <<OSA & tp=$!; ( sleep 45; kill $tp 2>/dev/null ) & wait $tp 2>/dev/null || echo "⚠ 主道巡超时, 下次再收" )
tell application "Reminders"
    set out to "── 未决 open · $LIST (iCloud 主道) ──" & linefeed
    tell $MAIN_PTR
        set NMO to name of (reminders whose completed is false)
        set BDO to body of (reminders whose completed is false)
        set DDO to due date of (reminders whose completed is false)
        repeat with i from 1 to count of NMO
            set dd to "·"
            try
                set dd to (item i of DDO as string)
            end try
            set out to out & "◻ " & (item i of NMO) & "  ⟪" & (item i of BDO) & "⟫  ⏱" & dd & linefeed
        end repeat
    end tell
    set out to out & "── 已办 done (批复, 最近 12) ──" & linefeed
    tell $MAIN_PTR
        set NMD to name of (reminders whose completed is true)
        set CMD to modification date of (reminders whose completed is true)
        set n2 to count of NMD
        set lo to n2 - 11
        if lo < 1 then set lo to 1
        repeat with i from lo to n2
            set out to out & "☑ " & (item i of NMD) & "  ⌚" & (item i of CMD as string) & linefeed
        end repeat
    end tell
    return out
end tell
OSA
    if [ "${MIRROR:-0}" = "1" ]; then
    echo "── MS To Do · 任务 (镜像在队, MIRROR=1 期间新账双挂; 参考并集) ──"
    ( todo 'set out to ""
        set NMO to name of (reminders whose completed is false)
        set BDO to body of (reminders whose completed is false)
        repeat with i from 1 to (count of NMO)
            set out to out & "◻[MS] " & (item i of NMO) & "  ⟪" & (item i of BDO) & "⟫" & linefeed
        end repeat
        if out is "" then return "(空)"
        return out' & tp=$!; ( sleep 45; kill $tp 2>/dev/null ) & wait $tp 2>/dev/null ) || echo "· 镜像巡未决, 主道无怗"
    fi
    ;;
done)
    PAT="${2:?标题片段}"
    osascript -e "tell application \"Reminders\" to tell $MAIN_PTR to set completed of (every reminder whose name contains \"$PAT\" and completed is false) to true" && echo "☑ 已销账: $PAT (Apple 主道)"
    todo "set completed of (every reminder whose name contains \"$PAT\" and completed is false) to true
        return \"TODO-OK\"" | grep -q TODO-OK && echo "☑ 已销账: $PAT (MS)" || true
    ;;
purge)
    PAT="${2:?标题片段}"
    osascript -e "tell application \"Reminders\" to tell $MAIN_PTR to delete (every reminder whose name contains \"$PAT\")" && echo "✖ 已删: $PAT (主道)"
    ;;
*)
    sed -n '2,25p' "$0"; exit 1;;
esac
