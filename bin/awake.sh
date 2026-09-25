#!/usr/bin/env bash
# 园子守夜灯 —— Amphetamine 管制器 (主人 0925 定正: 允许息屏, 不许休眠——displaySleepAllowed:true)
# 规矩: 本地长跑才点灯; 远程舱(Kaggle/AutoDL)任务一律不点 (哨会醒了收, 灯白烧)
# 用法: awake.sh status | on [分钟|0=无限] | off | wrap <命令...> | watch <pgrep模式> [秒]
set -u
AMP() { osascript -e "tell application \"Amphetamine\" to $1" 2>/dev/null; }
PIDF=~/.config/pocket/.awake-watch.pid
case "${1:-status}" in
  status) echo "active: $(AMP 'session is active') | 剩余秒: $(AMP 'session time remaining')";;
  on)     d="${1:-}" ; m="${2:-0}"; AMP "start new session with options {duration:$m, interval:minutes, displaySleepAllowed:true}"; sleep 1; echo "点灯 (时长 ${m}分, 0=无限)";;
  off)    AMP 'end session'; echo "收灯";;
  wrap)   shift; AMP "start new session with options {duration:0, interval:0, displaySleepAllowed:true}"
          echo "[awake] 灯亮, 执行: $*"
          "$@"; RC=$?
          AMP "end session" >/dev/null
          echo "[awake] 任务毕(退$RC) → 收灯";;
  watch)  shift; PAT=$1; INT=${2:-30}
          if [ -f "$PIDF" ] && kill -0 "$(cat $PIDF)" 2>/dev/null; then echo "已有哨在望"; exit 0; fi
          echo $$ > "$PIDF"
          HOLD=0
          while :; do
            MYPID=$$
            if [ "$(pgrep -f "$PAT" | grep -vw "$MYPID" | wc -l | tr -d ' ')" != "0" ]; then
              if [ "$HOLD" = 0 ]; then AMP "start new session with options {duration:0, interval:0, displaySleepAllowed:true}"; HOLD=1; echo "[$(date +%H:%M:%S)] 任务在跑 → 点灯"; fi
            else
              if [ "$HOLD" = 1 ]; then AMP 'end session'; HOLD=0; echo "[$(date +%H:%M:%S)] 任务终了 → 收灯"; break;
              else rm -f "$PIDF"; exit 0; fi
            fi
            sleep "$INT"
          done
          rm -f "$PIDF";;
  *) echo "用法: status|on [min]|off|wrap <cmd>|watch <pattern> [interval_s]";;
esac
