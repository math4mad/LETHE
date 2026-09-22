#!/bin/bash
# pocket.sh — 园报进袋: 飞书自定义机器人 (签名: base64(hmac_sha256(key="ts\nsecret", "")))
CFG=~/.config/pocket/feishu.env
[ -f "$CFG" ] || { echo "pocket: 缺配置 $CFG"; exit 2; }
source "$CFG"
[ -n "$1" ] || { echo "用法: pocket.sh <文本>"; exit 2; }
TS=$(date +%s)
KEY=$(printf '%s\n%s' "$TS" "$FEISHU_SECRET")
SIG=$(printf '' | openssl dgst -sha256 -hmac "$KEY" -binary | base64)
BODY=$(/opt/miniconda3/envs/default/bin/python -c "
import json,sys
print(json.dumps({'timestamp':str(sys.argv[1]),'sign':sys.argv[2],'msg_type':'text','content':{'text':sys.argv[3]}},ensure_ascii=False))" "$TS" "$SIG" "$1")
curl -s --max-time 15 -X POST -H "Content-Type: application/json" -d "$BODY" "$FEISHU_WEBHOOK"; echo
