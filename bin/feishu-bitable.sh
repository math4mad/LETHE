#!/usr/bin/env bash
# 园子↔飞书多维表格 桥器 —— 密钥不出 ~/.config/pocket/，token 缓存两小时自新
# 用法: feishu-bitable.sh tables | fields <tb> | recs <tb> [n] | add <tb> '<json-fields>' | del <tb> <rid>
set -u
source ~/.config/pocket/feishu.env
TOK=$(printf '%s' "$FEISHU_BASE_URL" | grep -oE 'base/[A-Za-z0-9]+' | cut -d/ -f2)
CACHE=~/.config/pocket/.ftok.cache
tok() {
  if [ -s "$CACHE" ] && [ $(( $(date +%s) - $(stat -f %m "$CACHE") )) -lt 6600 ]; then cat "$CACHE"; return; fi
  curl -s --max-time 15 -X POST "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal" \
    -H "Content-Type: application/json" -d "{\"app_id\":\"$FEISHU_APP_ID\",\"app_secret\":\"$FEISHU_APP_SECRET\"}" \
  | /usr/local/bin/python3 -c "import sys,json;print(json.load(sys.stdin).get('tenant_access_token',''))" > "$CACHE"
  chmod 600 "$CACHE"; cat "$CACHE"
}
H() { echo "Authorization: Bearer $(tok)"; }
API=https://open.feishu.cn/open-apis/bitable/v1/apps/$TOK
case "${1:-}" in
  tables) curl -s --max-time 20 "$API/tables" -H "$(H)";;
  fields) curl -s --max-time 20 "$API/tables/$2/fields" -H "$(H)";;
  recs)   curl -s --max-time 25 "$API/tables/$2/records?page_size=${3:-20}" -H "$(H)";;
  add)    curl -s --max-time 20 -X POST "$API/tables/$2/records" -H "$(H)" -H "Content-Type: application/json" -d "{\"fields\":$3}";;
  upd)    curl -s --max-time 20 -X PUT "$API/tables/$2/records/$3" -H "$(H)" -H "Content-Type: application/json" -d "{\"fields\":$4}";;
  del)    curl -s --max-time 20 -X DELETE "$API/tables/$2/records/$3" -H "$(H)";;
  garden) curl -s --max-time 20 "$API/tables/$GARDEN_TABLE/records?page_size=50" -H "$(H)";;
  gadd)   curl -s --max-time 20 -X POST "$API/tables/$GARDEN_TABLE/records" -H "$(H)" -H "Content-Type: application/json" -d "{\"fields\":$2}";;
  gupd)   curl -s --max-time 20 -X PUT "$API/tables/$GARDEN_TABLE/records/$2" -H "$(H)" -H "Content-Type: application/json" -d "{\"fields\":$3}";;
  *) echo "用法: tables|fields <tb>|recs <tb> [n]|add/upd/del <tb> …|garden|gadd '<json>'|gupd <rid> '<json>'"; exit 1;;
esac
