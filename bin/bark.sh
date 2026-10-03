#!/usr/bin/env bash
# bin/bark.sh — 园 ⇄ 主人 iPhone 的**直达线** (Bark 推送服务器, 绕开 Vibe Island)
#
# 分工 (与 bin/remind.sh 并列, 一账一道):
#   remind.sh : 园里的**日程与决策点** → iCloud 提醒事项 Concept-Space
#   bark.sh   : 「agent 正在等你」这类**即时一响** → Bark 直推 (无需 iCloud, 即时, 可带 ttl 自焚)
# 见 vibeisland.app/zh/guides/iphone-notifications/ (Vibe Island 走的是同一台服务器, 同一把密钥)
#
# 密钥来源 (二选一, 优先环境变量):
#   export BARK_KEY="<22位短码>";  或写在 ~/.zshrc 的 BARK_KEY="..."
#   短码正本 = iPhone Bark 首页示例 URL https://api.day.app/<短码>/... 中域名与第一个斜杠之间那一段
#   (Bark「设置」页里的 Device Token 是 Apple 推送令牌, 不是短码 —— 贴它必被服务器拒)
#
# 用法:
#   bin/bark.sh send "标题" "正文" ["分组"]   # 默认分组 Concept-Space, ttl=600
#   bin/bark.sh "标题" "正文"                  # send 可省
#   bin/bark.sh key                            # 只看密钥来源与长度校验 (不打印全码)
#   bin/bark.sh raw "<完整 URL 路径>"          # 逃生口: 直接打 api.day.app 的 path
#
# 退出码: 0 已受理 / 2 无密钥 / 3 网络失败 / 4 服务器拒收(码见输出 JSON)
set -euo pipefail

GROUP_DEFAULT="Concept-Space"
TTL_DEFAULT="600"

# 取密钥: $BARK_KEY 优先, 否则从 rc 文件里取最后一条 BARK_KEY="..."
bark_key() {
  if [[ -n "${BARK_KEY:-}" ]]; then printf '%s' "$BARK_KEY"; return 0; fi
  local f="${BARK_RC:-$HOME/.zshrc}"
  [[ -r "$f" ]] || return 0
  grep -o 'BARK_KEY="[^"]*"' "$f" 2>/dev/null | tail -1 | sed 's/.*="//;s/"$//' || true
}

urlenc() { python3 -c 'import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))' "$1"; }

cmd="${1:-}"
case "$cmd" in
  key)
    k=$(bark_key)
    if [[ -z "$k" ]]; then echo "无密钥: \$BARK_KEY 未设且 ~/.zshrc 里没有 BARK_KEY=\"...\""; exit 2; fi
    echo "来源: $([[ -n "${BARK_KEY:-}" ]] && echo '$BARK_KEY 环境变量' || echo "${BARK_RC:-$HOME/.zshrc}")"
    echo "长度: ${#k} (Bark 正本短码为 22 位; 不足 22 位基本可判为漏抄)"
    echo "前缀: ${k:0:4}…"
    exit 0
    ;;
  raw)
    k=$(bark_key); [[ -n "$k" ]] || { echo "无密钥" >&2; exit 2; }
    resp=$(curl -sS --max-time 20 -w $'\n%{http_code}' "https://api.day.app/${k}/${2:-}") || { echo "网络失败" >&2; exit 3; }
    ;;
  send) shift ;;
  "") echo "用法: bin/bark.sh send \"标题\" \"正文\" [分组]" >&2; exit 2 ;;
esac

title="${1:-园笔 lola}"
body="${2:-}"
group="${3:-$GROUP_DEFAULT}"
k=$(bark_key)
if [[ -z "$k" ]]; then
  echo "无密钥: 先 export BARK_KEY=<短码>, 或写进 ~/.zshrc 的 BARK_KEY=\"...\"" >&2
  exit 2
fi

t=$(urlenc "$title"); b=$(urlenc "$body"); g=$(urlenc "$group")
url="https://api.day.app/${k}/${t}/${b}?group=${g}&ttl=${TTL_DEFAULT}"

resp=$(curl -sS --max-time 20 -w $'\n%{http_code}' "$url") || { echo "网络失败 (DNS/超时/离线)" >&2; exit 3; }
http="${resp##*$'\n'}"; json="${resp%$'\n'*}"
echo "[bark] POST …/${k:0:4}…/${t}/${b} → HTTP ${http}"
echo "$json"
[[ "$http" == "200" ]] || exit 4
