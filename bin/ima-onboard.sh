#!/usr/bin/env bash
# ============================================================================
# ima-onboard.sh — 在任意 OpenClaw 机器上装好 ima-skill（笔记 / 知识库）并验证
#
# 正本声明
#   本文件是正本（source of truth）。技能包以 base64 内嵌在文件末尾的
#   __IMA_PAYLOAD__ 行之后，解出来即官方包 ima-skills-1.1.10 的内容。
#   派生面：解包目录 $TMPDIR/ima-skill/（临时，跑完即删）。
#   源件：math4mad 园子的 ~/.pi/agent/skills/ima-skill/（与内嵌包逐字节一致）。
#
# 一条 verify 命令（退出码即判据）
#   ./ima-onboard.sh --check      # 0 = 技能就位且真调通 IMA API
#
# 三条铁律（2026-10-09 在 macbook-pro 上踩实，别再犯）
#   1. 技能必须落在扫描根下**恰好一层**：<root>/ima-skill/SKILL.md
#      多套一层（如 skills/ima/ima-skill/）→ 永远扫不到，且**不报错**。
#   2. SKILL.md 必须带 YAML frontmatter（含 name + description）
#      否则只打印一行 "Skipping invalid skill ... description is required" 就丢弃。
#   3. 凭证只认 IMA_OPENAPI_CLIENTID / IMA_OPENAPI_APIKEY 两个名字
#      CLIENTID 是 32 位 hex（a2af…），APIKEY 是 77 位。写反了必 20004 鉴权失败。
#
# 用法
#   ./ima-onboard.sh              # 装技能 + 配凭证 + 验证
#   ./ima-onboard.sh --check      # 只验证，什么都不改
#   ./ima-onboard.sh --no-creds   # 只装技能，凭证稍后自己填
#   SKILL_ROOT=/tmp/x ./ima-onboard.sh    # 装到指定根（自测用）
# ============================================================================
set -euo pipefail

MARK=__IMA_PAYLOAD__
SKILL=ima-skill
ENV_FILE="$HOME/.openclaw/.env"
API=https://ima.qq.com/openapi/note/v1/list_note_folder_by_cursor
WORKDIR=""

DO_CHECK=0
NO_CREDS=0
for a in "$@"; do
  case "$a" in
    --check)     DO_CHECK=1 ;;
    --no-creds)  NO_CREDS=1 ;;
    -h|--help)   sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "未知参数: $a（试 --help）" >&2; exit 2 ;;
  esac
done

say()  { printf '%s\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
bad()  { printf '  \033[31m✗\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
hdr()  { printf '\n\033[1m%s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- 载荷解包
decode_payload() {
  if command -v python3 >/dev/null 2>&1; then
    python3 -c 'import base64,sys; sys.stdout.buffer.write(base64.b64decode(sys.stdin.buffer.read()))'
  elif command -v openssl >/dev/null 2>&1; then
    openssl base64 -d -A
  elif command -v base64 >/dev/null 2>&1 && base64 --help 2>&1 | grep -q -- '-d'; then
    base64 -d
  elif command -v base64 >/dev/null 2>&1; then
    base64 -D
  else
    echo "缺 base64 解码器（需要 python3 / openssl / base64 之一）" >&2
    return 1
  fi
}

extract_skill() {
  # $1 = 目标临时目录；产出 $1/ima-skill/
  mkdir -p "$1"
  awk -v m="$MARK" 'found{print} $0==m{found=1}' "$0" \
    | decode_payload \
    | tar xzf - -C "$1"
}

# ---------------------------------------------------------------- 凭证读取
load_env_file() {
  # shellcheck disable=SC1090
  [ -f "$ENV_FILE" ] && set -a && . "$ENV_FILE" && set +a
  return 0
}

shape_of() {
  # 粗判 32 位 hex（client id） / 77 位（api key）
  local v="$1"
  case "${#v}" in
    32) printf 'CLIENTID' ;;
    77) printf 'APIKEY' ;;
    *)  printf 'UNKNOWN' ;;
  esac
}

write_creds() {
  local cid="$1" key="$2"
  mkdir -p "$(dirname "$ENV_FILE")"
  touch "$ENV_FILE"; chmod 600 "$ENV_FILE"
  # 幂等：先删掉旧的两行，再追加
  if grep -q '^IMA_OPENAPI_' "$ENV_FILE" 2>/dev/null; then
    grep -v '^IMA_OPENAPI_' "$ENV_FILE" > "$ENV_FILE.tmp" && mv "$ENV_FILE.tmp" "$ENV_FILE"
    chmod 600 "$ENV_FILE"
  fi
  {
    echo "# IMA OpenAPI credentials for ima-skill (notes / knowledge-base)"
    echo "IMA_OPENAPI_CLIENTID=$cid"
    echo "IMA_OPENAPI_APIKEY=$key"
  } >> "$ENV_FILE"
}

# ---------------------------------------------------------------- 1. 装技能
install_skill() {
  hdr "① 安装 ima-skill"

  local tmp="$WORKDIR"
  extract_skill "$tmp" || { bad "内嵌载荷解包失败"; return 1; }
  [ -f "$tmp/$SKILL/SKILL.md" ] || { bad "解包后找不到 $SKILL/SKILL.md"; return 1; }
  ok "载荷解包完成（$(find "$tmp/$SKILL" -type f | wc -l | tr -d ' ') 个文件）"

  # frontmatter 自检 —— 这是最容易死的一关
  if head -1 "$tmp/$SKILL/SKILL.md" | grep -q '^---$' && grep -q '^description:' "$tmp/$SKILL/SKILL.md"; then
    ok "SKILL.md frontmatter 正常"
  else
    bad "SKILL.md 缺 frontmatter（name/description）—— OpenClaw 会静默丢弃这个技能"
    return 1
  fi

  # 首选官方安装器（显式给了 SKILL_ROOT 就走手工，便于自测不污染真环境）
  if command -v openclaw >/dev/null 2>&1 && [ -z "${SKILL_ROOT:-}" ]; then
    if openclaw skills install "$tmp/$SKILL" --global --force < /dev/null > "$tmp/install.log" 2>&1; then
      ok "openclaw skills install --global 成功"
      return 0
    fi
    warn "openclaw skills install 失败，回落到手工放置"
    sed -n '1,6p' "$tmp/install.log" | sed 's/^/      /'
  else
    warn "PATH 里没有 openclaw，回落到手工放置"
  fi

  # 手工放置到个人技能根（2026-10-09 在 OpenClaw 2026.9.8 上实测会被扫描）
  local root="${SKILL_ROOT:-$HOME/.agents/skills}"
  mkdir -p "$root"
  rm -rf "$root/$SKILL"
  cp -R "$tmp/$SKILL" "$root/$SKILL"
  ok "已放置到 $root/$SKILL"

  # 一层铁律自检
  if [ -f "$root/$SKILL/SKILL.md" ]; then
    ok "落点层次正确：<$root>/ima-skill/SKILL.md（恰好一层）"
  else
    bad "落点层次不对，检查 $root"
    return 1
  fi
}

# ---------------------------------------------------------------- 2. 配凭证
configure_creds() {
  hdr "② 配置 IMA 凭证"
  load_env_file

  local cid="${IMA_OPENAPI_CLIENTID:-}" key="${IMA_OPENAPI_APIKEY:-}"
  if [ -n "$cid" ] && [ -n "$key" ]; then
    ok "已存在：CLIENTID ${cid:0:4}…(${#cid}) / APIKEY ${key:0:4}…(${#key})"
    return 0
  fi

  if [ "$NO_CREDS" -eq 1 ] || [ ! -t 0 ]; then
    warn "未配置凭证。回头改 $ENV_FILE 加两行："
    warn "  IMA_OPENAPI_CLIENTID=<32位>"
    warn "  IMA_OPENAPI_APIKEY=<77位>"
    return 0
  fi

  say "  从 https://ima.qq.com/agent-interface 取 Client ID 与 API Key。"
  say "  （直接回车可跳过，稍后手填 $ENV_FILE）"
  printf '  IMA Client ID : '; read -r cid || true
  printf '  IMA API Key   : '; read -rs key || true; say ""
  if [ -z "$cid" ]; then warn "跳过凭证配置"; return 0; fi

  # 防「写反」——原 cfg.json 就栽在这：把 32 位 CLIENTID 填进了 api_key
  local s1 s2; s1=$(shape_of "$cid"); s2=$(shape_of "$key")
  if [ "$s1" = APIKEY ] && [ "$s2" = CLIENTID ]; then
    warn "检测到两项写反了，自动纠正"
    local t="$cid"; cid="$key"; key="$t"
  fi
  if [ "$(shape_of "$cid")" != CLIENTID ]; then warn "Client ID 长度 $((${#cid})) 不像 32 位"; fi
  if [ "$(shape_of "$key")" != APIKEY ];   then warn "API Key 长度 $((${#key})) 不像 77 位"; fi
  if [ -z "$key" ]; then warn "API Key 为空，跳过写入"; return 0; fi

  write_creds "$cid" "$key"
  ok "已写入 $ENV_FILE（chmod 600）"
}

# ---------------------------------------------------------------- 3. 验证
do_check() {
  hdr "③ 验证（退出码即判据）"
  local fail=0
  load_env_file

  # a) 技能可见性
  if command -v openclaw >/dev/null 2>&1; then
    local out; out=$(openclaw skills list --json 2>/dev/null || echo '[]')
    if printf '%s' "$out" | grep -q "\"name\": *\"$SKILL\""; then
      ok "openclaw 认到了 $SKILL"
      printf '%s' "$out" | python3 -c "
import sys,json
d=json.load(sys.stdin); items=d if isinstance(d,list) else d.get('skills',[])
for s in items:
    if s.get('name')=='$SKILL':
        print('      eligible=%s modelVisible=%s missing=%s' % (s.get('eligible'),s.get('modelVisible'),s.get('missing')))
        sys.exit(0 if s.get('eligible') else 3)
" 2>/dev/null || { bad "技能存在但不可用（eligible=false）—— 多半是凭证没进环境"; fail=1; }
    else
      bad "openclaw 没认到 $SKILL —— 检查是不是套了多余一层目录"
      fail=1
    fi
    openclaw skills list 2>&1 | grep -i "Skipping invalid skill" | grep -i "$SKILL" | sed 's/^/      /' || true
  else
    warn "PATH 里没有 openclaw，跳过技能可见性检查"
  fi

  # b) 真调一次 IMA API
  local cid="${IMA_OPENAPI_CLIENTID:-}" key="${IMA_OPENAPI_APIKEY:-}"
  if [ -z "$cid" ] || [ -z "$key" ]; then
    bad "没有凭证，无法真调 API"
    fail=1
  else
    local resp; resp=$(curl -s -m 20 -X POST "$API" \
      -H "ima-openapi-clientid: $cid" -H "ima-openapi-apikey: $key" \
      -H "Content-Type: application/json" \
      -d '{"cursor":"0","limit":5}' 2>/dev/null || echo '{}')
    if printf '%s' "$resp" | grep -q '"code":[[:space:]]*0'; then
      local n; n=$(printf '%s' "$resp" | grep -o '"folder_id"' | wc -l | tr -d ' ')
      ok "IMA API 通了（返回 $n 个笔记本）"
    else
      local code; code=$(printf '%s' "$resp" | grep -o '"code":[[:space:]]*[0-9-]*' | grep -o '[0-9-]\+$' | head -1)
      bad "IMA API 调用失败 (code=$code)  常见: 20004/200002=鉴权失败(凭证错/写反)  20002=限频"
      fail=1
    fi
  fi

  [ "$fail" -eq 0 ] && { hdr "全绿 ✓"; return 0; }
  hdr "有未通过项 ✗"; return 1
}

# ---------------------------------------------------------------- main
if [ "$DO_CHECK" -eq 1 ]; then
  do_check
  exit $?
fi

command -v openclaw >/dev/null 2>&1 || warn "PATH 里没有 openclaw —— 装完技能后本机 agent 可能读不到，先确认 openclaw 装了"

WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/ima-onboard.XXXXXX")
trap 'rm -rf "$WORKDIR"' EXIT

install_skill
if [ "$NO_CREDS" -eq 0 ]; then configure_creds; fi
do_check
exit 0

# ---- 内嵌载荷（官方 ima-skills-1.1.10 的 ima-skill/ 全量，tar.gz+base64）----
__IMA_PAYLOAD__
H4sIAEiLyWoAA+y9WXdUV7IwWN+3Vq9enf3c/bxv2t/VUFKmUgMysvFtDPiaa7C5Ble52uaSiXQE
aaRMVWaKoQzfkgBNICFhZiEMwkzGRhIGg5AA/Y1+duXJ4anua791DPvss8+QqQFB3S4rl7Eyz9lD
7NgRsSNix44d2hPvjNWmD8Q7On73uj51dXXNTU2C/q5bR3/hY/2tq6tvbBCRpvp19Q11TU2ROlEX
aYCnvxN1rw0i7dOdzsRSAEpnrLVsuUP7DaMcipyDEqsK4+v8/C//5//6u//+u99tj7WKT3eKL4T8
4LPf/W/wrx7+/Rn+4e/vl9bkxl27PpNfscZl+Pe/u4r8N/v5/9Ga7AzFuro6jFBXKnnQSMQSrcbv
/tt//91f2v/f/8zn//T/rMIg1z6lPjtihz8yYm1GKvz65MCi/A887+T/Rvj1O3F41SHx+fzG+b+h
TnRm4p3Ghkjz+kjT+ob6d94JReqa1q+rq697J9DULLZt/WDjZ5s+2vqHLaHDsUwmFfJj1w0b/33r
RqNzb9M7n7Yn/ti2L9C4XuyEStv+VK6SxuOBvzcefqsfxfXh19fHYvyPP5z8X7+uAfi/6fWBZH9+
4/xvz39oTyKZMdKvoY/l63+NoAGu6X9v5LOm//2mPzb/25rgasuBRfm/rtnF/03r1jWu6X9v4lO/
3k//W1f3TqS5aU39+8f/2PxPXP9atMDl638NUG5N/3sTH13/O5BIHuow2vYZtXtjaWP1+li+/gfy
v3lN/3sjnzX97zf98dP/VlsOLN//19wYaVrT/97Ex9//B1MRWVe/pgD+439s/ndy/WoqgsvX/5qa
6pvW9L838dH1v50fb922LdTZttp9AD7WNTYuy/9X31S/pv+9kc+a/veb/vjpf6stBxblf4/+t64O
4z/W9L/X//HX/+qbG5rXN61f0//+4T82/7+u1X8x/q9vXtfk9v831jWt+X/eyKe2tjaQiHUaLUJR
QqDNSLem4l2ZeDLRIo4GhMjPX8/O9uTHT4qt2zeKT7uMxMYdW0XuVE/hxIu/PR/OnZ/ODffmfzpf
mJrJT03mx/rNb4fz128XpvvNuXO5cyPZFxN/7TkODZkvzuXP38sNPs2NjpmDM6rMX3t6C7+czF28
wt+5Kfhi3howFy6bLy5ggamZ7Nxp7G7wYqGnL3fiUeFOb3b2VPb5jdzFgez8EyiTezpvnrqRf3G2
OPmLq33oPTc2kX98Uz00+/vMqWdYi56Hc7+MFu4Ohs3Bq+b8XDj//GLh5VmGJHfpCfSbfbGAwD+4
JQhNckQjj+RzGNTPk7mJodzlM/nJqcL046DqKQggB7mpIDRkjt4H0HMnR82rL3NPhszRUzwAHowL
bh5M7vyTwqVRHga/ZaDhSXb2fnZuDlrITX5vPrhsjl782/NB887xoDk7lRs8i2ib7cnOng5C2SDU
4l+qi+LAMAJNoH/xBYA3hEN9dt2cO1+4e8ccPesY8P5kp9EV2wfksj+T6Uq3hDFsLPTnP6OAD3Qa
mVhbLBNrASCTQCStHbFD+F0IozP5dbxFVPzn9fN3K+hJyvhzdzxlpFvEN8JIHGwRX1YAbe35dMeW
T4C29mzatnXLJ7u2bq6oEY7n8O/jLX+q2C2OUStdKeg+dWQLNuBfH4qljdbuVDxzhEFpTRltRiIT
j3Wk93SnaShHA6zz7tofT/NQFXiiO22kanHNiqeBHYw2BwdobYnKTR1x+C62bhaxRJvA1x8bR6pk
05mkiHVn9mPh1ljGEIfimf0Cfotke3u8FRqgdrFSLOOD2pBsZpPWYSxliE8/2fYnGB90Cx34Nmck
2rqScShQabdWJWJp8dGuXTvEflL40lbznyQdQ8IeOpL79hltNSKdScIbEU+I9niHka4RyZTIpGKJ
dGc8k4EXOMDEEZEEIFICBEgmnoihAOGmYx0dyUNG2562ZGcsnki3yP5qhUY9KIkCb2liKPB5It4e
d6GcXoXEpu5UCsDsOCLS3V1dyVQGCKm6mrawqqtr4KvTm1FdHYK23xI7jUx3VyDwPhTYKalCYKWW
6mp99vWpSttzVV3tRi9Uq4zaY4hWiV97zlPZNMhUmJnUwXirQQ+QkAAPKcD4EfyRDnlmM5nA8eBs
uuYHsat3QxSWMA5Ci068E3praIpohmD2YOCRkMgNnTOf9/hQVhg4IJGpBQoxUu0xgLVw5imIEBir
omcYJMhzeCJpuro6UB8SKGqO3zMHHhSme0HgZOeGiz1DIFtIgIwHAtXVuYvPzOejYiPhpNg3kn8x
xXIOiufO3CuMjHFZaC8QjUZhnvYHOg+0xVOitkv8zzBAl2iP70NYA0br/qQIHkl2p/a0Elh74m1B
8b6zVFi90svHuuJ7DhhHvKXlC+xag/YDgjZ/Ztq8ecIcvVwcGHVBaBxGchN+wmaDG0K/wizBNjiB
YyA24lSI7PPx3PBQ9vlls28wP3c3+/Ja7qdJc+ZaYfoCgKJDJn7t/9aBWJTQSOY8K8WbJ3Pf9wQC
uelRaKEwcwJWKZIJ5tAIrkLQPKxTU7e4tDk6De+hBVg6ct/BKnELFhaz53nxxIvs7Fz+hzms0juR
e3CTF3Tz2RNYxPO35uTaNzzE3CVyD24X799isAggC3NviW3JWJu/RHVIUijRxqKtHWjYplWWjzZv
QvWqAOKWJ2APzsDb3/jNTEvt25VQoRS9iPr3w23GwXCiu6Oj6liQ2sTqNFHOFnn6/NuTs+luLd4u
vhS1fxHBtx2wBsVucfSo/kp2CS/eRZGRACnJhJx/PmfOnCVEWQw3XJgugXNhL0TItpJpg9jY4XhG
RALtcSY3IBR8yYSRuzeZu7YAxDLUA4oANv7wOKt9MPvA+ySOdny6c5f4vfi3nZ9+Ij5ItoEkQF1h
vg+0BPNljzl1GZhIfADyVnz+2TYR9UqbKBGEOTWefTZUeNlnnvrBHHiRuzBT7F0w+0aKAyPmrREp
9IgPgXSBCgE289olVI36bptTw7kLjwtPp82XJ0nO2NS1Ua7bsPikxb4kU1CpdVFfDgPwHVmxskp8
A2jqSLZCwa5YZj9MfiQo9sJI4Vs9ohDWjA5Rmxa1XzAygj4S9W2sGhRf0SJX+5EI4pKG+hB0Ucs0
F29rEW5q8C8P/4CkZGlFIHbZTUmQ3CDAdx3pAk0GTWzJGeGv08mEKtkGJIbjCAaO8eTjEviJXPkQ
cfoaa+kMaRyuXGaBJ4WigZDY2i5ior070w3LVmeyrbvDsFWmmGiLt7cbWFGASrg/CepDVKE4KjrB
5BJ7DSBHgLxN6g6trUYXrHuyAmA/BQsorEm8bjMFmP2P8g8uFibvBQJHhRQ7pESL1f8clX06HhWm
53Ft9K8AMNW+to9wN+554HgHsEhjx7Kk2LiR5szET2hP0ULPT5QpxMaPqlVYeAH2FL9dItpIB9Mf
RDmwyHIvRJ0oK2PAFc+9zJ25rewz3SiSo/FYcnJM1vPswmSud1o9hyVOtj9+0q47eAkoCuB0aowI
uGtHzB7BUSTKt8Sv4zf+NjsqcpfP5Z4+zT3tNyfmclemA4Hs/G2wsPgXwAXvC9PT5uAtpigQmcWJ
nvzQM3Pwx9yje0DBJMcsigbDEQ3t5dOrOfVd8UpfSYYAinjwPS74T2Gdf7js9h1NvQKpl6Pb8i+X
0wcAGMydOlVYuJKb+oWpg2deN66/+AJsXyQFovY/gVgtgVhY2p7+DKtidTUXra728MVR2/qghQuU
K50BpVIdbUu2groRRVWqfwTVsShIbBCCYBi1Rn2wLAeBmlj/FYD9iy+4PQCczHlkXfup/wBWOoQy
oC1OHhLw/PSAFDke7JcC1gH2TL+UT32PCr3noTrAbzUAYFpgu409CX9bmx3SFmUzLtpptMVjezKw
Wm6IRBzjIrB1cbQ8aC2YuQW0J7CN6mq9FVWoFMigBoPJuYeAjJJ6v+nTnfTXPRoXti/OgNhGH52F
cETaiwtIO3NXeZIXGwJCxeIfgZ/hZjwlHBQS70TzZoUU4ueeYqi1B9mFa2BoogFiec8GcxP3c8MD
oEMyRfMLsCRRwpHwy86OsA8O2Ky6GmRt4U4vWzpEMj6sWpi+Wbw0JTWKy9OMBYWDwsJleCj5ZXDG
PHcfsOzsHNkZtBMAClYsc/R7OSuK6BXTOuDve5p9cY6bQ/hLT8x/tXGdOqW7D0GEfvEF0xjLo/LS
yBoUwJedP6MaYt3K6vIKqPqSAokub43z4inH5lqrpQNXqmfILqyFSFcuD9FNgmjz35g1F07gynzx
QeHuSXPwClo0sBjXivzVqdyNAcCPJTFhlDxGLAKYnUdtqf+KUpJgBu2OGVhXM9ZIoaXctUl4Qy35
KkBK9WHBJ4WgJklkb248qG41J7vmhjZH7yMZTNyDoY+OFhZm1NCyL9B0JIDQdeya3uLx75VgkR7i
/MR10CekK2D6WeHhZHYeyHNaTpoHFWRwqGnMzsMoJwnZ9paA5g+HgsruI+c7QaWTC085z4j57ImD
8BhCsI65yeLknDk3yn4UJhEA2Jw7j/PAPbDv/NmTYs8YWNK5obuFyWH0hPQNork5OGYOjZBMkhBh
RRo4U2th4SpQK0gsvYzlimENLzt3ujj5DOmquvrzXR/WviPyzy/mb5DjbL7PgSkAnvBSuNsLTYJB
/deeYalo6jXN53Pm4BPcRnh4/K89I/kfbxROwZfjelMIJ5nzzE9ysoZGYCIW+oqT8/CXhudoOHdj
snh/GEc2dgcYIgszNP28MPCYUZ199hDKZGfP5y7dyD26kF2YAosdx4oj03cxQHDnhnvNM9dzN57C
KH0J1RqqIgEeTqk1mHdhLOC5A1nqzHXz7mkJ4eyIOXqZPRPw9eUlmB6YTqC47IsLPMTCi59yIzcB
agkwqOtzwziLg0/YC8gQIA1NjACd5n+cBjQUBu5DO+atk/mxfm4HWzjXh89f/IRTMjpdOPEif+Ox
eW1UwkUQMfWKHZs/RIa++jI/NABfthxuNTpE/sFQ8dp3SIATP1nOUdzgkiwxgSTM+0toMEyNc8c4
sY4qQ9YU7EgeMlI79xtg0TeFItKPiYsO0cGiFOZbPfd9T+6X0zqR5R7cgnWlePOy+eKheQ7oiZtH
V4JyKEGnRIswVNfkw5iZQ129AY6L164U5y+D2scOKHIyCcAu+qD+9YOPLdongmRStPhsERYBaDZ9
tnXX1k0bt9ECAlxX7DkOYGbnzrjZj2UVjJGXBOibW8r13LUaI9KC6X76qLAAk3kcRo7T+EgSmItp
QOCSfyU7+yB3+WX+Fihqcww9teTgJWQ8ou/iwAhqdcS5OMj3CSoGujB8wrz6GKZQh5yEt66SAeYd
OjzgnamGG8W5kpTDbcOYbJz4MizTneJxoiKdW4FDyzOnzZTEMctiQZ3/aNZhjnTPtlzyWXV4OM/D
cyAkrKMDhp59dlq6wgloWBVAA2d/GiEvE890GIw3FqMPLuV/upOd/Rm+gF3J64SP8AwrAQN0AZNb
mJqTNui127m5MRAGIIqACuE/5vP81cew3BdenqO57oV1jtHCqxU8+aOx90Mj07pf5E6d4yfm0P3C
48e5awv5ny/DF9orv1g8wQ7+wsJ58+p33AP570dAMJm9A4Wpl6Df5edHUQsYGwSiY+gVCoA8Afes
VErOektI6gcpcP12brbPHLlg8bbyyyHOCZXAPEDtue9OFntuOGZEepTHT5I7FamNkE9ymlch9pqw
J5uUsEgI2fDabZh4RodDBRtHE5skk3wpF9RhKS54WsyxM7hCAx4BQLmdM6i8LugfuDwN2jgjWhfa
sFIqqwMVnYsDPA6mYKhVapVCHNTbkAs1dWH2nypE6EM5NwL6CK8fUtSFt8Uy8URthKWlnB7mmkCD
3bxcN4lyYNngIeZOPwdycnYBE5d7fImFlIWbB5dAnCOPfHW4vR3+Ma3jABqphxsDKOKJ1s2xYWA7
asliC4wRYGYHAakRkySasZO8ejDMwNRQAHl2x5HM/mRC2wIcNgduZJ+dYRbjOkCz8N27O/iW0CcK
53rgvnnqHpOBFPqgC2oEEJD8vOHtyi7quEHUtopggKWCSB9JBzBiQWyggIXKikxnF26eYthBam9F
VQi3ayurArgPZSRacfv7y4ruTHvtO1hi394D/Ke+IVKP3/bG9zXh3w6eu4rdvNGdSR2xdrwFdhlK
Z9qS3ZnQoVQ8Y1Ri/6E2ozXZZlRCH1VVquhe6P0Ax08cJvd45eeJOJbbTKW3pFLJVI3Ylkwe6O6i
H1V2NzjueKLbCAQd21G4w84bfHJpevoz8CuTDSxPKG9LEYqOylQ8kWkXFf8jXSGCb8vnQbDUdBzb
KH7XO2j5IJ4I7e3GvQKJaAsPFo4r4vsSyZRRUVUVrLI2aj+BAqGv0y7SsIFLwHtRa4ggPEpnRHt6
g9yZqKxoT1dUvcuPodsN7Wnq9kOY8J1HEq327Fe925VKthppF9BQJ5RJ7szA6PcRiO8wYIBSkKrX
byudHtU94GG5Fb500Hx7TRiHxC7jcIYnPVVJZKcwVXoMOs6Abg6Lys5Y66c7w9uALA5XlUQfEFji
oKhth4mtxIZE7V5RW9sZ7zRqgTqTMGX7hOyjKihqM5JG5COd2HB3EzdJrdIWMH+MJ9qSh9KasmkD
04XP0vhMMTuQouxDClvicJSDFuUB936580g6Y3SGtn4aQlTsbmn5DLCysaMDMaeztVUQn4e2yAFB
8c1Ge6y7I1MV8Lb0R5wGd1Mhmv4aYcFQumWA/Z0qa8dVKaplNG23rqqUMpcSr/RUXRfnMsvQwwHx
7zsbyn3Xa/Y9KY6PFc6/gJlxa+mzp0V0a+Jg8oBR+5mRzmynDbtoSe0dlBk5fSyX84/m8/PXxcZP
dm7VjGDQjC8OCIsy5BrIVgJH2IHqkn8xlZ3rF1Hnlmfr/lgqbWQ2kMSgVQnV6QuDoL7k58+BoEMf
F6+5E6cLvzyFddMcPVvs6UW7/tR1iuvrV0pm7jwt5tbSqbQll7rO1mTx/JXC9DRHQZCyh9QNxqyg
mBrQwsZQYQXlX0OhnGgyZaXaOXGveOci6LHapIFmJBe1oUFQ2EvwBxdxtE7F2f03cY+h1JplnYv1
VqXDYZCAizJg/WX3IlJZvF1Uvr1j5x+g33gysSu2F4hf/Qxtj30Na2MtcH4T754LQfxS+1ESJFzQ
2iIjSAGzLrpHJMz0s4HrWGxyF2by8+SeeT7K08VzGKQe3u5OG58D/31wBAewQbydScFKd0wYHQC8
DxATfXq/zb/HLbj5y6DjKvUX5hRGXLx5DbRoNjFK9NQegz6sTXSPvY3bfqwT0ZCkSksGjT4KMtVo
+wdo+aCRyuxK1v5bOpmICtCczfk5jrCA2hwXkRs6jVqOpdUhHz0bKn5/pnjlB/ZAsdruVHw1NFoh
WW76cQFvTg2ZffcYQg4HASh9hY9yLDnkzFCAQgwAS//XN4KURPgaZC0y+K6whbUlM9WzPaBfdcbw
VUQcAzXCiRVRuxmUn/0iUse0qE+JRXNvCUtktgilK6Ni6kCD5dMiaROGhWXjzk1bt0qFH9Tw0cto
hxCry9nXpr6cfA/9q5GhcpWEAlbhPEJS1H6eikOjGEIin+xAAq0lQal1VitFHEo4EXQHdbzrlHmw
CH8kwwTflvGCTlZ4y0n8AiO9yODEiBrLpF0RxDTdrw6sHZGS64HF4RyJO+ESnPYqyXpqKacT29vS
9uQtDpd4A0oFASeF70w/L1OLsBCIdzT7NJnMKASulovdBsFkJReOhT7ln2Ir0zMWWGNIa4SVrTA1
mX8+V7zyVFmXoJ/uw4kCGVC4+aPtZl31+P+S539De1IGBfG0vnIuqGXn/6ivq4us5X97M5+185+/
6U9J/rePg76yHFh2/o/6SFPDWv7fN/LxO/9Zt359c31T/fqGtfOf//Cfkvxvc/0rpwJZfv6P5nWN
a/nf3sinjP7Hh0BXIRHosvW/SHND07o1/e+NfNb0v9/0Zwn63yvLgRXof5G6tfy/b+Tjn/8j0hB5
pzHSvKb//cN/SvK/5PrVyAO3Av2vPtK4pv+9iU8Z/W/VEsLULTv/W3OkMbKm/72Rz5r+95v+LEH/
e2U5sCj/u+9/qK9b17yW//eNfPzvf6ivb3xn7f6H38KnJP+vYjq48vzfXFfXVO/O/9jcsLb+v5HP
W+Jja945c0elimSuwj35HSmDYhTT8Qwm60obhkglkxkRDYW0A/EYH5vGTCQ1eiaXGspTpKd+2G90
dFFOB9yyp1AjzJXRIqIy3UX4UPxAPHwwEg0Eij3jhYUBOjfxR3ioklNwgjkZN28HXcskdCtIJ2Cn
gXMlFaDEAOa3w66kAoXpO7kTfTKRCSYjwWiYCzO5kSmKNzsJNfiwpDmKwQNQvHC3V0Q1fzoMFLGm
johx6VdLbnHUyuPC5zRf7YOHMx8Vz0/xCF6xsRXkC1ilPACr2JBMKlAq6H8xdEZb9xutB/akjC48
3922BzMupvmAd4kz3593dWCyJL+j30f1U+xRDHkZHsoN/WA+vGCOjXDMpJ26Yg9ymcw5EMXAVerb
/pWO/wVPk5cbHNL39TE82civbvmkAqHjvJjX58WIXQqBN8eGoV08ZNCe7ABtEkERnMpHno1Vxbdu
9j8FDaR9t/evPcOqpJVRUh5IWzFN6tIhbL6cwnQhFweg0UWn9qg6+96d6oCJXPEHGqIWAEGRWlj6
AMvlZtAcnS72DAkNlxhIONGbv3DbHLySu/EMcGq+uECn4e0Bek4PLwpTqWQDyx6cM90CjgsDs/bE
E+3JkBX8F2/b8B4nxng/6j9yakyNB5NLIdUzOXx3Gka72ODKcmB29nThxQtujh4UfplxJIII5n7p
zd877dsuJrrC4/xjZ/GAZ/iPyVRbeMeOXfKwHZ+Jk4eR+ScHg1KIaTnU4Wj5xBUPc2yEzmzdkTmy
ymPdf6zL/UBDlIYp/eXukBIa6olDBJUUODbP44h80/MsAYx9RmaPs4Plj4aZtk2yWr1kNQy3Y5SW
y8BQskWcJb9URDKpUFi+LSU4PUPriKczKxya/wS0dqfSyRR97Yh3xjPRysj/bKqr8hUlroGZE/fs
mZp9wFoTHm4Z+1GO51tbJHuwBxCljViqdf+ryhFo6M/dRupIGSqTg1xKYzhhw0Os3RSmR5C/hl5q
auSgUiyLveeA6cwHl7Iv+oFI+Ilco3wGtzKqdAyuxFwtvTEc3PXb+YnT4excf+Hu93g67ulDPLh6
7n527uqiS5r/mHDtZiBx3c7/MKdihsuyjDW0FhEMvvLwbOFvnc7sr65WiVSqq1mVKDlAyWewquHp
BdcAJdORsvLwQv7WHDNvYeoHpgXgktzpH/zG96oTpg2PjgaVIqqD6SXAz2AiQkYuBQKUNAozeErj
grKQcZYyPPxAZfl4u0w+xtnKlge1kJmvl2NyQE9nrptXbyyKkeVr7stW9pdUgUBRqdmAsWxRPzaS
vzsjc5OoXD/ZubvmuV6ZUmTuXNBmkmVLDZZV1IslfYdBXOExGkzuOXYmP383P/+A1WGFNmk7nngk
85mzLGANflENsDSMZIn4LcRHZbIzBkumRKGFni1l78TaEJK0KiWmOJhfZkmyJPYS0KkElhQ9Q9ro
QHwJeifPyEtxb51ncWcF1Cf/Tq9XAv08CTShph7h13AsE5RJWBdlYDebDF4yB+bwyM21E8UrY97E
hXj82SWjHKQa9Mm5RmvZKTqTNIBrOiWeyE9OYTYqJ/ArAlnaeqhUYW6q4Qk6zegB06aFAIju+Wtc
vLoaNQ4qh0moKNMVy3cXx4FGLVM+LAohJiqicxpUugRlu5jMHHkEupFyznBygYfzYGjw8Y9AQJ5X
i1pb5F0po70jvm9/ppYU71Dr16BtF06O5yZ+4gPqnOBAbwR0J/PWXXNmlDNG4CkRKo+Zn+nQmMr/
jLz73ZidECFaWysNp1pSvvHscW50LPvyanaWUggApgZ/xDxZwLB6ScFNKpTwKDhXgadRmd9Ebwu4
pdjTA+SkfA0yEQEsw9S/cLciK8pVQtWi5AKwOhV6+vCgP9Ggo6fc6W+BKthG0g//IzPYfo5Befj/
zL3cxQd4+oxOV4daO2LdbUaYfNrp8KLhDT5zB4sCHWUG+wks/FBXWzuemkc4rc5zgxfVdx1TeF5u
ooe9Gw7co1S8cxyPwlIOiI+2bNwsjaHXAXhb8lACvUdg+vEJbse0OJIE4+Do9JU2njNMqHhAifJk
gezNP5qXeoIq52Z+2x6EH5ZpTz9f9VNCF3Cs3qvo6FOLfhSn3iPkIvqPqAuXrtLLH2c05E2meFQ0
lOyzM30omWpzVVhJn4edbZTr82CiLYT++sOdHXxuM11L6bUNaKa7E6Y9hDDJXAbxxL7OjpD1Jmr3
2dXlXUAay/bZma6l46uUm3qZQ1Z9esZZvs/y4wQexOsaqDiMUv8Ztfo87PEPHhVNi40T8310LHOI
2jihT884y/dZfpzpLswxkd5vGBkYJv2NuvtsTR9cZJwZ43Am7C22og/3iVtPYfwbSx1AoYfKaLO3
T/v1KvTZldjnaueoWO/oE0T3PiPsLbfyPr/u2scD/brL4Fb9+1SvV6HPQ8beLmdb/n16y628z8xh
r0yINGg/eD67OmLxV51M1efhznjCKUChz0bth4NXDtfK8mEvD9EblH0H9iaTBzxF/hLvilp029Xg
HWeTo8/utngy3LmK89nZGFtSn4drvSVX2uehmFcm+PXpLbfyPmMx7/rp16e33Mr6DID+y9ofWZGa
hqgpjHSiXZ4YP4vJcy0r7bZuX6AddPNH3shGNf3H02CL5OcH1Z0r1dV8Dl0ZV9ANJ7LIzs5JC59a
ovPxXNCdA3jsTnZ+RN2cRi0GrPR/8iI3MDKpFZklsr9486y6uAeJt5Gy44ViB+P8pTN50MocBr8/
iHfE98I/LLxXfqd7MQ7G24xkGA1yTDnyp2T3ru69BpY6kuzOwFcqdCiWad2PZezt+omfzIkZUJy3
bxM2GKjYtoSpNTBByIbA7IgUM8CXOHnNX0wJ45mDv/YMe6dBzUF2vo+/YzYCEHgiNzlcvHYz/+O0
2d/HPRQWrmo9jGiot8wyRDSYvsXJZ3h1Hc0HzwT7b+QcgLGJm0lOU9Pe8RLeVKicoyV341luZIo3
ojhfCmBXz5EjOMExN4wpFzlrntfSpESp8vImsDUdbWBaBJa++0EHAKPZ2tbSOx6eMwf7ZbKao9qL
ZXGU05JgqB3v2TCU9zWtgGNXZCqUtzZexfzgzZbhZ8W+ERHt7IKFN344nrAuk0mHAdeDF/3e/Auq
O+scmKE9bLPvp+zzS+boU97MVn7ale1ZU2TA/G37Np9Dhw6F/PhamM97zFuPHVP167VhEVln/WJR
wvvtqogmeDgZpJM3l8+A/hB7ZYwX3r8/xJZU84OtxPQQxPYvXVq+VnjNvifZ+YsOmYBBVtduLgpw
vf4jd2UKpLYLw69GryC96r3Sy+HIQ+GFOWbUtrwudTHqi0QMrDOctBl39fWUzTJtLnTUEJKIkDlr
nIu7d1lXbkVenIbo7jba58fOUkYt+3TEJvTpqHSqlLCLs3lJPzDPhitIx8rvjeujM4IDl2kdmXRt
qJa3l+IbOPDEyq+t1gdOn17sGcP1i0CV137NZ2d74BmlDcZ1l9Ln60qL+eyJjhDlGNaXQFZG+Ho7
Laswb4jBIiQnhpDHvQdUaAR07iKt3LVJDOJYlQ8GRMmRAb2gN2yxoA1Jf35rwWr6p5bbFu+e8kQr
/ZSQz2gVmlcPvZkT91Swlc4WwPjAGdnZB8tD4hL13VKo1NVPR7vlFFFZyd9VuURYVy4ZHYsGKkEK
5EWXUFrpl7ZsrRa0XvXagtZejEZGC1NTy5n214Nakgkg2q2oE9o/kTJB3p08NCKlF22r6AVxH+pJ
H+6k8cO587ThpOrlTt/JDd637g1AmTR6tvDLU87uZTm/NeL1H6amu9Ji3IOdU/+LKZ5OrtZ+UUVa
eoDGd32xC/7/BXoa4O926dhCV1s40hCONIabCYxIndj+AUGEKzVdOVB6atY7fjXoVR3L344d2Hfx
+iOgbryjgxUA7C3cEG4Mg5ENrIfre51sQs4YrJFcCcRo8cKCNWPymWXR6ZGCTWj/US5GmC9zkBdr
QCbUBu0BZ/3SE7KaYEZZPDvJSyYub1SkMiaDvBS1aOn/9Ul1wCHH1RRuDq+X2CXIZNSYVZ8ZHWrW
h9exBWzdnjMYicAD0hfoThlr19I/PE/CVgtqT1q93ICZLaNohnvN+IVr+QtXeB8Nd8PoFhpE8aXH
ucFH0DHlA7+YezIIbSoNMJi7upAbuRlkZUFm3ZOrKa7z5stBQBVtx3KOdbN3nHFjb1la+gqLDt4v
ZRuTsSPVAU69TmMuLEyAgsGBIrI3mI8xTGnHKS7N6WcKRDSSiVPVdQPMirg2LfSh72P8JCA9Pz/B
zKluvZ2dxdSpT2/DwECjYRooFdTMwfuDY7jUeXQxEDcAPUVbsOblyEzPtwqZ97/NvjgnA0ixD3VV
l66iSe+KS9FzLqu9zp3gXkaNVXhIBj/7Aaku5NB3SUFp0ovxfcbS9pddgeIGo2EEQA346FurrB3z
7ipeRaCe006rLuLQ06Bv9cm29J1SagtTq7JrxrUVyVulHEJTand0x2dbPty29V8/2mUl0F6NnVK8
FlXulgbDeAYlnEmG7Q3fYJW8Qvpt1XsQBtcVS6c3UAZY9H/IyAJSdTmRrUgZMUxYynFk+fkrzK+4
fQzLFg5QwSJkcuLRMbyPiHLsc/QhSBSK7Hngyq364dZtW/Z8snH7FkCDBzaB9246cou3bcAcsqEu
zL1Z6cgy7swXXldjJTL3zz7eFlKHBjCfOEGx5YtdfxcgwNJUMOzc+n+/YUzIvO8SFjw3QRnWt2/Z
vHXjnl1/2vH3AsfmR4Jn06ef7MIrhd88RG0hXRYgLAFe/B3yDC+8ci65w+Uk3E8/iUa83ppsYVrS
9fMyJNzk5ZqMh3gbSSWUxVK1owuzrSueRdB10iysNxcUQUyZ+1VQ0fxXwRb4+bZivq+CNXYBpAEs
8LaiSH6po0E2oM+K1YYnmJnLvndgLx6FcPQElK9DAgz4VTBwDEUSCBWOZDN7nrMwcSKiNZneYx/J
w2s5WNggMptC7uUR8baycBTop7abziwtSb7KAmkDYMvUAqzB95yQhvgVYsJVGG9YL1Ua3qnimeQB
I+EtSY9Vob3drQeMjLcUPycSUGVTxr540qdFfq6KISZ8YcSfOoCOldBJImrImVgKCsQ7DZ8h47s9
+E61aBzuAt5tK1FBvpVVkADWhYTDT8SJ42f6zac/M2HYGph1AautQJXkKEeTiqVsMUUsY4vNGu21
4gHrp2IDyideihuXwUl48gjfcmps5FWaFFmhxIxxC4vw/SJyA0ocQ5a1LoJ4S+SGntGtK8SB5GFi
I8UKfDNfnJMijG9D1I4TYWZo0OF+msz13JXHhDg9u2XukHYOxuc4WmP2sbJxRzgd9S/NjtLy0cdY
CYoKxF+QDwPBUL+k8X8TxLfwM6hxeo3Qph7eRY7VuAqnO+JtRpqCczylGz2l6bocDHDxlG06BkV3
Y3kvQWBNSQ40l0F17IXeqF/AF8cqAta1jajs02U5eEuUfbhumE/cCVVLSJn6lij2jphnUA9Md3dk
0jhT08+E254Tiq10qw5n5y1RK4LWrQE0lc5DOHhdz4PLeGPP2DCuqdJ4HrfxDTq+whDu9fmYitfp
wg3L2CJzEIgjSJ3rcbpBvVpQwUz3ZjlsTRWJzuTqbYjNUEcTsG5J4qZIaxhb/u588cot62IFu/Gz
L2Ck2u2s6BDXyBytwrEf9fOvuYkfzZmX8tLX6J4/wWf79s2bP/qoszONx7+Gsy9P4w09Ucbanvq6
+nV1DZHmSFMD8AoF/SkXwpKOiNoRwY4NC+Yvc/o5nnWlFtCE9tui005/8uHGIdfFd2CJyRgDqTM5
Gdna6dZoFPGCRqmiUbrKoxSTa3BbvL04C1Fpi/eDltfSOBzrxEQf6LGEBSre2mFI6amK+GxyHsYP
ysjdyH+OMVEIuHbQbNA1LDTPrd/SObr6Yy0pLpaJCWuEb8nzB5akELnLN8yZkzCx30D70N77uEYJ
+FIjULHBu55qbM3uGKwlb6kXG+oEsBBYnOpKG7wzqU4eBhNGKtWZ3mffQFTqVDCsOLT0a/c/D0qz
3bq4HvB7ge1RGUjhUQvUXSp8qta69lGj16XqDTw1zrUjQuhWR4gJR0H7HDFNjjxKHBS0dLDigC/k
XX58HcniqwTOkgtlnjAMOwhdu4gQfcPMxLSEq6uYMUKeQiKs2A10rvHV09O3tXPEKCUvT7PA4Psr
aLW3zyi7tyXlcWN7U9J9oTUmCFBRHpYeQIcJsNfC02nz5UlzuE8d7FTClOUtikqLpmOpw/GDoWRq
H0Y7h1FuhiL19evw4iXcv6anUcWFw479DJ0f7AVL1uSv+flzILrZi2Yhk08o8PU50jfEV9Swe0vf
5+X7CVvpupL0VlG7TRArRSUnyGsWbz1GPq31Cd7GfgG1sm+fUla4NY1rkRjaam7POhEe5t1jR9Pa
3jkW5TnnUxV8N4iDomQwDe+PStgc01GtRiBfV8vFrBe+UYw3f6UwaP7KkcIa2jnkXzfQua1/jWc+
6t4rUrFDcggcn8VTpeWfYNqjFdwBuwzaoY1mucPrcFnmztzEu6zsCCi3I0Of1eBi5Ihejn1AY6IW
5Mx/6IZWC77KxOLQVgS+xQ4dAEFD9xmKt+uPVeDLlKhtExVfpSrwZj920yEPuKjFRaTSxaexKZ4F
nH1MfkJckgO7tmzfsWfz1s9gLJ0HMkYnANeGPcj4LUe8AtIXzTi5XJW6gyeu1TEaiZBtohYdPFbr
YGxjhh2ysRdHk/TQuNzTPgdYBN+Hat4aIPE/rEByOY6zoNmxpNFO/LwRL64vBsob2ct18koHFJB7
sNS2gkwVYcWPjDudVd7cLo41D5pHV/jocbqZ+1Tu4rPS/ahtDukYawK4ZvugU6Y5KWZSnaI21a4h
J6j0azWBRGKSJWvlITF0Vat79jbH013JdJxPNSgZys4m1QxUNfuegEVK9/2pZQUPGVJBeStm1LNu
0LlS+wFJLBxTLe2ejp2xLodDwQArIz8D23fqF+5B/N6XgwqToJz02dzCy7gzg0wg4ErbACuRKsLJ
nPLz8B/f8G1XnvipMPcjXjSItwD35O+OcDO5a5PA6hxiNXHPP7UDuuR8EiPgrW58HpXEDWANryMb
G8adTW1h1jMuWCYKCpBZc+FE7s5xc+EXnEJHdhhbc8YIlvPkNrgxANByqM/d00DaHHrGlaIkDGD9
piWctAeaN/k6trc1Ut8QlTcz4igXDTTywwM+9t8FxbwptE2IFiHvz/IZUGeyILrsnmdRGT2sWPHm
PdoIemoLzmTF9+4xR5MJD5qXtfcyZG/8z/T7pdOQIlB4Exkhzv0mW07p+ElKFZBCHRVdoXRA+UNq
ZCsosV/uRnSyAg/6IoYi0MHt7AsUyjhrlGWMqR1V8ms3kSYenmXOp8uWnYmXVBoLZD2woKzn1n6c
rpgWph/bvvfBGfHFF0IjMcBvvzl6Pz9/JTvXr7fDx3BVDDKeAOeD6U7ic1ipdAW0iCBN+mVgc1wJ
bV1R5zqbq+cbKW1IuPnLsiXo4Dfq+K6RLMEaoBKc7wEfSy+Rcr7jHKNBQhNPF8ERNjiq13aiguwg
GUEjvAjy0rboJHEp7AVsjNUjxjiPjEXq7DQp9oyBQs85TEpjw0uZSzd7tRHTb0pzga62Oh8EODux
seBi3aUM3rqk2p6oiXvmgzEePN//x4RHsQUWALY/TmLm6UPkTs3jqh3g97oW8BAsOlH5oneWQksK
pfQXZXzNoIxu88oiF3e4Fno3aK9iNNfYj+QUK+960Gki627b1fCLKI+7ZaiTOx2LhkIh9N8qdzrC
Wd/Q2GQ983Eko/ucXEMOj9oyUPUaHT9lfT5el5ZzxdIXJI0H/stO+epPqvKz+CYIW4LryJuXBPES
jLfx7Fjw79Y9bzgEl99NFsNxxdtqBMJbA2bQQZgQ8sS1GWyNgCaMbjmY5E4jgcfsYWVJ49M0eeZs
t5FfbjDHmugU68sV4ovMllN+29K7vk4SJPe+VP5ZIQylOagsfGB6L8xTYN046v8J43BmDxcHlomn
9wDieTMFdeTeCTTPFN59U5ctgYz8FAdNa+D0IFYmGoR4Gfh3uBNd+g+t5g660HOUMfRLBtom/3KQ
L0IZ5C9z5cRBtdJKn7Ms/csHokVAcAkEUAe9+W8srFVXZ+f7bH3Wk6ZHpQlzJQlDbwXZlhRJyvqG
Sj8DGq85MOdJ9TQMtq/K/rKkbDIY3tKD/vjcg1usUrjC9fjud3Zplue9MoluCLmvhZmkdSnNfhk6
otu7qCGpFNJoz9ABT04fLE2r8ai1+qDpw4GTlEgPdSdaiUiL6oql+C5wS1MiTEnaBTXSmnu73500
BUbb0vtftMfe6P74vv3kaNojfUcIdPHHy9m5KctQGZYqJKn3mI6IVhTLAe9avhwIwhTbpYCU/VOq
TdIqrYWHfmlrD/32XX4kytzWhm5u+vXsRIACgBft7s69Rop/czHtiS8CgZ4yya6oliGbe0Ir9dnQ
IsmwzcF+9IFbqriVg1uaf+wZkFybm50FfuYK0n8VCTn4Se4GAJlTKjDcSWAnp2aZM/VHCQ3o9IvK
vTTcepkYyl19bN4a5/Te6Jg3Z/q1uhofWe4B3Lm3urcsf3yNWY9kX2KDoKhum9mYOXn80r/GISCM
BY4Elf6h6+hyIt+UiH6DulQGtwCB0euA6XmrT9PFMBaA9Bn4jbpJVLr5orLehjryEVn7huT3wzoA
HLntsrPj5qlJyfd2tV+HblBF89bDwuPb5EqSG4LoO2Eoou4w2OpqDhWX8fNkvxUG7hcmh2H8hemz
xfNXCtPT+RsUuS0DCEBSFO8PB8jdUxwfK1waBdYrnrgntm6mKUdnyNhZlcWNe4QlH6hjZrawMCEP
Yo8/Lk787OfUQXp1CAfNTss/GNI6wzWCvWVEfq6lAc+p894Z7S+qn8rHAI3JxQWTRc2zaIA2t24G
I7qvZNwvYsqFRuQU3JH5daJPRIO4X+tz0H3YlYbwrz0jv06cAxagmteGS9UU7OBrM9obm9YJu4p0
89Dur+sUutmndon90gPKFVwtoP1yPDSl+Z9f5qd6CgtXzbk7NJ88X2C+5x7c5lnjKWBXn5Vr8cH3
6A/QwvB/7bnDD+1QfHgUZIlsjt7PnaIz/zwBfY9gQQNiscTzOI1QN8BRGAQdDdomC7YrvdYrQr4e
UaL1w49LNb5Im8yGKE4ItRoHEsOxs5SjvWQeCWT6hwO5nnl2OHN+Mui0Qdh53hfrHypFsLglBAZV
4FGL44gWH+bAEKMoQ3LjOSULuMjQcpaJ6Xk0/B7OM4C41vPGG+hb87flxQrDF2VVJ0MoFQszi949
LTCamcScQApzqovkAS6luoWXklBwSDb9vkeLpxMjDocUZunkZLJavkJ/BsmNjhYWZiySBLbvwdOz
NFBaGoVAhUyI/7x+btzqWU7doNn3UNDEaar6ENM17fK7pg6z7GGc54PLufMv+V3+6ixmebPSYHLR
wi8ncxevQCO0AZ87BW+m0SU5Oag1MnayOInWq/6aKQsq0k52/sfvoAdYpnPfjVFFORbX9Gh5s3mS
fDzqNvYVMo6L8oxnB6BQ8V5RmHpZmJrkImH74FulRvk1gun6wZgS4lVW9SzQ6tRUfu4+iCm9eqTe
ri8LnxQMDshODFegHimboHyro4y259ULUjcYY8RPtM1ZKzAYPiJIbgzr6gmPEYvoqNWJxJ/sHdTs
syulCJn1SN6JEq+uTPtqz6xh54cG1G6MPs/nRxxGve9UK0sfNzTOfJubuI4HCCxJIgcKDEE4/vcI
FwFyIxSLSjXXLYJJmm+aCVdRVBdgCFQoRgHglupWVwMlmYPf4u7d9J38fD+zL5RjpqGOcCNt9Kzs
bOpeYWoOacDRnYOmqphviNi4Vu7eZO7aQthRRbmMqny5iTPv+vGRzJDr4aNx4UIoP74mWDChrrdk
efGf17+dFLyXU7w0BeqQDFih5Q+Y4SLacg7euDydne/JPhv82/PrqqAuUKzcwPxemaYUpJudOw3z
FfBuwkl7HG2GS8JK+U8piq+aIxf4u8g+u46hWMN9he/7MEvEhZn8/MlSG3qk0HOqb1EYPmFefYzb
Fpj0mxc2ZXddntZNVRwt7RNzfNIr7QuX3xTmfRF5/pKNLpmTx76DAA0KTc0Ns70Xdlh7YZetpyI+
qLISAGHJ/rxfD/Tnt+tIZ3VVJLWlBPvvKWJAHS3lnGgf1RI6zq3pmF4kUTCWH1LlJhxFNNHU6XuT
qIny8U5Wu2VIQK1zq+Y17HDjJF4bwGAauoHBssKH/fazHVsDTvDVPjfuYcvdbdAUHAYMGEQ9zxfb
4y+3w29lslr2ljis8jZHqCMOQtKL4FkFmwisHqFdS8S6oFSWXJfQsDKqn4JQrREKrcQlMt0wR7AB
BkgpZD+EHjfuChrXDv6zTZo7fR3d4tKcO64Bycwq+8bhYtrrHlQYep5TpEgeLE4w6p//kL/+AI+D
k3UrD/ZSNDtQLM8p53QB+wZ4lJO7UWi6gpOQywflKWkgku3gFVEKmOKJFxIHGP6g1cNoGmdop0yT
8OB7zLItI3SG2Qx1XC8lVL5q2auBORrVgdXcpRv548/M0aeWBWGH79hH8Di1tTITgL2LV59gLumB
ObD2MWJfi9zgkwT+NOBolejJkwwb7AUiQHv+p74rXukT27du3yLUIX8C1R3+ySTP8gdzEoGli0JD
L5ZszRiZ2nQmZcQ6ERrf+F6PRSjtZEuSMUa5LKdhobhfFdwrJZ0dRGUnMvm9HgDrCHiSZwYcMZd2
FPbYmcIvMxJtMtfPVb9DDbKi33GGtSs6/yE/i8ZdhvY4j3auoI9l3/9dXx9pbl67//ONfNbu//5N
fxblf/se8BXLgUX5H3jeyf8NzfVr93+/kU9Dnd/935FIXV1jpHntAvB/+M8yEzusqI/y/B+JRBoa
XfwfaV7XsLb+v4nPW/8U7k6nwnvjibCROEipYQIV3WlDgJETb81UvBsIcJaY1tSRrkxSbBAqQwyW
beHHFVXvymLtaU8RzCRjvaY4SU8JeoqFAuEwebo3pvbRzRMCk9LEE/vIt93enWhFK4weGlAkXRlL
7TtYRbkcuHn4ja1/c+xdeISXkld2GBkRh0f178Kf97DAwVCHkdiX2Y8Pfg8vqmQuCG4BM3dsoGJf
xneHUkZXR6zVqAz/R21tuEZUIIx22YOxDlUWTLPIbn4ZbxeV/4Tvjh7FIpyoI/3HeGZ/ZUVtbUWV
1SE3kwSZaKRSyVRldHuc7kvBSt0GgV9b+/Y3ANGxqOxXCCvzjnE4nqmMyMfH6P84+C+h9G4ACpp4
N8AvUkamO5Wgt+8GjlnT+dmWf/9862dbNkPZLyvQsK+A4al0LNoPaBB/UdoU/MKZUfAb5z3BbzLl
ScVuewY3Mbnwhe9p5/ztB17buT8WqYQ6NQJ3DRklElImqRAfnfoIylZWpKE0dATlq0LdXVDDqKRq
obb4PiOdqazYbxzGuYHhqV6wTuUijcfS+2Xji7ZrjQuU5I3dmf3JVPwvMR4NaUeicsfnu8Sne782
WjNVNFyo8JkV5tIirADh1o5kd1sogw8TGQoTtm5ZCcPUtnW3ZsKNDevCzc3N79hD2dsd72hz9Fr5
jeD52dpWI799jMjsNKAQPMJzJhy+yuCloRSS4a44PpOZYPCHOKbzDyCYHm4Q0be/URWOvfv2N1qV
Y9F3cesAxhcJiZ3xfYmPiWc+2r5xU+3OjzZGKnfa8HzMDVapHtKqvKICDXzZf5XVQX1IfASI41xX
UIeH91XCGh5+wxQkXyXkML9KcMUPgXlgQmpAoMn7/rjguxY+xCGDYIGZSaYzNeqwI8sGBS6XBthQ
rPDshgDIdKVspiqUTqYylcSHtoT7SPaxQasf6ox1VVYeqBIb3kfkHghlktvwNqNNsNhWVh3bAChO
YGjN559t3ZTs7EomABqrly8P7K4CGRD6OhlPVFb8c4WrO4UeaJcx5Gr8q8Tb31gogx/4UwMTnqgZ
bYAZpdZ2JXFmoU3kjq8Sch6/StAE23NSRQiXU+usGOWaJMCIbPA7MaUNc5XedyNTUwz41HDSE9NM
jQM0J0lZlWyisuroUCm6agqJD5CnnKzsmvVteIql1BS65s+am3d5bqSw4YQQ0T/XIjS1sY590FVm
f+cGxEK0xnoZOwCTb3HzMfs5VSLd3Mah/RaelH7JMNfiTgS8t4ejFelOddQST3ApZ7eEToTK+s4V
d+sUaMtEeZ71YDwmbCHoFPmsQeJ6ndbljTVqXkbTX2or0O53XaU+tlZmu5hcc6xy3wjO+XVMltPf
yERffq9kXi/vK1jW9E61Vc4qgQvnDmAsWYZSAxKNWRwhJagNuEroVbEb9QOZxW87tBFq70iCFrAZ
FqEQqOKVVQIELKiSVRq364LbalJP+rX0RkFfaVgHfzVoEXjpUobG29PO7H/WUHXZA5ITRQpLHsbw
MUyaFXr7G0bqsVDnkT/zkgdLXdSuakkjrBp++xvGtL2wWAIUNAgkQQer28KV1agKp+iuaLHGr41H
qnxVTOQId4uCHp8d09AQc6zuG3wXX2pGLcDaLxQ59JPlcIuoAI6o4EdqTWYgeBgt+phkS2qhpp/a
nBOoSoh9Dlp6Zr8hYq2ZbtA26Ri8xIWg4+6giKK1HW8zQCNIQtHUoTjUaY91dIi9sdYDiF59R0Mj
fGqFDlLb1G/vszCZVZTaGanQsJns4nMx1mzpWAeUwOLZIhobG2wMtbjw5INIhTpLj67Qd0KAADT4
a9xFtllU4iUPq6xjslucBKHaO0yJ/mDWu0GmH+Fsg9Aq/eVCx9yUBUYPrieoCYbQAEINU+KnRlSm
jDStLDwotFv2JttQ+lRUsJYPBUJAfBWooYIyXNm6vzvBi1EllQRjhh9VOcsbCdTnK7XG2UbB90Br
me70JkzI8P4Gytz9z/8sXC/eEw0oMKyqttnSkdxXGZXCP93dinZJe3eHqKQsvigC9GaOVdlGjMuM
qVMvjgkD8y54u5IWEmrfvJQAFcP8tZXsrAWeIVaOlew1YvfK00W/JXvB9DDusF/EHnzREFgaLHqC
ncOXUCd0FttnKBh8+tf64yyqGllKVeLPIZjBSseKuz0WTzgX2E54UuljCtu2stU52cwaVXZKq3OD
sglxGcsYKUvV+Sc2LHcTOEg4soYlVP1xYhmz0tJvQ4DIzkkjdqwmdKWYjF1bzQW7skphztm4nLfo
54jdFs5n60pAKtOCvIfC5H1HptH30vG2953pRN9LY5ZHlTP0Pc4RaqcHfY//vm8nAX1PJv3U8n2+
R2186Uo58l4nZtvcjc+1ZJ7vZdL8zJGxk55KRawMxThXbK/6IT0Q7VQrnUk7l+8S04XrPCAS2kx2
J9pwiqwaZakXYdF1OyJSpsV313y6/z/4LGH/1yfzzvL6WP7+b0OkuW7N//tGPmv7v7/pzzL2f1cs
B5a//9vYHFm3tv/7Jj4l9n+b1tc1r69f2//9h/+sJO/ecvsoz//A/c3N7vivtf3fN/RZwv5vGI9s
VYsd6uoaIgTan2TVH828GPufMkmxdftGoc67CzzwHoLq2MIfYh1x3GNLy7LonhGYKKZGxBJgOh/O
pGKtmbRA/1SnkYmho0MkDKMNrEfoDZtw5FPESs77C/DobCtUT1tdfmaAkdPNu8apOLrfj7TgczoI
tLXdnSAynlYeM2odD9nvSwCEnK0R8RLPiEpHnVR3B4wIz+rjCIxEGjqr4j7qS/TRnbAbrnH641QT
3EIDtZBIuhqpIVBcZRu5rBFHTx+gIYGOnGTHQYNgR3+JRAqbzVyJbGe/3JrSgPa5vGOZFfGCOqQS
o20PvXChwxWFLUH8tDvT1Q2YxrOcQB0dh2JH2BtMF85UtchiQuyIpdNyKN/wH7qXII2Jf/CUfY16
SNHrCBQejw/H9hJhGH6Xk7jq+F1q4CwBE4EFfN7oaZDW2e882Zrkc/3mGGzRhZwgFzzmGL2oTGhk
o3Yy+YYui5qrVgVHrqkshahFiklsvWFUfQj0XwoJlPhhyVigyzrx9tFS4y9ZQI7c+YbvDMPnf8B6
JBthUrF+lYil2CuT7u5C4jPaQngM0TW4LYfj6LNvMyxOqCPHH5AGndElrgOpg9s5RzSxLeUg7vVg
PmS7bMrADTx0rnYnVMdSXJOcw6kSdA6uRhiZ1pAl7qApcifZTSmPEkoskDr8nmtBpXBgaWFDXezk
chbAh1bQ0K/ne/g/QIbkBJJ62mV1v3dyhqrxWv+zxrfli117tm/cIbdBgD4xJ4cNHFK2A7wW5+4K
VKjg/YS2ZKu7akPZqpyy3K59eHnVfRKbJ9vb462GFbYSwualaxAUgc6OkPVG9tnVlXF32bhol53p
2i7cXe9KxvWGPMAv3lJ54GEBS8NfKg6g6z9lr4c70u5Om5YCvoHp3u02PJAv3kh5yNNdyM9p0Eoz
ADj9lb21pg8u3hmlnoeSsk5nm7tKs3+VTnnlrFVP/lxZ7a7EPnfF9Z6KYKHsA+Gb2Ccrfd215Epf
dxl2LWMl1Q4Ze7uWWg3LymqZwx6aj3j5jFDS1RGLK1rDe309FcvT+OFaqmXNR1eDp74PpXW3xZPh
TnuYnY2xJVc7XAulLfTEPKRWsh6UlbViMY8MK1kLylKtY24578j27RL1lSkDlinQkDuSyQPdXVWv
S9hb0l3PKm+JeQCYAlC5xJc1HNW5WyTbrTgygD4VN9KVcm2QOzMYvBZPYawHvD4iDsUTaVpODxhG
V5r2+0GzTyZg+jtcqh4szDYOrF0gN2hfEhiOmxp3W1tCSynLsaXw1O6KN4KO4fRsQRMO1PU4WH1p
WvwpiRloD3uPWGtgQA4f9JP4vkSlu9MaAoZ5A0jNEhgoUvC5Y2lr839crg4KVmIYXLcO7AX6qCAW
c5f7S7yLX+CEUAWyqPYaAt4EjnnUjp1KJUrjWHVi3Hskg1v7b0bh8CPP7R/ArEXq6htBS8M/llaF
N9ft2bZ1+9ZdO6Vi0tSCV4dXQw0aN19ZEhabdv4BXjY7X1qXj8ObSIPz1a4vduHTRudTurUcnq9v
wUvG1eOtKDyRvxmozVs+3Pj5tl17bOAEByZghXexwo7NH9bQtSo1YseOXTViIwoKqVK6dUGc0Him
44jQNVllLKX/jpPy+Sc7P9+x49PPdm3ZvOcPWzdv+RRv9YShJoxDYqeRqfyyAgwA3ASPHYzjn87k
QfpzgP4c6qQ/7R38y9jbSS8b6Weq8+Be/ov/b9jXVYF79yX73eToFhmBDJgw9W//BGZM0zf9Iejj
rQco8sxZMpZJJdMHYu76tQS3/oxGYD/gkQR2e/jrIxlI/uYnbLWnX4VsoNVVyfd/2YEbVlwNXcnL
savx9iOV35A9Z1mqaANyRXGMo308e/N6JDxrkCihLFn0jVwe6Kd4DziryopZjb79jXwcJmFRFcok
P4wfNtqg2WPi4w+iWnyrXhaacJTcjiWPuWbRc67j7z4fy5yzVTt68vvfW6suzoN13sR9VASjseho
ibO+HYxFoTmlT6vs9jufIvjgS8Q+O+J3TMQxbxRv9Hefh9WbzyWGSJEGpSJrnEzKcTMVeghSGW+o
jEMqER1U4eXh+irtsI4W54PthKRft9IG7V2t5CccV0slcSsH3VJ6+K4VSpzZHsu0YptWpVAnPqgM
fxWq/PI/Qrt/X/V2WC+O3h2r0r+or19GdjtD4UULRSxytXiiqzuzaWnhpPZZM4QaeUkoT1yLwkGN
UL62FgW5fGig2YWQSmshEhKbeK8C54CjoQLIlhgp+G4A9WucUvzFEc/4zR3tfAwUPxwyhwBakhPD
+9DlJjZs2CAqtnzyKeiwFasUW1VfZZ3dyuxPJQ+hz8xiyXprRNgc6KO6WkMrqLB0TsIDiCEd0VVE
0b6aR2h/LF0JVaQVQmsTZY3FucBk9+ikbBFRp48SzwVljvm6KT9NgNZlAwfyAzeG2oz0gUyyC73+
oSgFPx4rAdMmBslNQMuC7+1v3NVfFdYAn5H5TG6saGo+bhfplhKWRA+9tevkvE6tHU28GjTnEqX2
f5BOqX3JN4nujo536akzPJufW6c0M9u1Kh7u+xevjecusxv4F6jUaI8njDaHsOjq4jBRJKx/say5
L+GXqwpNqA7IPzGQyr7dpGNC32IDM1du9aDZi34JbTBai/IcgYYE9yiQbSmUGGHx0JAvIGrjb293
xrFDR3ChrFBzQ1OGM2YJA4UeSwLogNtvXUazexBaQZ2SSBjogdFM/HItt1hA/lSM8Lk+AsexAB+u
ID6GSY0SGUtpaQ1WcnmUBPuxkNi1P67t4OLOgpOdZASrjKcOaKADyj9JOhlBYZ22DErshL5ZJMvO
qhzY9hE1Dgz742wJyLLw5J1gb5cVtJKAWBT6lh/NmWeLWGEWVoxYAjttMzJGCmxww4YnVGFPE8m2
RseKifs8mmJBPo4NtFyG8JUlHfD7NnR9wEvNp/Clmh9a372GvRQVqun37ZZ0KR/wkLnaL2xRcNXY
FCEdiKr3Gp0K5EvPCRE1qx9aA8dl2jacrH7AskHPvtHGvjhHGRv6Y+wKIjsg46AA5g57MWkKiY1A
9KQwpsnIM9oCuiEIpd22IGsMZA/K7dvXjqBjVTUBgNlzbOPvHb6y9nnFT8n4L+2KhtAevqRhpX0s
P/47EmlqWov/eiOftfjv3/RnKfxvh4CvTA4sP/67vrmuYS3++018/OO/6yINzeub1q/Ff//Df5bC
/6+2+i/G/011kSb3+l/X3Li2/r+Rz1vo6rKv/9m4Yyul1v91/MbfZkcxT3dhej4/dyd37TpfyPef
14fHRG5ixDw1ad3OiYm+McybUgdzinsr2xFQV+jPf6bcD3YxdHxysbDrmjsuZI6MFqboHvVxOty9
49Odu/72fBitIDoSz4UooTde9vTyNDeGxYS3Y3cfPneFRq2hnf9WFKZuFaZ7rZu3ONGyzJU9NmfO
3pGpKej6hKPyh3B/jorC9OPc5TPux1Cj1vcj/F5Q+SgyqBxCbWtHHGwyzLJ+VGyi73iFgaN9R3n4
d8A4EiWIMDQes5m44InqcTRRbQSOvNdfg20MncJc19byNWULPxZ7rpv9j/IPLiIiZBbsk6Pm1Zce
dKg2+f62Uu+X8SmNyfIoXcmHOsMLR5w3NGt3kYw4R1ki7z/d6qSfX+BHmDtA5lCgEq6s9/6du+5s
1UA4Ksy+Qdd98NQwZak/9bfng5wf3XmbwNBSEK7uiffLHO5ByFFn+vBXm2vo2vdqYBfuqVe/S0BW
3KvvDb4ibF2dq0+Ap2t5ZdGgOTpN+fi1SxLwxrO+2/oFQ75zIIHwvc7Wb+jeuzFW8rGG7ndJ7VJ6
XRnOHdNc5sJXBcHRxW/rWt546WZIdfUCp+bHKxtujThGXZLBlz9iJU75xie+5owXI8+NoXj7rvti
0SEUvXwXg28H8sID+cO9JpURoprwdMtRucq0+Y73qEy3xp2riTs/nZ3twaWK6vK9FUutS/fCWOBq
d6KWrmvO9OK1kLAQoUJi1dTvTy1Vk+86cmKoxEWrquKXu2lhsy87YjKl6XVMpGMSrZtZS8+gNnlL
nTo1Vf5zpq7t8Q4eYKB7FG11gmrIa0X80AVjpqtRXMjyXALmqDHUYz78zl6WiCAISfb1tI6bil4F
SctCEQOvQe0zXB1sUYqS/SpYJMx9aLcsWVXiicy6RqGqgDDQMOq8iMlVQV9BqNrik5AffOKagaPC
urLXMZC9yWQHcwVJwvyLqeLkEwaL5mwjC14/QeUrvv2FVukZtCfQf9J0QaSPT9kyOE/aNPmVkVMj
R+S5Tlobkvca6vJyeCnjWuLg3IRVcoQlC6phOiWozv4OmVkKHyVxsTinlpdnzpL+61JJ6bYkUVdO
1nl69xF85SSft76XAxeRg94mvDcn6k047yP3v1ERVUo5k9LIs9iTSpdTIpY4We6ZWsIUWUTIe40u
2STVa7z7bZCNDBgM3oEFRKmNZivZFfBscywTg6JIsub0c9CorasevWNa0YiWPB5bIdFHwyCNn7T0
ENcMpwyc2zYSTSDRG+qxdt0GdRk3XoNXt4GvzdLr2cTt6Yzr4nVP1tWdNq1bSy0sQI6FVqMDz5Vr
zuvb+Foo6fpYvtJSlnSWz+OtyfQe6d1QvSt0oFVtTj8rPJx0Oz0sLKot8qhdvZsX1qOO64uRYR5c
Kpw67jTRqJGOWDqzpzPZFm8/sge96FHHaj7RAzORXZjKncdLEIuX8Kq9/N1v83N3+Wdu8BFTNUmM
NB3UtIfjVSWm++m6NLy9LDcx5AOOdi1cyUaUPu1BCkuKlNGGPqZYB3RkX7ZmDjwoTPcykTh9GJas
kZfHuRwVhb6Xhanp7NxZnJGSlPNKZLMcmqHEfr4ozs4+xgvldn368ZZPfHuNcr5Atah463Lac7c0
1+ta5Fqyrp+DjmLCJXUJp/ZHk4K3ON49vSiByUyDVkOeZgoLA3jX7GLNYOiUP5FSZvyuLr9VlWpy
DkVFnq6auclSpCll5T7dcPNWxjV14h7ednr9urd6a3c6k+zc05bE1IQOuVkYuI/esmdDUA9v2PPt
3SVqli5niKVU2CKyLoljpvbcd+PZ2ZfWYtXznObCjYOjzkN+4fLssRofyWIebirLXq/hQ4BECAU7
Nn/ogNB7d+NrxAdjpJ76ZaXEAcon4Y3ILnRRK9pESlnGC0oPGXv3xGGxDdkKz4b3QFl4fyk+17Lw
NFDneArOAU3Ue/Sf7rVfpSP9US8gjdTvjh27RGlAPCf6VwBTmZP6UQKkifrlI4uLAEJn81cAg8+Z
e2rFOkzPgKyjftk7bvb9lH1+yRx9ym7yV6GXYVJzScUHHbArdMiIH44nrF2udHmKQriaCS48trk5
eShhI8hxMt8ej3341oOpTi9R2WUXp931jKCrL/NDA06iUWf9ozXWDzyMb//CM/aLdbHUD8mXCLH1
T+cLU06pW3aaEknQfzzzBGQCf1fO2gQPGQEbt+KV8NPXlgpPGhk1mfCCpF6sBCyCpwE73/WFm7vt
1AXRFY11+YCQqkJHiB2A+KRC8GVs55lvT5G/xJdCVgRIExHviwvF64/0V9CeyqoQrezsaqiqsR5R
xgR41hiznx2KHYxWwv/sJ7FYa7QS/le1dJysI335bn/x5tnC3e9z340pYOji7tz56dxwr7xOmII8
5OXEdNX0B/GO+F74F/5TsntX914jjHc0T8x8tGv7NryYe76Pq+MV6MB7oGOBQXoz/+M0GOncSmHh
qu2KKgWlvcVBe7+F6Tu5E32sF0VCwhy8as7PsVIUCNAOvme/Xjc3AoHq6sLdO+boWXNiLndlmnb+
S23Monfqp5+ysz25B7fxYmfaWRJumwYwQeC8JW9V5rvUl+jHMhf6zMkfFzNarPnyVXKcxsuiSlYp
Q89h6V2epge6xYfm/kRP8cICJwKAseV/umOJhBIGscMidrZZzjJWLeqaa7QklNrN3KVR53f5vE9b
Ttdoycbsi8yXgr4zebCw6M74UredS03/LWkMM+GUIKGjzrF6SMePUISvU0onCLfDU9vWEl6PpzJq
WpW1j+i0bX9p2eh8ovHyW3gMUTr4CeGlWNfhTlou70r3icW+ucGLvPrpMQjoWVsuA3umwOZj3wkp
8ym5gyr8XfzLMaL0ec04YnNsd6H8xQSrz3Q5dnKC7+cqd5EPmaV37N6cTkFGHUzQJpLTZOwK5X7U
Nst8fOrejvShlHWzewfiIyHKt7+4sHB24N0eXBKm2MOPyttEb/7CbXPwClMwhvHceJa/OgW6hC5A
eUPH0jCdfWl+fFdfrMLqnv1Fx2MZG1HHm3J9EM85dw8GbQrdUC80ehjydhTydftbyLo2iciyFlSb
vxftQ/Wjq8HRpQyItexlIY2WDu/EWF51/uXDLha6SnFP4W6v5ppf4opSwm+66CZGuX0Efb9AyfuG
kPCNvyol+L0xWIuIaFeoytL1Kn24Xjm7FHFrbbV6Yjr8dAoO7oB5jNTW18H6eB+mDjRtDhNa2ry5
Vx//MXlHrs3qEobl476jYQJ50UA7Y11fvcejrfEGG331vj5wnuvc5RvmzEkHWTSGhG+A3NLIAkO0
3oDqvUqat+VITqWTVuSGBZVXc5ydhcULt/DuXMz9NAmsnv9hjlXu7OzPTtCidFI06gTZR+++MFMc
GC1eGTMHnxD5NdV5tO7VVJRLrXi6qCApV3qZA14xB+Yca9zK9WT5zGeyF1WYS7GMG18cMnjUGbFF
okAPcrDCuywcxdN7jIQDQTJwRnvCETS4HfbyJdfPTfxozryUaz1YIns0wnKujPwkO3sad8OJrpxz
BBUp1gCzpESprB1MBcAL9lacM4dGCk+nzZcntdo2IzcBI7vCPcsysjvk8/Wx8atxL6OILgCOumDw
Mi1HtfQ9Kp6fKkyPuGBYHc5fBQ5dKguVW2aWY+Ms+vHMUGlJunTrZ9GPvZypUN+j/gFLvJgzdcvw
kEFb96mxDJMan6idGr8wHLUV6uB8L9vbyC7J/q+CcqfY8MoMu3tf2bHyrm2psc4rNWh0S5UaS1EM
xSvIixUJiZVLhhWKA2v9X96ivxpigN8vVRQsg88Xf7tcRrYUU19mZk4ld2iNHuC4XE7lt8vj1uWw
Ire/dHbUea3ZssDKHI0op3aXOSbh65iTB7pOPNI9dLnBi9J3wfGGmsMO7aAX/WAK5SenzKlxPPMy
f1s5Osxz97Nz9m4BGL7oQSX9kHUT2d3EUO7aCaB3vRd9nEvy9b0ZefFfmuu9VFdWCpQRAmUOiSyB
z0u+lMFE5c7uHC0ZYU5CQPGBl0qkiu6rXLvGXYZVy3Kqr+rvql5GzCwqZXTWfweWWefZJHY3lNww
8zmdtCznO/AjOtypT7B4YPQamw+ao6f0U2t4II3H8vRn88FlDMIaG8bzU1QGGDY738ehgfyECdHh
1BORcEO4MdwUbg6vD0cawpFGaJcdK1yTnYFQpz68Dl79taeX3Z3IHRF4kH8w9ArbeL7zs2ruhdLv
luZxAKU01uk8XXaUc3R9JmcYE2Cmd2Axh9/KfNknp3D8pDXHmvOqTrqvhhY3SkoRsFgdT4I//hdx
LkiOcDkXqqtLYAaovbp65YHu/lPtXi8WmUl9v3bpcbkWBvWdIC1eXN/vAfygG9lOvKgiDJe0cCxn
kXCgYqlLgBXKiYmryQnpnazP6CXTMRMvaXml5paLl5rcpc9uad22PB+7p3Xp82otIJaYjgrtxFUU
07pFBTBr/tacLk6VjHUsEOtDMvwft0tKrQraYWzfxUDfVgU9zzd8zXnK+3ju6kJ2/nvPkXXc9ovq
WzbhdVFKEPDoAi4rFFWSG3oGOg5DjQ8poUT+x2kMxz11D3sY/JGOTfAEvpJ0X+Z2wlJE+hIluE4m
5QXsckIYbPopL1W9QRveo0beNn1O6/tui1A0or2gRKzdkNz0KHwpXvtO13lXSQQt13O1dCm1PN+U
R5A5dlMcx5NoJwW+/9r/LYsy3kZBPWrsR+0MEOg0tlNKoqvWCp2ypo/HHgi4vLRoaanDpg+P5+fu
8kny/Dz8N4CMqs6IXsaT2Pm7I/mhZ7mpU57ze+OYzcUvjYG1o09DAJUPhBLuXg734Siqq7kDlCOD
UYfzDYdifjusigAMXMp2UkdZ2ZNhai5/NhMH6PrS+NUc2QA8HlkEEB6elbuRtX4pEHRvAUCefXY9
+3ycIedoWls/uvUMmuDMLyymHCzGQgdLUygbjsIp+XAO9ZwX+NsHl/jYN4MAI4JVHGfPoO+cG8m+
mFD6DkwqoNT6hRJXFRcYvTd3RngEDiJ+GDAoI/Hu9IIgdXZDYpKM80EKb9IHI1VyTGThI8rwpFnP
c3PkEU4VLVsKOJmux5GRhFYcAIXNfnP0fm50LPvyquvINLsJ+NSZ78wOD8mCNMUySAjL+lFwsWcM
T7TQfiU2O3/GRdO5oZe4wJHBoyDhGBYvL2L8G5vogYAMkxoakYmCbkwW7w87A+WGC0/60K/BD+fO
wxKuauVO38kN3peHo3QjyV8AanaT4ENm0Ad1s5ix7hRxbsWMIueBNnd9sQv+T5G28Ne65QdKN7Ft
FuY4chD42z+whKwnltsF83rHrwa96o7NH0I3eJAB/uzYgX0Xrz8q3jwLdqbZ9yQ7f5F6Iwsx0oTE
DT/xPiBqAqTkEoxDWI10vOYu3dAnEHiJe+QyxZvXzFsXLffLuIyarBdQHgiSNBGLGMxzIzCXuRvP
zeejVsooSrRkUdZ1lLYki2l6o9EoZlIKYAbfIIh+lPzBFlGHGXaDRirVmd6H94DyKc4gPcU7kIN4
W1ooFKIszdAGCWlZfUNdFIG0zowCVYso1gHuGB0DiZmdHQfNSq62drVfh25QRT5dChWrqzm6zpzp
F1EGBQTCwwvAyfn5K8ynJEAAecWJHtDTCpPD+YX5wvTZ4vkrhenp/I1ejUvIZwKvYWaYfwt3TxaG
TwQCkRBIA/aLsSrHybssD43bVxYNBqOBessBYoeVWb4XrEt34ljacmHhMkrvq4/NW+OcVSXQAOrx
TL9W1+ETAqmanZ3Lzp5WEFEZ+TrQGLL6EhsEq+W48PVO5B7c5PHZo9YQcdT+4dVZgF/m5wpTU+at
k/mx/vKKiyhhcpbWS+qsTpgotF4xxXp5ZSkSqauri5B3Ddc65JtHF6zmeAo0/Rvtzek7aHIywVjO
BGqlHqrg4vpiyqcV1vf5dSk4QMCQCw5oCVg8Pz/BGGXXX3FgpDB9ofxAIhHVQLFnvvDyrGogOzui
2kDZrA/C3QiOg1dMoPzchUHnOHgpdW6su1toQJCnbgIDYfT86MXck0EPQtmFBosEHgb1aaS+jhoZ
Mvvu5YbOmQPzDjBYDZRtLFwDXvNtI0KkiN2hsDtzV7ZRvAKqxRn1In9mwBw740aw1UgDAoKYID+9
Iq385FRh6pZUUvRXWgN/71SHax+fj53/E0NOMde3nfhztfrAvJ9NTcvI/x1ZV98UWcv/+UY+a/m/
f9MfN//bub5XTw4syv/u/N+RZvis5f9+Ex+//N9165vX1zVH1r2zlv/7H/7j5n8t7feq9bEY/+MP
1/ofqQP+b1o1CMp8fuP879X/dn68ddu2V8r37v4s+/6XSFNjXf2a/vdGPmv632/6U1r/Wz05sOz7
XyLNkYa6Nf3vTXz89b931q17p75u7f6Xf/yPm/9Xf/VfjP/rG9fVR9zrf6S5YW39fxOft8QnOO+i
kjfNqgKB98WOFJgAf+6Op+PwpkWkDUOkksmMiIZCij6idLdl2sh0d9UIO41EuoZuI8VsTXhpYGVV
VOw3OrqMVCgQwI0xutmbL/SOWhFKSHh0+UtAxgDg5cufwkuskJ+azI/18/5TdvZ+dm7O2t2TkUSF
6Xlz9KLKSvzXnl7rnoBeGTMgk+RiHILZf4UCjgZzF2fM+Tkss/DCPHWD9g2PBwLyVpnxk3pCfqjH
fn7e9ZC7BVHPBUlRauJ93N8aH8vf7S3cPWkOXqHgNMfxfPPhdxjTSkPikrgvP3Ev//JWofcUh2Rh
iMXofd5/4zwICMXZy4U7WDh/pzf34KbcnaOWcw++z10ckGFScr+Cb2ehcwFLu6Bl9T5HRWHmBHZZ
+sIXvKgED9QwUlep28WC1Vfzs4rxREvrLmCdfAnjztPQS0/WstX/2Ld7UEIKTp3leE3HpjghArDV
v+MvyoPASTFXkAqNhnn9dn7itKTsiZ+0I7ev6UMHMtIZHqWMiNl7RDvSY+0HV6qEDcU7F3Fz+/mN
aLAuGK0Sv7fOdMgiS0njRWPlm2Suj6nhFgeGKXUV/qxRQTKAgcLCTDA30QOyKwiSC78WFs4Gs89O
5x/OYzRUD57awcCFqXEUjz9P4k748EDu8hnVNofUFE+8MOfOF6Zvyz3/iXtm373iiXtcDPo3z8zn
Tg850QL4cMUbOiIQK4s9QzDsmvwPc9nZOb096JbQ44dCPh6Du/x+GET8sHzXpdzrJAQaFEYPYTI/
lXjd+Zrz/EURXlAd9hnqaOgeThupjZDvBYnWRfNz0wA5YaIMKdCihCEj0wMqhkUKbpjDyanC9OMg
Fwpz5jQuFcSU778Adm5ZkWkrzS1qX9oEg4z6vVY4+b2d38szbPPqHNBgNEKTijvtPUMl0ueosZsv
e8ynPwPB8pB4WVb57eV1dNXVuQujuZPXeZ9ZO6cmA68ISXw3ljy3YukKdNYG1v5B3qg2+0e4jfJJ
MylHoJFoK4kNjRSWi5hSferX7/Fci4PpkGCEUAhHYXIYOIy1Cwzl5lKKYPQZhOGB8sEBmDpK8Uie
hm0KZbRHyrUoVvQW6FygFAE7Y8zEGAomC3HDGFe00Ac8iUf+hufMwX4KcsIYX0muQJMM2/hJDCYf
fVqYHgkEpDybfpydv52dPZ2dnweSZ/HG4smKRWIdwkmSTPfQLENONIDxUEELD1CZmQJFZHU1l3c/
7L/CXKaeO1vQWRDVthcXCgtXsnNXGXlB1xgZscseozU6ncTkHDlnhwJm8XDl/EhxoocJ2EXeoJBi
oMvFZ5bOOWShJXfqFMCem/qlMH1N0cDgDGaAPPXFF3/tOa1kPuEAloLi8e/V4AEldCoNKmVnT2F9
bMqJhnuT+UdSayXyZF4EVitM3yxemmJMBAKMCEYBNHvpJkaKnngBVKZ08eyz6+ohQ0oRZ7F9eNEh
ZbQ0X17iyDOzd6Aw9ZIjWq3+zLGz3BmjyMKAOTuVGzwLg4EZAAiIBHhStQfZhWvmg8vAVhqdwIPc
4JhGIAqXjABsZHDGiT914tXxGKf/xQWOZsxN3JeUA6h4+VLK82Bu8GKQxxxkJR3oiZEVCEhE0l2X
OKr3oZ/j9xBzJx5ZFM4k6+CLYYyFuzytzTpqGdMDbtq6LmfSl/iwF03m4h0RGr1m0bjhaKrct7eK
53vMiRm+8AA4i+cCNJkzM5qkARl4A1YqnrXCzL1c3xmOg6P5ovBAfcljaZ6d69cpHqQTURodMpay
goDH44YvT5t3jmNjIE1dDKBGCMSPub+mpvJz98GkcrIB13Nxgs4HDi5A3pw9Rdw3gFYb4Y5hV1qc
UoYDGMuoBsgKWolFS44R5rcEfdMUA+4BfhmNOTyR++W0FwVKFvlRJ90NyQQGFYiuTjyyKeHcfRfB
ALlgpBiFa0vMkHkK9FuY+gEYDdkNGpHG+PGgBwaaD1xWqAfQXs2hER3TDBLecYFR2TZmAQTz1rg9
IYMX1WlzjHwjSLkqnyvvuQJqhyXDf5Dmds9Q7vQPbKSbZ66DqJQHYa2G+GB63yCgA4A3p3qBxqBX
+AkMZo5ipCweIGCsT08DoSsyd3ANYGPgPoWWzpvPb7gWLes0k/jP6xef0zpPmXc5iFrl7QVWcyx8
sIg7eI/DVC2tgyN8VdZepW+GrQBuWnNU43qPlA04EGDfiMNRQeeApW7FgYbPnjCzck3z+QXrQhhi
XLxDYv6Ws23mVJpNSfZMO4AwPrwCPEOHMfRqHFFpHUP4py93V2LevZZwOBwKhaowywM+C3+eNlJp
x6NNLV/RT4wT5xMLwG1354tXbskVXINbYyBzph+ks44BRAnxqnn2VB7MURkATaPhBik5KfDTPRhQ
sf9bLiGZDiS0JASOHB0/rjAvD11rg6WwbVwcJXBPfy7c/JFRieHNhw8fDmFGdDw2cuTIkdDXXfj9
ODAZyz+UDDP9FvHQmQJgcIqYxdnrxzPUMoD23EvMnzr6rT7KIMZHA2UsXMtfuMLlHFMHPQi6zxmw
HxVA2ELd7gxU97zHvIX+Mq7CHcjcDw++N2dnuVnLLwUPABjz6W2MEv2lVyoQ5MwAyipepsumCTAg
xz7p0ZN5m1HXpqNRckg+ZhqbimwkWvH2e2Pp/QHKNZ0bHpKiipoNSBelCLr8kGG3xyMoKr4Jyod4
7IIi9kXQ9nxgfH6Qsghh9L6+sASPQUG6wEVWAu6Fb/V1xyoAKPsICrAyjCT95W68aSEEMMdbOWMm
DVqQN/CivjrQmRTKw+sc9KBifD6VAOhSJzj4CR7KGRpR5IiLUklMuDBMiGCTBwfK8BCEQRiary2M
w4ax0qEFmmxOF+jn6uApPz2kXrLbx5p+/bpiJAEoiQ4MqzCwj2rJO/e8JKjCgCd23wj2SeCxAxGs
C5ZFRmkHEeFFfm3BdgAb5MmwpppnCiDGmeKV3gJFA9oPpqWCpDtnCBz7F0DUDXKSTkHtAWGC0NnA
emC1p0ozK3V08nOQMcX5y6CGZF9gEH9pGO01jNHkJo8IgiPpC+B5S6b9/SrxVYKJWupZjEbVuQuN
r79/KObAqep66+YgcTOzsmD+IJfDmfw88O0057AQ9tpt47iMTW6jvOTQ7BbdnEktAGD22EoO+KsE
wjEJy0gfw4EjVyYty5Kggy5AjhJipBxdHM4lCNSIV6BqMBYn8Sxh7sy3uYnr5USqhBGmcNZcOMGn
s6yDxmBzagdB+RCqlWZqMzs90J1EmOPjpKyE4PWDUadURk83VPkAH7CSIbcyqCtcs3mtwhWbk8vh
t3R3Z2csdYS+28JM+0WJBOhwKN94QDd14fWGifhhkZv+MX/3Wz7fFtVvnKOmM7FMN543Pc6H1+S5
XC15nQW2vDeSzr0C3AeMI8Ia4B4GFZ4ejHV0G4JaeM/ofJ/rFKZH3gvDr6i1V2b5TnCtoePDjFPc
TvwAJpkP+Eq0ehDKQ3bh1Fm1DHqd6FN4IwKTF+W68eiHNtkM5dRA0OuiG6SdMnAfpAzerBKJbtBd
2fioProBbHhzsD//cF7Oh41/Gy0yqY8PZuxDz168aEu/P73R5gpaOW+KAMvTG45Xqs0aX2l2S9jl
WATgpKC0nKdqN5Yy/NsqDmqV1sXhzv1ZeSicrm0rtxWLGudgPx/RQ98xHUrkJ6zHOzUNhr3MPhCD
z7sW1nYGrfdRJe5JMdXPFyLF8PnBDa7TgwhiGbD8YXJswpSA59XAMXtH0b4ZGGWIGADPJqCzaxLH
JI2hYZTF2A+fjsQbBHu+NW9+V75nktqUrcbsec7T5fai80Y6Maml40oO1Uxc5M5/2/npJ2i8W3v9
5chRasOW2SygOahhN4hkpyIMPBYHvOcdHgRLV76ZKV2at9K5LVRHtTVQG560Ur7voTPtg6xmcYs4
Wrn40mvZDI7O3cjVx8BV6jpHdyPSIyFfs1CjbvFHA7y3LngB1PjtAVtXwQ4xBE5JquZpMWE6qOdv
cstWpolH93InR7Nzp0EFoGPP2i4S+x1JbEoeVM1hnhML6fDa1n+/wa/xtmNRGtoZoUO+IUJOFdeq
xvYHj1R3jpA7CpEENnfxyhgf8Uer3bp+GWgJI4HWEzeOTudO98O4cGv01jgxje7LYVqFHviL5Rnx
cdaQ8UNlrHQDXqcLryfoq9F8F+qwbrCs1ymodhAQGi3CRK5nwLTOuBRYDzguxfx22HnXrfStsJNE
ViHaxSQbVIYFOWaU0XQc7c7RYVkLUPriJ0Td6DQwozxvXys1bg7Vwbe/DJu3L6nj9rq6ZzkQSlnY
JK/UaoCFFa1LeYs1PUpLyGGU6nhytYG0Bw34ru42BEzsmNWCkrjwlV7ayXJ2vbkPl4OeBuug7/bl
YsfMS58zX0aYCx4MrnMeG+dz1m5ovAfIeSYxLos2SFUmINVmvTxsfGHQJ3OROv1M9xP5Hya3m2qg
ZBl47Ny8cg8YDMSQC86j6DQ0X/b5nH72NocJO/GyLJJCY4N4wD0s8j/MAenik1vXbfA4amz4hHn1
cen2mnyOVetD1Q9Yo112edqWrk8fKrGlN7lOqGthnv5sDt4oXrnlpg9NPihnCoLPOcY8QL6DLQ4N
Io/0/5y/3+uBEvAGq44edwfIVJvt3gbXY6QYiU09IYdzjheXm9Te/9fe1fZEdUTh7/6Km/ilmoIs
oDYkbWIKH4gWTTUpTdOAqSQ1wTSp9IMpJkstBhQRDTWNxaK2BRusmtQqrlh/TFnAf9E5c+blnLkz
c+fuIq3t3sSou3fn/Zw5r8/p0tAFxCTkTsSeGXaRYOI8OrL18Drl6RPvC4Hy8Mg5xdsl2IoY5Ku7
1zL6Zrd6E/SpV9OPxS6ysu3eo+r2qhrc4fjPSP5X+xAK0033UT7/52B35UAr/ndHnlb+z//6idC/
TQVqkg+Uzv/prFS6u1v5PzvxBPJ/OjveEbdpK//nP/9E6H+bbv8i+u/qPljZ797/nZXu1v2/E89u
yLZRusOhY/00+lQoZeCPry1t3FrU8Srz1zOhhggZVsPdKXUOY0TmavXVpQwvDe2EBbdQm/IItX02
ehryhE71ZF+LboeOHusbEJ0OvX+kv2/gRH/vefYyytPOq+LP4b6Pz+9SRR3bAOe6J6OlvCXCnvFW
ySEjcJaOmDV2HF8ejoyRAWuGzscByFBiwaAmIAycgfC+SYUEODslRH8b+IaaiNR9cJ04euDLlVfV
RczSKZWiowHJ6lMr9bkr0VeZypWWZlIubUU2/Fd1RllbVFTJFfhk+k/0Qw0OmrUQ35g5DO8r8hMO
O+MXjTqOfNXTwrQqMEI+j/QTTfVQ05EZKGLkvkgF23pKH568Cexj/fnsGWUhpr52nBNCY7NPSMQ/
frI5v2gCVMUnvmFEo/lxGI4PmnW5YFI4IPoRwwdh9r6uoqHyqiusIYruHW0rx65omgeOBVfZ11M0
P4OAUJIcPouWvjujrrR4SYACvOlgPYAoGSF4NPHuiXf1kKC/5YnsE/3/T9/aLSYq391DId/tJPwY
6GEc7cZK1YemU8gtcLJo4SRj8Fd+kOi3myu3s9OnYmNzK0lHmiuqII1lg5VftLC5zQfV9eexuq1Y
GIg4T+XPsL5PvrmCB6s+EM9r040pt639kThEx+VnTmO4DZuXnmxU4U6U/iQZzqe9OMae6IOhj26v
wqEP7TBtTxdWSGgvUj2DbvHQ2bFzoyM5wHb8G2veyi3euP20vnY1uJS28oJCMR8bOQPeP+44kuHe
0bRCE/xnXQ8qUHxGubiXvllfvayQob2VDkrVL/GsTISkS0oCoYoEpCKGnab266cMkRfYCLcZKaDi
aZOEiug2FWnRNi9Oqn/LklzFzTq0z5vlPlcnrCfarMMFeLPc6Vam2Vy9S7qw66uXoJCAXgosv4M+
QnS0BVqnxwCr1dgB4z90sZqbDQTceBmZ07xiWzcjbMs3bHK5mjTmSH0Kh9z8pSYYDTmXFy0Vg/eU
DIBTB0mLNvQtzH7Vb5Hxkji2EtWU9TfbWAvWt64p1Zkb4jjlHiOJDPPZs8Aqul6oE0vZS4mSmGhn
pTP4CZHQQJhLyfomK+NG6EVupq2Hv8C9ggKyrlthQvMwlq+HB/JNgYB95ycd2DGtQ/t6Mh0g6Anv
k6Uc8PaprvFjxuPzwlUX81N9HWfMW7lxJ84UZXGwY56wRXmYqCtSxvlByvutH4l/srb5a00fqkAr
4nzBXQXqMPZIjlpkc5IVnOZ2KaIBNbMHPhXJMz+tLXm+8i9b8ZIllNBt8DDHy+c2dEw9IheTlVME
7tBQA2XN1HqUFrpIw+zyZjoIaThF9wi0z2QwpjBxESx9PTxCmNMulcHKtstEUafdZOnT0y6TwcaV
qiLLApJz0aQs5iGmVKaz/dxmW9gMEwdM/MxibfNO1TDxYiGAWqGwRMTDpY0Lk6g0VkzZdJXowQv2
hayhkTLNYiWNCdYYX5ktFnPsrDnWmNgSy+rleGDJknoN7a+XRW7jha74EQnCzXTpcUUnpgqpjLbd
uPEMU+pU5tNqzcbMdtAAWhOlmzRlEr8rOzsu/iv7z+gIZq/Xa1c9I2BxvjP50N8S4b44GoLoBB1b
SCcyGjx2qIpogjAvCmqQX6TKw/pW+NJYch0uaEojyojyolrXiY/slZVv3rFe1Qpj187VqlcYV7DE
pR3uMbVkeF4ubIy6E+zHjRCxUeCo4j9OdV9dNlaQKjgyTEwk2OD0KSWvi3N6Vv1PtLrHlGSlh8Kt
4q0LdwPlLVRlCnh1Y/oZ6ocpCzM89sXYydGhz0+PwZ3vXPiZ5TX1ay/WV39TJSOrz5Mg8qhlsjOX
9Bq6Z8IWysIbJ+6NQynBXEAU0U0XMUy4fpjJRd06JU+mV1NsxM6pBEyDCmdMM3v3SqAV2D2ZzSOj
wYWCV0WIOJOggFk5kKI/N1u/+sgYgFHVHtcAaDC/r/Bc0Lap+wxlwly8bO5AJNK+s9L0w8SFTq+D
nrjgRDqmcfh5XVCSvUcRzOmAisRZ+Tc9y5yLAUuHkZpwW388rS9fxv0NzN/lHtgO5yBZeSZC6bpL
AhlYV4PMKaeZB8VkTh3SxUKl6winXnDDAhDRxXGWExnzvYwBAtK6pTyjJ5SsMyNBJWa9PIdU9iNt
IbGkFo5ulMs4p748d8np8ISroCzA9GJphrd5TmhC4meQnGwvh8IauoZPGRDGXN1FzpGypphSCU7k
YUNJTCjJ0pzAe1y+g9Vjx5mCm2c5jq1Oq4EedpPjNWU4jZfL5FhMGQZDmUt3O2CSZDZvkUFA+PmK
jTAp5CXBtkMhLwa7y/7MAWpL12Czxom8+ImevUYkjejZdBJjM+L8IoKIDK6x9h2WJ2vzQXMJsB8c
+vBw79GPBsB/IMQSyZIrw7q057Abc+NlMRStFW+BtzMo8iy7MDFPMvkrZWVzXmUvj6xP/r41MU+9
6Y2oTwmMKupzj8uabBs1eCh18d14JD41me/9vYw897e7gCFx0iSII4WS/MI9QWkO2h3tzBBkNjiY
5XHk1C0fxmnFttYl1loeLZACn0DqNoULi6LkgZNL4onZ5QQYAQJMp1HAQjB6HHMPs1YNzKSB1KMo
l/Xa/Pra99CSTK9U2PSqSjkNuOMsLk0QyZ+wN5hR6S2x8/IxCx/mQ3/v2xmUmV6aAADKhfuA+09y
HAtm+O/nkAAfyDF2muCTzfKsLJFtxThXYBcZ/zrQnnkjQEP8y4nzLGRi3hhSE3ZK+RucJR5/Kjib
wop8cX9jas4IHKUEDLPjr4t8Mz8Fv0ZCDtKymWsqSWeN07Q2nXnR1v3EjUNwSZyDh0wheIdFxSDk
TwHyCK5IkiLTTFCef5+b3tBwVJHZJcqGYG1mpoMrjiwJlwD5nnV3WRAXWCaGTTLFnRjTct2qa3GW
FGRCHTCNEMwJvlKBV7w+X/l1p2whHyMsv+ySv0W/CP9STYxht0xxD9G2TC0AA0OmRqFgcgPMwedM
cWooGGPxWD1jPnbkUP/Aib7BEwD5qwmNrqoctr5AxUsORA7ZF0ltbfiXcwXqGVqX9rB/Lvmxk7FS
bzc9LsRQRY+J8Xpn7LhLJR1dRCgCbi1/uzVzQQHionEFbw55XRnrjM/gAoCzCiTDmrSUsi9hz06O
nh0Zzrbu3Nv8uQbg3kIu/uEx4CJIxR6AXOuPLpLfMtODRCOuOaYGYy7aJXR/1Vf2buYgNOH87KwN
/IiLRdJwBHJJviZ/2aEaERL/XP3SYqleUxBLCn4tsUlkxhsilMBqpkU0J+KRFDQAZjhl/hU9cywS
APt5MgmGWQk7Ioh9nzrvHKJkF0Efgby3Cy+UgLY8YfQxMEg9eFl/cBfjH7J98Cbgj+hvIbutOklG
xoBHtu6uBON1AjMrgBkp+DVgitSvfGcLOHDojri/JBVHJPz7NOCQ+O8JnIgPTST46386mbT1tJ7W
80Y9fwOVYn6yABYCAA==
