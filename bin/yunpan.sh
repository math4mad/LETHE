#!/usr/bin/env bash
# yunpan.sh · 园笔大件上云器 (CloudDrive2 → 阿里云盘 CHORA-VAULT 库)
# 用法: put <本地件> [云端名] | get <云端名> <本地目标] | ls | mkdir <名> | stat
# 凭据: ~/.cd2_credentials (主人自填; 容「user pass」空格体或「username: u / password: p」键值体)
#       亦可用环境变量 CD2_USER/CD2_PASS 覆。园笔不落盘不抄录口令。
# 云端库: CD2_DAV 可覆, 默认 http://127.0.0.1:19798/dav/<库名默认 CHORA-VAULT>
set -euo pipefail
BASE="${CD2_BASE:-http://127.0.0.1:19798}"
MOUNT="${CD2_MOUNT:-阿里云盘Open/CHORA-VAULT}"
CRED="${CD2_CRED:-$HOME/.cd2_credentials}"

cred_parse() {  # 稳: 用户名取含@段, 口令取其余末段
  python3 - "$CRED" <<'PY'
import re, sys
raw = [l.strip() for l in open(sys.argv[1]) if l.strip()]
toks = [w for l in raw for w in re.split(r'[\s:]+', l) if w]
u = next((t for t in toks if '@' in t), toks[0] if toks else '')
p = [t for t in toks if t.lower() not in ('username', 'password') and '@' not in t]
print(u); print(p[-1] if p else '')
PY
}
if [ -z "${CD2_USER:-}" ] && [ -f "$CRED" ]; then
  CD2_USER=$(sed -n 1p <(cred_parse)); CD2_PASS=$(sed -n 2p <(cred_parse))
  export CD2_USER CD2_PASS
fi
: "${CD2_USER:?缺用户名 (CD2_USER 或 ~/.cd2_credentials)}"
: "${CD2_PASS:?缺口令 (CD2_PASS 或 ~/.cd2_credentials)}"
DAV="$BASE/dav/$MOUNT"
AUTH=(-u "$CD2_USER:$CD2_PASS")

cmd="${1:?put|get|ls|mkdir|stat}"; shift || true
case "$cmd" in
  stat)
    curl -s -m 8 -o /dev/null -w "web     HTTP %{http_code}\n" "$BASE/" || true
    curl -s -m 8 -o /dev/null -w "dav根   HTTP %{http_code}\n" "${AUTH[@]}" -X PROPFIND "$BASE/dav/" -H "Depth: 0" || true
    curl -s -m 8 -o /dev/null -w "库 $MOUNT HTTP %{http_code}\n" "${AUTH[@]}" -X PROPFIND "$DAV/" -H "Depth: 0" || true
    ;;
  mkdir)
    d="${1:?库名}"; curl -s -m 15 -o /dev/null -w "MKCOL $d → %{http_code}\n" "${AUTH[@]}" -X MKCOL "$BASE/dav/$d"
    ;;
  ls)
    curl -s -m 20 "${AUTH[@]}" -X PROPFIND "$DAV/" -H "Depth: 1" \
    | python3 -c "
import sys, re
t = sys.stdin.read()
for blk in re.findall(r'<d:response>(.*?)</d:response>', t, re.S):
    nm = re.search(r'<d:displayname>([^<]+)<', blk)
    cl = re.search(r'<d:getcontentlength>(\d+)<', blk)
    if nm and cl and nm.group(1) != '$MOUNT':
        print(f'{int(cl.group(1))/1e6:9.1f}MB  {nm.group(1)}')
"
    ;;
  put)
    f="${1:?本地文件}"; name="${2:-$(basename "$f")}"
    [ -s "$f" ] || { echo "☠ 空文件或不存在: $f"; exit 1; }
    sha=$(shasum -a 256 "$f" | awk '{print $1}')
    echo " 上传 $(basename "$f") ($(( $(stat -f %z "$f") / 1024 / 1024 ))MB) sha ${sha:0:16}…"
    code=$(curl -s -m 3600 -o /dev/null -w "%{http_code}" "${AUTH[@]}" -T "$f" "$DAV/$name")
    case "$code" in 201|204|200) ;; *) echo "☠ 上传 HTTP $code"; exit 2;; esac
    rcode=$(curl -s -m 600 -o /tmp/yp_verify -w "%{http_code}" "${AUTH[@]}" "$DAV/$name")
    [ "$rcode" = "200" ] || { echo "☠ 云端回读 HTTP $rcode"; exit 3; }
    rsha=$(shasum -a 256 /tmp/yp_verify | awk '{print $1}'); rm -f /tmp/yp_verify
    if [ "$sha" = "$rsha" ]; then echo "✔ 在架对撞成: $DAV/$name  sha256=${sha:0:16}…"; else echo "☠ 对撞破! $sha ≠ $rsha"; exit 4; fi
    ;;
  get)
    name="${1:?云端名}"; dst="${2:?本地目标}"
    curl -sS -m 3600 "${AUTH[@]}" -o "$dst" "$DAV/$name"
    echo " 落 $dst sha256=$(shasum -a 256 "$dst" | awk '{print substr($1,1,16)}')…"
    ;;
esac
