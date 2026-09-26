#!/usr/bin/env bash
# yunpan.sh · 园笔大件上云器 (CloudDrive2 → 阿里云盘Open / CHORA-VAULT 库)
# 用法: put <本地件> [云端名] | get <云端名> <本地目标> | ls | mkdir <路径> | stat
# 凭据: ~/.cd2_credentials (容「user pass」空格体或「username: u / password: p」键值体;
#       用户名取含@段, 口令取其外末段)。口令永不打印、永不入库。
# 覆量: CD2_BASE (默认 http://127.0.0.1:19798), CD2_MOUNT (默认 阿里云盘Open/CHORA-VAULT)
set -euo pipefail
BASE="${CD2_BASE:-http://127.0.0.1:19798}"
MOUNT="${CD2_MOUNT:-阿里云盘Open/CHORA-VAULT}"
CRED="${CD2_CRED:-$HOME/.cd2_credentials}"
DAV="$BASE/dav/$MOUNT"

if [ -z "${CD2_USER:-}" ] && [ -f "$CRED" ]; then
  CD2_USER=$(python3 -c "
import re
raw=[l.strip() for l in open('$CRED') if l.strip()]
toks=[w for l in raw for w in re.split(r'[\s:]+',l) if w]
print(next((t for t in toks if '@' in t), toks[0]))")
  CD2_PASS=$(python3 -c "
import re
raw=[l.strip() for l in open('$CRED') if l.strip()]
toks=[w for l in raw for w in re.split(r'[\s:]+',l) if w]
p=[t for t in toks if t.lower() not in ('username','password') and '@' not in t]
print(p[-1] if p else '')")
  export CD2_USER CD2_PASS
fi
: "${CD2_USER:?缺用户名}"; : "${CD2_PASS:?缺口令}"
A=(-u "$CD2_USER:$CD2_PASS")

cmd="${1:?put|get|ls|mkdir|stat}"; shift || true
case "$cmd" in
  stat)
    curl -s -m 8  -o /dev/null -w "web   HTTP %{http_code}\n" "$BASE/"
    curl -s -m 8  -o /dev/null -w "dav   HTTP %{http_code}\n" "${A[@]}" -X PROPFIND "$DAV/" -H "Depth: 0"
    ;;
  mkdir)
    curl -s -m 15 -o /dev/null -w "MKCOL $1 → %{http_code}\n" "${A[@]}" -X MKCOL "$BASE/dav/$1"
    ;;
  ls)
    curl -s -m 20 "${A[@]}" -X PROPFIND "$DAV/" -H "Depth: 1" | python3 -c '
import sys, re
t = sys.stdin.read()
for blk in re.findall(r"<[dD]:response>(.*?)</[dD]:response>", t, re.S):
    if re.search(r"<[dD]:collection\s*/>|<[dD]:collection>", blk): continue
    cl = re.search(r"<[dD]:getcontentlength>([0-9]+)<", blk)
    hp = re.search(r"<[dD]:href>([^<]+)<", blk)
    nm = re.search(r"<[dD]:displayname>([^<]*)<", blk)
    import urllib.parse as up
    name = up.unquote(nm.group(1) if nm and nm.group(1) else (hp.group(1).split("/")[-1] if hp else "?"))
    size = f"{int(cl.group(1))/1e6:8.1f}MB" if cl else "     -   "
    print(f"{size}  {name}")'
    ;;
  put)
    f="${1:?本地文件}"; name="${2:-$(basename "$f")}"
    [ -s "$f" ] || { echo "☠ 空文件或不存在: $f"; exit 1; }
    sha=$(shasum -a 256 "$f" | awk '{print $1}')
    echo " 上传 $name ($(( $(stat -f %z "$f")/1024/1024 ))MB) sha ${sha:0:16}…"
    code=$(curl -s -m 3600 -o /dev/null -w "%{http_code}" "${A[@]}" -T "$f" "$DAV/$name")
    case "$code" in 201|204|200) ;; *) echo "☠ 上传 HTTP $code"; exit 2 ;; esac
    rcode=$(curl -s -m 900 -o /tmp/yp_verify -w "%{http_code}" "${A[@]}" "$DAV/$name")
    [ "$rcode" = "200" ] || { echo "☠ 云端回读 HTTP $rcode"; exit 3; }
    rsha=$(shasum -a 256 /tmp/yp_verify | awk '{print $1}'); rm -f /tmp/yp_verify
    if [ "$sha" = "$rsha" ]; then echo "✔ 在架对撞成: $DAV/$name  sha256=${sha:0:16}…"
    else echo "☠ 对撞破! $sha ≠ $rsha"; exit 4; fi
    ;;
  get)
    name="${1:?云端名}"; dst="${2:?本地目标}"
    curl -sS -m 3600 "${A[@]}" -o "$dst" "$DAV/$name"
    echo " 落 $dst sha256=$(shasum -a 256 "$dst" | awk '{print substr($1,1,16)}')…"
    ;;
esac
