#!/bin/bash
# bin/pin.sh — the Receipt Writer · 入城收据
#     FOUNDLING §2 的 standing invitation 落地成鞭，§3 规程第 2 步的用印机：
#     Pinned on arrival — path, sha256, producer repo@sha, purpose —
#     before any experiment touches the bytes. A byte without a row is quarantine.
#
# 用法:
#   bin/pin.sh <artifact-path> <producer-repo@sha> <purpose>   # 收货打条
#   bin/pin.sh --verify <artifact-path>                         # 复查（对照 sidecar）
#
# 侧车: <artifact>.lethe-pin.json —— 打条即写，verify 即比。
set -euo pipefail

die() { echo "✘ $*" >&2; exit 1; }

if [ "${1:-}" = "--verify" ]; then
    P="${2:?用法: bin/pin.sh --verify PATH}"
    [ -f "$P" ] || die "无此物: $P"
    PIN="$P.lethe-pin.json"
    [ -f "$PIN" ] || die "无收据 · quarantine — $P 未经 §3 规程，verify 无从谈起"
    HAVE=$(shasum -a 256 "$P" | awk '{print $1}')
    WANT=$(python3 -c "import json;print(json.load(open('$PIN'))['sha256'])")
    if [ "$HAVE" = "$WANT" ]; then
        echo "✔ 字节如故 · $P"
        echo "    sha256 $HAVE"
        echo "    pinned $(python3 -c "import json;d=json.load(open('$PIN'));print(d['received'], '·', d['producer'], '· for:', d['purpose'])")"
        exit 0
    else
        echo "✘ 字节已变 · $P"
        echo "    pinned $WANT"
        echo "    now    $HAVE"
        echo "    → 金库与格模之一必有一改；先对账（Letter 回执 + git log），再论引用。"
        exit 1
    fi
fi

P="${1:?用法: bin/pin.sh PATH PRODUCER PURPOSE}"
PRODUCER="${2:?需 producer repo@sha · 口头宝藏不是宝藏}"
PURPOSE="${3:?需用途 · 宝藏必 purpose-bound §3.3}"
[ -f "$P" ] || die "无此物: $P（先入城门，再打车；字母在前，字模在后）"

SHA=$(shasum -a 256 "$P" | awk '{print $1}')
NOW=$(date -u +%Y-%m-%dT%H:%M:%S+00:00)
PIN="$P.lethe-pin.json"

python3 - "$PIN" "$P" "$SHA" "$PRODUCER" "$PURPOSE" "$NOW" <<'EOF'
import json, sys
pin, path, sha, producer, purpose, now = sys.argv[1:7]
json.dump({"path": path, "sha256": sha, "producer": producer,
           "purpose": purpose, "received": now,
           "law": "FOUNDLING §3 — hash before we announce; no re-export"},
          open(pin, "w"), ensure_ascii=False, indent=2)
EOF

echo "⚏ 收据已打 · $PIN"
echo "| $(basename "$P") | \`$SHA\` | $PRODUCER | $PURPOSE | $NOW |"
echo
echo "↑ 此行可直接贴入 website/FOUNDLING.md（§3 收货台账）。announce by letter, not by memory."
