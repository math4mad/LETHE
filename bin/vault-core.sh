#!/usr/bin/env bash
# bin/vault-core.sh — 核心资产保险柜 (主人 0923 夜裁定: 对话与论文必保, 实验室是锦上添花)
#
# 园律「凡宝必三存」落地: 本脚本管第 2、3 存之**清点与封存**:
#   1) 扫核心资产 (对话母本 / 手记 / 论文源件 / 预注册册) → sha256 清单 (只登哈希, 不搬正文)
#   2) 打成本地第二存 ~/Cora-Vault/core-YYYYMMDD.tar.gz (防回退防误删; 同盘不防盘死)
#   3) 打印"三存缺口"表: 何物几存、何物单存 (单存即险, 报警)
#
# 用法:
#   bash bin/vault-core.sh            # 清点 + 封存 + 缺口报告
#   bash bin/vault-core.sh verify     # 只验哈希 (盘上物对清单)
#   bash bin/vault-core.sh manifest   # 只出清单 (stdout, 可贴 LEDGER)
set -uo pipefail

VAULT_DIR="${VAULT_DIR:-$HOME/Cora-Vault}"
MANIFEST="$VAULT_DIR/MANIFEST.tsv"
mkdir -p "$VAULT_DIR"

# ---- 核心资产源 (对话与论文优先; 实验产物不在必保之列) ----
SOURCES=(
  "$HOME/Programming/code-2026/Concept-Space-Sphere/external|对话与手记母本|主人自产 docx/对话 pdf (gitignore 罩着, 盘上唯一份)"
  "$HOME/Programming/code-2026/cora-atlas/papers|论文源件|tex/pdf/md 诸稿 (公开 git 在册)"
  "$HOME/Programming/code-2026/chora/docs|论文与册|chora 侧文册"
  "$HOME/Programming/code-2026/chora/letters|信箱|君臣对话录与学者信件"
  "$HOME/Programming/code-2026/cora-atlas/iterations|迭代手记|对话之公开化文本"
)
PREREG=("$HOME/Programming/code-2026/chora/experiments")

# 排除: 第三方版权书与压缩包 (不入库不上传, 只登哈希备查)
EXCL_RE='(z-lib|Springer|\.zip$|pouchdb-bookmark)'

scan() {
  for spec in "${SOURCES[@]}"; do
    IFS='|' read -r dir cls note <<<"$spec"
    [ -d "$dir" ] || { echo "SKIP 无此目录: $dir" >&2; continue; }
    find "$dir" -type f \( -iname "*.docx" -o -iname "*.pdf" -o -iname "*.tex" -o -iname "*.md" -o -iname "*.txt" \) \
      ! -path "*/.git/*" ! -name "*.lethe-pin.json" -print0 2>/dev/null \
    | while IFS= read -r -d '' f; do
        clsx="$cls"; [[ "$f" =~ $EXCL_RE ]] && clsx="$cls/第三方禁上传"
        printf '%s\t%s\t%s\t%s\n' "$(shasum -a 256 "$f" | awk '{print $1}')" "$f" "$clsx" "$note"
      done
  done
  for p in "${PREREG[@]}"/PREREG_*.md; do
    [ -f "$p" ] || continue
    printf '%s\t%s\t%s\t%s\n' "$(shasum -a 256 "$p" | awk '{print $1}')" "$p" "预注册册" "判据先冻之案卷 (公开 git 在册)"
  done
}

case "${1:-seal}" in
  manifest) scan | sort -k2; exit 0 ;;
  verify)
    [ -f "$MANIFEST" ] || { echo "✘ 无清单可验: $MANIFEST"; exit 1; }
    bad=0; miss=0
    while IFS=$'\t' read -r sha path _cls _note; do
      [ -f "$path" ] || { echo "✘ 失物 $path"; miss=$((miss+1)); continue; }
      got=$(shasum -a 256 "$path" | awk '{print $1}')
      [ "$got" = "$sha" ] || { echo "✘ 变字节 $path"; bad=$((bad+1)); }
    done < <(tail -n +2 "$MANIFEST")
    echo "验讫: 变字节 $bad · 失物 $miss (清单 $(wc -l < "$MANIFEST" | tr -d ' ') 件)"
    [ "$bad" -eq 0 ] && [ "$miss" -eq 0 ] ;;
  *)
    TMP="$VAULT_DIR/.MANIFEST.new"
    scan | sort -k2 > "$TMP"
    n=$(wc -l < "$TMP" | tr -d ' ')
    printf 'sha256\tpath\tclass\tnote\n' > "$MANIFEST"
    cat "$TMP" >> "$MANIFEST"; rm -f "$TMP"
    DAY=$(date +%Y%m%d)
    TAR="$VAULT_DIR/core-$DAY.tar.gz"
    # 封存时剔除第三方版权件 (只留哈希在册, 字节不外存)
    awk -F'\t' '$3 !~ /第三方/ {print $2}' "$MANIFEST" | (cd "$HOME" && tar -czf "$TAR" -T - 2>/dev/null)
    TS=$(shasum -a 256 "$TAR" | awk '{print $1}')
    echo "⚏ 核心资产清点: $n 件 · 清单 $MANIFEST"
    echo "⚏ 本地第二存: $TAR ($(du -sh "$TAR" | awk '{print $1}')) sha256 ${TS:0:16}…"
    echo
    echo "── 单存之险 (只有盘上一份者, 即对话母本) ──"
    awk -F'\t' 'NR>1 && $2 ~ /external\// {print $3"\t"$2}' "$MANIFEST" \
      | sed "s|$HOME/Programming/code-2026/Concept-Space-Sphere/external/||" | head -40
    echo
    echo "── 三存缺口 ──"
    echo "  ① 盘上工作树: 有"
    echo "  ② 本地第二存: $TAR (同盘, 防误删/防回退, 不防盘死)"
    echo "  ③ off-disk : 未配置 (Time Machine 无目的盘; iCloud Drive 拒访; 无私有仓令牌路)"
    echo "     → 待主人一指点: 挂外置盘 / 开 iCloud / 立私有仓 (园笔荐: 私有仓只收哈希与自产文, 第三方书永不上传)"
    ;;
esac
