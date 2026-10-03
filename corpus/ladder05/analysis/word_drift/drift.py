#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""drift.py — 儿童语义·词义漂移探针 (v0)

主人 1003 令「再考虑儿童语义的变化情况」→ 主人圈 **B · 词义漂移**。

【预注册（先冻后用，铁律 5；判据在本脚本 commit 之后不再改动）】
  轴＝**泛灵／去泛灵**（承 LADDER-1：泛灵→去泛灵迁移 0.983→0.009；承 PREREG_LADDER-3：
  「月亮先离动物球」）。
  语料＝本地 CHILDES（Brown + Bernstein），仅取 **CHI（目标儿童自身产出）**，带真实月龄。
  探针＝**同话轮共现**（utterance-level co-occurrence）：
      anim(t,b) = #utt(含 t 且含任一「生物词」) / #utt(含 t)
      obj (t,b) = #utt(含 t 且含任一「物件词」) / #utt(含 t)
      **泛灵坐标 y(t,b) = anim − obj** ∈ [−1, +1]
  预言（可证伪）：
    P1  **moon** 的 y 早期显著为正、随月龄**下降**（泛灵物→去泛灵）。
    P2  对照锚 **dog / baby** 的 y **恒高**（真正的生物，不漂）。
    P3  对照锚 **ball / car / book** 的 y **恒低**（真物件，不漂）。
  失败条款：若 moon 曲线平直（或锚词与 moon 无法区分）→「泛灵漂移」于此语料不成立，**照登**。

跑法: .venv/bin/python drift.py        →  table.md / drift.json / 控制台
"""
import os, re, glob, json, collections

HERE = os.path.dirname(os.path.abspath(__file__))
CHAT = os.path.join(HERE, "..", "..", "layers", "childes", "chat")

# ── 探针词表（义素轴的两极）──────────────────────────────────────────
ANIM = set("""mommy mama mummy daddy dada baby dog doggie puppy kitty cat bird fish
boy girl man lady people person guy duck bunny rabbit horse cow pig bear""".split())
OBJ = set("""ball block cup shoe book car truck milk water juice bottle box
chair table spoon fork paper crayon pencil flower tree rock stone""".split())

# ── 被追的目标词 ────────────────────────────────────────────────────
TARGETS = ["moon", "sun", "star", "flower", "dog", "baby", "ball", "car", "book", "light", "milk"]

BANDS = [(13, 23), (24, 33), (34, 43), (44, 53), (54, 62)]
BANDLAB = [f"{a}–{b}" for a, b in BANDS]


def age_of(txt):
    m = re.search(r'^@ID:\s*eng\|[^|]*\|CHI\|(\d+);(\d+)\.(\d+)\|', txt, re.M)
    if not m:
        return None
    return int(m.group(1)) * 12 + int(m.group(2))


def utterances(txt):
    """抽 *CHI 话轮，切成 token。"""
    out = []
    for line in txt.splitlines():
        if not line.startswith("*CHI:"):
            continue
        toks = []
        for raw in line[5:].strip().split():
            t = raw.lower()
            t = re.sub(r'\(.*?\)', '', t)
            t = t.replace("'s", "").replace("'", "")
            t = re.sub(r'[^a-z]+', '', t)
            if t:
                toks.append(t)
        if toks:
            out.append(toks)
    return out


def band_of(m):
    for i, (a, b) in enumerate(BANDS):
        if a <= m <= b:
            return i
    return None


def main():
    # 逐会话累积: 每 band 里, 含目标词的 utt 中, 是否出现 ANIM / OBJ
    hit_t = collections.defaultdict(lambda: [0, 0, 0])   # (band,target) -> [n_utt, n_anim, n_obj]
    tot = collections.Counter()                           # band -> 总 utt 数 (作规模参照)
    nsess = collections.Counter()
    for f in sorted(glob.glob(os.path.join(CHAT, "*.cha"))):
        txt = open(f, encoding="utf-8", errors="ignore").read()
        m = age_of(txt)
        if m is None:
            continue
        b = band_of(m)
        if b is None:
            continue
        nsess[b] += 1
        for toks in utterances(txt):
            tot[b] += 1
            has_t = [t for t in TARGETS if t in toks]
            if not has_t:
                continue
            a = any(w in ANIM for w in toks)
            o = any(w in OBJ for w in toks)
            for t in has_t:
                rec = hit_t[(b, t)]
                rec[0] += 1
                rec[1] += 1 if a else 0
                rec[2] += 1 if o else 0

    table = {}
    for t in TARGETS:
        row = []
        for b in range(len(BANDS)):
            n, an, ob = hit_t.get((b, t), [0, 0, 0])
            if n == 0:
                row.append(None)
            else:
                anim, obj = an / n, ob / n
                row.append(dict(n=n, anim=round(anim, 3), obj=round(obj, 3),
                                y=round(anim - obj, 3)))
        table[t] = row

    meta = dict(sessions={BANDLAB[b]: nsess[b] for b in range(len(BANDS))},
                utt={BANDLAB[b]: tot[b] for b in range(len(BANDS))},
                bands=BANDS, anim=sorted(ANIM), obj=sorted(OBJ), targets=TARGETS)
    json.dump(dict(meta=meta, table=table),
              open(os.path.join(HERE, "drift.json"), "w"), ensure_ascii=False, indent=1)

    # 控制台 + markdown
    lines = ["# 词义漂移 · 泛灵坐标 y = P(生物词|同话轮) − P(物件词|同话轮)", "",
             "| 词 | " + " | ".join(BANDLAB) + " |", "|---|" + "---|" * len(BANDLAB)]
    for t in TARGETS:
        cells = []
        for b, r in enumerate(table[t]):
            cells.append("—" if r is None else f"{r['y']:+.2f}*(n={r['n']})*")
        lines.append(f"| **{t}** | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("会话/带: " + " · ".join(f"{BANDLAB[b]}={nsess[b]}" for b in range(len(BANDS))))
    md = "\n".join(lines)
    open(os.path.join(HERE, "table.md"), "w").write(md + "\n")
    print(md)
    print("\nDRIFT_DONE")


if __name__ == "__main__":
    main()
