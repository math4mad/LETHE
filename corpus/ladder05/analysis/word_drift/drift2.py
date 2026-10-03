#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""drift2.py — 儿童语义·词义漂移探针 v1（**句法层版**）

缘起：v0（drift.py，同话轮共现）判 moon 无漂移，但原始语料里 moon 常被当**施事**用
（`*CHI: moon pick dirt up .` / `%mor: noun|moon verb|pick...`）。
本机 264 份 .cha **全带 %mor / %gra 句法层**，故把探针从"同话轮共现"升级为
**句法位置**——英文 SVO，名词后紧邻动词 ⇒ 主语位；`-Acc` ⇒ 宾语位。

【预注册 v1（先冻后用）】
  对一词 t，在 CHI 自身产出中：
    agent(t) = #(t 后紧邻 verb) / #(t 出现)          ← 施事/主语位
    patnt(t) = #(t 带 -Acc) / #(t 出现)              ← 受事/宾语位
    **施事坐标 a(t) = agent − patnt ∈ [−1, +1]**
  预言：
    P1  **moon** 的 a 早期为正、随月龄**下降**（"月亮能做事"→"月亮是物"）——承 LADDER-1 泛灵→去泛灵。
    P2  锚 **dog / baby** a 恒高；P3 锚 **ball / car / book** a 恒低。
  失败条款：moon 平直 → 泛灵漂移在句法探针上亦不立，**照登**（不挪靶）。

跑法: .venv/bin/python drift2.py
"""
import os, re, glob, json, collections

HERE = os.path.dirname(os.path.abspath(__file__))
CHAT = os.path.join(HERE, "..", "..", "layers", "childes", "chat")

ANIM = set("""mommy mama mummy daddy dada baby dog doggie puppy kitty cat bird fish
boy girl man lady people person guy duck bunny rabbit horse cow pig bear""".split())
OBJ = set("""ball block cup shoe book car truck milk water juice bottle box
chair table spoon fork paper crayon pencil flower tree rock stone""".split())
TARGETS = ["moon", "sun", "star", "light", "flower",
           "dog", "baby", "kitty", "ball", "car", "book", "milk"]
BANDS = [(13, 23), (24, 33), (34, 43), (44, 53), (54, 62)]
BANDLAB = [f"{a}–{b}" for a, b in BANDS]


def age_of(txt):
    m = re.search(r'^@ID:\s*eng\|[^|]*\|CHI\|(\d+);(\d+)\.(\d+)\|', txt, re.M)
    return None if not m else int(m.group(1)) * 12 + int(m.group(2))


def band_of(m):
    for i, (a, b) in enumerate(BANDS):
        if a <= m <= b:
            return i
    return None


def mor_tokens(line):
    """%mor 行 → [(pos, lemma, raw)]"""
    s = line.split(":", 1)[1].strip()
    out = []
    for tok in s.split():
        if "|" not in tok:
            continue
        pos, rest = tok.split("|", 1)
        lemma = re.split(r"[-&]", rest, 1)[0].lower()
        out.append((pos.lower(), lemma, tok))
    return out


def main():
    # (band, word) -> [n, next_is_verb, has_acc, same_utt_anim, same_utt_obj]
    acc = collections.defaultdict(lambda: [0, 0, 0, 0, 0])
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
        cur = None
        for line in txt.splitlines():
            if line.startswith("*CHI:"):
                cur = line
            elif line.startswith("%mor:") and cur is not None:
                toks = mor_tokens(line)
                lemmas = [x[1] for x in toks]
                has_anim = any(x in ANIM for x in lemmas)
                has_obj = any(x in OBJ for x in lemmas)
                for i, (pos, lem, raw) in enumerate(toks):
                    if lem in TARGETS:
                        nxt = toks[i + 1] if i + 1 < len(toks) else None
                        r = acc[(b, lem)]
                        r[0] += 1
                        r[1] += 1 if (nxt and nxt[0].startswith("verb")) else 0
                        r[2] += 1 if raw.endswith("-Acc") else 0
                        r[3] += 1 if has_anim else 0
                        r[4] += 1 if has_obj else 0
                cur = None

    table = {}
    for t in TARGETS:
        row = []
        for b in range(len(BANDS)):
            n, nv, na, an, ob = acc.get((b, t), [0, 0, 0, 0, 0])
            if n == 0:
                row.append(None)
                continue
            row.append(dict(n=n, agent=round(nv / n, 3), patnt=round(na / n, 3),
                            a=round((nv - na) / n, 3),
                            anim=round(an / n, 3), obj=round(ob / n, 3)))
        table[t] = row

    json.dump(dict(bands=BANDS, bandlab=BANDLAB,
                   sessions={BANDLAB[b]: nsess[b] for b in range(len(BANDS))},
                   table=table),
              open(os.path.join(HERE, "drift2.json"), "w"), ensure_ascii=False, indent=1)

    lines = ["# 词义漂移 v1 · 句法施事坐标 a = P(后邻动词) − P(带 -Acc)", "",
             "| 词 | " + " | ".join(BANDLAB) + " |", "|---|" + "---|" * len(BANDLAB)]
    for t in TARGETS:
        cells = []
        for r in table[t]:
            cells.append("—" if r is None else f"**{r['a']:+.2f}** (n={r['n']})")
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    lines += ["", "会话/带: " + " · ".join(f"{BANDLAB[b]}={nsess[b]}" for b in range(len(BANDS)))]
    md = "\n".join(lines)
    open(os.path.join(HERE, "table2.md"), "w").write(md + "\n")
    print(md)
    print("\nDRIFT2_DONE")


if __name__ == "__main__":
    main()
