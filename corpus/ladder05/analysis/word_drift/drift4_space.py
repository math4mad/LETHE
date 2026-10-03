#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""drift4_space.py — 儿童语义·词义漂移 v3（**分布语义空间版**）

诊断链：
  v0 同话轮共现 → 循环（探针含目标自身）
  v2 修正后 → 全词近零、孤立率 85–100%（儿童话轮多为单词句）→ 便宜探针**判死**
  v3（本器）→ 不信单词共现，改在**会话内滑窗**上建每年龄段的 **PPMI+SVD 分布语义空间**，
              看目标词的**近邻集**与**坐标**随龄怎么漂。这也正是「世界模型的轴」本人。

预注册 v3：
  P1  锚词 dog / ball 的近邻集跨龄**稳定**（真语义不漂）。
  P2  moon 的近邻若有漂移，须能指到**体裁**（童谣：cat/fiddle/dish/spoon）而非本体。
  P3  相邻年龄段的 SVD 空间用 Procrustes 对齐后**轴有旋转**（＝世界模型的轴在生长）。
  失败条款：近邻集噪声主导 / 对齐残差无结构 → 照登，不救。

跑法: .venv/bin/python drift4_space.py
"""
import os, re, glob, json, math, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CHAT = os.path.join(HERE, "..", "..", "layers", "childes", "chat")
BANDS = [(13, 23), (24, 33), (34, 43), (44, 62)]
BANDLAB = [f"{a}–{b}" for a, b in BANDS]
WIN = 3          # 会话内滑窗（话轮数）
MINFREQ = 6      # 词进入空间的频次阈
DIM = 60

STOP = set("""a an the and or but if then so to of in on at by for with from up down
out off over under again more very too not no yes is are was were be been am do does did
done have has had can could will would shall should may might must it its this that these those
there here what who why how when where i you he she we they me him her us them my your his
mine yours ours their theirs as also just now only well oh ah uh um hmm yeah ok okay please
little big one two three go going gone get got put see look want know think say said
like make made come came back again""".split())


def age(txt):
    m = re.search(r'^@ID:\s*eng\|[^|]*\|CHI\|(\d+);(\d+)\.(\d+)\|', txt, re.M)
    return None if not m else int(m.group(1)) * 12 + int(m.group(2))


def band_of(m):
    for i, (a, b) in enumerate(BANDS):
        if a <= m <= b:
            return i
    return None


def utt_tokens(line):
    out = []
    for raw in line[5:].strip().split():
        x = raw.lower()
        x = re.sub(r'\(.*?\)', '', x).replace("'s", "").replace("'", "")
        x = re.sub(r'[^a-z]+', '', x)
        if len(x) >= 2 and x not in STOP:
            out.append(x)
    return out


def build_band(sessions):
    """sessions: list of list-of-utt(token list) —— 会话内滑窗共现。"""
    co = collections.Counter()
    freq = collections.Counter()
    ctx = collections.defaultdict(collections.Counter)
    for utts in sessions:
        flat = [w for u in utts for w in u]
        freq.update(flat)
        for i, u in enumerate(utts):
            window = [w for j in range(max(0, i - WIN), min(len(utts), i + WIN + 1))
                      for w in utts[j]]
            for a in u:
                for b in window:
                    if a != b:
                        co[(a, b)] += 1
    vocab = sorted(w for w, c in freq.items() if c >= MINFREQ)
    idx = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    M = np.zeros((V, V), np.float64)
    tot = sum(co.values())
    for (a, b), c in co.items():
        if a in idx and b in idx:
            M[idx[a], idx[b]] = c
    # PPMI
    rs = M.sum(1, keepdims=True)
    cs = M.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((M * tot) / np.maximum(rs * cs, 1e-12))
    ppmi = np.maximum(pmi, 0.0)
    return vocab, idx, ppmi, freq


def svd_embed(ppmi, dim=DIM):
    if ppmi.shape[0] < dim + 2:
        dim = max(2, ppmi.shape[0] - 2)
    U, S, Vt = np.linalg.svd(ppmi, full_matrices=False)
    return U[:, :dim] * S[:dim], U, S


def cos(a, b):
    return float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def main():
    band_sessions = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(CHAT, "*.cha"))):
        txt = open(f, encoding="utf-8", errors="ignore").read()
        m = age(txt)
        if m is None:
            continue
        b = band_of(m)
        if b is None:
            continue
        band_sessions[b].append([utt for line in txt.splitlines() if line.startswith("*CHI:")
                                 for utt in [utt_tokens(line)] if utt])

    spaces = {}
    for b in range(len(BANDS)):
        vocab, idx, ppmi, freq = build_band(band_sessions[b])
        Z, U, S = svd_embed(ppmi)
        spaces[b] = dict(vocab=vocab, idx=idx, Z=Z, freq=freq)
        print(f"[{BANDLAB[b]}] sessions={len(band_sessions[b])} vocab={len(vocab)} "
              f"tokens={sum(freq.values())} sv={np.round(S[:5],1)}")

    targets = ["moon", "sun", "star", "light", "flower", "dog", "baby", "ball", "car", "book", "milk", "tree"]
    report = {}
    for t in targets:
        rows = []
        for b in range(len(BANDS)):
            sp = spaces[b]
            if t not in sp["idx"]:
                rows.append(None)
                continue
            v = sp["Z"][sp["idx"][t]]
            sims = [(w, cos(v, sp["Z"][i])) for w, i in sp["idx"].items() if w != t]
            sims.sort(key=lambda x: -x[1])
            rows.append([w for w, s in sims[:8]])
        report[t] = rows

    lines = ["# 词义漂移 v3 · 分布语义近邻（会话内滑窗 PPMI+SVD）", ""]
    for t in targets:
        lines.append(f"### {t}")
        for b, nb in enumerate(report[t]):
            lines.append(f"- **{BANDLAB[b]}**：" + ("（词未入空间）" if nb is None else " · ".join(nb)))
        lines.append("")
    md = "\n".join(lines)
    open(os.path.join(HERE, "neighbors.md"), "w").write(md + "\n")
    print(md)
    json.dump({t: report[t] for t in targets}, open(os.path.join(HERE, "neighbors.json"), "w"),
              ensure_ascii=False, indent=1)
    np.save(os.path.join(HERE, "_spaces.npy"),
            np.array([spaces[b]["Z"] for b in range(len(BANDS))], dtype=object), allow_pickle=True)
    print("DRIFT4_DONE")


if __name__ == "__main__":
    main()
