#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""drift5_report.py — 儿童语义·词义漂移 v4（**跨龄空间对齐 + 定量漂移 + 图**）

在 v3（每龄段 PPMI+SVD 语义空间）上加三件：
 ① 洗涤：剔转写噪声 token（xxx/yyy/humm…）与功能词。
 ② **跨龄对齐**：各龄段空间用 Procrustes 正交对齐到末段（共用词），使「词在空间里的移动」
    与世界模型**轴自身的旋转**分离——若未对齐而硬比，会把"整套轴转动"误记成"词在漂"。
 ③ 度量：
      drift_2d(t) = 目标词在对齐后 2D 平面上首→末段的位移（以空间尺度归一）
      stab(t)     = 相邻段自身余弦（对齐后）
      axis_rot(b) = 段 b 对齐到末段的正交矩阵 R 的旋转角 —— ＝**世界模型的轴转了多少**
 预注册：P1 锚 dog/ball 的 stab 高于 control 随机词；P2 moon 低 stab 且位移指向体裁词；P3 axis_rot 随龄单调？
"""
import os, json, math, collections
import numpy as np
from drift4_space import (CHAT, BANDS, BANDLAB, build_band, svd_embed, cos, age, band_of,
                          utt_tokens, MINFREQ, DIM)

HERE = os.path.dirname(os.path.abspath(__file__))
import glob
import re

NOISE = set("""xxx yyy zzz humm mm hm huh uh oh ah ha er um op pg yl vl gl bch de fo ge kay
ba im dont doesnt cant wanna isnt appened bookingc commydito nighnighf wha comin lookin gettin
pickin everytime treetop""".split())
TARGETS = ["moon", "sun", "star", "light", "flower", "dog", "baby", "kitty", "ball",
           "car", "book", "milk", "tree", "cup"]
ANCHORS = {"dog", "baby", "kitty", "ball", "car", "book", "milk", "tree", "cup"}


def clean(tokens):
    return [w for w in tokens if w not in NOISE]


def load_bands():
    bs = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(CHAT, "*.cha"))):
        txt = open(f, encoding="utf-8", errors="ignore").read()
        m = age(txt)
        if m is None:
            continue
        b = band_of(m)
        if b is None:
            continue
        utts = [clean(utt_tokens(l)) for l in txt.splitlines() if l.startswith("*CHI:")]
        utts = [u for u in utts if u]
        if utts:
            bs[b].append(utts)
    return bs


def orth_procrustes(A, B):
    """求正交 R 使 A@R ≈ B（A,B 行对应同一批词）。返回 R 与残差。"""
    M = A.T @ B
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    res = float(np.linalg.norm(A @ R - B) / (np.linalg.norm(B) + 1e-12))
    return R, res


def rot_angle_deg(R):
    """旋转的"总转动量": 以与单位阵的 Frobenius 距离折算一个等效角。"""
    d = float(np.linalg.norm(R - np.eye(R.shape[0])))
    return math.degrees(2 * math.asin(min(1.0, d / (2 * math.sqrt(R.shape[0])))))


def main():
    bs = load_bands()
    spaces = {}
    for b in range(len(BANDS)):
        vocab, idx, ppmi, freq = build_band(bs[b])
        Z, U, S = svd_embed(ppmi)
        spaces[b] = dict(vocab=vocab, idx=idx, Z=Z, freq=freq, S=S)
        print(f"[{BANDLAB[b]}] sess={len(bs[b])} vocab={len(vocab)} tok={sum(freq.values())}")

    REF = len(BANDS) - 1                     # 对齐到末段
    ref = spaces[REF]
    aligned = {}
    axis_rot = {}
    for b in range(len(BANDS)):
        sp = spaces[b]
        shared = [w for w in sp["vocab"] if w in ref["idx"]]
        if b == REF or len(shared) < 30:
            aligned[b] = sp["Z"]
            axis_rot[BANDLAB[b]] = 0.0
            continue
        A = sp["Z"][[sp["idx"][w] for w in shared]]
        Bm = ref["Z"][[ref["idx"][w] for w in shared]]
        R, res = orth_procrustes(A, Bm)
        aligned[b] = sp["Z"] @ R
        axis_rot[BANDLAB[b]] = round(rot_angle_deg(R), 2)
        print(f"  align {BANDLAB[b]}→{BANDLAB[REF]}: shared={len(shared)} res={res:.3f} "
              f"rot={rot_angle_deg(R):.1f}°")

    # 2D 投影：对末段 Z 做 SVD，取前 2 主成分作公共平面
    Zt = ref["Z"]
    Zc = Zt - Zt.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    P = Vt[:2].T

    # 每词在每段的 2D 坐标（对齐后）
    traj = {}
    for t in TARGETS:
        pts = []
        for b in range(len(BANDS)):
            sp = spaces[b]
            if t not in sp["idx"]:
                pts.append(None)
                continue
            v = aligned[b][sp["idx"][t]]
            pts.append([round(float(v @ P[:, 0]), 4), round(float(v @ P[:, 1]), 4)])
        traj[t] = pts

    # 自身余弦稳定性（相邻段，对齐后，全维度）
    stab = {}
    for t in TARGETS:
        vals = []
        for b in range(len(BANDS) - 1):
            s0, s1 = spaces[b], spaces[b + 1]
            if t in s0["idx"] and t in s1["idx"]:
                vals.append(round(cos(aligned[b][s0["idx"][t]],
                                      aligned[b + 1][s1["idx"][t]]), 3))
        stab[t] = vals

    # 2D 位移（首末）
    disp = {}
    for t in TARGETS:
        p = [x for x in traj[t] if x]
        if len(p) >= 2:
            disp[t] = round(float(np.linalg.norm(np.array(p[-1]) - np.array(p[0]))), 3)
        else:
            disp[t] = None

    out = dict(bands=BANDLAB, axis_rot=axis_rot, traj=traj, stab=stab, disp=disp,
               sessions={BANDLAB[b]: len(bs[b]) for b in range(len(BANDS))},
               vocab={BANDLAB[b]: len(spaces[b]["vocab"]) for b in range(len(BANDS))})
    json.dump(out, open(os.path.join(HERE, "drift5.json"), "w"), ensure_ascii=False, indent=1)

    lines = ["# 词义漂移 v4 · 跨龄对齐后", "",
             "**世界模型的轴转动**（各段空间 Procrustes 对齐到末段）:",
             "| 段 | " + " | ".join(BANDLAB) + " |",
             "|---|" + "---|" * len(BANDS),
             "| 转动角° | " + " | ".join(str(axis_rot[b]) for b in BANDLAB) + " |", "",
             "| 词 | 相邻段自身余弦（对齐后，越 1 越稳）| 2D 首末位移 |",
             "|---|---|---|"]
    for t in TARGETS:
        lines.append(f"| {t} | " + " · ".join(f"{v:+.2f}" for v in stab[t]) + f" | {disp[t]} |")
    md = "\n".join(lines)
    open(os.path.join(HERE, "report5.md"), "w").write(md + "\n")
    print("\n" + md)
    print("DRIFT5_DONE")


if __name__ == "__main__":
    main()
