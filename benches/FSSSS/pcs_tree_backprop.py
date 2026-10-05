# -*- coding: utf-8 -*-
"""
树先验 × 后验反哺 (PCS 扩展 A) — 最小实验
=========================================
对应 ima 会话 `external/diag-20261005.txt`：
  · 「WordNet is-a 树接入 ExplicitPrior」——先验沿树传播而非平铺查表
  · 下一步「后验沿树回传成先验」——把在线增量贝叶斯在树上闭合

概念树（用园引擎 16 义素词表 / 5 概念空间）::

    存在(根, prior=1.0)
    ├── 市井域 (2/3)
    │   ├── 超市       (1/2)
    │   └── 路边摊     (1/2)
    └── 叙事域 (1/3)
        ├── 觉醒循环   (1/3)
        ├── 接待员日常 (1/3)
        └── 福特剧场   (1/3)

初始先验 = 根先验 × 沿路径 split 累乘 → 复现园引擎 prior
（超市=1/3 · 摊=1/3 · 叙事各=1/9）。

反哺律（本实验核心）::

    今天叶后验 q(leaf)
    ──向上汇总──▶ 内部节点各子节点质量 m_i = Σ_{leaf∈child_i} q(leaf)
    新 split  w_i' = (1−η)·w_i + η·(m_i / Σ m)
    明天先验 = 用新 split 从根重新传播

守恒: Σ叶子先验 = 1 ; Σ叶子后验 = 1 ; 反哺后每节点 Σsplit = 1。

运行:  python3 benches/FSSSS/pcs_tree_backprop.py
依赖:  numpy（复用根目录 cognitive_engine.py）
"""
import os
import sys
import json
import copy

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

import numpy as np
from cognitive_engine import SemeBasedCognitiveEngine

ETA = 0.5          # 反哺学习率
ROOT = "存在"

# 树：节点 -> [(子节点, split 权重)]
TREE0 = {
    ROOT: [("市井域", 2 / 3), ("叙事域", 1 / 3)],
    "市井域": [("超市", 1 / 2), ("路边摊", 1 / 2)],
    "叙事域": [("觉醒循环", 1 / 3), ("接待员日常", 1 / 3), ("福特剧场", 1 / 3)],
}
LEAVES = ["超市", "路边摊", "觉醒循环", "接待员日常", "福特剧场"]
INTERNAL = ["市井域", "叙事域"]          # 带 split 的内部节点（不含根）


# ── 树操作 ──────────────────────────────────────────────────────────────
def children(node, tree):
    return tree.get(node, [])


def descendants(leaf, tree):
    """叶子向上到根的路径（不含根）。"""
    parent = {}
    for p, cs in tree.items():
        for c, _ in cs:
            parent[c] = p
    path = []
    cur = leaf
    while cur in parent:
        path.append(cur)
        cur = parent[cur]
    return path


def propagate(tree, root_prior=1.0):
    """前向：先验沿树传播，返回 {叶子: 先验}。"""
    out = {}

    def walk(node, acc):
        cs = children(node, tree)
        if not cs:
            out[node] = acc
            return
        for c, w in cs:
            walk(c, acc * w)

    walk(ROOT, root_prior)
    return out


def split_table(tree):
    """{内部节点: {子: 权重}} —— 便于展示/反哺。"""
    return {p: {c: w for c, w in cs} for p, cs in tree.items()}


def backprop(tree, leaf_posterior, eta=ETA):
    """后验反哺（证据加权版）:

        M_p = 该节点子树的总后验质量（= 有多少证据落在这一支）
        cond_i = m_i / M_p  （节点内条件份额）
        w_i' = (1 − η·M_p)·w_i + (η·M_p)·cond_i

    M_p→0（无证据的旁支）⇒ 权重不动；M_p→1 ⇒ 按条件后验收敛。
    同节点内 Σw=1 自守；跨节点独立。
    """
    new = copy.deepcopy(tree)
    for p, cs in tree.items():
        if not cs:
            continue
        mass = [sum(leaf_posterior.get(l, 0.0) for l in collect_leaves(c, tree))
                for c, _ in cs]
        M = sum(mass)
        if M <= 0:
            continue
        cond = [m / M for m in mass]
        alpha = eta * M           # 证据权重
        new[p] = [(c, (1 - alpha) * w + alpha * cd) for (c, w), cd in zip(cs, cond)]
        s = sum(w for _, w in new[p]) or 1.0
        new[p] = [(c, w / s) for c, w in new[p]]
    return new


def collect_leaves(node, tree):
    cs = children(node, tree)
    if not cs:
        return [node]
    out = []
    for c, _ in cs:
        out += collect_leaves(c, tree)
    return out


# ── 推断 ────────────────────────────────────────────────────────────────
def cosine_lik(engine, concept_vecs, word, floor=0.01):
    x = np.array([float(v) for v in engine.encode_word(word)])
    return {n: max(float(np.dot(cv, x) / (np.linalg.norm(cv) * np.linalg.norm(x) + 1e-8)), floor)
            for n, cv in concept_vecs.items()}


def posterior_step(engine, concept_vecs, prior, word):
    lik = cosine_lik(engine, concept_vecs, word)
    u = {n: lik[n] * prior[n] for n in lik}
    m = sum(u.values()) or 1.0
    return {n: u[n] / m for n in u}, lik


def mass_check(name, d):
    s = sum(d.values())
    ok = abs(s - 1.0) < 1e-9
    print(f"   {name}: Σ = {s:.12f}  {'✓' if ok else '✗ 不守恒!'}")
    return ok


def main():
    engine = SemeBasedCognitiveEngine()
    tree = copy.deepcopy(TREE0)
    concept_vecs = {n: np.array([float(v) for v in engine.get_concept_vector(n)]) for n in LEAVES}

    print("=" * 86)
    print("树先验 × 后验反哺（最小实验）— 园 16 义素词表 · 5 概念空间")
    print("=" * 86)

    # 0 · 初始先验：树传播 vs 园引擎
    prior = propagate(tree)
    print("\n① 初始先验（根=1，沿树 split 传播）:")
    for n in LEAVES:
        eng = engine.concept_spaces[n]["prior_prob"]
        mark = "✓" if abs(prior[n] - eng) < 1e-9 else f"✗(园={eng:.4f})"
        print(f"   {n:6s} 树={prior[n]:.4f}   园={eng:.4f}   {mark}")
    mass_check("初始先验", prior)

    # 1 · 两轮观测: 反哺前后各跑一遍，看先验如何被昨天的后验改写
    stream = ["塑胶凳", "煤气罐", "三轮车", "大排档", "冷柜", "购物车"]
    print("\n② 观测流: " + " ".join(stream))
    print("   （先验 → 逐词后验 → 每步后验反哺 → 明天先验）")

    for rnd in range(1, 4):
        print(f"\n{'─' * 86}\n   第 {rnd} 轮  先验 = " +
              "  ".join(f"{n}:{prior[n]:.3f}" for n in LEAVES))
        post = dict(prior)
        for w in stream:
            post, lik = posterior_step(engine, concept_vecs, post, w)
            print(f"     [{w:<4s}] 后验 = " + "  ".join(f"{n}:{post[n]:.3f}" for n in LEAVES))
        mass_check("本轮末后验", post)
        # 反哺：今天后验 → 新 split → 明天先验
        tree = backprop(tree, post)
        prior = propagate(tree)
        print(f"   反哺后 split: 市井域=" +
              " ".join(f"{c}:{w:.3f}" for c, w in tree["市井域"]) +
              " | 叙事域=" + " ".join(f"{c}:{w:.3f}" for c, w in tree["叙事域"]))
        print(f"   明天先验 = " + "  ".join(f"{n}:{prior[n]:.3f}" for n in LEAVES))
        mass_check("反哺后先验", prior)

    # 2 · 守恒与学习方向自检
    print("\n" + "=" * 86)
    print("③ 自检")
    print("=" * 86)
    # 学习方向：观测流里 路边摊 证据（塑胶凳/煤气罐/三轮车/大排档）为主
    root_w = {c: w for c, w in tree[ROOT]}
    print(f"   根 split 市井:叙事 = {root_w['市井域']:.4f} : {root_w['叙事域']:.4f}  "
          f"（初始 2/3:1/3，市井证据 → 市井升）")
    pm = {c: w for c, w in tree["市井域"]}
    print(f"   摊/超市 split: 摊={pm['路边摊']:.4f} 超市={pm['超市']:.4f}  " +
          f"{'✓ 摊升（观测方向正确）' if pm['路边摊'] > 0.5 else '（无偏）'}")
    nm = {c: w for c, w in tree["叙事域"]}
    print(f"   叙事域 split（旁支，证据≈0）: " +
          "  ".join(f"{c}:{w:.4f}" for c, w in nm.items()) +
          "   ← 应基本不动（证据加权反哺的抗漂移）")
    print("   注：三轮观测同一先验 → 后验沿树回传，把『路边摊』的 split 抬高 =")
    print("       在线增量贝叶斯在树上闭合（今天后验 = 明天先验）。")

    out = {
        "eta": ETA,
        "tree_final": {p: {c: w for c, w in cs} for p, cs in tree.items()},
        "prior_final": prior,
    }
    path = os.path.join(os.path.dirname(__file__), "report_pcs_tree_backprop.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n报告 → {path}")


if __name__ == "__main__":
    main()
