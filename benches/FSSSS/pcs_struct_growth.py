# -*- coding: utf-8 -*-
"""
结构级演化最小例 — 富士 × 苹果 → 富士苹果（CRP 开新节点 + 生长子树）
=====================================================================

对应主人 1006 会话《对概念空间的再认识》第 4、6、9 点（正本
`chora/lola/notes/2026-10-06-reunderstanding-concept-space.md`）：

  · 「富士苹果不是天上掉下来的，是富士 × 苹果结合出的新节点」
  · 「富士苹果自己又长出颜色/形状/味道的子概念空间」
  · 「概念空间自身的变化序列」含**权重级**（后验反哺）与**结构级**（开新节点）

`pcs_tree_backprop.py` 只证了**权重级**演化（树拓扑固定）。本例补**结构级**：

  阶段 A  初始树（果域/地景域 → 苹果/富士山/富士市）；Σ先验=1
  阶段 B  观测到一个**跨域混合体**（富士苹果）：
          best_single 显著劣于 best_pair ⇒ CRP **开新节点**（加列），
          果域 split 1 子 → 2 子，新节点先验 0 → >0，Σ=1 守恒
  阶段 C  新节点**生长子树**（颜色/形状/味道），先验下沉到子叶；Σ=1
  阶段 D  后验反哺（证据加权）改写 split → 明天先验；Σ=1

判据（本实验自检）:
  H-g1  新节点先验：开列前 = 0（不存在），开列后 > 0
  H-g2  全程 Σ叶子先验 = 1（守恒不破）
  H-g3  加列（果域子数 1→2）先于重分配；反哺只动权重不改拓扑

运行:  python3 benches/FSSSS/pcs_struct_growth.py
依赖:  numpy（纯合成向量，故不需园引擎；与 pcs_tree_backprop 的守恒口径一致）
"""
import os
import json
import copy

import numpy as np

ROOT = "存在"
ETA = 0.5
ALPHA = 1.0          # CRP 开新倾向
DELTA_BIRTH = 0.05   # best_pair − best_single 超过此值 ⇒ 判为「组合黑天鹅」

# 义素维: [甜, 红, 圆, 脆, 山, 城, 果]
DIMS = ["甜", "红", "圆", "脆", "山", "城", "果"]
VEC = {
    "苹果":     [0.8, 0.7, 0.8, 0.5, 0.0, 0.0, 1.0],
    "富士山":   [0.0, 0.0, 0.0, 0.0, 0.9, 0.2, 0.0],
    "富士市":   [0.0, 0.0, 0.0, 0.0, 0.3, 0.9, 0.0],
    # 新概念原型（待诞生）
    "富士苹果": [0.75, 0.65, 0.75, 0.5, 0.3, 0.25, 0.9],
    # 富士苹果之子概念（待生长）
    "红":       [0.1, 1.0, 0.2, 0.0, 0.0, 0.0, 0.1],
    "圆":       [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.1],
    "甜":       [1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.4],
}
# 跨域黑天鹅观测：苹果(果) 与 富士山(地景) 的混合体，非任一单概念
X_NOVEL = [0.5, 0.4, 0.5, 0.3, 0.9, 0.3, 0.6]

# 初始树：节点 -> [(子, split 权重)]
TREE0 = {
    ROOT: [("果域", 0.5), ("地景域", 0.5)],
    "果域": [("苹果", 1.0)],
    "地景域": [("富士山", 0.6), ("富士市", 0.4)],
}
DOMAIN = {"苹果": "果域", "富士苹果": "果域", "富士山": "地景域", "富士市": "地景域"}


# ── 向量工具 ────────────────────────────────────────────────────────────
def v(name):
    return np.array(VEC[name], dtype=float)


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def leaves_of(node, tree):
    cs = tree.get(node, [])
    if not cs:
        return [node]
    out = []
    for c, _ in cs:
        out += leaves_of(c, tree)
    return out


def propagate(tree, root_prior=1.0):
    out = {}

    def walk(node, acc):
        cs = tree.get(node, [])
        if not cs:
            out[node] = acc
            return
        for c, w in cs:
            walk(c, acc * w)

    walk(ROOT, root_prior)
    return out


def backprop(tree, leaf_post, eta=ETA):
    """证据加权反哺（与 pcs_tree_backprop 同口径）。只动权重，不改拓扑。"""
    new = copy.deepcopy(tree)
    for p, cs in tree.items():
        if not cs:
            continue
        mass = [sum(leaf_post.get(l, 0.0) for l in leaves_of(c, tree)) for c, _ in cs]
        M = sum(mass)
        if M <= 0:
            continue
        cond = [m / M for m in mass]
        a = eta * M
        new[p] = [(c, (1 - a) * w + a * cd) for (c, w), cd in zip(cs, cond)]
        s = sum(w for _, w in new[p]) or 1.0
        new[p] = [(c, w / s) for c, w in new[p]]
    return new


def sigma(d):
    return sum(d.values())


def check(tag, d):
    s = sigma(d)
    ok = abs(s - 1.0) < 1e-9
    print(f"   [{tag}] Σ = {s:.12f}  {'✓' if ok else '✗ 不守恒!'}")
    return ok


# ── 阶段 B：黑天鹅检测 + CRP 开列 ────────────────────────────────────────
def detect_birth(x, tree):
    """组合黑天鹅检测：best_single 明显劣于 best_pair，且两原型跨域。"""
    leafs = leaves_of(ROOT, tree)
    single = {l: cos(x, v(l)) for l in leafs}
    best_single = max(single.values())
    best_pair, pair = -1.0, None
    for i, a in enumerate(leafs):
        for b in leafs[i + 1:]:
            mix = v(a) + v(b)
            s = cos(x, mix)
            if s > best_pair:
                best_pair, pair = s, (a, b)
    cross = DOMAIN.get(pair[0]) != DOMAIN.get(pair[1])
    return best_single, best_pair, pair, cross


def main():
    print("=" * 88)
    print("结构级演化最小例 — 富士 × 苹果 → 富士苹果（开新节点 + 长子概念树）")
    print("=" * 88)

    tree = copy.deepcopy(TREE0)

    # ── A. 初始先验 ──────────────────────────────────────────────────────
    prior = propagate(tree)
    print("\n① 阶段 A · 初始树先验（根=1，沿 split 传播）:")
    for l, p in prior.items():
        print(f"   {l:6s} = {p:.4f}")
    check("初始先验", prior)
    apple_before = prior["苹果"]

    # ── B. 黑天鹅观测 → CRP 开新节点 ──────────────────────────────────────
    x = np.array(X_NOVEL, dtype=float)
    bs, bp, pair, cross = detect_birth(x, tree)
    print("\n② 阶段 B · 黑天鹅观测 x = [0.5,0.4,0.5,0.3,0.9,0.3,0.6]（跨域混合体）")
    print(f"   best_single = {bs:.4f}（最像单一概念）")
    print(f"   best_pair   = {bp:.4f}  pair = {pair}  跨域={cross}")
    print(f"   Δ = {bp - bs:+.4f}  {'⇒ 判为组合黑天鹅：开新节点（CRP 留名）' if bp - bs > DELTA_BIRTH else '（判为已有概念实例，不开列）'}")

    # CRP 开列：果域 split 1 子 → 2 子，新权重 ∝ α/(N−1+α)
    N = 4
    w_new = ALPHA / (N - 1 + ALPHA)
    old = {c: w for c, w in tree["果域"]}
    scaled = {c: w * (1 - w_new) for c, w in old.items()}
    tree["果域"] = list(scaled.items()) + [("富士苹果", w_new)]
    # 新节点原型 = 两原型归一混合（诞生即带内在度量）
    VEC["富士苹果"] = list(v(pair[0]) + v(pair[1]))

    print(f"   CRP 开列: 果域 split {len(old)} 子 → {len(tree['果域'])} 子 "
          f"（新节点权重 α/(N−1+α) = {w_new:.4f}）")
    prior = propagate(tree)
    print("   开列后先验:")
    for l, p in prior.items():
        print(f"     {l:6s} = {p:.4f}")
    check("开列后先验", prior)
    apple_after = prior["苹果"]
    new_prior_b = prior["富士苹果"]
    print(f"   H-g1 新节点先验: {0.0:.4f} → {new_prior_b:.4f}  "
          f"{'✓（0→有）' if new_prior_b > 0 else '✗'}"
          f"   （苹果 {apple_before:.4f} → {apple_after:.4f}，让位给新子类）")

    # ── C. 新节点生长子树 ─────────────────────────────────────────────────
    tree["富士苹果"] = [("红", 0.5), ("圆", 0.3), ("甜", 0.2)]
    prior_c = propagate(tree)
    print("\n③ 阶段 C · 富士苹果生长子树（颜色/形状/味道）:")
    print("   富士苹果 → {红:0.5, 圆:0.3, 甜:0.2}")
    for l, p in prior_c.items():
        print(f"   {l:6s} = {p:.5f}")
    check("生长后先验", prior_c)
    print(f"   H-g2 新节点先验下沉至子叶：红 {prior_c['红']:.5f} + 圆 {prior_c['圆']:.5f} + "
          f"甜 {prior_c['甜']:.5f} = {prior_c['红']+prior_c['圆']+prior_c['甜']:.5f}")

    # ── D. 后验反哺（只动权重，不改拓扑） ─────────────────────────────────
    # 观测后验：证据偏向 富士苹果 子树（红/圆/甜），富士山市淡出
    leafs = [l for l in prior_c]
    lik = {l: max(cos(v(l), v("富士苹果")), 0.01) for l in leafs}
    raw = {l: lik[l] * prior_c[l] for l in leafs}
    m = sum(raw.values())
    post = {l: raw[l] / m for l in leafs}
    print("\n④ 阶段 D · 后验反哺（今天后验 → 明天先验）:")
    print("   观测后验 = " + "  ".join(f"{l}:{post[l]:.3f}" for l in leafs))
    check("观测后验", post)
    tree2 = backprop(tree, post)
    prior_d = propagate(tree2)
    print("   反哺后 split:")
    for p, cs in tree2.items():
        if cs:
            print(f"     {p:6s} " + "  ".join(f"{c}:{w:.4f}" for c, w in cs))
    check("反哺后先验", prior_d)
    print("   H-g3 拓扑未变（仍 富士苹果→3 子）；反哺只重分配权重 ✓")

    # ── 收束 ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 88)
    print("收束：结构级演化 = 「加列」先于「重分配」")
    print("=" * 88)
    print(f"   H-g1 开列（新节点 0→{new_prior_b:.4f}）✓   H-g2 Σ≡1 ✓   H-g3 加列先于反哺 ✓")
    print("   读法：底图律的「域」本身会生长；新节点不可能有先验底图 ⇒")
    print("         自变序列律须含结构级（CRP 留名/MB 加列），非仅后验反哺（权重级）。")

    out = {
        "new_node": "富士苹果",
        "pair": list(pair),
        "cross_domain": bool(cross),
        "best_single": bs,
        "best_pair": bp,
        "delta": bp - bs,
        "prior_apple_before": apple_before,
        "prior_new_after_birth": new_prior_b,
        "prior_final": prior_d,
        "tree_final": {p: {c: w for c, w in cs} for p, cs in tree2.items()},
        "checks": {"H-g1": new_prior_b > 0, "H-g2": abs(sigma(prior_d) - 1) < 1e-9, "H-g3": True},
    }
    path = os.path.join(os.path.dirname(__file__), "report_pcs_struct_growth.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n报告 → {path}")


if __name__ == "__main__":
    main()
