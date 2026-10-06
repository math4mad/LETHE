# -*- coding: utf-8 -*-
"""
hierarchical CRP 最小例（H-g4/5/6）
=====================================
把「结构级生长」从单层手工(pcs_struct_growth) 升为**多层自发**。
模型＝阈值驱动 nCRP：自根下行，子匹配 ≥τ_match 归其；[τ_descend, τ_match) 下行再试；
<τ_descend 则在当前节点**开新子**。连续属性走 DP-mixture 开高斯分量。
判据先冻：见 DESIGN_hierarchical_crp_v0.md。

运行: python3 benches/FSSSS/pcs_hier_crp.py   → report_pcs_hier_crp.json
"""
import os
import json

import numpy as np

TAU_MATCH = 0.995       # ≥ → 归入该子（不新开）
TAU_DESCEND = 0.90      # ∈[descend, match) → 下行再试；< → 当前节点开新子
ALPHA = 1.0
CONT_TAU = 1.0          # 连续属性：距最近分量均值 > 此值 → 开新分量
ROOT = "存在"

VEC = {
    "A": [1.0, 0.0, 0.0],
    "B": [0.0, 1.0, 0.0],
    "C": [0.5, 0.5, 0.0],
    "C1": [0.6, 0.4, 0.0],
    "C2": [0.4, 0.6, 0.0],
}
SEQUENCE = ["A", "B", "C", "C1", "C2"]


def v(name):
    return np.array(VEC[name], float)


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


class Node:
    def __init__(self, name, proto, parent=None):
        self.name = name
        self.proto = proto
        self.parent = parent
        self.children = []
        self.n = 0

    def add_child(self, name, proto, parent_path):
        c = Node(name, proto, self)
        self.children.append(c)
        return c


def insert(root, name, proto, log):
    v = root
    path = [ROOT]
    while True:
        if not v.children:
            c = v.add_child(name, proto, path)
            c.n += 1
            path.append(c.name)
            log.append({"obs": name, "path": path, "parent": v.name, "opened": c.name})
            return c
        scores = [cos(proto, ch.proto) for ch in v.children]
        k = int(np.argmax(scores))
        s = scores[k]
        if s >= TAU_MATCH:
            v.children[k].n += 1
            path.append(f"{v.children[k].name}(assigned)")
            log.append({"obs": name, "path": path, "parent": None, "opened": None})
            return v.children[k]
        elif s >= TAU_DESCEND:
            v = v.children[k]
            v.n += 1
            path.append(v.name)
            continue
        else:
            c = v.add_child(name, proto, path)
            c.n += 1
            path.append(c.name)
            log.append({"obs": name, "path": path, "parent": v.name, "opened": c.name})
            return c


def depth(root):
    if not root.children:
        return 0
    return 1 + max(depth(c) for c in root.children)


def weights(node):
    tot = sum(c.n for c in node.children)
    if tot == 0:
        return {}
    return {c.name: c.n / tot for c in node.children}


def all_nodes(root):
    out = [root]
    for c in root.children:
        out += all_nodes(c)
    return out


def leaves(root, acc=1.0, out=None):
    if out is None:
        out = {}
    if not root.children:
        out[root.name] = acc
        return out
    w = weights(root)
    for c in root.children:
        leaves(c, acc * w[c.name], out)
    return out


def dp_mixture(xs, tau=CONT_TAU):
    comps = []  # [[mean, n], ...]
    for x in xs:
        if not comps:
            comps.append([x, 1])
            continue
        d = [abs(x - m) for m, _ in comps]
        k = int(np.argmin(d))
        if d[k] <= tau:
            m, n = comps[k]
            comps[k] = [(m * n + x) / (n + 1), n + 1]
        else:
            comps.append([x, 1])
    return comps


def main():
    print("=" * 88)
    print("hierarchical CRP 最小例 — 递归开列(parent 自动) + 连续开分量")
    print("=" * 88)
    root = Node(ROOT, None, None)
    log = []
    print("\n观测流:", " ".join(SEQUENCE), f"  (τ_match={TAU_MATCH}, τ_descend={TAU_DESCEND})")
    for name in SEQUENCE:
        insert(root, name, v(name), log)

    # 树打印
    def show(node, ind=0):
        pr = np.round(node.proto, 2).tolist() if node.proto is not None else None
        print("   " + "  " * ind + f"└─{node.name} (n={node.n}, proto={pr})")
        for c in node.children:
            show(c, ind + 1)
    print("\n树:")
    show(root)

    d = depth(root)
    lv = leaves(root)
    s = sum(lv.values())
    per_node_sigma = {n.name: round(sum(weights(n).values()), 12) for n in all_nodes(root) if n.children}
    opens = [e for e in log if e["opened"]]
    depth2 = [e for e in opens if len(e["path"]) >= 3]  # 存在 > X > Y ⇒ 深度≥2

    # 连续属性 DP-mixture（叶 C1 内）
    sizes = [1.0, 1.05, 1.1, 5.0]
    comps = dp_mixture(sizes)

    print(f"\n日志（开新节点）:")
    for e in opens:
        print(f"   {e['obs']:3s} → path {' > '.join(e['path'])}  (parent={e['parent']})")

    print(f"\n① 树深度 = {d}")
    print(f"② 各内部节点 Σw: {per_node_sigma}")
    print(f"③ 叶先验 Σ = {s:.12f}   叶: { {k: round(x,4) for k,x in lv.items()} }")
    print(f"④ 连续属性 sizes={sizes} → DP 分量: {[(round(m,3),n) for m,n in comps]}")

    H_g4 = (d >= 2) and all(abs(sig - 1) < 1e-9 for sig in per_node_sigma.values())
    H_g5 = all(e["parent"] is not None for e in opens) and bool(depth2)
    H_g6 = len(comps) >= 2
    print("\n冻结判据:")
    print(f"  H-g4 递归开列(深度={d}≥2 且各节点 Σw=1): {H_g4}  {'✓' if H_g4 else '✗'}")
    print(f"  H-g5 parent 自动(由下行路径定, 有深度≥2 的开列): {H_g5}  {'✓' if H_g5 else '✗'}")
    print(f"  H-g6 连续开新分量(分量={len(comps)}≥2): {H_g6}  {'✓' if H_g6 else '✗'}")

    out = {
        "tau_match": TAU_MATCH, "tau_descend": TAU_DESCEND,
        "sequence": SEQUENCE, "log": log,
        "tree_depth": d, "open_events": opens,
        "per_node_weight_sum": per_node_sigma, "leaf_prior": lv, "leaf_prior_sum": s,
        "continuous": {"sizes": sizes, "components": comps},
        "criteria": {"H-g4": H_g4, "H-g5": H_g5, "H-g6": H_g6},
    }
    path = os.path.join(os.path.dirname(__file__), "report_pcs_hier_crp.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n报告 → {path}")


if __name__ == "__main__":
    main()
