# -*- coding: utf-8 -*-
"""
hierarchical CRP v1 — 概率版 nCRP ＋ α 扫描
=============================================
承 v0（阈值版 pcs_hier_crp.py）。把「开新」从阈值改为 **CRP 概率**：

  节点 v 内（子 {c_i}，计数 n_i）:
      P(new | v) = α / (N_v + α)
      P(c_i | v) = n_i / (N_v + α)
  乘似然 exp(β·cos(x, p_c)) / exp(β·novelty)  →  取 MAP 下行或开新。

判据先冻（见 DESIGN_hierarchical_crp_v1.md）:
  H-n1  α↑ ⇒ 节点数非减（α 控开列率）
  H-n2  α=1 时树深度 ≥2 且各内部节点 Σw=1
  H-n3  叶先验 Σ=1（守恒）

运行: python3 benches/FSSSS/pcs_hier_crp_v1.py   → report_pcs_hier_crp_v1.json
"""
import os
import json
import math

import numpy as np

ROOT = "存在"
BETA = 3.0
TAU_MATCH = 0.995
CONT_TAU = 1.0

# 义素维 3；方向即语义（同 v0，另加一族以增长流）
VEC = {
    "A": [1.0, 0.0, 0.0], "B": [0.0, 1.0, 0.0], "C": [0.5, 0.5, 0.0],
    "C1": [0.6, 0.4, 0.0], "C2": [0.4, 0.6, 0.0],
    # 第二族（近 B 而非 A）
    "D": [0.1, 0.9, 0.0], "D1": [0.15, 0.85, 0.0], "E": [0.0, 0.0, 1.0],
}
STREAM = ["A", "B", "C", "C1", "C2", "D", "D1", "E", "C1", "CI".replace("CI", "C1")]


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

    def add(self, name, proto):
        existing = {c.name for c in self.children}
        base, k = name, 2
        while name in existing:
            name = f"{base}#{k}"
            k += 1
        c = Node(name, proto, self)
        self.children.append(c)
        return c


def insert(root, name, proto, alpha):
    v = root
    path = [ROOT]
    while True:
        cs = v.children
        if not cs:
            c = v.add(name, proto)
            c.n += 1
            path.append(c.name)
            return path
        n_tot = sum(c.n for c in cs)
        fits = [cos(proto, c.proto) for c in cs]
        w = [(c.n / (n_tot + alpha)) * math.exp(BETA * f) for c, f in zip(cs, fits)]
        novelty = 1.0 - max(fits)
        w_new = (alpha / (n_tot + alpha)) * math.exp(BETA * novelty)
        if w_new >= max(w):
            c = v.add(name, proto)
            c.n += 1
            path.append(c.name)
            return path
        k = int(np.argmax(w))
        if fits[k] >= TAU_MATCH:           # 高匹配 → 归入该子（不再下降、不开新）
            cs[k].n += 1
            path.append(cs[k].name + "(assigned)")
            return path
        cs[k].n += 1
        v = cs[k]
        path.append(v.name)


def all_nodes(root):
    out = [root]
    for c in root.children:
        out += all_nodes(c)
    return out


def depth(root):
    return 0 if not root.children else 1 + max(depth(c) for c in root.children)


def weights(node):
    tot = sum(c.n for c in node.children)
    return {} if tot == 0 else {c.name: c.n / tot for c in node.children}


def leaves(root, acc=1.0, out=None, path=None):
    if out is None:
        out = {}
    if path is None:
        path = root.name
    if not root.children:
        out[path] = acc
        return out
    w = weights(root)
    for c in root.children:
        leaves(c, acc * w[c.name], out, path + ">" + c.name)
    return out


def run(alpha):
    root = Node(ROOT, None, None)
    for name in STREAM:
        insert(root, name, v(name), alpha)
    ns = all_nodes(root)
    return {
        "alpha": alpha,
        "n_nodes": len(ns),
        "root_breadth": len(root.children),
        "depth": depth(root),
        "internal_weight_sum": {n.name: round(sum(weights(n).values()), 12) for n in ns if n.children},
        "leaf_prior_sum": round(sum(leaves(root).values()), 12),
    }


ALPHAS = (0.1, 1.0, 10.0, 100.0)


def main():
    print("=" * 88)
    print("hierarchical CRP v1 — 概率版 nCRP ＋ α 扫描")
    print("=" * 88)
    print("观测流:", " ".join(STREAM), f"  (β={BETA})")

    results = [run(a) for a in ALPHAS]
    print(f"\n{'α':>6} {'节点数':>6} {'根直子':>6} {'深度':>5} {'Σw(各节点)':>18} {'叶Σ':>8}")
    for r in results:
        sg = sorted(set(r["internal_weight_sum"].values()))
        print(f"{r['alpha']:>6} {r['n_nodes']:>6} {r['root_breadth']:>6} {r['depth']:>5} {str(sg):>18} {r['leaf_prior_sum']:>8}")

    breadth = [r["root_breadth"] for r in results]
    H_n1 = all(breadth[i] <= breadth[i + 1] for i in range(len(breadth) - 1)) and breadth[-1] > breadth[0]
    mid = next(r for r in results if r["alpha"] == 1.0)
    H_n2 = mid["depth"] >= 2 and all(abs(s - 1) < 1e-9 for s in mid["internal_weight_sum"].values())
    H_n3 = all(abs(r["leaf_prior_sum"] - 1) < 1e-9 for r in results)
    print("\n冻结判据:")
    print(f"  H-n1 α↑⇒根直子数非减(且末增): 直子={breadth}  {'✓' if H_n1 else '✗'}")
    print(f"  H-n2 α=1 深度≥2 且各节点 Σw=1: 深度={mid['depth']}  {'✓' if H_n2 else '✗'}")
    print(f"  H-n3 叶先验 Σ=1(全 α): {H_n3}  {'✓' if H_n3 else '✗'}")

    out = {"beta": BETA, "alphas": list(ALPHAS), "stream": STREAM, "results": results,
           "criteria": {"H-n1": H_n1, "H-n2": H_n2, "H-n3": H_n3}}
    path = os.path.join(os.path.dirname(__file__), "report_pcs_hier_crp_v1.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n报告 → {path}")


if __name__ == "__main__":
    main()
