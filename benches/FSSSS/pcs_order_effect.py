# -*- coding: utf-8 -*-
"""
nCRP 上的 PB 式序效应 — 同一袋观测，不同喂序 ⇒ 结构是否不同
==============================================================
承 hierarchical CRP。PB 族问：「序本身在参数/结构里留下什么痕」。本器把**同一多集**的观测
按不同次序喂入 nCRP，比较所得**树结构**（顶层共属关系），检验「序即数据」。

判据先冻（见 DESIGN_order_effect_crp.md）:
  H-o1 序改结构：原序与至少一个其它序的**顶层共属关系**不同（对称差 > 0）
  H-o2 逆序有别：逆序与原序结构不同（共属差 > 0 或 直子/深度 不同）
  H-o3 守恒：各序叶先验 Σ=1

运行: python3 benches/FSSSS/pcs_order_effect.py   → report_order_effect_crp.json
"""
import os
import sys
import json
import math
import random

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pcs_hier_crp_v1 as b  # Node / cos / v / VEC / STREAM / ROOT / BETA / TAU_MATCH / leaves

ALPHA = 1.0


def grow(stream, alpha=ALPHA):
    """返回 root 与每个观测出现位置(按 name 的第 k 次)的顶层祖先。"""
    root = b.Node(b.ROOT, None, None)
    occ = {}
    top_of = {}          # key -> 顶层祖先名
    for name in stream:
        k = occ.get(name, 0)
        occ[name] = k + 1
        key = (name, k)
        path = b.insert(root, name, b.v(name), alpha)   # v1 的 MAP 插入，返回 path(名串)
        top = (path[1] if len(path) > 1 else path[0]).split("(")[0]
        top_of[key] = top
    return root, top_of


def co_membership(top_of):
    keys = sorted(top_of)
    out = set()
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if top_of[keys[i]] == top_of[keys[j]]:
                out.add(frozenset((keys[i], keys[j])))
    return out


def struct_signature(root):
    nodes = b.all_nodes(root)
    return {"n_nodes": len(nodes), "root_breadth": len(root.children), "depth": b.depth(root),
            "leaf_sum": round(sum(b.leaves(root).values()), 12)}


def main():
    print("=" * 88)
    print("nCRP 上的 PB 式序效应 — 同袋观测 · 不同喂序")
    print("=" * 88)
    base = list(b.STREAM)
    orderings = {"original": base, "reversed": list(reversed(base)), "grouped": sorted(base)}
    for s in range(5):
        perm = base[:]
        random.Random(100 + s).shuffle(perm)
        orderings[f"shuffle{s}"] = perm

    print(f"\n袋（多集）: {sorted(base)}  (α={ALPHA})")
    print(f"\n{'序':>10} {'节点数':>6} {'直子':>5} {'深度':>5} {'Σ叶':>8} {'共属差vs原序':>12}")
    ref = None
    rows = {}
    for label, stream in orderings.items():
        root, top_of = grow(stream)
        sig = struct_signature(root)
        cm = co_membership(top_of)
        if ref is None:
            ref = cm
            dist = 0
        else:
            dist = len(cm ^ ref)
        rows[label] = {"signature": sig, "co_membership_dist_vs_original": dist}
        print(f"{label:>10} {sig['n_nodes']:>6} {sig['root_breadth']:>5} {sig['depth']:>5} "
              f"{sig['leaf_sum']:>8} {dist:>12}")

    dists = {k: v["co_membership_dist_vs_original"] for k, v in rows.items() if k != "original"}
    H_o1 = any(d > 0 for d in dists.values())
    rev = rows["reversed"]
    orig = rows["original"]["signature"]
    H_o2 = (rev["co_membership_dist_vs_original"] > 0) or \
           (rev["signature"]["root_breadth"] != orig["root_breadth"]) or \
           (rev["signature"]["depth"] != orig["depth"])
    H_o3 = all(abs(v["signature"]["leaf_sum"] - 1) < 1e-9 for v in rows.values())

    print("\n冻结判据:")
    print(f"  H-o1 序改结构（共属差>0）: max={max(dists.values())}  {'✓' if H_o1 else '✗'}")
    print(f"  H-o2 逆序有别（共属差={rev['co_membership_dist_vs_original']}, "
          f"直子 {rev['signature']['root_breadth']} vs {orig['root_breadth']}）: {H_o2}  {'✓' if H_o2 else '✗'}")
    print(f"  H-o3 各序 Σ叶=1: {H_o3}  {'✓' if H_o3 else '✗'}")

    out = {"alpha": ALPHA, "bag": sorted(base), "rows": rows,
           "criteria": {"H-o1": H_o1, "H-o2": H_o2, "H-o3": H_o3}}
    path = os.path.join(HERE, "report_order_effect_crp.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o))
    print(f"\n报告 → {path}")


if __name__ == "__main__":
    main()
