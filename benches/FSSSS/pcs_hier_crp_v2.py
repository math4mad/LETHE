# -*- coding: utf-8 -*-
"""
hierarchical CRP v2 — 采样版 nCRP ＋ 多种子
=============================================
承 v1（MAP 确定性版）。把「开新 vs 复用哪子」改为**真 CRP 采样**（每种子一 rng），
以多籽统计结构分布。判据见 DESIGN_hierarchical_crp_v2.md。

判据先冻:
  H-s1 采样致结构随机（某种子集下 n_nodes 有方差 >0）
  H-s2 α 控平均开列率：mean(根直子) 随 α 非减且末增
  H-s3 全籽全 α 叶先验 Σ=1（守恒）

运行: python3 benches/FSSSS/pcs_hier_crp_v2.py  → report_pcs_hier_crp_v2.json
"""
import os
import sys
import json
import math
import random

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pcs_hier_crp_v1 as b  # Node / cos / VEC / STREAM / weights / leaves / all_nodes / depth

SEEDS = list(range(20))
ALPHAS = (0.1, 1.0, 10.0, 100.0)


def insert_sampling(root, name, proto, alpha, rng):
    v = root
    while True:
        cs = v.children
        if not cs:
            c = v.add(name, proto)
            c.n += 1
            return
        N = sum(c.n for c in cs)
        fits = [b.cos(proto, c.proto) for c in cs]
        ws = [(c.n / (N + alpha)) * math.exp(b.BETA * f) for c, f in zip(cs, fits)]
        novelty = 1.0 - max(fits)
        w_new = (alpha / (N + alpha)) * math.exp(b.BETA * novelty)
        r = rng.random() * (sum(ws) + w_new)
        acc, chosen = 0.0, None
        for i, w in enumerate(ws):
            acc += w
            if r <= acc:
                chosen = i
                break
        if chosen is None:
            c = v.add(name, proto)
            c.n += 1
            return
        if fits[chosen] >= b.TAU_MATCH:
            cs[chosen].n += 1
            return
        cs[chosen].n += 1
        v = cs[chosen]


def one_run(alpha, seed):
    root = b.Node(b.ROOT, None, None)
    rng = random.Random(seed)
    for name in b.STREAM:
        insert_sampling(root, name, b.v(name), alpha, rng)
    return {"n_nodes": len(b.all_nodes(root)),
            "root_breadth": len(root.children),
            "depth": b.depth(root),
            "leaf_sum": round(sum(b.leaves(root).values()), 12)}


def stats(xs):
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    return round(m, 3), round(math.sqrt(var), 3)


def main():
    print("=" * 88)
    print("hierarchical CRP v2 — 采样版 nCRP ＋ 多种子")
    print("=" * 88)
    print(f"种子 {len(SEEDS)} 个 · α ∈ {list(ALPHAS)} · β={b.BETA} · 观测流 {' '.join(b.STREAM)}")

    table = []
    all_leaf_ok = True
    for a in ALPHAS:
        runs = [one_run(a, s) for s in SEEDS]
        nn = [r["n_nodes"] for r in runs]
        br = [r["root_breadth"] for r in runs]
        dp = [r["depth"] for r in runs]
        all_leaf_ok &= all(abs(r["leaf_sum"] - 1) < 1e-9 for r in runs)
        table.append({"alpha": a,
                      "n_nodes_mean": stats(nn)[0], "n_nodes_std": stats(nn)[1],
                      "breadth_mean": stats(br)[0], "breadth_std": stats(br)[1],
                      "depth_mean": stats(dp)[0], "depth_std": stats(dp)[1],
                      "breadth_vals": br})

    print(f"\n{'α':>6} {'节点数(μ±σ)':>14} {'根直子(μ±σ)':>14} {'深度(μ±σ)':>12}")
    for t in table:
        print(f"{t['alpha']:>6} {t['n_nodes_mean']:>8}±{t['n_nodes_std']:<5} "
              f"{t['breadth_mean']:>8}±{t['breadth_std']:<5} {t['depth_mean']:>6}±{t['depth_std']:<5}")

    H_s1 = any(t["n_nodes_std"] > 0 or t["breadth_std"] > 0 or t["depth_std"] > 0 for t in table)
    bmean = [t["breadth_mean"] for t in table]
    H_s2 = all(bmean[i] <= bmean[i + 1] for i in range(len(bmean) - 1)) and bmean[-1] > bmean[0]
    H_s3 = all_leaf_ok
    print("\n冻结判据:")
    print(f"  H-s1 采样致结构随机(有方差>0): {H_s1}  {'✓' if H_s1 else '✗'}")
    print(f"  H-s2 α 控平均开列率(mean根直子非减且末增): {bmean}  {'✓' if H_s2 else '✗'}")
    print(f"  H-s3 全籽全α Σ=1: {H_s3}  {'✓' if H_s3 else '✗'}")

    out = {"seeds": SEEDS, "alphas": list(ALPHAS), "beta": b.BETA, "stream": b.STREAM,
           "table": table, "criteria": {"H-s1": H_s1, "H-s2": H_s2, "H-s3": H_s3}}
    path = os.path.join(HERE, "report_pcs_hier_crp_v2.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n报告 → {path}")


if __name__ == "__main__":
    main()
