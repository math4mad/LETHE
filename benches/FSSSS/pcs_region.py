# -*- coding: utf-8 -*-
"""
PCS 区域概念版 — 让归一化常数 size(C) 真正生效
================================================

`pcs_prototype.py` 里概念退化为「原型点」，此时 Z_C = size(C) 只依赖 c/权重/维数、
与位置无关，故五个 Z 相等，后验中约掉 —— size 没起作用。

本脚本改用**区域概念**（锚点包围盒 → 凸体核心），使 Z_C 因概念的"体积/松紧"而异：

    p(x|C) = μ_C(x) / Z_C          ← 正确（归一化密度）
    p̃(x|C) = μ_C(x)                ← 遗忘 Z（园引擎式的"未归一似然"）

对比两者后验，展示 **忘掉 Z 会把信念偏向体积大的概念**（先验被 Z 暗中加权）。
并给出序贯 + 留出选 c。

运行: /path/to/venv132/bin/python benches/FSSSS/pcs_region.py
"""
import os
import sys
import json
import math

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "external", "ConceptualSpaces-1.3.2-py3"))

import numpy as np
from cognitive_engine import SemeBasedCognitiveEngine
from conceptual_spaces.cs import cs as space
from conceptual_spaces.cs.cuboid import Cuboid
from conceptual_spaces.cs.core import Core
from conceptual_spaces.cs.concept import Concept
from conceptual_spaces.cs.weights import Weights

PAD = 1e-6
C_SENS = 8.0
DIM_EPS = 1e-300

SUPERMARKET_ANCHORS = ["沃尔玛", "收银台", "货架", "购物车", "冷柜"]
STALL_ANCHORS = ["烧烤摊", "折叠桌", "三轮车", "大排档", "煤气罐"]
NARRATIVE = ["觉醒循环", "接待员日常", "福特剧场"]
CONCEPT_NAMES = ["超市", "路边摊"] + NARRATIVE


def build_space(engine):
    names = list(engine.basis_dict.keys())
    n = len(names)
    dom = {"市井": list(range(0, 10)), "叙事": list(range(10, n))}
    space.init(n, dom)
    dim_w = {d: {i: 1.0 for i in ds} for d, ds in dom.items()}
    return names, dom, Weights({d: 1.0 for d in dom}, dim_w)


def _region_cuboid(engine, names, dom, anchors):
    Vs = [[float(x) for x in engine.encode_word(w)] for w in anchors]
    lo = [min(v[i] for v in Vs) for i in range(len(names))]
    hi = [max(v[i] for v in Vs) for i in range(len(names))]
    for i in range(len(lo)):
        if abs(hi[i] - lo[i]) < PAD:
            lo[i] -= PAD; hi[i] += PAD
    return Cuboid(lo, hi, dom)


def _point_cuboid(engine, names, dom, prot):
    p = [float(x) for x in prot]
    return Cuboid([x - PAD for x in p], [x + PAD for x in p], dom)


def concept_region(engine, names, dom, w, anchors, c=C_SENS):
    return Concept(Core([_region_cuboid(engine, names, dom, anchors)], dom), 1.0, c, w)


def concept_point(engine, names, dom, w, prot, c=C_SENS):
    return Concept(Core([_point_cuboid(engine, names, dom, prot)], dom), 1.0, c, w)


def size_of(concept):
    z = getattr(concept, "_cached_size", None)
    if z is None:
        z = concept.size(); concept._cached_size = z
    return z


def density(concept, x):
    """p(x|C) = μ(x)/Z —— 真归一化密度。"""
    return concept.membership_of(x) / size_of(concept)


def raw(concept, x):
    """μ(x) —— 未归一（遗忘 Z）。"""
    return concept.membership_of(x)


def posterior(liks, priors, forget_z=False):
    u = {n: liks[n] * priors[n] for n in liks}
    m = sum(u.values()) or DIM_EPS
    return {n: u[n] / m for n in u}


def run(concepts, priors, words, engine, forget_z=False):
    post = dict(priors); traj = []; ev = 0.0
    for w in words:
        x = [float(v) for v in engine.encode_word(w)]
        liks = {n: (raw(concepts[n], x) if forget_z else density(concepts[n], x)) for n in concepts}
        u = {n: liks[n] * post[n] for n in liks}
        m = sum(u.values()) or DIM_EPS
        ev += math.log(m)
        post = {n: u[n] / m for n in u}
        traj.append({"word": w, "posterior": dict(post)})
    return traj, ev, post


def main():
    engine = SemeBasedCognitiveEngine()
    names, dom, w = build_space(engine)
    concepts = {
        "超市": concept_region(engine, names, dom, w, SUPERMARKET_ANCHORS),
        "路边摊": concept_region(engine, names, dom, w, STALL_ANCHORS),
    }
    for n in NARRATIVE:
        concepts[n] = concept_point(engine, names, dom, w, engine.get_concept_vector(n))
    priors = {n: engine.concept_spaces[n]["prior_prob"] for n in CONCEPT_NAMES}

    print("=" * 82)
    print("PCS 区域概念版: 归一化常数 Z_C = size(C) 是否改变推断?")
    print("=" * 82)
    Zs = {n: size_of(concepts[n]) for n in CONCEPT_NAMES}
    print("\n  Z_C = size(C) (16D):")
    for n in CONCEPT_NAMES:
        print(f"    {n:6s} Z = {Zs[n]:.6e}")
    print(f"  ⇒ 区域概念 Z 不等: Z(超市)/Z(路边摊) = {Zs['超市']/Zs['路边摊']:.3f}")
    print("    (叙事概念退化为点, Z 相等 —— 与 prototype 版一致)")

    probes = ["塑胶凳", "塑胶袋", "吸管", "煤气罐", "购物车", "大排档", "冷柜", "三轮车"]
    print("\n  单步后验: 正确(÷Z) vs 遗忘Z, 先验均匀")
    print(f"    {'词':<6}| {'正确 argmax':>10} {'p(超市)':>9} {'p(摊)':>8} | "
          f"{'遗忘Z argmax':>12} {'p(超市)':>9} {'p(摊)':>8} | 异?")
    flip = 0
    for wd in probes:
        x = [float(v) for v in engine.encode_word(wd)]
        lk_ok = {n: density(concepts[n], x) for n in CONCEPT_NAMES}
        lk_no = {n: raw(concepts[n], x) for n in CONCEPT_NAMES}
        po = posterior(lk_ok, priors); pn = posterior(lk_no, priors)
        ao = max(po, key=po.get); an = max(pn, key=pn.get)
        diff = "**异**" if ao != an else ""
        flip += int(ao != an)
        print(f"    {wd:<6}| {ao:>10} {po['超市']:>9.4f} {po['路边摊']:>8.4f} | "
              f"{an:>12} {pn['超市']:>9.4f} {pn['路边摊']:>8.4f} | {diff}")

    stream = ["塑胶凳", "煤气罐", "三轮车", "大排档", "我不要再被人摆布"]

    # 全词表扫描: ÷Z vs 遗忘Z 的 argmax 翻转
    print("\n  全词表扫描 (÷Z vs 遗忘Z):")
    flips = []
    for wd in engine.vocab:
        x = [float(v) for v in engine.encode_word(wd)]
        po = posterior({n: density(concepts[n], x) for n in CONCEPT_NAMES}, priors)
        pn = posterior({n: raw(concepts[n], x) for n in CONCEPT_NAMES}, priors)
        if max(po, key=po.get) != max(pn, key=pn.get):
            flips.append((wd, max(po, key=po.get), max(pn, key=pn.get)))
    print(f"    翻转 {len(flips)} / {len(engine.vocab)} 词")
    for wd, a, b in flips:
        print(f"      {wd:<12} ÷Z→{a:<4}  遗忘Z→{b}")
    print(f"\n  忘掉 Z 等价于暗中把先验改成 p̃(C) ∝ Z_C·p(C) ——")
    print(f"    体积大的概念被凭空放大 (超市/路边摊 = {Zs['超市']/Zs['路边摊']:.3f}×)。")

    tr_ok, ev_ok, fin_ok = run(concepts, priors, stream, engine, forget_z=False)
    tr_no, ev_no, fin_no = run(concepts, priors, stream, engine, forget_z=True)
    print("\n  序贯终态 (÷Z vs 遗忘Z):")
    print("    ÷Z  :", {n: round(fin_ok[n], 4) for n in CONCEPT_NAMES})
    print("    遗忘:", {n: round(fin_no[n], 4) for n in CONCEPT_NAMES})

    print("\n  留出选 c (前 3 词训练 / 后 2 词测试), ÷Z:")
    train, test = stream[:3], stream[3:]
    c_scan = {}
    for c in [2.0, 4.0, 8.0, 12.0, 16.0]:
        cs_ = {
            "超市": concept_region(engine, names, dom, w, SUPERMARKET_ANCHORS, c),
            "路边摊": concept_region(engine, names, dom, w, STALL_ANCHORS, c),
        }
        for n in NARRATIVE:
            cs_[n] = concept_point(engine, names, dom, w, engine.get_concept_vector(n), c)
        tr_traj, tr_ev, _ = run(cs_, priors, train, engine)
        _, te_ev, _ = run(cs_, {n: tr_traj[-1]["posterior"][n] for n in CONCEPT_NAMES}, test, engine)
        c_scan[c] = {"train": tr_ev, "test": te_ev, "total": tr_ev + te_ev}
        print(f"     c={c:>5.1f}  train={tr_ev:>8.3f}  test={te_ev:>8.3f}")
    best_c = max(c_scan, key=lambda k: c_scan[k]["test"])
    print(f"   ⇒ 留出测试最优 c = {best_c}")

    report = {"Z": Zs, "single_step_flips": flip, "sequential_correct": fin_ok,
              "sequential_forgetZ": fin_no, "c_scan": c_scan, "best_c": best_c}
    out = os.path.join(os.path.dirname(__file__), "report_pcs_region.json")
    json.dump(report, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=float)
    print(f"\n报告 → {out}")


if __name__ == "__main__":
    main()
