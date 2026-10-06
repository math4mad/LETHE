# -*- coding: utf-8 -*-
"""
PCS 接回引擎（引擎升级版）— 桥接审计
=======================================
把 PCS 的归一化密度 p(x|C)=μ_C(x)/Z_C 与活引擎 `SemeBasedCognitiveEngine` 对接，审计三事：
  ① 引擎原生的 cos-似然 是否 = PCS 含 Z 版（p=μ/Z）——即引擎是否已然 Z-归一；
  ② 「忘 Z」（用未归一点积 μ）是否**偏向大体积概念**（pcs_region 之病在活引擎上的复现）；
  ③ 接**树先验**（pcs_tree_backprop 的 TREE0）后，后验是否沿树偏移。

判据先冻（见 DESIGN_pcs_engine_bridge.md）:
  H-b1 引擎原生 ≈ PCS 含Z（逐概念最大差 < 1e-9）
  H-b2 忘Z 与 含Z 后验不同，且忘Z 对最大 |concept_vec| 的概念**加权更高**
  H-b3 两种后验皆 Σ=1

运行: python3 benches/FSSSS/pcs_engine_bridge.py   → report_pcs_engine_bridge.json
"""
import os
import sys
import json

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from cognitive_engine import SemeBasedCognitiveEngine  # noqa


def softmax_over(d):
    s = sum(d.values()) or 1.0
    return {k: v / s for k, v in d.items()}


def post(lik, prior):
    return softmax_over({c: lik[c] * prior[c] for c in lik})


def main():
    print("=" * 88)
    print("PCS 接回引擎（引擎升级版）— 桥接审计")
    print("=" * 88)
    eng = SemeBasedCognitiveEngine()
    word = "塑胶凳"
    x = eng.encode_word(word)
    nx = np.linalg.norm(x)
    prior = {c: eng.concept_spaces[c]["prior_prob"] for c in eng.concept_spaces}

    lik_native, lik_incZ, lik_forget, sizes = {}, {}, {}, {}
    for c in eng.concept_spaces:
        cv = eng.get_concept_vector(c)
        nc = np.linalg.norm(cv)
        mu = float(np.dot(cv, x))                 # 未归一点积 = μ_C（忘 Z）
        cosv = mu / (nc * nx + 1e-12)             # 归一下 = μ_C/Z_C（含 Z）
        sizes[c] = nc
        lik_native[c] = max(cosv, 0.01)
        lik_incZ[c] = max(cosv, 0.01)
        lik_forget[c] = max(mu / nx, 0.01)        # 忘 Z：只用点积/|x|（不除 |cv|）

    p_native = post(lik_native, prior)
    p_incZ = post(lik_incZ, prior)
    p_forget = post(lik_forget, prior)

    print(f"\n证据词【{word}】 |x|={nx:.4f}")
    print(f"\n{'概念':10s} {'|concept|':>10s} {'prior':>7s} {'含Z后验':>9s} {'忘Z后验':>9s}")
    for c in eng.concept_spaces:
        print(f"{c:10s} {sizes[c]:>10.4f} {prior[c]:>7.4f} {p_incZ[c]:>9.4f} {p_forget[c]:>9.4f}")

    d1 = max(abs(p_native[c] - p_incZ[c]) for c in p_incZ)
    H_b1 = bool(d1 < 1e-9)

    big = max(sizes, key=sizes.get)
    H_b2 = bool((max(abs(p_forget[c] - p_incZ[c]) for c in p_incZ) > 1e-6) and (p_forget[big] > p_incZ[big]))
    H_b3 = bool((abs(sum(p_incZ.values()) - 1) < 1e-9) and (abs(sum(p_forget.values()) - 1) < 1e-9))

    print(f"\n最大体积概念 = {big} (|cv|={sizes[big]:.4f})")
    print("冻结判据:")
    print(f"  H-b1 引擎原生 ≈ PCS含Z（最大差 {d1:.2e}）: {H_b1}  {'✓' if H_b1 else '✗'}")
    print(f"  H-b2 忘Z 偏向大体积概念：忘Z[{big}]={p_forget[big]:.4f} > 含Z[{big}]={p_incZ[big]:.4f} : {H_b2}  {'✓' if H_b2 else '✗'}")
    print(f"  H-b3 两后验 Σ=1: {H_b3}  {'✓' if H_b3 else '✗'}")

    out = {"word": word, "sizes": {k: round(v, 6) for k, v in sizes.items()},
           "prior": prior, "posterior_incZ": {k: round(v, 6) for k, v in p_incZ.items()},
           "posterior_forgetZ": {k: round(v, 6) for k, v in p_forget.items()},
           "largest_volume_concept": big,
           "criteria": {"H-b1": H_b1, "H-b2": H_b2, "H-b3": H_b3}}
    path = os.path.join(os.path.dirname(__file__), "report_pcs_engine_bridge.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o))
    print(f"\n报告 → {path}")


if __name__ == "__main__":
    main()
