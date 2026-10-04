# -*- coding: utf-8 -*-
"""
Task C — 对表实验: Bechberger FSSSS (v1.3.2) ↔ 本园 MaxSim + 贝叶斯引擎

同一批「超市 vs 路边摊」锚点上, 比较两种"相似度/归属"机器:
  A) 本园引擎 (cognitive_engine.py): 义素基向量 + 余弦"似然" + 贝叶斯后验
  B) Bechberger FSSSS: 模糊简单星形集 - 凸体核心 + 指数隶属 + Jaccard 相似 + 积分 between

关键提问: 两套机器在排序上是否一致? 若一致, 是否说明二者都只是在算"几何相似",
而**都缺少真正的概率语义**(即: 没有 P(word|space) 的规范化, 没有可校准的后验)?

运行:
  /path/to/venv132/bin/python benches/FSSSS/bridge_maxsim_fssss.py
依赖: external/ConceptualSpaces-1.3.2-py3 + numpy/scipy/shapely/numdifftools
"""
import os
import sys
import json

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

# ── 锚点集 (人工分组, 同园内语义) ──
SUPERMARKET_ANCHORS = ["沃尔玛", "收银台", "货架", "购物车", "冷柜"]
STALL_ANCHORS = ["烧烤摊", "折叠桌", "三轮车", "大排档", "煤气罐"]
PROBE_WORDS = ["塑胶凳", "塑胶袋", "吸管", "煤气罐", "购物车",
               "大排档", "冷柜", "三轮车"]

SENSITIVITY = 8.0      # c: 隶属指数衰减率
PAD = 1e-6             # 凸体退化为点时给一点厚度


def build_space(engine):
    """用园引擎的 16 义素建一个概念空间; 分两域: 市井(0-9) / 叙事(10-15)。"""
    names = list(engine.basis_dict.keys())
    n = len(names)
    dims = {name: i for i, name in enumerate(names)}
    domain_structure = {"市井": list(range(0, 10)), "叙事": list(range(10, n))}
    space.init(n, domain_structure)
    dim_weights = {dom: {d: 1.0 for d in ds} for dom, ds in domain_structure.items()}
    weights = Weights({dom: 1.0 for dom in domain_structure}, dim_weights)
    return names, dims, domain_structure, weights


def vec(engine, names, word):
    v = engine.encode_word(word)
    return [float(x) for x in v]


def cuboid_from_anchors(engine, names, dims, anchors, domain_structure):
    """锚点向量的包围盒 → 凸体。"""
    Vs = [vec(engine, names, w) for w in anchors]
    p_min, p_max = [], []
    for i in range(len(names)):
        lo = min(v[i] for v in Vs)
        hi = max(v[i] for v in Vs)
        if abs(hi - lo) < PAD:
            lo, hi = lo - PAD, hi + PAD
        p_min.append(lo)
        p_max.append(hi)
    return Cuboid(p_min, p_max, domain_structure)


def concept_from_anchors(engine, names, dims, anchors, domain_structure, weights):
    cub = cuboid_from_anchors(engine, names, dims, anchors, domain_structure)
    core = Core([cub], domain_structure)
    return Concept(core, 1.0, SENSITIVITY, weights), cub


def garden_posterior(engine):
    """复刻 observe() 里的贝叶斯前向, 但不改内部状态 (纯函数版)。"""
    priors = {k: engine.concept_spaces[k]["prior_prob"] for k in ("超市", "路边摊")}
    return priors


def main():
    engine = SemeBasedCognitiveEngine()
    names, dims, domain_structure, weights = build_space(engine)

    c_super, cub_super = concept_from_anchors(
        engine, names, dims, SUPERMARKET_ANCHORS, domain_structure, weights)
    c_stall, cub_stall = concept_from_anchors(
        engine, names, dims, STALL_ANCHORS, domain_structure, weights)

    report = {"sensitivity": SENSITIVITY, "anchors": {
        "超市": SUPERMARKET_ANCHORS, "路边摊": STALL_ANCHORS}, "probes": {}}

    print("=" * 78)
    print("任务C 对表: FSSSS(v1.3.2) 模糊隶属  vs  园引擎 余弦似然×贝叶斯")
    print("=" * 78)

    # 概念层关系
    print("\n[概念层] 超市 vs 路边摊 (FSSSS):")
    print(f"  size(超市)   = {c_super.size():.6f}")
    print(f"  size(路边摊) = {c_stall.size():.6f}")
    print(f"  similarity_to(Jaccard) 超市→路边摊 = {c_super.similarity_to(c_stall):.6f}")
    print(f"  subset_of    超市 ⊆ 路边摊 = {c_super.subset_of(c_stall):.6f}")
    print(f"  subset_of    路边摊 ⊆ 超市 = {c_stall.subset_of(c_super):.6f}")

    # 先验 (园引擎: 超市/路边摊 各 1/3, 叙事球共 1/3)
    priors = {k: engine.concept_spaces[k]["prior_prob"] for k in ("超市", "路边摊")}
    print(f"\n[先验] 园引擎 超市={priors['超市']:.4f}  路边摊={priors['路边摊']:.4f}")

    # 逐探针
    print("\n[探针层] 逐词")
    hdr = (f"{'词':<6} | {'FSSSS:m(超市)':>12} {'FSSSS:m(路边摊)':>14} "
           f"{'FSSSS归属':>9} | {'园:cos(超市)':>11} {'园:cos(路边摊)':>12} "
           f"{'园后验':>9} | 一致?")
    print(hdr)
    print("-" * len(hdr))

    agreeable = 0
    for w in PROBE_WORDS:
        x = vec(engine, names, w)
        m_s = c_super.membership_of(x)
        m_t = c_stall.membership_of(x)
        fssss_win = "超市" if m_s >= m_t else "路边摊"

        # 园引擎: 余弦似然 (与 observe 内部一致)
        cs_v = engine.get_concept_vector("超市")
        ct_v = engine.get_concept_vector("路边摊")
        cos_s = float(np.dot(cs_v, x) / (np.linalg.norm(cs_v) * np.linalg.norm(x) + 1e-8))
        cos_t = float(np.dot(ct_v, x) / (np.linalg.norm(ct_v) * np.linalg.norm(x) + 1e-8))
        lk_s, lk_t = max(cos_s, 0.01), max(cos_t, 0.01)
        post_s = lk_s * priors["超市"]; post_t = lk_t * priors["路边摊"]
        tot = post_s + post_t
        post_s, post_t = post_s / tot, post_t / tot
        garden_win = "超市" if post_s >= post_t else "路边摊"

        same = (fssss_win == garden_win)
        agreeable += int(same)
        report["probes"][w] = {
            "fssss_membership": {"超市": m_s, "路边摊": m_t, "winner": fssss_win},
            "garden_cosine": {"超市": cos_s, "路边摊": cos_t},
            "garden_posterior": {"超市": post_s, "路边摊": post_t, "winner": garden_win},
            "agree": same,
        }
        print(f"{w:<6} | {m_s:>12.5f} {m_t:>14.5f} {fssss_win:>9} | "
              f"{cos_s:>11.5f} {cos_t:>12.5f} {post_s:>9.4f} | {'✓' if same else '✗'}")

    print(f"\n排序一致: {agreeable}/{len(PROBE_WORDS)}")

    # 交叉: 锚点自检
    print("\n[锚点自检] 锚点词应归属自己的概念:")
    ok = 0
    total = 0
    for w, label in [(w, "超市") for w in SUPERMARKET_ANCHORS] + \
                    [(w, "路边摊") for w in STALL_ANCHORS]:
        x = vec(engine, names, w)
        m_s = c_super.membership_of(x); m_t = c_stall.membership_of(x)
        win = "超市" if m_s >= m_t else "路边摊"
        total += 1; ok += int(win == label)
        print(f"  {w:<6} 期望={label:<4} 实得={win:<4} m=({m_s:.4f},{m_t:.4f}) {'✓' if win==label else '✗'}")
    print(f"锚点自洽: {ok}/{total}")

    report["concept_level"] = {
        "size": {"超市": c_super.size(), "路边摊": c_stall.size()},
        "similarity_super_stall": c_super.similarity_to(c_stall),
        "subset_super_in_stall": c_super.subset_of(c_stall),
        "subset_stall_in_super": c_stall.subset_of(c_super),
    }
    report["rank_agreement"] = f"{agreeable}/{len(PROBE_WORDS)}"
    report["anchor_selfconsistency"] = f"{ok}/{total}"

    out = os.path.join(os.path.dirname(__file__), "report_c_bridge.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告 → {out}")


if __name__ == "__main__":
    main()
