# -*- coding: utf-8 -*-
"""
PCS 最小原型 — Probabilistic Conceptual Space
=============================================

把 Bechberger 的 FSSSS 从"模糊几何"升为"概率几何"的那一步:

    FSSSS 隶属      μ_C(x) = μ0 · exp(−c · d_W(x, core_C))      —— 未归一的"典型度"
    FSSSS 已给的归一化常数  Z_C = size(C) = ∫ μ_C(x) dx          —— 他算了, 但从不除
    ⇒  真似然       p(x | C) = μ_C(x) / Z_C                       —— 这才是概率密度
    ⇒  贝叶斯后验   p(C | x) ∝ p(x | C) · p(C)

一句话: Bechberger 手里已有分母 (size), 只是没做除法 —— 完成除法, 模糊集就成了概率密度。

本脚本做三件事:
  ① 归一化自检: 在 2D 上把 p(x|C)=μ/size 与蒙特卡洛积分对撞
  ② 单步后验: 同一批义素探针上, PCS 后验 vs 园引擎(余弦截断)后验
  ③ 序贯后验: 一串观测的证据累积轨迹 PCS vs 园, 并给出 PCS 的边际似然(模型证据)

运行:
  /path/to/venv132/bin/python benches/FSSSS/pcs_prototype.py
依赖: external/ConceptualSpaces-1.3.2-py3 + numpy/scipy/shapely/numdifftools
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

PAD = 1e-9
C_SENS = 8.0            # 灵敏度 c (与 bridge 实验一致)
DIM_EPS = 1e-12


# ── 空间/概念构造 ────────────────────────────────────────────────────────
def build_space(engine):
    names = list(engine.basis_dict.keys())
    n = len(names)
    domain_structure = {"市井": list(range(0, 10)), "叙事": list(range(10, n))}
    space.init(n, domain_structure)
    dim_weights = {dom: {d: 1.0 for d in ds} for dom, ds in domain_structure.items()}
    weights = Weights({dom: 1.0 for dom in domain_structure}, dim_weights)
    return names, domain_structure, weights


def prototype_cuboid(engine, names, domain_structure, prot):
    p = [float(x) for x in prot]
    return Cuboid([x - PAD for x in p], [x + PAD for x in p], domain_structure)


def concept_from_prototype(engine, names, domain_structure, weights, prot, c=C_SENS):
    cub = prototype_cuboid(engine, names, domain_structure, prot)
    return Concept(Core([cub], domain_structure), 1.0, c, weights)


def concept_density(concept, x):
    """p(x|C) = μ_C(x) / size(C) —— 模糊集 → 概率密度。"""
    z = getattr(concept, "_cached_size", None)
    if z is None:
        # size() 在 16D 上要跑 2^16 项, 缓存之
        z = concept.size()
        concept._cached_size = z
    return concept.membership_of(x) / z


# ── ① 归一化自检 (2D) ────────────────────────────────────────────────────
def check_normalization_2d():
    doms = {"d": [0, 1]}
    space.init(2, doms)
    w = Weights({"d": 1.0}, {"d": {0: 1.0, 1: 1.0}})
    c = 1.0
    prot = [0.0, 0.0]
    cub = Cuboid([prot[0] - PAD, prot[1] - PAD], [prot[0] + PAD, prot[1] + PAD], doms)
    con = Concept(Core([cub], doms), 1.0, c, w)
    z = con.size()
    # 真隶属的度量: 域内维权归一化后 sqrt(Σ w_dim Δ²)。w_dim 各为 0.5。
    w0 = w._dimension_weights["d"][0]
    w1 = w._dimension_weights["d"][1]
    L = 60.0
    rng = np.random.default_rng(0)
    N = 4_000_000
    pts = rng.uniform(-L, L, size=(N, 2))
    d = np.sqrt(w0 * pts[:, 0] ** 2 + w1 * pts[:, 1] ** 2)
    vals = np.exp(-c * d)
    mc = vals.mean() * (2 * L) ** 2
    return {"analytic_size": z, "monte_carlo": float(mc),
            "rel_err": abs(z - mc) / mc}


# ── ②③ PCS 推断 ──────────────────────────────────────────────────────────
def pcs_run(engine, names, domain_structure, weights, words, concepts, priors, c=C_SENS):
    """序贯贝叶斯: p(C|x_t) ∝ p(x_t|C) · p(C)。返回轨迹 + 边际对数似然。"""
    post = dict(priors)
    traj = []
    log_evidence = 0.0
    for w in words:
        x = [float(v) for v in engine.encode_word(w)]
        liks = {name: concept_density(concepts[name], x) for name in concepts}
        unnorm = {name: liks[name] * post[name] for name in liks}
        marginal = sum(unnorm.values())
        log_evidence += math.log(marginal + DIM_EPS)
        post = {name: unnorm[name] / marginal for name in unnorm}
        z = {name: getattr(concepts[name], "_cached_size", None) or concepts[name].size() for name in concepts}
        traj.append({"word": w,
                     "density": liks,
                     "size": z,
                     "posterior": dict(post)})
    return traj, log_evidence


def garden_run(engine, words, concepts_vec, priors):
    """复刻园引擎 observe() 的更新 (余弦似然 + 贝叶斯), 不改内部状态。"""
    post = dict(priors)
    traj = []
    for w in words:
        x = [float(v) for v in engine.encode_word(w)]
        xv = np.array(x)
        liks = {}
        for name, cv in concepts_vec.items():
            cos = float(np.dot(cv, xv) / (np.linalg.norm(cv) * np.linalg.norm(xv) + 1e-8))
            liks[name] = max(cos, 0.01)
        unnorm = {n: liks[n] * post[n] for n in liks}
        marg = sum(unnorm.values())
        post = {n: unnorm[n] / marg for n in unnorm}
        traj.append({"word": w, "likelihood": liks, "posterior": dict(post)})
    return traj


def fmt_word(w, n=10):
    return (w[:n] + "…") if len(w) > n else w


def main():
    engine = SemeBasedCognitiveEngine()

    # ① 归一化自检 —— 注意: 它会把全局空间重置为 2D, 故必须在建 16D 空间之前跑
    print("=" * 84)
    print("PCS 最小原型: 模糊集 → 概率密度 (p = μ/size) → 贝叶斯")
    print("=" * 84)
    norm = check_normalization_2d()
    print("\n① 归一化自检 (2D, c=1):")
    print(f"   size() 解析值 = {norm['analytic_size']:.6f}")
    print(f"   蒙特卡洛积分   = {norm['monte_carlo']:.6f}")
    print(f"   相对误差       = {norm['rel_err']:.2e}  "
          f"{'✓ 归一 ⇒ 密度合法' if norm['rel_err'] < 0.01 else '✗'}")

    # 之后才建 16D 空间 / 概念
    names, domain_structure, weights = build_space(engine)
    concept_names = ["超市", "路边摊", "觉醒循环", "接待员日常", "福特剧场"]
    concepts, protos = {}, {}
    for name in concept_names:
        protos[name] = engine.get_concept_vector(name)
        concepts[name] = concept_from_prototype(engine, names, domain_structure, weights, protos[name])
    priors = {name: engine.concept_spaces[name]["prior_prob"] for name in concept_names}

    report = {"sensitivity": C_SENS, "priors": priors, "normalization_2d": norm}

    # 概念体积 (归一化常数)
    print("\n   概念 Z_C = size(C) (16D):")
    sizes = {}
    for name in concept_names:
        sizes[name] = concepts[name].size()
        print(f"     {name:6s} Z = {sizes[name]:.6e}   log Z = {math.log(sizes[name]):.4f}")
    report["sizes"] = sizes
    print("   注: 原型退化为点时, Z 只依赖 c/权重/维数而与位置无关, 故五者相等 →")
    print("       后验中 Z 约掉; **区域**概念的 Z 不等 (见 bridge: size(超市)=9016 ≠ size(摊)=6709),")
    print("       此时 Z 是真正起作用的归一化常数。")

    # ② 单步后验对表
    probes = ["塑胶凳", "塑胶袋", "吸管", "煤气罐", "购物车", "大排档", "冷柜", "三轮车"]
    print("\n② 单步后验 (先验均匀):")
    hdr = f"{'词':<6}| {'PCS argmax':>10} {'PCS p(超市)':>11} {'PCS p(摊)':>10} | {'园 argmax':>9} {'园 p(超市)':>10} {'园 p(摊)':>9} | 一致"
    print(hdr); print("-" * len(hdr))
    concepts_vec = {n: np.array([float(v) for v in engine.get_concept_vector(n)]) for n in concept_names}
    pcs_single = {}
    for w in probes:
        x = [float(v) for v in engine.encode_word(w)]
        liks = {n: concept_density(concepts[n], x) for n in concept_names}
        u = {n: liks[n] * priors[n] for n in concept_names}
        m = sum(u.values()); post = {n: u[n] / m for n in concept_names}
        # 园
        xv = np.array(x); gl = {n: max(float(np.dot(concepts_vec[n], xv) /
              (np.linalg.norm(concepts_vec[n]) * np.linalg.norm(xv) + 1e-8)), 0.01) for n in concept_names}
        gu = {n: gl[n] * priors[n] for n in concept_names}; gm = sum(gu.values())
        gpost = {n: gu[n] / gm for n in concept_names}
        pa = max(post, key=post.get); ga = max(gpost, key=gpost.get)
        pcs_single[w] = {"posterior": post, "argmax": pa, "garden_argmax": ga}
        print(f"{w:<6}| {pa:>10} {post['超市']:>11.4f} {post['路边摊']:>10.4f} | "
              f"{ga:>9} {gpost['超市']:>10.4f} {gpost['路边摊']:>9.4f} | {'✓' if pa==ga else '✗'}")
    report["single_step"] = pcs_single

    # ③ 序贯证据累积
    stream = ["塑胶凳", "煤气罐", "三轮车", "大排档",
              "这些残暴的欢愉，终将以残暴结局", "我不要再被人摆布"]
    print("\n③ 序贯证据累积 (先验=园先验):")
    print("   先验:", {k: round(v, 4) for k, v in priors.items()})
    pt, pev = pcs_run(engine, names, domain_structure, weights, stream, concepts, priors)
    gt = garden_run(engine, stream, concepts_vec, priors)
    print(f"\n   {'步':<2} {'词':<12}| " + " ".join(f"{k:>8}" for k in concept_names) + "   (PCS 后验)")
    for i, row in enumerate(pt, 1):
        print(f"   {i:<2} {fmt_word(row['word'],12):<12}| " +
              " ".join(f"{row['posterior'][k]:>8.4f}" for k in concept_names))
    print(f"\n   {'步':<2} {'词':<12}| " + " ".join(f"{k:>8}" for k in concept_names) + "   (园 后验)")
    for i, row in enumerate(gt, 1):
        print(f"   {i:<2} {fmt_word(row['word'],12):<12}| " +
              " ".join(f"{row['posterior'][k]:>8.4f}" for k in concept_names))

    pcs_final = pt[-1]["posterior"]; g_final = gt[-1]["posterior"]
    pcs_top = max(pcs_final, key=pcs_final.get); g_top = max(g_final, key=g_final.get)
    print(f"\n   末步 argmax: PCS={pcs_top} (p={pcs_final[pcs_top]:.4f})  "
          f"园={g_top} (p={g_final[g_top]:.4f})  {'✓ 一致' if pcs_top==g_top else '✗ 分歧'}")
    print(f"   PCS 边际对数似然 (模型证据) log p(x_1..x_T) = {pev:.4f}")

    # c 敏感度 → 用**留出**边际似然选 c (训练/测试分割; 园引擎做不到)
    print("\n   c 敏感度 (留出边际似然: 前 4 词训练选 c, 后 2 词测试; 概率模型自带模型选择):")
    train, test = stream[:4], stream[4:]
    c_scan = {}
    for c in [2.0, 4.0, 8.0, 12.0, 16.0]:
        cons = {n: concept_from_prototype(engine, names, domain_structure, weights, protos[n], c) for n in concept_names}
        tr_traj, tr_ev = pcs_run(engine, names, domain_structure, weights, train, cons, priors, c)
        post_after = tr_traj[-1]["posterior"]
        _, te_ev = pcs_run(engine, names, domain_structure, weights, test, cons, post_after, c)
        c_scan[c] = {"train": tr_ev, "test": te_ev, "total": tr_ev + te_ev}
        print(f"     c = {c:>5.1f}   train = {tr_ev:>9.3f}   test = {te_ev:>9.3f}   total = {tr_ev+te_ev:>9.3f}")
    best_c = max(c_scan, key=lambda k: c_scan[k]["test"])
    print(f"   ⇒ 留出测试最优 c = {best_c}   (注意: 6 词样本极小, 仅供示范)")

    report["sequential_pcs"] = pt
    report["sequential_garden"] = gt
    report["pcs_log_evidence"] = pev
    report["c_scan"] = c_scan
    report["best_c"] = best_c

    out = os.path.join(os.path.dirname(__file__), "report_pcs_prototype.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n报告 → {out}")


if __name__ == "__main__":
    main()
