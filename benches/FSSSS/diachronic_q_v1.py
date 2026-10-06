# -*- coding: utf-8 -*-
"""
历时提问测试 v1 — 组合泛化与**边界**（"i-" 族）
=================================================
主人 1006 示：i+Mac ⇒ iPhone/iPad…，但 **iCar 不出现**在概念空间。
问 ① 能否泛化出同域兄弟 ② 能否拒斥越域 iCar ③ 无边界基线是否过生成 iCar。
判据先冻：见 DESIGN_diachronic_q_v1.md（H-v1-1..H-v1-5）。

四系统：S_exp(显式+边界) / S_free(无边界) / S_sync(共时) / S_shuf(去历史)
运行: python3 benches/FSSSS/diachronic_q_v1.py   → report_diachronic_q_v1.json
"""
import os
import json

ROOT = "存在"
PREFIX = "i"
SEED = "Mac"                                   # 种子：i×Mac
TRUE_SIBLINGS = {"iPhone", "iPad", "iPod"}     # 真值泛化（不含种子 iMac）
NEGATIVE = "iCar"                              # 越域反事实负例

TREE = {
    ROOT: [("消费电子域", 0.8), ("载具域", 0.2)],
    "消费电子域": [("Mac", 0.25), ("Phone", 0.25), ("Pad", 0.25), ("Pod", 0.25)],
    "载具域": [("Car", 1.0)],
}


def domain_of(node, tree):
    for p, cs in tree.items():
        for c, _ in cs:
            if c == node:
                return p
    return None


def all_bases(tree):
    out = []
    for p, cs in tree.items():
        if p != ROOT:
            out += [c for c, _ in cs]
    return out


def propagate(tree, root_prior=1.0):
    out = {}

    def walk(n, acc):
        cs = tree.get(n, [])
        if not cs:
            out[n] = acc
            return
        for c, w in cs:
            walk(c, acc * w)

    walk(ROOT, root_prior)
    return out


def combine(base, tree, seed_domain, boundary=True):
    """i×base 准入 ⇔ 域(base)==域(种子)（边界=结构性域树成员，非白名单）。"""
    if boundary and domain_of(base, tree) != seed_domain:
        return None
    return PREFIX + base


def s_exp(tree, boundary=True):
    sd = domain_of(SEED, tree)
    out = set()
    for b in all_bases(tree):
        r = combine(b, tree, sd, boundary=boundary)
        if r:
            out.add(r)
    return out


def s_sync():
    return set()      # 旧字母表 {Mac,Phone,Pad,Pod,Car}——无 i-节点概念


def s_shuf():
    return set()      # A∪B 打乱拟单一分布——无阶段对比


def main():
    print("=" * 88)
    print('历时提问测试 v1 — i×Mac ⇒ iPhone/iPad… 但 iCar 不出现')
    print("=" * 88)
    print("\n域树:")
    for p, cs in TREE.items():
        print(f"  {p} → " + ", ".join(c for c, _ in cs))
    print(f"\n种子: {PREFIX}×{SEED} = iMac（阶段 A 已生）")

    sd = domain_of(SEED, TREE)
    print(f"种子域 = {sd}  ⇒ 边界规则: 域(base)=={sd}")

    exp = s_exp(TREE, boundary=True)      # {iMac,iPhone,iPad,iPod}
    free = s_exp(TREE, boundary=False)    # + iCar
    newly = exp - {"iMac"}                # 阶段 A→B 新增
    syn, shf = s_sync(), s_shuf()

    # 守恒自检
    leaves = propagate(TREE)
    s = sum(leaves.values())
    cons = abs(s - 1.0) < 1e-9

    print(f"\nS_exp  输出: {sorted(exp)}")
    print(f"S_free 输出: {sorted(free)}   ← 无边界 ⇒ 过生成 {NEGATIVE}")
    print(f"S_sync 输出: {sorted(syn)}   S_shuf 输出: {sorted(shf)}")
    print(f"\n阶段 A→B 新出现(S_exp): {sorted(newly)}")
    print(f"守恒自检: Σ叶子先验={s:.12f}  {'✓' if cons else '✗'}")

    gen_recall = len(TRUE_SIBLINGS & newly) / len(TRUE_SIBLINGS)
    boundary_correct = NEGATIVE not in exp
    overgen_free = NEGATIVE in free
    sync_recall = len(TRUE_SIBLINGS & syn) / len(TRUE_SIBLINGS)
    shuf_recall = len(TRUE_SIBLINGS & shf) / len(TRUE_SIBLINGS)

    H1 = gen_recall == 1.0
    H2 = boundary_correct
    H3 = overgen_free
    H4 = (sync_recall == 0.0 and shuf_recall == 0.0)
    H5 = cons

    print("\n计分:")
    print(f"  S_exp  兄弟召回 {gen_recall:.3f} | 拒 iCar {boundary_correct}")
    print(f"  S_free 过生成 iCar {overgen_free}")
    print(f"  S_sync 结构召回 {sync_recall:.3f} | S_shuf 结构召回 {shuf_recall:.3f}")

    print("\n冻结判据:")
    print(f"  H-v1-1 显式可泛化同域兄弟(召回=1): {gen_recall:.3f}  {'✓' if H1 else '✗'}")
    print(f"  H-v1-2 显式拒斥越域 iCar: {boundary_correct}  {'✓' if H2 else '✗'}")
    print(f"  H-v1-3 无边界必过生成 iCar: {overgen_free}  {'✓' if H3 else '✗'}")
    print(f"  H-v1-4 共时/去历史不可答: sync={sync_recall:.3f} shuf={shuf_recall:.3f}  {'✓' if H4 else '✗'}")
    print(f"  H-v1-5 可复算/守恒: {cons}  {'✓' if H5 else '✗'}")

    out = {
        "ask": "from seed iMac: which i-compounds appear? does iCar appear?",
        "seed": "iMac", "seed_domain": sd,
        "ground_truth": {"new": sorted(TRUE_SIBLINGS), "excluded": [NEGATIVE]},
        "systems": {
            "S_exp": {"output": sorted(exp), "new": sorted(newly)},
            "S_free": {"output": sorted(free)},
            "S_sync": {"output": sorted(syn)},
            "S_shuf": {"output": sorted(shf)},
        },
        "scores": {"S_exp": {"generalization_recall": gen_recall, "boundary_correct": boundary_correct},
                   "S_free": {"overgeneration": overgen_free},
                   "S_sync": {"structural_recall": sync_recall},
                   "S_shuf": {"structural_recall": shuf_recall}},
        "criteria": {"H-v1-1": H1, "H-v1-2": H2, "H-v1-3": H3, "H-v1-4": H4, "H-v1-5": H5},
    }
    path = os.path.join(os.path.dirname(__file__), "report_diachronic_q_v1.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n报告 → {path}")


if __name__ == "__main__":
    main()
