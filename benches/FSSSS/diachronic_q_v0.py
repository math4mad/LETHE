# -*- coding: utf-8 -*-
"""
历时提问测试 v0（Diachronic Question Test, v0）
=================================================
问：系统能否答出「概念 C 在阶段 A→B 变了什么」，并给**结构级**答案
    （加列／开新节点／子树生长），而非仅共时频率漂移？
判据先冻：见 DESIGN_diachronic_q_v0.md（H-t1..H-t4）。

三系统（理想化，为可复算；非评测真 LLM）：
  S_exp  园显式 PCS（树＋split）      → 结构事件 ＋ 权重事件
  S_sync 共时频率基线（旧字母表）      → 仅频率漂移；无"新节点"概念
  S_shuf 去历史基线（A∪B 打乱拟单一分布）→ 报「无差异」

运行: python3 benches/FSSSS/diachronic_q_v0.py     → report_diachronic_q_v0.json
"""
import os
import json

import numpy as np

ROOT = "存在"
# 义素维 [甜,红,圆,脆,山,城,果]
VEC = {
    "苹果":     [0.8, 0.7, 0.8, 0.5, 0.0, 0.0, 1.0],
    "富士山":   [0.0, 0.0, 0.0, 0.0, 0.9, 0.2, 0.0],
    "富士市":   [0.0, 0.0, 0.0, 0.0, 0.3, 0.9, 0.0],
    "富士苹果": [0.75, 0.65, 0.75, 0.5, 0.3, 0.25, 0.9],
    "红":       [0.1, 1.0, 0.2, 0.0, 0.0, 0.0, 0.1],
    "圆":       [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.1],
    "甜":       [1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.4],
}
TREE_A = {
    ROOT: [("果域", 0.5), ("地景域", 0.5)],
    "果域": [("苹果", 1.0)],
    "地景域": [("富士山", 0.6), ("富士市", 0.4)],
}
TREE_B = {
    ROOT: [("果域", 0.5), ("地景域", 0.5)],
    "果域": [("苹果", 0.75), ("富士苹果", 0.25)],
    "地景域": [("富士山", 0.6), ("富士市", 0.4)],
    "富士苹果": [("红", 0.5), ("圆", 0.3), ("甜", 0.2)],
}


def v(name):
    return np.array(VEC[name], dtype=float)


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def nodes(tree):
    return set(tree.keys()) | {c for cs in tree.values() for c, _ in cs}


def edges(tree):
    return {(p, c) for p, cs in tree.items() for c, _ in cs}


def outdeg(tree, node):
    return len(tree.get(node, []))


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


# ── 真值（由构造给出，冻） ──────────────────────────────────────────────
GT_NODES = {"富士苹果", "红", "圆", "甜"}
GT_EDGES = {("果域", "富士苹果"), ("富士苹果", "红"), ("富士苹果", "圆"), ("富士苹果", "甜")}
ALPHA_A = ["苹果", "富士山", "富士市"]          # 阶段 A 的旧字母表


def ground_truth():
    pa, pb = propagate(TREE_A), propagate(TREE_B)
    weight = {l: (pa.get(l, 0.0), pb.get(l, 0.0)) for l in pa if l in pb and abs(pa[l] - pb[l]) > 1e-9}
    sa, sb = dict(TREE_A["果域"]), dict(TREE_B["果域"])
    return {
        "struct_nodes": sorted(GT_NODES),
        "struct_edges": sorted(list(e) for e in GT_EDGES),
        "outdeg_delta": {"果域": [outdeg(TREE_A, "果域"), outdeg(TREE_B, "果域")]},
        "weight": {k: list(x) for k, x in weight.items()},
        "split": {"果域.苹果": [sa["苹果"], sb["苹果"]]},
    }


# ── S_exp：显式 PCS ─────────────────────────────────────────────────────
def s_exp():
    na, nb = nodes(TREE_A), nodes(TREE_B)
    ea, eb = edges(TREE_A), edges(TREE_B)
    pa, pb = propagate(TREE_A), propagate(TREE_B)
    ev = {("+node", n) for n in (nb - na)} | {("+edge",) + e for e in (eb - ea)}
    weight = {l: [pa[l], pb[l]] for l in pa if l in pb and abs(pa[l] - pb[l]) > 1e-9}
    return {"events": ev, "weight": weight,
            "outdeg": {p: [outdeg(TREE_A, p), outdeg(TREE_B, p)] for p in set(TREE_A) | set(TREE_B)}}


# ── S_sync：共时频率基线（旧字母表） ────────────────────────────────────
STREAM_A = ["苹果", "富士山", "富士市", "富士山", "苹果", "富士市"]
STREAM_B = ["富士苹果", "红", "圆", "甜", "富士苹果", "苹果"]


def hist(stream, alpha):
    c = {a: 0 for a in alpha}
    for w in stream:
        c[max(alpha, key=lambda a: cos(v(w), v(a)))] += 1
    n = sum(c.values()) or 1
    return {a: c[a] / n for a in alpha}


def s_sync():
    ha, hb = hist(STREAM_A, ALPHA_A), hist(STREAM_B, ALPHA_A)
    weight = {a: [ha[a], hb[a]] for a in ALPHA_A if abs(ha[a] - hb[a]) > 1e-9}
    # 强命「新东西叫什么」→ 只能在旧字母表里挑最像者（别名，非新节点）
    alias = max(ALPHA_A, key=lambda a: cos(v("富士苹果"), v(a)))
    return {"events": set(), "weight": weight, "alias_new_as": alias}


# ── S_shuf：去历史基线（A∪B 打乱，拟单一分布） ──────────────────────────
def s_shuf():
    # 打乱后只剩一个分布，无 A/B 对比 ⇒ 报「无差异」
    return {"events": set(), "weight": {}, "no_change": True}


# ── 计分 ────────────────────────────────────────────────────────────────
def struct_recall(events):
    gt = {("+node", n) for n in GT_NODES} | {("+edge",) + e for e in GT_EDGES}
    return len(gt & set(events)) / len(gt)


def false_new(events):
    return sum(1 for e in events if e[0] == "+node" and e[1] not in GT_NODES)


def main():
    print("=" * 88)
    print("历时提问测试 v0 — 富士：阶段 A（无富士苹果）→ 阶段 B（有）")
    print("=" * 88)

    gt = ground_truth()
    print("\n真值变更集（构造，冻）:")
    print("  结构 +节点:", gt["struct_nodes"])
    print("  结构 +边  :", gt["struct_edges"])
    print("  出度变化  :", gt["outdeg_delta"], " 叶先验漂移:", gt["weight"])

    exp, syn, shf = s_exp(), s_sync(), s_shuf()

    # 守恒自检
    sa, sb = sum(propagate(TREE_A).values()), sum(propagate(TREE_B).values())
    cons = abs(sa - 1) < 1e-9 and abs(sb - 1) < 1e-9
    print(f"\n守恒自检: Σ_A={sa:.12f}  Σ_B={sb:.12f}  {'✓' if cons else '✗'}")

    rows = []
    for name, sysd in (("S_exp", exp), ("S_sync", syn), ("S_shuf", shf)):
        rec = struct_recall(sysd["events"])
        wdr = len(sysd["weight"]) > 0
        fn = false_new(sysd["events"])
        nc = bool(sysd.get("no_change", False))
        rows.append((name, rec, wdr, fn, nc))

    print("\n提问 Q(C) = 「C 在 A→B 变了什么？」——各系统答案:") 
    print(f"  S_exp : 结构事件 {sorted(exp['events'])}")
    print(f"  S_sync: 结构事件 {sorted(syn['events'])} （强命新物→别名 '{syn['alias_new_as']}'）")
    print(f"  S_shuf: 无差异 = {shf['no_change']}")

    print("\n计分:")
    print(f"  {'系统':8s} {'结构召回':>8s} {'权重漂移':>8s} {'幻觉新节点':>10s} {'报无差异':>8s}")
    for name, rec, wdr, fn, nc in rows:
        print(f"  {name:8s} {rec:8.3f} {str(wdr):>8s} {fn:10d} {str(nc):>8s}")

    exp_rec = struct_recall(exp["events"])
    syn_rec = struct_recall(syn["events"])
    H_t1 = exp_rec >= 1.0
    H_t2 = syn_rec == 0.0
    H_t3 = shf["no_change"] and struct_recall(shf["events"]) == 0.0
    H_t4 = cons
    print("\n冻结判据:")
    print(f"  H-t1 显式PCS可答结构变化: 结构召回={exp_rec:.3f}  {'✓' if H_t1 else '✗'}")
    print(f"  H-t2 共时基线不能命名新节点: 结构召回={syn_rec:.3f}  {'✓' if H_t2 else '✗'}")
    print(f"  H-t3 去历史基线报无差异: {shf['no_change']}  {'✓' if H_t3 else '✗'}")
    print(f"  H-t4 可复算/守恒: {cons}  {'✓' if H_t4 else '✗'}")

    out = {
        "ask": "Q(C): what changed from stage A to stage B?",
        "ground_truth": gt,
        "systems": {
            "S_exp": {"struct_events": sorted(list(e) for e in exp["events"]),
                      "weight": exp["weight"], "outdeg": exp["outdeg"]},
            "S_sync": {"struct_events": sorted(list(e) for e in syn["events"]),
                       "weight": syn["weight"], "alias_new_as": syn["alias_new_as"]},
            "S_shuf": {"struct_events": [], "no_change": True},
        },
        "scores": {n: {"structural_recall": r, "weight_drift_recovered": w,
                       "false_new": f, "no_change_reported": c}
                   for n, r, w, f, c in rows},
        "criteria": {"H-t1": H_t1, "H-t2": H_t2, "H-t3": H_t3, "H-t4": H_t4},
    }
    path = os.path.join(os.path.dirname(__file__), "report_diachronic_q_v0.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n报告 → {path}")


if __name__ == "__main__":
    main()
