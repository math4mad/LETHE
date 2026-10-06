# -*- coding: utf-8 -*-
"""
自变序列 × CRP 合流 — 变化序列的可审计记录 + 历时提问
========================================================
承 hierarchical CRP（nCRP）。把每次插入产生的事件（开列/归入/下行）记为**变化序列**
（＝《再认识》「概念空间自身携带的变化序列」的可审计记录），并据此回答
「概念在 t1→t2 变了什么」——把 v0/v1/v2 历时提问测试，改用**本记录**作答。

判据先冻（见 DESIGN_evolution_crp.md）:
  H-e1 序列可还原：仅凭 open 事件回放，可重建终态节点/边集（与实树完全一致）
  H-e2 历时可答：what_changed(t1,t2) 报出的新节点/边 = 真值 diff
  H-e3 守恒：叶先验 Σ=1

运行: python3 benches/FSSSS/pcs_evolution.py   → report_evolution_crp.json
"""
import os
import sys
import json
import math

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pcs_hier_crp_v1 as b  # Node / cos / v / VEC / STREAM / ROOT / BETA / TAU_MATCH / leaves

ALPHA = 1.0


def grow(stream, alpha=ALPHA):
    root = b.Node(b.ROOT, None, None)
    events = []
    snaps = []
    for step, name in enumerate(stream):
        proto = b.v(name)
        v, vpath = root, b.ROOT
        while True:
            cs = v.children
            if not cs:
                c = v.add(name, proto)
                c.n += 1
                cpath = vpath + ">" + c.name
                events.append({"step": step, "obs": name, "kind": "open",
                               "parent": vpath, "child": cpath})
                break
            N = sum(c.n for c in cs)
            fits = [b.cos(proto, c.proto) for c in cs]
            ws = [(c.n / (N + alpha)) * math.exp(b.BETA * f) for c, f in zip(cs, fits)]
            novelty = 1.0 - max(fits)
            w_new = (alpha / (N + alpha)) * math.exp(b.BETA * novelty)
            if w_new >= max(ws):
                c = v.add(name, proto)
                c.n += 1
                cpath = vpath + ">" + c.name
                events.append({"step": step, "obs": name, "kind": "open",
                               "parent": vpath, "child": cpath})
                break
            k = int(np.argmax(ws))
            if fits[k] >= b.TAU_MATCH:
                cs[k].n += 1
                events.append({"step": step, "obs": name, "kind": "assign",
                               "parent": vpath, "child": vpath + ">" + cs[k].name})
                break
            cs[k].n += 1
            v = cs[k]
            vpath = vpath + ">" + cs[k].name
            events.append({"step": step, "obs": name, "kind": "descend",
                           "parent": None, "child": vpath})
        snaps.append(snapshot(root))
    return root, events, snaps


def snapshot(root):
    nodes, edges = set(), set()

    def walk(node, path):
        nodes.add(path)
        for c in node.children:
            cp = path + ">" + c.name
            edges.add((path, cp))
            walk(c, cp)
    walk(root, b.ROOT)
    return {"nodes": nodes, "edges": edges}


def replay(events):
    """仅凭 open 事件回放重建节点/边集。"""
    nodes, edges = {b.ROOT}, set()
    for e in events:
        if e["kind"] == "open":
            nodes.add(e["child"])
            edges.add((e["parent"], e["child"]))
    return {"nodes": nodes, "edges": edges}


def what_changed(sa, sb):
    return {"new_nodes": sorted(sb["nodes"] - sa["nodes"]),
            "new_edges": sorted([list(x) for x in (sb["edges"] - sa["edges"])]),
            "removed_nodes": sorted(sa["nodes"] - sb["nodes"])}


def main():
    print("=" * 88)
    print("自变序列 × CRP 合流 — 变化序列可审计 + 历时提问")
    print("=" * 88)
    root, events, snaps = grow(b.STREAM, ALPHA)
    final = snapshot(root)
    opened = [e for e in events if e["kind"] == "open"]
    print(f"\n观测流 {' '.join(b.STREAM)}  (α={ALPHA})")
    print(f"事件总数 {len(events)}：open {len(opened)} / "
          f"assign {sum(e['kind']=='assign' for e in events)} / "
          f"descend {sum(e['kind']=='descend' for e in events)}")
    print("\n变化序列（open 事件，即结构级开列）:")
    for e in opened:
        print(f"   step{e['step']:>2} [{e['obs']:>3}] open {e['parent']} > {e['child'].split('>')[-1]}")

    # H-e1 回放还原
    rec = replay(events)
    H_e1 = (rec["nodes"] == final["nodes"]) and (rec["edges"] == final["edges"])

    # H-e2 历时：step2 快照 → 终态
    t1, t2 = 2, len(snaps) - 1
    chg = what_changed(snaps[t1], snaps[t2])
    # 真值：由 events 在 (t1,t2] 区间的 open 事件取
    gt_nodes = sorted({e["child"] for e in opened if t1 < e["step"] <= t2})
    gt_edges = sorted([e["parent"], e["child"]] for e in opened if t1 < e["step"] <= t2)
    H_e2 = (chg["new_nodes"] == gt_nodes) and (chg["new_edges"] == gt_edges)

    lv = b.leaves(root)
    H_e3 = abs(sum(lv.values()) - 1) < 1e-9

    print(f"\nH-e1 回放还原（仅 open 事件）: 节点{len(rec['nodes'])}/{len(final['nodes'])} "
          f"边{len(rec['edges'])}/{len(final['edges'])}  {'✓' if H_e1 else '✗'}")
    print(f"H-e2 历时提问 t1=step{t1}→t2=step{t2}:")
    print(f"   报出新节点 {chg['new_nodes']}")
    print(f"   真值新节点  {gt_nodes}   {'✓' if H_e2 else '✗'}")
    print(f"H-e3 叶先验 Σ={sum(lv.values()):.12f}  {'✓' if H_e3 else '✗'}")

    out = {"alpha": ALPHA, "stream": b.STREAM,
           "events": events, "snap_nodes_count": [len(s["nodes"]) for s in snaps],
           "obtained": {"new_nodes": chg["new_nodes"], "new_edges": chg["new_edges"]},
           "ground_truth": {"new_nodes": gt_nodes, "new_edges": gt_edges},
           "criteria": {"H-e1": H_e1, "H-e2": H_e2, "H-e3": H_e3}}
    path = os.path.join(HERE, "report_evolution_crp.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n报告 → {path}")


if __name__ == "__main__":
    main()
