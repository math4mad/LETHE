#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kaggle 臂产出 vs 本地正本 对账 (退出码即判据, 三样交付律之 verify 命令)。

用法:
  python3 verify_friends_kaggle.py <kaggle_metrics.json>
    默认对表本地 metrics_friends_s1234.json, 容差: 余弦类 1e-2 / 探针 0.02 / 流计数逐位
    (PREREG.md 冻结)

退出码:
  0 = 全部在容差内 (K-1 通过: 平台迁移成立)
  1 = 有超差项 (打印明细)
  2 = 文件/结构错误
"""
import json, os, sys

TOL_COS = 1e-2
TOL_PROBE = 0.02

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL = os.path.join(HERE, "..", "metrics_friends_s1234.json")


def check(name, kg, lc, tol, fails):
    if kg is None or lc is None:
        fails.append(f"{name}: None value (kg={kg} lc={lc})")
        return
    d = abs(float(kg) - float(lc))
    flag = "OK " if d <= tol else "FAIL"
    print(f"  [{flag}] {name}: kaggle={kg:.6f} local={lc:.6f} |Δ|={d:.2e} (tol {tol:g})")
    if d > tol:
        fails.append(f"{name}: |Δ|={d:.2e} > tol {tol:g}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    kg = json.load(open(sys.argv[1]))
    lc = json.load(open(LOCAL))
    fails = []

    print("== Rachel timeline (tol cos 1e-2 / probe 0.02) ==")
    for i, s in enumerate(lc["rachel_timeline"]["seasons"]):
        tag = f"S{s:02d}"
        check(f"{tag} rachel_within_last",
              kg["rachel_timeline"]["rachel_within_last"][i],
              lc["rachel_timeline"]["rachel_within_last"][i], TOL_COS, fails)
        check(f"{tag} others_within_mean",
              kg["rachel_timeline"]["others_within_mean"][i],
              lc["rachel_timeline"]["others_within_mean"][i], TOL_COS, fails)
        check(f"{tag} rachel_centroid_dist",
              kg["rachel_timeline"]["rachel_centroid_dist"][i],
              lc["rachel_timeline"]["rachel_centroid_dist"][i], TOL_COS, fails)
        check(f"{tag} probe_lead_vs_rest",
              kg["rachel_timeline"]["probe_lead_vs_rest"][i],
              lc["rachel_timeline"]["probe_lead_vs_rest"][i], TOL_PROBE, fails)

    print("== per-season within_last (all roles, tol 1e-2) ==")
    for s, d in lc["season_metrics"].items():
        for r, v in d["within_last"].items():
            check(f"S{s} {r} within_last",
                  kg["season_metrics"][s]["within_last"][r], v, TOL_COS, fails)

    print("== flow (tol 1e-2; counts exact) ==")
    for k in ("adj_same_role_sim", "adj_diff_role_sim"):
        check(f"flow {k}", kg["flow_stats"][k], lc["flow_stats"][k], TOL_COS, fails)
    for k in ("n_sentences", "n_tokens", "n_adj_same", "n_adj_diff"):
        a, b = kg["flow_stats"][k], lc["flow_stats"][k]
        ok = a == b
        print(f"  [{'OK ' if ok else 'FAIL'}] flow {k}: kaggle={a} local={b} (exact)")
        if not ok:
            fails.append(f"flow {k}: {a} != {b}")

    print()
    if fails:
        print(f"VERIFY_FAIL ({len(fails)} items over tolerance):")
        for f_ in fails:
            print("  -", f_)
        sys.exit(1)
    print("VERIFY_OK — all metrics within frozen tolerances (K-1 passed)")
    sys.exit(0)


if __name__ == "__main__":
    main()
