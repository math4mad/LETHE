#!/usr/bin/env python3
"""verify — Geometry of Meaning 7B Kaggle 臂回收对表 (三样交付律②)

用法:
  python3 verify_geometry_kaggle.py <kaggle-output-dir>
  (目录须含 metrics_geometry_kaggle.json; 可选 log.txt 哨兵检查)

判据:
  V-1  文件齐: metrics_geometry_kaggle.json + figs_geometry_kaggle/ 四件以上
  V-2  语料对撞: metrics 内 corpus.sha256_16 == 正本 corpus_paraphrase.json 前 16
  V-3  规模: n_sentences=200, n_concepts=50, layer_indices 含末层
  V-4  复算: 三策略 verdict 的 G1/G3 布尔与阈值复算一致
  V-5  哨兵: 若给 log.txt, 须含 GEOM_DONE 且无 GEOM_ERROR

退出码: 0 = 全过; 1 = 有败 (每条打印 PASS/FAIL)
"""
import gzip
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_SRC = os.path.expanduser("~/CodeBuddy/Geometry-Of-Meaning/corpus_paraphrase.json")
G1_THRESHOLD = 0.75
G3_THRESHOLD = 0.85
STRATS = ["last_token", "mean_pool", "entity"]


def main(out_dir):
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not cond:
            ok = False

    mpath = os.path.join(out_dir, "metrics_geometry_kaggle.json")
    check("V-1a metrics_geometry_kaggle.json 存在", os.path.isfile(mpath))
    if not os.path.isfile(mpath):
        print("verify FAIL: 无 metrics 文件")
        return 1
    m = json.load(open(mpath, encoding="utf-8"))

    figs = os.path.join(out_dir, "figs_geometry_kaggle")
    have_figs = (os.path.isdir(figs) and len(os.listdir(figs)) >= 4)
    check("V-1b figs_geometry_kaggle/ >= 4 件", have_figs,
          f"{len(os.listdir(figs)) if os.path.isdir(figs) else 0} 件")

    corpus_sha = hashlib.sha256(open(CORPUS_SRC, "rb").read()).hexdigest()
    got = m.get("corpus", {}).get("sha256_16", "")
    check("V-2 语料 sha256 对撞正本", got == corpus_sha[:16],
          f"臂 {got} vs 正本 {corpus_sha[:16]}")

    n_s = m.get("corpus", {}).get("n_sentences")
    n_c = m.get("corpus", {}).get("n_concepts")
    li = m.get("layer_indices", [])
    check("V-3a 规模 200 句 / 50 概念", n_s == 200 and n_c == 50,
          f"{n_s} 句 / {n_c} 概念")
    check("V-3b 层采样含末层 (28)", len(li) >= 10 and li[-1] == 28,
          f"{len(li)} 层, 末层 {li[-1] if li else None}")

    for strat in STRATS:
        v = m.get("strategies", {}).get(strat, {}).get("verdict", {})
        if not v:
            check(f"V-4 {strat} verdict 在", False)
            continue
        g1_calc = v["last3_pos_mean"] > G1_THRESHOLD and v["last3_pos_mean"] > v["last3_neg_mean"]
        g3_calc = v["G3_cka"] > G3_THRESHOLD
        check(f"V-4 {strat} G1 复算一致",
              bool(v["G1_pass"]) == g1_calc,
              f"pos={v['last3_pos_mean']:.4f} neg={v['last3_neg_mean']:.4f} "
              f"margin={v['last3_margin']:.4f} -> G1={'PASS' if g1_calc else 'FAIL'}")
        check(f"V-4 {strat} G3 复算一致",
              bool(v["G3_pass"]) == g3_calc,
              f"cka={v['G3_cka']:.4f} -> G3={'PASS' if g3_calc else 'FAIL'}")

    g2 = m.get("g2_aux", {})
    for strat in ["last_token", "mean_pool"]:
        if strat in g2:
            r = g2[strat]["intra_over_inter"]
            print(f"  [参考] G-2 辅助读数 {strat}: intra/inter={r:.3f}"
                  + (" (<1 同概念更紧)" if r < 1 else ""))

    logp = os.path.join(out_dir, "log.txt")
    if os.path.isfile(logp):
        log = open(logp, encoding="utf-8", errors="replace").read()
        check("V-5 哨兵 GEOM_DONE 在且无 GEOM_ERROR",
              "GEOM_DONE::" in log and "GEOM_ERROR::" not in log)
    else:
        print("  [跳过] V-5 无 log.txt (哨兵检查跳过)")

    print()
    if m.get("strategies"):
        print("判据裁决汇总 (末 3 层):")
        for strat in STRATS:
            v = m.get("strategies", {}).get(strat, {}).get("verdict", {})
            if v:
                print(f"  {strat:10s} pos={v['last3_pos_mean']:.4f} "
                      f"neg={v['last3_neg_mean']:.4f} margin={v['last3_margin']:+.4f} "
                      f"cka={v['G3_cka']:.4f}  G1={'PASS' if v['G1_pass'] else 'FAIL'} "
                      f"G3={'PASS' if v['G3_pass'] else 'FAIL'}")

    print()
    print("verify:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
