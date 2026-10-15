#!/usr/bin/env python3
"""冒烟：B0 路径-星图 vs C1 义素超立方体（同模型、同训练循环）。

stage0 形制模型（n_layer=4, d_model=192, n_head=6），训练步数
截短至 SMOKE_STEPS（正式对照夜航跑满 40k）。退出码 = 管线是否
跑通，非门限判据；门限判读在预注册 gates.json（夜航正式跑后）。

产出 report_smoke.json（派生面；实测正本为训练日志）。
"""
import os
os.environ.setdefault("TQDM_DISABLE", "1")

import json, sys, time
from pathlib import Path

REPO = "/Users/mac/Documents/GitHub/llm-geometry-analysis"
sys.path.insert(0, REPO)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from src.config import StageConfig
from src.graph import (build_path_star_graph, build_vocab as build_star_vocab,
                       build_train_test_leaves)
from src.data import build_dataset
from src.model import PathStarTransformer
from src.train import train
from src.metrics import (eval_hardest_token_acc, eval_embedding_auc,
                         eval_alignment_delta, steps_to_pct,
                         compute_stable_rank, compute_eigenspace_gap,
                         compute_principal_angles, _build_graph_node_index)
import seme_adapter as sa
from seme_graph import build_seme_hypercube

SMOKE_STEPS = 3000
EVAL_EVERY = 250
SEED = 42
OUT = Path(__file__).resolve().parent / "report_smoke.json"


def run_condition(name, graph, vocab, train_leaves, test_leaves,
                  max_seq_len, geom):
    cfg = StageConfig(
        stage_name=name, n_layer=4, d_model=192, n_head=6,
        graph_degree=len(graph["paths"]),
        path_length=max(len(p) for p in graph["paths"]) - 1,
        train_leaf_fraction=len(train_leaves) /
            max(len(train_leaves) + len(test_leaves), 1),
        train_steps=SMOKE_STEPS, batch_size=16, grad_accum_steps=2,
        learning_rate=3e-4, weight_decay=0.01, edge_path_mix_ratio=1.0,
        notes="smoke (truncated steps)",
    )
    dataset = build_dataset(graph, vocab, train_leaves, test_leaves,
                            max_seq_len, vocab["pad_id"])
    model = PathStarTransformer(vocab["vocab_size"], cfg.d_model,
                                cfg.n_head, cfg.n_layer, max_seq_len)
    device = torch.device("cpu")
    art = Path("/tmp/seme-smoke") / name
    art.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model, log = train(model, cfg, vocab, graph, dataset, test_leaves,
                       device, art, seed=SEED, eval_every=EVAL_EVERY)
    wall = time.time() - t0

    per_hop = sa.alignment_delta_per_hop(model, vocab, graph)
    metrics = {
        "condition": name,
        "n_paths": len(graph["paths"]),
        "vocab_size": vocab["vocab_size"],
        "n_train_leaves": len(train_leaves),
        "n_test_leaves": len(test_leaves),
        "hardest_token_acc_final": log[-1]["hardest_token_acc"],
        "hardest_token_acc_best": max(e["hardest_token_acc"] for e in log),
        "steps_to_50": steps_to_pct(log, 0.50),
        "steps_to_90": steps_to_pct(log, 0.90),
        "arm_cluster_acc": geom["arm_cluster"](model, graph, vocab, device),
        "alignment_delta_mean": float(np.mean(per_hop)),
        "alignment_per_hop": per_hop,
        "stable_rank": geom["stable_rank"](model, graph, vocab)["stable_rank"],
        "eigenspace_gap": geom["eigenspace_gap"](model, graph, vocab)["gap"],
        "principal_angle_mean_deg":
            geom["principal_angles"](model, graph, vocab)["mean_angle_deg"],
        "pe_norm_final": log[-1]["pe_norm"],
        "wall_sec": wall,
        "train_steps": SMOKE_STEPS,
        "seed": SEED,
    }
    if "hop_level_cluster" in geom:
        metrics["hop_level_cluster_acc"] = \
            geom["hop_level_cluster"](model, graph, vocab)
    return model, metrics


def selftest(model_b0, star_graph, star_vocab):
    """h 包装器 vs 原仓库实现（B0 基线对表）。

    确定性指标容差 1e-6；eigsh 依赖指标的噪声地板 =
    原实现自身两跑之差（ARPACK 随机起始向量，原实现不可
    精确复现），容差 = max(1e-6, 20 x 噪声地板)。
    """
    node_tokens, _, _ = _build_graph_node_index(star_graph, star_vocab)
    dev = torch.device("cpu")
    ok = True

    # 确定性指标（svd / 质心 / 逐 hop 余弦，无随机源）
    det_checks = {
        "stable_rank": (
            compute_stable_rank(model_b0, star_graph, star_vocab)["stable_rank"],
            sa.stable_rank_h(model_b0, node_tokens, star_graph,
                             star_vocab)["stable_rank"]),
        "alignment_delta": (
            eval_alignment_delta(model_b0, star_vocab, star_graph),
            float(np.mean(sa.alignment_delta_per_hop(
                model_b0, star_vocab, star_graph)))),
        "arm_cluster_acc": (
            eval_embedding_auc(model_b0, star_vocab, star_graph, dev),
            sa.arm_cluster_acc_h(model_b0, star_graph, star_vocab, dev)),
    }
    for k, (a, b) in det_checks.items():
        d = abs(a - b)
        status = "OK" if d < 1e-6 else "MISMATCH"
        if d >= 1e-6:
            ok = False
        print(f"  selftest {k:22s} orig={a:.6f} h={b:.6f} "
              f"|diff|={d:.2e} {status}")

    # eigsh 依赖指标：噪声地板 = 原实现自身两跑之差
    eig_checks = {
        "eigenspace_gap": (
            lambda: compute_eigenspace_gap(
                model_b0, star_graph, star_vocab)["gap"],
            lambda: sa.eigenspace_gap_h(
                model_b0, node_tokens, star_graph, star_vocab)["gap"]),
        "principal_angle_mean": (
            lambda: compute_principal_angles(
                model_b0, star_graph, star_vocab)["mean_angle_deg"],
            lambda: sa.principal_angles_h(
                model_b0, node_tokens, star_graph,
                star_vocab)["mean_angle_deg"]),
    }
    for k, (orig_fn, h_fn) in eig_checks.items():
        o1, o2 = orig_fn(), orig_fn()
        floor = abs(o1 - o2)
        h = h_fn()
        d = abs(o1 - h)
        tol = max(1e-6, 20.0 * floor)
        status = "OK" if d <= tol else "MISMATCH"
        if d > tol:
            ok = False
        print(f"  selftest {k:22s} orig={o1:.6f} h={h:.6f} "
              f"|diff|={d:.2e} floor={floor:.2e} tol={tol:.2e} {status}")
    return ok


def main():
    # ---- B0：路径-星图基线（stage0 形制：degree=128, path_length=4）
    star_graph = build_path_star_graph(128, 4)
    star_vocab = build_star_vocab(128, 4)
    train_leaves, test_leaves = build_train_test_leaves(
        star_graph, 0.75, seed=SEED)
    geom_star = {
        "arm_cluster": lambda m, g, v, d: eval_embedding_auc(m, v, g, d),
        "stable_rank": lambda m, g, v: compute_stable_rank(m, g, v),
        "eigenspace_gap": lambda m, g, v: compute_eigenspace_gap(m, g, v),
        "principal_angles": lambda m, g, v: compute_principal_angles(m, g, v),
    }
    print("== B0 路径-星图基线 ==")
    model_b0, m_b0 = run_condition(
        "B0", star_graph, star_vocab, train_leaves, test_leaves,
        4 + 6, geom_star)

    # ---- C1：义素超立方体（D=4, V=4；测试块 = 第 3 维 == 3）
    seme_graph = build_seme_hypercube(n_semes=4, n_values=4)
    orig_graph = sa.to_orig_graph(seme_graph)
    names = sa.seme_node_names(seme_graph)
    seme_vocab = sa.build_vocab(orig_graph, names)
    test_leaves_s = [names[i] for i in range(seme_graph["n_nodes"])
                     if seme_graph["node_semes"][i][3] == 3]
    train_leaves_s = [n for n in names[1:] if n not in set(test_leaves_s)]
    geom_seme = {
        "arm_cluster": lambda m, g, v, d: sa.arm_cluster_acc_h(m, g, v, d),
        "hop_level_cluster": lambda m, g, v: sa.hop_level_cluster_acc(
            m, g, v, seme_graph),
        "stable_rank": lambda m, g, v: sa.stable_rank_h(m, names, g, v),
        "eigenspace_gap": lambda m, g, v: sa.eigenspace_gap_h(m, names, g, v),
        "principal_angles": lambda m, g, v: sa.principal_angles_h(
            m, names, g, v),
    }
    print("== C1 义素超立方体 ==")
    _, m_c1 = run_condition(
        "C1", orig_graph, seme_vocab, train_leaves_s, test_leaves_s,
        4 + 6, geom_seme)

    # ---- selftest：h 包装器在 B0 上对表
    print("== selftest（h 包装器 vs 原实现，B0 基线）==")
    ok = selftest(model_b0, star_graph, star_vocab)

    report = {
        "schema": "seme-hypercube-smoke/1",
        "note": "冒烟：步数截短至 3000（正式夜航跑满）。"
                "退出码只判管线通断，门限判读在 gates.json。",
        "selftest_pass": ok,
        "conditions": {"B0": m_b0, "C1": m_c1},
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    keys = ["hardest_token_acc_final", "hardest_token_acc_best",
            "steps_to_50", "steps_to_90",
            "arm_cluster_acc", "alignment_delta_mean", "stable_rank",
            "eigenspace_gap", "principal_angle_mean_deg", "wall_sec"]
    print("\n== 对照（B0 星图 / C1 超立方体）==")

    def _fmt(x):
        return "-" if x is None else f"{x:.4f}"

    for k in keys:
        print(f"  {k:26s} {_fmt(m_b0[k])} / {_fmt(m_c1[k])}")
    print(f"  hop_level_cluster_acc   {'-':>8} / {m_c1['hop_level_cluster_acc']:.4f}")
    print("  per-hop alignment B0:", [f"{x:.3f}" for x in m_b0["alignment_per_hop"]])
    print("  per-hop alignment C1:", [f"{x:.3f}" for x in m_c1["alignment_per_hop"]])
    print(f"report -> {OUT}")
    print("SMOKE-DONE" if ok else "SMOKE-SELFTEST-FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
