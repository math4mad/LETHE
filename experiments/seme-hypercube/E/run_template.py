#!/usr/bin/env python3
"""义素超立方体夜航臂 —— CONDITION=E（α 插值扫描，E0..E6 × 3 seeds）。

预注册：benches/seme-hypercube/prereg.md §7（2026-10-15
冻结，先于首跑）。统一设置：stage1_local_core 形制（8 层
d256，lr=2e-4，wd=0.01，batch=8，grad_accum=4，40k 步，
max_seq_len=12）；数据集 = C1 对称格式（edge_path_mix
1.0），训练/评估机制与 C1 完全一致——唯一变量 = 嵌入
初始化 emb(α) = α·DW + (1−α)·rnd_s（vendor/alphascan.py）。

每种子跑一次 DeepWalk（与 C3 同源），七档条件复用；
每条件前 torch.manual_seed(seed) 使非嵌入初始化跨条件
一致 —— α 为条件间唯一变量。判读协变量：每条件训练前
init_probe（arm/hop 聚类准确率）。

舱内结构法：每训完即 json.dump 落 /kaggle/working 并
print REPORT_LINE(b64)；汇总只读已落盘数据（中途挂也
不丢已完成 (条件, 种子)）。
"""
import subprocess
import sys
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
               check=False)
# E 依赖：DeepWalk 走 gensim Word2Vec（vendor/src/deepwalk.py）。
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gensim"],
               check=False)

import os
os.environ.setdefault("TQDM_DISABLE", "1")

import base64
import hashlib
import json
import time
from pathlib import Path

WORK = Path(os.environ.get("NIGHT_WORK", "/kaggle/working"))
HERE = Path(__file__).resolve().parent

# ---- vendor 内嵌加载（build_arm.py 注入 _VENDOR；本机开发时
# _VENDOR 为空，直接用旁边的 vendor/ 目录） ----
_VENDOR = {}  # BUILD_INJECT_VENDOR


def _materialize_vendor() -> Path:
    if not _VENDOR:
        return HERE / "vendor"
    vdir = WORK / "vendor_runtime"
    if not all((vdir / rel).exists() for rel in _VENDOR):
        vdir.mkdir(parents=True, exist_ok=True)
        for rel, (b64, sha) in _VENDOR.items():
            raw = base64.b64decode(b64)
            assert hashlib.sha256(raw).hexdigest() == sha, \
                f"vendor sha mismatch: {rel}"
            target = vdir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        print(f"[vendor] materialized {len(_VENDOR)} files -> {vdir}",
              flush=True)
    return vdir


sys.path.insert(0, str(_materialize_vendor()))
sys.path.insert(0, str(HERE))

import numpy as np
import torch

from src.config import StageConfig
from src.graph import (build_path_star_graph,
                       build_vocab as build_star_vocab,
                       build_train_test_leaves)
from src.data import build_dataset
from src.model import PathStarTransformer
from src.train import train
from src.metrics import (steps_to_pct, eval_alignment_delta,
                         eval_embedding_auc, compute_stable_rank,
                         compute_eigenspace_gap, compute_principal_angles)
import seme_adapter as sa
from seme_graph import build_seme_hypercube
from src.deepwalk import run_deepwalk
from alphascan import build_alpha_embedding

BASE_CONDITION = "E"  # 条件标签 = E0..E6（α = i/6）
ALPHAS = [i / 6 for i in range(7)]  # E0..E6 七档（§7 冻结）
# 正式数值即预注册 §2/§7 统一设置（stage1_local_core 形制）；
# 环境变量覆盖仅供本机短验，Kaggle 夜航不设覆盖。
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43,44").split(",")]
TRAIN_STEPS = int(os.environ.get("TRAIN_STEPS", "40000"))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "500"))
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", "5000"))
MAX_SEQ_LEN = 12


def rss_mark(tag):
    try:
        import resource
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"RSS-MARK {tag}: {rss_mb:.0f} MB", flush=True)
    except Exception:
        print(f"RSS-MARK {tag}: unavailable", flush=True)


def report_line(payload):
    b64 = base64.b64encode(
        json.dumps(payload, ensure_ascii=False).encode()).decode()
    print(f"REPORT_LINE:{b64}", flush=True)


def build_condition(seed):
    """按预注册构图 / 词表 / 训练-测试划分（测试块整维隔离）。

    E 臂与 C1 同图同划分：测试块 = 第 3 维 == 3 的整块
    64 节点（从未出现在任何训练路径中间）。"""
    seme_graph = build_seme_hypercube(n_semes=4, n_values=4)
    orig_graph = sa.to_orig_graph(seme_graph)
    names = sa.seme_node_names(seme_graph)
    vocab = sa.build_vocab(orig_graph, names)
    test_leaves = [names[i] for i in range(seme_graph["n_nodes"])
                   if seme_graph["node_semes"][i][3] == 3]
    test_set = set(test_leaves)
    train_leaves = [n for n in names[1:] if n not in test_set]
    return orig_graph, vocab, train_leaves, test_leaves, seme_graph, names


def measure(model, log, graph, vocab, device, seme_graph, names):
    """全套度量（与 C1/C3 同口径；泛化 hardest/auc 走
    test 叶，训练循环内算；几何量为超立方体感知版 *_h）。"""
    per_hop = sa.alignment_delta_per_hop(model, vocab, graph)
    arm_cluster = sa.arm_cluster_acc_h(model, graph, vocab, device)
    stable_rank = sa.stable_rank_h(
        model, names, graph, vocab)["stable_rank"]
    eigenspace_gap = sa.eigenspace_gap_h(
        model, names, graph, vocab)["gap"]
    principal_angle = sa.principal_angles_h(
        model, names, graph, vocab)["mean_angle_deg"]
    alignment_delta = float(np.mean(per_hop))
    hop_level = sa.hop_level_cluster_acc(model, graph, vocab, seme_graph)
    return {
        "hardest_token_acc_final": log[-1]["hardest_token_acc"],
        "hardest_token_acc_best": max(e["hardest_token_acc"] for e in log),
        "steps_to_50": steps_to_pct(log, 0.5),
        "steps_to_90": steps_to_pct(log, 0.9),
        "arm_cluster_acc": arm_cluster,
        "hop_level_cluster_acc": hop_level,
        "alignment_delta": alignment_delta,
        "alignment_per_hop": per_hop,
        "hop1_alignment": per_hop[0],
        "hop2to6_alignment": float(np.mean(per_hop[1:])),
        "stable_rank": stable_rank,
        "eigenspace_gap": eigenspace_gap,
        "principal_angle_mean_deg": principal_angle,
        "pe_norm_final": log[-1]["pe_norm"],
    }


def summarize(results):
    """3 seeds 汇总：标量取均值/min/max；列表逐元素。"""
    out = {}
    keys = results[str(SEEDS[0])].keys()
    for k in keys:
        if k in ("condition", "alpha", "seed", "wall_sec", "train_steps"):
            continue
        vals = [r[k] for r in results.values()]
        if vals[0] is None:
            out[k] = None
            continue
        if isinstance(vals[0], (list, tuple)):
            arr = np.array(vals, dtype=float)
            out[k] = {"mean": arr.mean(axis=0).tolist(),
                      "min": arr.min(axis=0).tolist(),
                      "max": arr.max(axis=0).tolist()}
        else:
            arr = np.array(vals, dtype=float)
            out[k] = {"mean": float(arr.mean()),
                      "min": float(arr.min()),
                      "max": float(arr.max())}
    return out


def metrics_flat(condition, summary):
    """夜航汇总 metrics.json：扁平键 <COND>_<指标>
    （check_gates.py --stage training 的判读输入）。
    门限别名按原仓库口径（notebooks/02 cell 38：加载
    best ckpt 后评估）：hardest 门限 = best-checkpoint 值
    （best ckpt 按 hardest_token_acc 选取）；final_auc =
    arm_cluster_acc（本仓在训练终态模型上评估，原仓库在
    best ckpt 上评估——ckpt 按回收纪律已清，口径差诚实
    标注于聚合器 _provenance，不作事后追测）。"""
    flat = {"bench": "seme-hypercube", "condition": condition}
    for k, v in summary.items():
        if v is None:
            continue
        flat[f"{condition}_{k}"] = v["mean"] if isinstance(v, dict) else v
    flat[f"{condition}_hardest_token_acc"] = flat.get(
        f"{condition}_hardest_token_acc_best")
    flat[f"{condition}_final_auc"] = flat.get(
        f"{condition}_arm_cluster_acc")
    return flat


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    graph, vocab, _, _, seme_graph, names = build_condition(SEEDS[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{BASE_CONDITION}] device={device} "
          f"n_paths={len(graph['paths'])} "
          f"vocab={vocab['vocab_size']} "
          f"max_seq_len={MAX_SEQ_LEN} alphas={ALPHAS}",
          flush=True)

    all_results = {}  # cond -> {seed: metrics}
    for seed in SEEDS:
        _, _, tr_leaves, te_leaves, _, _ = build_condition(seed)
        cfg = StageConfig(
            stage_name=f"{BASE_CONDITION}-seed{seed}",
            n_layer=8, d_model=256, n_head=8,
            graph_degree=len(graph["paths"]),
            path_length=max(len(p) for p in graph["paths"]) - 1,
            train_leaf_fraction=len(tr_leaves)
                / (len(tr_leaves) + len(te_leaves)),
            train_steps=TRAIN_STEPS, batch_size=8, grad_accum_steps=4,
            learning_rate=2e-4, weight_decay=0.01,
            edge_path_mix_ratio=1.0,
            notes=f"night flight seme-hypercube {BASE_CONDITION}",
        )
        dataset = build_dataset(graph, vocab, tr_leaves, te_leaves,
                                MAX_SEQ_LEN, vocab["pad_id"])
        # DeepWalk 每种子一次（Hamming 图 H(4,4) 边集，
        # 经 to_orig_graph 转换，与 C3 每种子同源 →
        # α=1 ≡ C3 恒等端点）。七档条件复用。
        dw_emb = run_deepwalk(graph, vocab, cfg.d_model, seed=seed)
        dw_norms = np.linalg.norm(dw_emb, axis=1)
        print(f"[{BASE_CONDITION}] DeepWalk (Hamming H(4,4) edge set): "
              f"emb_norm mean={dw_norms.mean():.3f} "
              f"zero_rows={int((dw_norms < 1e-12).sum())} "
              f"(special tokens)", flush=True)
        for i, alpha in enumerate(ALPHAS):
            cond = f"E{i}"
            # 非嵌入初始化跨条件一致（同种子同种子状态）→
            # 条件间唯一变量 = 嵌入初始化 emb(α)。
            torch.manual_seed(seed)
            model = PathStarTransformer(vocab["vocab_size"], cfg.d_model,
                                        cfg.n_head, cfg.n_layer,
                                        MAX_SEQ_LEN)
            model = model.to(device)  # 原仓库约定：调用者搬设备
            emb = build_alpha_embedding(dw_emb, alpha, i, seed)
            model.emb.weight.data.copy_(
                torch.from_numpy(emb).to(device))
            # init_probe（训练前几何量，判读协变量）：
            # 若 α 与 hardest 呈倒 U 而 init_probe 单调，
            # 则训练动态（而非初始几何）是主因。
            pre_hop = sa.alignment_delta_per_hop(model, vocab, graph)
            pre_arm = sa.arm_cluster_acc_h(model, graph, vocab, device)
            pre_hop_level = sa.hop_level_cluster_acc(
                model, graph, vocab, seme_graph)
            emb_norms = np.linalg.norm(emb, axis=1)
            print(f"[{cond}] alpha={alpha:.4f} init: "
                  f"hop1={pre_hop[0]:.3f} "
                  f"hop2to6={np.mean(pre_hop[1:]):.3f} "
                  f"arm={pre_arm:.3f} "
                  f"hop_level={pre_hop_level:.3f} "
                  f"emb_norm={emb_norms.mean():.3f}", flush=True)
            art = WORK / f"{cond}-seed{seed}"
            art.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            model, log = train(model, cfg, vocab, graph, dataset,
                               te_leaves, device, art, seed=seed,
                               eval_every=EVAL_EVERY,
                               ckpt_every=CKPT_EVERY)
            wall = time.time() - t0
            m = measure(model, log, graph, vocab, device,
                        seme_graph, names)
            m.update({"condition": cond, "alpha": alpha, "seed": seed,
                      "wall_sec": wall, "train_steps": TRAIN_STEPS})
            # init_probe 展平（标量键，汇总器直接处理）
            m["init_hop1_alignment"] = float(pre_hop[0])
            m["init_hop2to6_alignment"] = float(np.mean(pre_hop[1:]))
            m["init_arm_cluster_acc"] = pre_arm
            m["init_hop_level_cluster_acc"] = pre_hop_level
            m["init_emb_norm_mean"] = float(emb_norms.mean())
            all_results.setdefault(cond, {})[str(seed)] = m
            # 舱内结构法：每训完即落盘 + REPORT_LINE
            (WORK / f"{cond}_seeds.json").write_text(
                json.dumps(all_results[cond], indent=2,
                           ensure_ascii=False))
            report_line(m)
            rss_mark(f"{cond} seed {seed} done, wall={wall:.0f}s")

    # 每条件 3-seed 汇总 + 全条件 flat metrics.json
    flat_all = {"bench": "seme-hypercube",
                "condition": BASE_CONDITION}
    for i in range(len(ALPHAS)):
        cond = f"E{i}"
        summary = summarize(all_results[cond])
        (WORK / f"{cond}_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False))
        flat = metrics_flat(cond, summary)
        flat.pop("bench", None)  # flat_all 已带 bench
        flat.pop("condition", None)  # 保留基条件标签 "E"
        flat_all.update(flat)
        report_line({"summary": True, "condition": cond,
                     "metrics_flat": flat})
    (WORK / "metrics.json").write_text(
        json.dumps(flat_all, indent=2, ensure_ascii=False))
    rss_mark("all conditions x seeds done")
    # 回收只需小文件（log + 汇总）：清 ckpt，防 output 包过大
    # 传输断连（1014 实测：28MB 包反复 IncompleteRead）。
    # ckpt 非判读依据——度量全在 REPORT_LINE / metrics.json。
    for i in range(len(ALPHAS)):
        cond = f"E{i}"
        for seed in SEEDS:
            art_dir = WORK / f"{cond}-seed{seed}"
            for f in art_dir.glob("*.pt"):
                f.unlink()
    rss_mark("ckpts cleaned")
    print("NIGHT-DONE", flush=True)


if __name__ == "__main__":
    main()
