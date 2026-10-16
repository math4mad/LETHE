#!/usr/bin/env python3
"""义素超立方体夜航臂 —— CONDITION=C4（第四格消融：
DeepWalk 初始化 + 嵌入冻结）。

预注册：benches/seme-hypercube/prereg.md（§2 条件矩阵 /
§4 2×2 因子矩阵第四格 / §68-71）。统一设置：
stage1_local_core 形制（8 层 d256，lr=2e-4，wd=0.01，
batch=8，grad_accum=4，40k 步，max_seq_len=12）；
B0 与 C1 同架构、同训练循环、同诊断（src/metrics.py
原样复用）。

第四格（§4）：与 C3 同装置（DeepWalk 在 Hamming 边集
初始化嵌入层），随后嵌入层 requires_grad=False ——
结构完全固定，仅其余网络学习利用该结构。判据三态：
C4 hardest ≥ 0.95 → 结构本身足以承担能力（学习非必要，
C3 结论推广）；C4 ≈ C2（≤ 0.35）→ 学习仍必要，结构
只是先验；C4 介于 C2/C3 之间 → 结构与学习各贡献部分。

疫苗二代（0919 矩阵全灭真死因）：先卸 torchao。
舱内结构法：每训完即 json.dump 落 /kaggle/working 并
print REPORT_LINE(b64)；汇总只读已落盘数据（不排在
训练循环之后依赖——中途挂也不丢已完成种子）。
"""
import subprocess
import sys
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
               check=False)
# C4 依赖：DeepWalk 走 gensim Word2Vec（vendor/src/deepwalk.py）。
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

CONDITION = "C4"  # 第四格消融：DeepWalk 初始化 + 嵌入冻结
# 正式数值即预注册 §2 统一设置（stage1_local_core 形制）；
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
    """按 CONDITION 构图 / 词表 / 训练-测试划分（测试块整维隔离）。"""
    if CONDITION == "B0":
        graph = build_path_star_graph(512, 6)
        vocab = build_star_vocab(512, 6)
        train_leaves, test_leaves = build_train_test_leaves(
            graph, 0.75, seed=seed)
        return graph, vocab, train_leaves, test_leaves, None, None
    seme_graph = build_seme_hypercube(n_semes=4, n_values=4)
    # 默认 test_dim=3, test_value=3：第 3 维 == 3 的整块 64 节点
    # 从未出现在任何训练路径中间（典范测地线最后才改该维）。
    orig_graph = sa.to_orig_graph(seme_graph)
    names = sa.seme_node_names(seme_graph)
    vocab = sa.build_vocab(orig_graph, names)
    test_leaves = [names[i] for i in range(seme_graph["n_nodes"])
                   if seme_graph["node_semes"][i][3] == 3]
    test_set = set(test_leaves)
    train_leaves = [n for n in names[1:] if n not in test_set]
    return orig_graph, vocab, train_leaves, test_leaves, seme_graph, names


def measure(model, log, graph, vocab, device, seme_graph, names):
    """全套度量：泛化（hardest/auc 走 test 叶，训练循环内算）
    + 嵌入几何（超立方体感知版 *_h，已在 B0 基线上
    selftest 对表）。"""
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
        "emb_frozen": True,
    }


def summarize(results):
    """3 seeds 汇总：标量取均值/min/max；列表逐元素。"""
    out = {}
    keys = results[str(SEEDS[0])].keys()
    for k in keys:
        if k in ("condition", "seed", "wall_sec", "train_steps"):
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


def metrics_flat(summary):
    """夜航汇总 metrics.json：扁平键 <COND>_<指标>
    （check_gates.py --stage training 的判读输入）。
    门限别名按原仓库口径（notebooks/02 cell 38：加载
    best ckpt 后评估）：hardest 门限 = best-checkpoint 值
    （best ckpt 按 hardest_token_acc 选取）；final_auc =
    arm_cluster_acc（本仓在训练终态模型上评估，原仓库在
    best ckpt 上评估——ckpt 按回收纪律已清，口径差诚实
    标注于聚合器 _provenance，不作事后追测）。"""
    flat = {"bench": "seme-hypercube", "condition": CONDITION}
    for k, v in summary.items():
        if v is None:
            continue
        flat[f"{CONDITION}_{k}"] = v["mean"] if isinstance(v, dict) else v
    flat[f"{CONDITION}_hardest_token_acc"] = flat.get(
        f"{CONDITION}_hardest_token_acc_best")
    flat[f"{CONDITION}_final_auc"] = flat.get(
        f"{CONDITION}_arm_cluster_acc")
    return flat


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    graph, vocab, _, _, seme_graph, names = build_condition(SEEDS[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{CONDITION}] device={device} n_paths={len(graph['paths'])} "
          f"vocab={vocab['vocab_size']} max_seq_len={MAX_SEQ_LEN}",
          flush=True)

    results = {}
    for seed in SEEDS:
        _, _, tr_leaves, te_leaves, _, _ = build_condition(seed)
        cfg = StageConfig(
            stage_name=f"{CONDITION}-seed{seed}",
            n_layer=8, d_model=256, n_head=8,
            graph_degree=len(graph["paths"]),
            path_length=max(len(p) for p in graph["paths"]) - 1,
            train_leaf_fraction=len(tr_leaves)
                / (len(tr_leaves) + len(te_leaves)),
            train_steps=TRAIN_STEPS, batch_size=8, grad_accum_steps=4,
            learning_rate=2e-4, weight_decay=0.01,
            edge_path_mix_ratio=1.0,
            notes=f"night flight seme-hypercube {CONDITION}",
        )
        dataset = build_dataset(graph, vocab, tr_leaves, te_leaves,
                                MAX_SEQ_LEN, vocab["pad_id"])
        model = PathStarTransformer(vocab["vocab_size"], cfg.d_model,
                                    cfg.n_head, cfg.n_layer, MAX_SEQ_LEN)
        model = model.to(device)  # 原仓库约定：调用者搬设备
        # C4（第四格消融）：与 C3 同装置 —— 在夜航版边集
        # （Hamming 图 H(4,4)，经 to_orig_graph 转换）上跑
        # DeepWalk 初始化嵌入层；随后**嵌入层冻结**
        # （requires_grad=False）：结构完全固定，仅其余
        # 网络学习利用该结构。冻结后优化器对该层无梯度
        # （AdamW 对 p.grad=None 的参数整体跳过，连
        # weight decay 也不作用）。
        dw_emb = run_deepwalk(graph, vocab, cfg.d_model, seed=seed)
        model.emb.weight.data.copy_(
            torch.from_numpy(dw_emb).to(device))
        model.emb.weight.requires_grad_(False)
        n_frozen = int(model.emb.weight.numel()) if not model.emb.weight.requires_grad else 0
        dw_norms = np.linalg.norm(dw_emb, axis=1)
        print(f"[C4] DeepWalk prealign + FROZEN embedding: "
              f"emb_norm mean={dw_norms.mean():.3f} "
              f"zero_rows={int((dw_norms < 1e-12).sum())} "
              f"(special tokens) frozen_params={n_frozen}", flush=True)
        # 判读点（与 C3 同口径）：(a) pre 诊断 = 训练前 DeepWalk
        # 嵌入的几何量（机械注入基线）；(b) perm 对照 = 同样嵌入
        # 随机置换节点映射（保留分布形状、打破义素结构）——
        # 若真版 pre 诊断 >> 置换版，则 DeepWalk 编码了义素组合
        # 结构；C4 训练后 (post) 与 pre 之差 = 0（嵌入冻结，
        # 几何量不变），网络学习贡献单独由 hardest 体现。
        pre_hop = sa.alignment_delta_per_hop(model, vocab, graph)
        pre_arm = sa.arm_cluster_acc_h(model, graph, vocab, device)
        pre_hop_level = sa.hop_level_cluster_acc(
            model, graph, vocab, seme_graph)
        node_ids = [vocab["token2id"][n] for n in names]
        rng = np.random.default_rng(seed)
        saved = model.emb.weight.data[node_ids].clone()
        perm_order = rng.permutation(len(node_ids))
        model.emb.weight.data[node_ids] = saved[perm_order]
        perm_hop = sa.alignment_delta_per_hop(model, vocab, graph)
        perm_arm = sa.arm_cluster_acc_h(model, graph, vocab, device)
        perm_hop_level = sa.hop_level_cluster_acc(
            model, graph, vocab, seme_graph)
        model.emb.weight.data[node_ids] = saved  # 恢复真映射
        deepwalk_probe = {
            "pre_train": {
                "hop1_alignment": float(pre_hop[0]),
                "hop2to6_alignment": float(np.mean(pre_hop[1:])),
                "arm_cluster_acc": pre_arm,
                "hop_level_cluster_acc": pre_hop_level,
            },
            "perm_ctrl": {
                "hop1_alignment": float(perm_hop[0]),
                "hop2to6_alignment": float(np.mean(perm_hop[1:])),
                "arm_cluster_acc": perm_arm,
                "hop_level_cluster_acc": perm_hop_level,
            },
        }
        print(f"[C4] probe pre: hop1={pre_hop[0]:.3f} "
              f"hop2to6={np.mean(pre_hop[1:]):.3f} "
              f"arm={pre_arm:.3f} hop_level={pre_hop_level:.3f} | "
              f"perm: hop1={perm_hop[0]:.3f} arm={perm_arm:.3f}",
              flush=True)
        art = WORK / f"{CONDITION}-seed{seed}"
        art.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        model, log = train(model, cfg, vocab, graph, dataset, te_leaves,
                           device, art, seed=seed, eval_every=EVAL_EVERY,
                           ckpt_every=CKPT_EVERY)
        wall = time.time() - t0
        m = measure(model, log, graph, vocab, device, seme_graph, names)
        m.update({"condition": CONDITION, "seed": seed,
                  "wall_sec": wall, "train_steps": TRAIN_STEPS})
        # C4 判读点数据展平（标量键，汇总器直接处理）
        for scope, dd in deepwalk_probe.items():
            for dk, dv in dd.items():
                m[f"dw_{scope}_{dk}"] = dv
        results[str(seed)] = m
        # 舱内结构法：每训完即落盘 + REPORT_LINE
        (WORK / f"{CONDITION}_seeds.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False))
        report_line(m)
        rss_mark(f"{CONDITION} seed {seed} done, wall={wall:.0f}s")

    summary = summarize(results)
    (WORK / f"{CONDITION}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    flat = metrics_flat(summary)
    (WORK / "metrics.json").write_text(
        json.dumps(flat, indent=2, ensure_ascii=False))
    report_line({"summary": True, "condition": CONDITION,
                 "metrics_flat": flat})
    rss_mark("all seeds done")
    # 回收只需小文件（log + 汇总）：清 ckpt，防 output 包过大
    # 传输断连（1014 实测：28MB 包反复 IncompleteRead）。
    # ckpt 非判读依据——度量全在 REPORT_LINE / metrics.json。
    for seed in SEEDS:
        art_dir = WORK / f"{CONDITION}-seed{seed}"
        for f in art_dir.glob("*.pt"):
            f.unlink()
    rss_mark("ckpts cleaned")
    print("NIGHT-DONE", flush=True)


if __name__ == "__main__":
    main()
