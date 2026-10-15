#!/usr/bin/env python3
"""义素超立方体夜航臂 —— CONDITION=D0（界面实验：方向性格式）。

预注册：benches/seme-hypercube/prereg.md（§4 界面实验判据 /
§6 D0 臂规格）。界面实验（原文 Phase 1.8b 形制）：
**仅改输入格式**——训练与评估都用方向性格式
[EDGE] prev curr [SEP] next [EOS]（预测位 logits[3]），
不改架构、不改权重更新规则。数据知识源与 C1 同
（全边集 + 训练路径），唯一差异是格式携带的上下文
(prev)——即「界面」本身。判据（prereg §4 已冻）：
D0 hardest（best-ckpt）− C1 hardest（best-ckpt）< 0.05
→「界面瓶颈」主张在语义图下不成立。

疫苗二代（0919 矩阵全灭真死因）：先卸 torchao。
舱内结构法：每训完即 json.dump 落 /kaggle/working 并
print REPORT_LINE(b64)；汇总只读已落盘数据（不排在训练循环之后
依赖——中途挂也不丢已完成种子）。
"""
import subprocess
import sys
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
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
from src.model import PathStarTransformer
from src.metrics import (steps_to_pct, eval_alignment_delta,
                         eval_embedding_auc, compute_stable_rank,
                         compute_eigenspace_gap, compute_principal_angles)
import seme_adapter as sa
import directional as dz
from seme_graph import build_seme_hypercube

CONDITION = "D0"  # 界面实验臂（方向性格式）
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
    + 嵌入几何（超立方体感知版 *_h，与 C1 同口径）
    + 方向性遍历行为量（D0 专有：teacher forcing 叶→原点）。"""
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
    # D0 专有：方向性格式下的行为遍历（teacher forcing）
    test_leaves = [names[i] for i in range(seme_graph["n_nodes"])
                   if seme_graph["node_semes"][i][3] == 3]
    traversal_full, traversal_per_hop = dz.eval_traversal_dir(
        model, vocab, graph, test_leaves, device)
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
        "traversal_full_acc": traversal_full,
        "traversal_per_hop": traversal_per_hop,
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
    口径同 C1：hardest 门限 = best-checkpoint 值
    （best ckpt 按 hardest_token_acc 选取）；final_auc =
    arm_cluster_acc（训练终态模型上评估——口径差诚实标注）。"""
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
            notes=f"night flight seme-hypercube {CONDITION} (directional interface)",
        )
        # 方向性格式数据集（知识源与 C1 同：全边集游走三元组
        # + 训练路径三元组；见 vendor/directional.py 头注）
        triples = dz.build_directional_dataset(
            graph, vocab, seme_graph, tr_leaves, MAX_SEQ_LEN)
        print(f"[{CONDITION}] directional triples: {triples.size(0)}",
              flush=True)
        model = PathStarTransformer(vocab["vocab_size"], cfg.d_model,
                                    cfg.n_head, cfg.n_layer, MAX_SEQ_LEN)
        model = model.to(device)  # 原仓库约定：调用者搬设备
        art = WORK / f"{CONDITION}-seed{seed}"
        art.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        model, log = dz.train_directional(
            model, cfg, vocab, triples, graph, te_leaves, device, art,
            seed=seed, eval_every=EVAL_EVERY, ckpt_every=CKPT_EVERY)
        wall = time.time() - t0
        m = measure(model, log, graph, vocab, device, seme_graph, names)
        m.update({"condition": CONDITION, "seed": seed,
                  "wall_sec": wall, "train_steps": TRAIN_STEPS})
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
