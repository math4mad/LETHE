"""方向性格式（界面实验 D0）——原文 Phase 1.8b 形制的义素超立方体适配。

界面实验（prereg §6）：**仅改输入格式**，不改架构、不改权重
更新规则、不同的只是序列里携带的上下文。对照 C1：

  C1 边样本   [EDGE] u      [SEP] v     [EOS]   （对称邻接）
  C1 路径样本 [PATH] leaf   [SEP] hop...  [EOS]   （根→叶单向）
  D0 三元组   [EDGE] prev curr [SEP] next [EOS]   （方向性转移）

知识源与 C1 **同**（全边集 + 训练路径），唯一差异是格式所
携带的上下文 (prev)——即「界面」本身。原文出处：
llm-geometry-analysis notebooks/08_phase1_8_directional.ipynb
Exp 1.8b（encode_edge_directional / build_dataset_edges_directional，
seq = [edge, prev, curr, sep, next, eos]，预测位 = logits[3]）。

与原文的一处有意差异（prereg §6 审阅者注）：原文 1.8b 的
方向性数据集取**全部**路径（含测试叶）的三元组；本臂取
**训练路径**（测试块 64 叶的三元组不入训），以保持与 C1
同等的「未见组合」泛化需求——否则 hardest_token_acc 会是
对测试叶首跳的记忆而非泛化，与 C1 0.583 不可比。
"""
from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.config import StageConfig
import seme_adapter as sa


def build_directional_dataset(
    orig_graph: dict,
    vocab: dict,
    seme_graph: dict,
    train_leaves: list,
    max_seq_len: int,
) -> torch.Tensor:
    """方向性格式数据集（[N, max_seq_len] 的 token id 张量）。

    ① 游走三元组（对应 C1 的边样本）：每条有向边 (u→v)
       携带全部前驱语境 p ∈ N(u)\\{v} 与起始语境 [BOS] ——
       (p, u, v) 与 ([BOS], u, v)。覆盖全边集（1536 边
       × 双向 × 12 语境）。
    ② 路径三元组（对应 C1 的路径样本）：**训练**路径的
       双向典范测地线三元组（根→叶与叶→根），首 hop 以
       [BOS] 作 prev。
    """
    t2id = vocab["token2id"]
    pad_id, bos_id = vocab["pad_id"], vocab["bos_id"]
    eos_id, sep_id = vocab["eos_id"], vocab["sep_id"]
    edge_id = vocab["edge_id"]
    names = sa.seme_node_names(seme_graph)
    seqs: list[list[int]] = []
    # ① 游走三元组：全边集 × 前驱语境
    for u in range(seme_graph["n_nodes"]):
        u_id = t2id[names[u]]
        for v in seme_graph["adj"][u]:
            v_id = t2id[names[v]]
            for p in seme_graph["adj"][u]:
                if p == v:
                    continue
                seqs.append([edge_id, t2id[names[p]], u_id,
                             sep_id, v_id, eos_id])
            seqs.append([edge_id, bos_id, u_id, sep_id, v_id, eos_id])
    # ② 路径三元组：训练路径双向
    train_set = set(train_leaves)
    for path in orig_graph["paths"]:
        if path[-1] not in train_set:
            continue
        ids = [t2id[n] for n in path]
        for line in (ids, ids[::-1]):
            for i in range(len(line) - 1):
                prev = bos_id if i == 0 else line[i - 1]
                seqs.append([edge_id, prev, line[i], sep_id,
                             line[i + 1], eos_id])
    tensor = torch.full((len(seqs), max_seq_len), pad_id,
                        dtype=torch.long)
    for i, s in enumerate(seqs):
        tensor[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return tensor


def eval_hardest_token_acc_dir(
    model, vocab: dict, graph: dict, test_leaves, device,
) -> float:
    """方向性格式最难 token：给定叶节点，预测向原点第一跳。
    查询 [EDGE] [BOS] leaf [SEP]，预测位 = logits[0, 3, :]
    （与原文 1.8b 的位置约定一致）。"""
    model.eval()
    t2id = vocab["token2id"]
    leaf_parent = {path[-1]: path[-2] for path in graph["paths"]}
    correct = 0
    with torch.no_grad():
        for leaf in test_leaves:
            query = [[vocab["edge_id"], vocab["bos_id"],
                      t2id[leaf], vocab["sep_id"]]]
            logits = model(torch.tensor(query, dtype=torch.long,
                                        device=device))
            pred = logits[0, 3, :].argmax().item()
            correct += int(pred == t2id[leaf_parent[leaf]])
    return correct / max(len(test_leaves), 1)


def eval_traversal_dir(
    model, vocab: dict, graph: dict, test_leaves, device,
) -> tuple:
    """叶→原点 方向性遍历（teacher forcing：每步喂真前驱）。
    返回 (full_path_acc, per_hop_acc)：full = 全程全对才算；
    per_hop[k] = 第 k 跳独立准确率（对应原文 1.8b 的
    eval_directional 遍历评估）。"""
    model.eval()
    t2id = vocab["token2id"]
    max_hops = max(len(p) for p in graph["paths"]) - 1
    leaf_rev = {path[-1]: list(reversed(path))
                for path in graph["paths"]}
    per_hop = [[] for _ in range(max_hops)]
    full_correct = 0
    with torch.no_grad():
        for leaf in test_leaves:
            rev = leaf_rev[leaf]
            hops_ok = []
            for i in range(len(rev) - 1):
                prev = vocab["bos_id"] if i == 0 else t2id[rev[i - 1]]
                query = [[vocab["edge_id"], prev, t2id[rev[i]],
                          vocab["sep_id"]]]
                logits = model(torch.tensor(query, dtype=torch.long,
                                            device=device))
                ok = int(logits[0, 3, :].argmax().item()
                         == t2id[rev[i + 1]])
                hops_ok.append(ok)
                per_hop[i].append(ok)
            full_correct += int(all(hops_ok))
    per_hop_acc = [float(np.mean(h)) if h else 0.0 for h in per_hop]
    return full_correct / max(len(test_leaves), 1), per_hop_acc


def get_lr(step: int, warmup_steps: int, total_steps: int,
           peak_lr: float, min_lr_fraction: float = 0.1) -> float:
    """与 src/train.py 同形的 lr 调度（线性 warmup + 余弦衰减）。"""
    min_lr = peak_lr * min_lr_fraction
    if step < warmup_steps:
        return peak_lr * max(step, 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return min_lr + (peak_lr - min_lr) * cosine


def train_directional(
    model,
    cfg: StageConfig,
    vocab: dict,
    triples: torch.Tensor,
    graph: dict,
    test_leaves,
    device,
    artifact_dir: Path,
    seed: int = 42,
    eval_every: int = 500,
    ckpt_every: int = 5000,
) -> tuple:
    """与 src/train.py train() 同循环（AdamW betas=(0.9,0.95) /
    warmup=500 + 余弦 / grad_accum / clip=1.0 / best-ckpt /
    3000 步无改进 early stop），唯二差异：
    数据集 = 方向性格式三元组（随机等概抽样）；训练期评估 =
    方向性格式最难 token。log 条目键与 train() 一致
    （step/loss/hardest_token_acc/lr/pe_norm），故
    steps_to_pct / hardest_best 口径与 C1 可比。"""
    no_decay_keywords = {"bias", "LayerNorm.weight", "ln_f.weight",
                         "ln1.weight", "ln2.weight"}
    param_groups = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(kw in n for kw in no_decay_keywords)
                    and p.requires_grad],
         "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in model.named_parameters()
                    if any(kw in n for kw in no_decay_keywords)
                    and p.requires_grad],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        param_groups, lr=cfg.learning_rate, betas=(0.9, 0.95),
        eps=1e-8)

    log: list[dict] = []
    best_acc = -1.0
    best_step = 0
    lr = cfg.learning_rate
    optimizer.zero_grad()
    n_triples = triples.size(0)
    gen = torch.Generator().manual_seed(seed)

    pbar = tqdm(range(1, cfg.train_steps + 1),
                desc=cfg.stage_name, dynamic_ncols=True)
    for step in pbar:
        model.train()
        idx = torch.randint(0, n_triples,
                            (cfg.batch_size,), generator=gen)
        batch = triples[idx].to(device)

        logits = model(batch[:, :-1])
        targets = batch[:, 1:].contiguous()
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=vocab["pad_id"],
        )

        (loss / cfg.grad_accum_steps).backward()

        if step % cfg.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            lr = get_lr(step, warmup_steps=500,
                        total_steps=cfg.train_steps,
                        peak_lr=cfg.learning_rate)
            for g in optimizer.param_groups:
                g["lr"] = lr
            optimizer.step()
            optimizer.zero_grad()

        if step % eval_every == 0 or step == cfg.train_steps:
            model.eval()
            with torch.no_grad():
                acc = eval_hardest_token_acc_dir(
                    model, vocab, graph, test_leaves, device)

            pe_norm = model.pe_norm_ratio()
            if pe_norm > 3.0:
                print(f"\n[{cfg.stage_name}] WARNING step {step}: "
                      f"pe_norm={pe_norm:.2f} > 3.0 — geometry "
                      f"analysis may be unreliable.")

            entry = {
                "step": step,
                "loss": float(loss.item()),
                "hardest_token_acc": float(acc),
                "lr": float(lr),
                "pe_norm": float(pe_norm),
            }
            log.append(entry)
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             acc=f"{acc:.4f}", lr=f"{lr:.2e}",
                             pe=f"{pe_norm:.2f}")

            if step % ckpt_every == 0:
                torch.save(
                    {"step": step, "model_state": model.state_dict(),
                     "log": log, "cfg": asdict(cfg)},
                    artifact_dir / f"{cfg.stage_name}_step{step:06d}.pt")

            if acc > best_acc:
                best_acc = acc
                best_step = step
                torch.save(
                    {"step": step, "model_state": model.state_dict(),
                     "log": log, "cfg": asdict(cfg)},
                    artifact_dir / f"{cfg.stage_name}_best.pt")

            if step >= cfg.train_steps * 0.5 and (step - best_step) >= 3000:
                print(f"\n[{cfg.stage_name}] Early stop at step {step}: "
                      f"no improvement for {step - best_step} steps. "
                      f"Best = {best_acc:.4f} at {best_step}.")
                break

    print(f"\n[{cfg.stage_name}] Done. Best hardest_token_acc = "
          f"{best_acc:.4f} at step {best_step}.")
    return model, log
