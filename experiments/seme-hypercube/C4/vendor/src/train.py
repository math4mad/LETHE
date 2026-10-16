"""Training loop for the path-star transformer."""
from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .config import StageConfig
from .data import MixedBatchSampler
from .metrics import eval_hardest_token_acc


def get_lr(step: int, warmup_steps: int, total_steps: int, peak_lr: float, min_lr_fraction: float = 0.1) -> float:
    """Linear warmup followed by cosine decay to min_lr_fraction * peak_lr."""
    min_lr = peak_lr * min_lr_fraction
    if step < warmup_steps:
        return peak_lr * max(step, 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return min_lr + (peak_lr - min_lr) * cosine


def train(
    model,
    cfg: StageConfig,
    vocab: dict,
    graph: dict,
    dataset: dict,
    test_leaves: set,
    device: torch.device,
    artifact_dir: Path,
    seed: int = 42,
    freeze_embeddings: bool = False,
    eval_every: int = 500,
    ckpt_every: int = 2000,
) -> tuple:
    """
    Full training loop.

    Returns (trained_model, log) where log is a list of dicts with keys:
    step, loss, hardest_token_acc, lr, pe_norm.
    """
    if freeze_embeddings:
        model.emb.weight.requires_grad_(False)
        print(f"[{cfg.stage_name}] Embeddings FROZEN.")

    no_decay_keywords = {"bias", "LayerNorm.weight", "ln_f.weight", "ln1.weight", "ln2.weight"}
    param_groups = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(kw in n for kw in no_decay_keywords) and p.requires_grad
            ],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(kw in n for kw in no_decay_keywords) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        param_groups, lr=cfg.learning_rate, betas=(0.9, 0.95), eps=1e-8
    )

    sampler = MixedBatchSampler(dataset, cfg.batch_size, cfg.edge_path_mix_ratio, seed=seed)

    log: list[dict] = []
    best_acc = -1.0
    best_step = 0
    lr = cfg.learning_rate
    optimizer.zero_grad()

    pbar = tqdm(range(1, cfg.train_steps + 1), desc=cfg.stage_name, dynamic_ncols=True)

    for step in pbar:
        model.train()
        batch = sampler.sample_batch().to(device)

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
            lr = get_lr(step, warmup_steps=500, total_steps=cfg.train_steps, peak_lr=cfg.learning_rate)
            for g in optimizer.param_groups:
                g["lr"] = lr
            optimizer.step()
            optimizer.zero_grad()

        if step % eval_every == 0 or step == cfg.train_steps:
            model.eval()
            with torch.no_grad():
                acc = eval_hardest_token_acc(model, vocab, graph, test_leaves, device)

            pe_norm = model.pe_norm_ratio()
            if pe_norm > 3.0:
                print(
                    f"\n[{cfg.stage_name}] WARNING step {step}: "
                    f"pe_norm={pe_norm:.2f} > 3.0 — geometry analysis may be unreliable."
                )

            entry = {
                "step": step,
                "loss": float(loss.item()),
                "hardest_token_acc": float(acc),
                "lr": float(lr),
                "pe_norm": float(pe_norm),
            }
            log.append(entry)
            pbar.set_postfix(
                loss=f"{loss.item():.4f}", acc=f"{acc:.4f}",
                lr=f"{lr:.2e}", pe=f"{pe_norm:.2f}",
            )

            if step % ckpt_every == 0:
                torch.save(
                    {"step": step, "model_state": model.state_dict(), "log": log, "cfg": asdict(cfg)},
                    artifact_dir / f"{cfg.stage_name}_step{step:06d}.pt",
                )

            if acc > best_acc:
                best_acc = acc
                best_step = step
                torch.save(
                    {"step": step, "model_state": model.state_dict(), "log": log, "cfg": asdict(cfg)},
                    artifact_dir / f"{cfg.stage_name}_best.pt",
                )

            if step >= cfg.train_steps * 0.5 and (step - best_step) >= 3000:
                print(
                    f"\n[{cfg.stage_name}] Early stop at step {step}: "
                    f"no improvement for {step - best_step} steps. Best = {best_acc:.4f} at {best_step}."
                )
                break

    print(f"\n[{cfg.stage_name}] Done. Best hardest_token_acc = {best_acc:.4f} at step {best_step}.")
    return model, log
