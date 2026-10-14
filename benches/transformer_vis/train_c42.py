#!/usr/bin/env python3
"""射击 3 · 训练级 C42 —— 有序 vs 打乱语料各训 LoRA, 量 ΔW 夹角
(dag06 非交换律的训练级形态; 量法沿用 exp12c 园线)

园线 (exp12c-curriculum 冻结): r=16 / α=32 / 6 ep / lr=1e-4 / Adam β2=0.95
臂: ord (D5_baby 50 卡按年龄阶序) vs shuf (同 50 卡打乱, seeds 14/15/16)
任务: 卡片流上的因果语言建模 (next-token)

判据 (先冻, 跑前登记):
  K-1  ord~shuf ΔW 夹角 < 3°            → 时序课程序在 LoRA 尺度不可感
  K-2  3°–10°                            → 序效应真实但弱
  K-3  > 10°                             → 强效应
  K-4  归零条款: 臂间夹角 ≤ shuf 间散布   → 一律归零
  K-5  训后探针: ord 臂末层 ρ ≥ shuf 臂均值 + 0.05 → 训练把时序轴刻进权重

输出: metrics_c42.json + figs_c42/ + 本目录
"""
import json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "corpus_baby.json")
MODEL = os.path.expanduser("~/Programming/code-2026/first-lora/model")
DECK = json.load(open(CORPUS))["deck"]
FIGS = os.path.join(HERE, "figs_c42"); os.makedirs(FIGS, exist_ok=True)

# ---------- 园线旋钮 (逐条列出, 园规「参数意愿律」) ----------
SEED_FIXED = 13
SEEDS_SHUFFLE = [14, 15, 16]
EPOCHS = 6                 # exp12c 园线
LR = 1e-4                  # exp12c 园线
LORA_R = 16                # exp12c 园线
LORA_ALPHA = 32            # exp12c 园线
ADAM_BETA2 = 0.95          # exp12c 园线 "b2a4" 转写 (β2=0.95), 如实标注
BLOCK = 128                # 流切块 (非重叠)
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]   # 注意力四矩阵

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

corp = json.load(open(CORPUS))
items = corp["items"]
n_stages = len(corp["stages"])
stage_names = corp["stages"]
texts = [f"{it['name']}：{it['desc']}。" for it in items]
N = len(texts)
stages = np.array([it["stage"] for it in items])
dev = "mps" if torch.backends.mps.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(MODEL)

def spearman_rho(x, y):
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).statistic)

def all_hidden(ids, model):
    with torch.no_grad():
        out = model(**ids, output_hidden_states=True)
    return out.hidden_states

def encode_isolated(texts, model):
    reps = []
    for t in texts:
        ids = tok(t, return_tensors="pt", add_special_tokens=False).to(dev)
        hs = all_hidden(ids, model)
        reps.append(np.stack([h[0].float().mean(0).cpu().numpy() for h in hs]))
    return np.stack(reps)

def delta_vector(model):
    """所有 LoRA 模块的 B@A·scaling 拉成一个大向量
    (peft 0.21: lora_A/lora_B 为 ModuleDict, scaling 为 dict)"""
    def vals(c):
        return list(c.values()) if hasattr(c, "values") else list(c)
    vecs = []
    for m in model.modules():
        if hasattr(m, "lora_A"):
            s = m.scaling
            s = s["default"] if isinstance(s, dict) else (s[0] if isinstance(s, (list, tuple)) else s)
            for a, b in zip(vals(m.lora_A), vals(m.lora_B)):
                d = (b.weight @ a.weight) * s
                vecs.append(d.detach().flatten().float().cpu())
    return torch.cat(vecs).numpy()

def angle_deg(v1, v2):
    c = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    c = max(-1.0, min(1.0, c))
    return float(np.degrees(np.arccos(c)))

def train_arm(order, seed):
    """按 order 拼流, 训 LoRA; 返回 (model, ΔW 向量, 逐 epoch 损失)"""
    torch.manual_seed(seed)
    base = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager").to(dev)
    cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=TARGETS,
                     lora_dropout=0.0, task_type="CAUSAL_LM")
    lm = get_peft_model(base, cfg)
    stream = "\n".join(texts[i] for i in order)
    ids = tok(stream, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
    blocks = ids[0].unfold(0, BLOCK, BLOCK).clone()      # (nblocks, 128)
    opt = torch.optim.AdamW(lm.parameters(), lr=LR, betas=(0.9, ADAM_BETA2), weight_decay=0.0)
    lm.train()
    losses = []
    t0 = time.time()
    for ep in range(EPOCHS):
        ep_loss, nb = 0.0, 0
        for blk in blocks:
            x = blk.unsqueeze(0)
            opt.zero_grad()
            out = lm(x, labels=x)
            out.loss.backward()
            opt.step()
            ep_loss += float(out.loss); nb += 1
        losses.append(ep_loss / nb)
        print(f"    arm seed={seed} ep={ep+1}/{EPOCHS} loss={losses[-1]:.4f} ({time.time()-t0:.0f}s)", flush=True)
    lm.eval()
    return lm, delta_vector(lm), losses

print(f"[1/4] 训练 ord 臂 (SEED_FIXED={SEED_FIXED}) …", flush=True)
ord_order = list(range(N))
lm_ord, dw_ord, loss_ord = train_arm(ord_order, SEED_FIXED)
print(f"[2/4] 训练 shuf 臂 ×{len(SEEDS_SHUFFLE)} …", flush=True)
lms_shuf, dws_shuf, losses_shuf = [], [], []
for sd in SEEDS_SHUFFLE:
    rng = np.random.default_rng(sd)
    lm_s, dw_s, loss_s = train_arm(rng.permutation(N).tolist(), sd)
    lms_shuf.append(lm_s); dws_shuf.append(dw_s); losses_shuf.append(loss_s)

print("[3/4] ΔW 夹角 …", flush=True)
ang_ord_shuf = [angle_deg(dw_ord, d) for d in dws_shuf]
ang_shuf_shuf = []
for i in range(len(dws_shuf)):
    for j in range(i + 1, len(dws_shuf)):
        ang_shuf_shuf.append(angle_deg(dws_shuf[i], dws_shuf[j]))
ord_mean = float(np.mean(ang_ord_shuf))
shuf_spread = float(np.mean(ang_shuf_shuf))

print("[4/4] 训后探针 (逐层 ρ) …", flush=True)
R_ord = encode_isolated(texts, lm_ord)
rho_ord_per = [spearman_rho(
    __import__("sklearn.decomposition", fromlist=["PCA"]).PCA(n_components=2, random_state=SEED_FIXED)
    .fit_transform(R_ord[:, li])[:, 0], stages) for li in range(R_ord.shape[1])]
rho_shuf_per = []
for lm_s in lms_shuf:
    R_s = encode_isolated(texts, lm_s)
    rho_shuf_per.append([spearman_rho(
        __import__("sklearn.decomposition", fromlist=["PCA"]).PCA(n_components=2, random_state=SEED_FIXED)
        .fit_transform(R_s[:, li])[:, 0], stages) for li in range(R_s.shape[1])])
rho_shuf_per = np.array(rho_shuf_per)
rho_shuf_mean = rho_shuf_per.mean(0)

# ---------- 判据裁决 ----------
if ord_mean <= shuf_spread:
    verdict = "K-4 归零: 臂间夹角 ≤ shuf 间散布"
elif ord_mean < 3.0:
    verdict = "K-1 时序课程序在 LoRA 尺度不可感"
elif ord_mean <= 10.0:
    verdict = "K-2 序效应真实但弱"
else:
    verdict = "K-3 强效应"
k5 = float(rho_ord_per[-1]) >= float(rho_shuf_mean[-1]) + 0.05

# ---------- 图 ----------
fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
labels = [f"ord~shuf{s}" for s in SEEDS_SHUFFLE] + [f"shuf{SEEDS_SHUFFLE[i]}~shuf{SEEDS_SHUFFLE[j]}" for i in range(3) for j in range(i+1, 3)]
vals = ang_ord_shuf + ang_shuf_shuf
colors = ["#21918c"]*3 + ["#b8860b"]*3
ax.bar(labels, vals, color=colors, edgecolor="#333333", linewidth=0.6)
ax.axhline(3.0, color="#d62728", ls="--", lw=1.0)
ax.axhline(10.0, color="#d62728", ls=":", lw=1.0)
ax.text(len(vals)-0.5, 3.2, "K-1 阈 3°", color="#d62728", fontsize=8, ha="right")
ax.text(len(vals)-0.5, 10.2, "K-3 阈 10°", color="#d62728", fontsize=8, ha="right")
ax.set_ylabel("ΔW 夹角 (度)")
ax.set_title("训练级 C42: 有序 vs 打乱 ΔW 夹角")
ax.tick_params(axis="x", labelsize=7.5, rotation=20)
for i, v in enumerate(vals):
    ax.text(i, v + 0.15, f"{v:.1f}°", ha="center", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_c42_angles.png")); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.8, 4.6), dpi=150)
lay = np.arange(len(rho_ord_per))
ax.plot(lay, -np.array(rho_ord_per), "s-", color="#21918c", lw=1.6, ms=3, label="ord 臂 (训后)")
ax.plot(lay, -rho_shuf_mean, "^-", color="#b8860b", lw=1.6, ms=3, label="shuf 臂 (训后, 3 seeds)")
ax.fill_between(lay, -(rho_shuf_mean + rho_shuf_per.std(0)), -(rho_shuf_mean - rho_shuf_per.std(0)), color="#b8860b", alpha=0.18)
ax.axhline(0, color="#666666", lw=0.8)
ax.set_xlabel("layer (0 = embedding, L = last)")
ax.set_ylabel("−Spearman ρ（PC1 vs 年龄阶）")
ax.set_title("训后探针: 时序轴在 LoRA 权重里的残影（ρ 取反仅轴符号约定）")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_c42_rho.png")); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=150)
ax.plot(range(1, EPOCHS+1), loss_ord, "s-", color="#21918c", lw=1.6, ms=4, label=f"ord (seed {SEED_FIXED})")
for i, sd in enumerate(SEEDS_SHUFFLE):
    ax.plot(range(1, EPOCHS+1), losses_shuf[i], "^-", lw=1.4, ms=3, label=f"shuf (seed {sd})")
ax.set_xlabel("epoch"); ax.set_ylabel("mean block loss")
ax.set_title("训练曲线 (因果 LM, 卡片流)")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_c42_loss.png")); plt.close(fig)

metrics = {
    "deck": DECK, "minted": "2026-10-13",
    "protocol": {"lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "epochs": EPOCHS,
                 "lr": LR, "adam_beta2": ADAM_BETA2, "block": BLOCK,
                 "targets": TARGETS, "seeds_shuffle": SEEDS_SHUFFLE, "seed_fixed": SEED_FIXED},
    "angles": {"ord_shuf": ang_ord_shuf, "ord_shuf_mean": ord_mean,
               "shuf_shuf": ang_shuf_shuf, "shuf_spread": shuf_spread},
    "losses": {"ord": loss_ord, "shuf": losses_shuf},
    "rho_per_layer_post": {"ord": rho_ord_per, "shuf": rho_shuf_per.tolist(),
                            "shuf_mean": rho_shuf_mean.tolist()},
    "verdict": verdict, "K5_ord_shuf_rho_gap": float(rho_ord_per[-1] - rho_shuf_mean[-1]),
    "K5_pass": bool(k5),
}
json.dump(metrics, open(os.path.join(HERE, "metrics_c42.json"), "w"),
          ensure_ascii=False, indent=1)
print(f"\n裁决: {verdict}")
print(f"ord~shuf {ord_mean:.2f}° vs shuf 散布 {shuf_spread:.2f}° · K5 末层 ρ 差 {metrics['K5_ord_shuf_rho_gap']:+.3f} ({'过' if k5 else '不过'})")
print(f"DONE — figs_c42/ + metrics_c42.json")
