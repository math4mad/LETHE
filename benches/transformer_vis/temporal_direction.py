#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E-TDV 时序方向向量实验 —— v_temporal = mean(H_C1) − mean(H_C0)

出处: 主人 Qwen 帖「Transformer时序信息引入」(2026-10-13)
问题: 末层表征中, 有序流相对孤立流的平均偏移向量, 是不是模型内部
      「语义演化曲线」的真实切线方向? (而非上下文长度伪影)

三判据 (PREREG_E-TDV_temporal_direction.md 先冻):
  K-1 探针    score = R0l·v̂ 对年龄阶的 Spearman ρ (对照 v̂_shuf)
  K-2 时间机器 H_new = H_C0 + α·v̂, 平均预测阶随 α 单调漂移
  K-3 破 C42  LoRA 引起的表示偏移 ΔH 在 v̂ 上的投影, ord > shuf
              (维度修正: ΔW 在权重空间, v̂ 在表示空间, 故量表示偏移)

正本: 本脚本 (源 of truth)。复现:
  /opt/miniconda3/envs/default/bin/python temporal_direction.py  (退出码 0 即过)
产物: metrics_temporal_direction.json + figs_tdv/ (派生)
"""

import json, os, sys, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

# ---------- 旋钮区 (园规: 每个数一行) ----------
HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "corpus_baby.json")
MODEL = os.path.expanduser("~/Programming/code-2026/first-lora/model")
SEED_FIXED = 13
SHUFFLE_SEEDS = [14, 15, 16]
ALPHA_GRID = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
LORA_R, LORA_ALPHA = 16, 32
EPOCHS, LR, ADAM_BETA2, BLOCK = 6, 1e-4, 0.95, 128
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
# ------------------------------------------------

corp = json.load(open(CORPUS))
items = corp["items"]
stages = np.array([it["stage"] for it in items])
n_stages = len(corp["stages"])
stage_names = corp["stages"]
texts = [f"{it['name']}：{it['desc']}。" for it in items]
N = len(texts)
FIGS = os.path.join(HERE, "figs_tdv"); os.makedirs(FIGS, exist_ok=True)

torch.manual_seed(SEED_FIXED); np.random.seed(SEED_FIXED)

CJK = None
for cand in ("PingFang SC", "Arial Unicode MS", "STHeiti", "Songti SC",
             "Heiti TC", "Noto Sans CJK SC"):
    try:
        f = font_manager.findfont(cand, fallback_to_default=False)
        if "DejaVu" not in f:
            CJK = cand; plt.rcParams["font.family"] = cand
            plt.rcParams["axes.unicode_minus"] = False
            break
    except Exception:
        pass

from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(MODEL)
dev = "mps" if torch.backends.mps.is_available() else "cpu"

def fresh_model():
    return AutoModelForCausalLM.from_pretrained(
        MODEL, attn_implementation="eager").to(dev).eval()

model = fresh_model()
L = model.config.num_hidden_layers
HID = model.config.hidden_size

def all_hidden(ids, m):
    with torch.no_grad():
        out = m(**ids, output_hidden_states=True)
    return out.hidden_states

def encode_isolated(texts_, m):
    """C0: 逐卡孤立编码, 全层 mean-pool → (n, L+1, d)"""
    reps = []
    for t in texts_:
        ids = tok(t, return_tensors="pt", add_special_tokens=False).to(dev)
        hs = all_hidden(ids, m)
        reps.append(np.stack([h[0].float().mean(0).cpu().numpy() for h in hs]))
    return np.stack(reps)

def encode_stream(texts_, order, m):
    """C1/C2: 按 order 拼流, offset mapping 切回每张卡, 全层 mean-pool → (n, L+1, d)"""
    stream = "\n".join(texts_[i] for i in order)
    spans, pos = {}, 0
    for i in order:
        t = texts_[i]
        spans[i] = (pos, pos + len(t))
        pos += len(t) + 1
    enc = tok(stream, return_tensors="pt", return_offsets_mapping=True,
              add_special_tokens=False).to(m.device if hasattr(m, "device") else dev)
    offs = enc.pop("offset_mapping")[0].tolist()
    hs = all_hidden(enc, m)
    reps = {}
    for i in order:
        lo, hi = spans[i]
        sel = [k for k, (s, e) in enumerate(offs) if s >= lo and e <= hi and e > s]
        if not sel:
            sel = [k for k, (s, e) in enumerate(offs) if lo <= s < hi]
        reps[i] = np.stack([h[0][sel].float().mean(0).cpu().numpy() for h in hs])
    return np.stack([reps[i] for i in range(N)])

def spearman(x, y):
    return float(spearmanr(x, y).statistic)

t0 = time.time()
print(f"[1/7] C0 孤立编码 {N} 卡 …", flush=True)
R0 = encode_isolated(texts, model)                 # (n, L+1, d)
print(f"[2/7] C1 有序流 …", flush=True)
R1 = encode_stream(texts, list(range(N)), model)
R2s = []
for sd in SHUFFLE_SEEDS:
    print(f"[3/7] C2 打乱流 seed={sd} …", flush=True)
    rng = np.random.default_rng(sd)
    R2s.append(encode_stream(texts, rng.permutation(N).tolist(), model))
print(f"      编码完成 {time.time()-t0:.0f}s", flush=True)

R0l, R1l = R0[:, -1], R1[:, -1]
R2l = [R2[:, -1] for R2 in R2s]

# ---------- 时序方向向量 ----------
diff_ord = R1l - R0l                               # 逐卡 C1−C0
v_temporal = diff_ord.mean(0)
diff_shuf = np.mean([R2l_ - R0l for R2l_ in R2l], axis=0)
v_shuf = diff_shuf.mean(0)
v̂ = v_temporal / np.linalg.norm(v_temporal)
v̂_s = v_shuf / np.linalg.norm(v_shuf)
cos_vv = float(np.dot(v̂, v̂_s))

# 逐阶段平均偏移在 v̂ 上的投影 (切线应随阶单调)
proj_by_stage = [float((diff_ord[stages == s] @ v̂).mean())
                 for s in range(n_stages)]
proj_shuf_by_stage = [float((diff_shuf[stages == s] @ v̂).mean()) for s in range(n_stages)]

# ---------- K-1 探针 ----------
score_t = R0l @ v̂
score_s = R0l @ v̂_s
rho_probe_t = spearman(score_t, stages)
rho_probe_s = spearman(score_s, stages)
# 对照: PCA-PC1 (D5_baby 正本读法)
rho_pca0 = spearman(PCA(n_components=2, random_state=SEED_FIXED)
                    .fit_transform(R0l)[:, 0], stages)

# ---------- K-2 时间机器 ----------
centroids = np.stack([R0l[stages == s].mean(0) for s in range(n_stages)])
centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
def predict_rank(alpha):
    q = R0l + alpha * v̂
    qn = q / np.linalg.norm(q, axis=1, keepdims=True)
    sims = qn @ centroids.T                          # (n, n_stages)
    pred = sims.argmax(1)
    return pred, float(pred.mean())
ranks, rank_curve = [], []
for a in ALPHA_GRID:
    pred, mr = predict_rank(a)
    ranks.append(pred); rank_curve.append(mr)
rank_curve = [float(r) for r in rank_curve]
mono = all(rank_curve[i] <= rank_curve[i+1] + 1e-9 for i in range(len(ALPHA_GRID)-1))
span = rank_curve[-1] - rank_curve[0]
true_rank_mean = float(stages.mean())

# ---------- K-3 破 C42 (LoRA 表示偏移在 v̂ 上的投影) ----------
from peft import LoraConfig, get_peft_model

def delta_vector(m):
    def vals(c):
        return list(c.values()) if hasattr(c, "values") else list(c)
    vecs = []
    for mm in m.modules():
        if hasattr(mm, "lora_A"):
            s = mm.scaling
            s = s["default"] if isinstance(s, dict) else (s[0] if isinstance(s, (list, tuple)) else s)
            for a, b in zip(vals(mm.lora_A), vals(mm.lora_B)):
                d = (b.weight @ a.weight) * s
                vecs.append(d.detach().flatten().float().cpu())
    return torch.cat(vecs).numpy()

def train_lora(order, seed):
    torch.manual_seed(seed)
    base = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager").to(dev)
    cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=TARGETS,
                     lora_dropout=0.0, task_type="CAUSAL_LM")
    lm = get_peft_model(base, cfg)
    stream = "\n".join(texts[i] for i in order)
    ids = tok(stream, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
    blocks = ids[0].unfold(0, BLOCK, BLOCK).clone()
    opt = torch.optim.AdamW(lm.parameters(), lr=LR, betas=(0.9, ADAM_BETA2), weight_decay=0.0)
    lm.train()
    losses = []
    tt = time.time()
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
        print(f"      arm seed={seed} ep={ep+1}/{EPOCHS} loss={losses[-1]:.4f} ({time.time()-tt:.0f}s)", flush=True)
    lm.eval()
    return lm, delta_vector(lm), losses

print(f"[4/7] LoRA ord 臂 (seed={SEED_FIXED}) …", flush=True)
lm_ord, dw_ord, loss_ord = train_lora(list(range(N)), SEED_FIXED)
print(f"[5/7] LoRA shuf 臂 ×{len(SHUFFLE_SEEDS)} …", flush=True)
lms_shuf, dws_shuf, losses_shuf = [], [], []
for sd in SHUFFLE_SEEDS:
    rng = np.random.default_rng(sd)
    lm_s, dw_s, loss_s = train_lora(rng.permutation(N).tolist(), sd)
    lms_shuf.append(lm_s); dws_shuf.append(dw_s); losses_shuf.append(loss_s)

print(f"[6/7] LoRA 表示偏移投影 …", flush=True)
H_ord = encode_isolated(texts, lm_ord)[:, -1]
dH_ord = H_ord - R0l
proj_ord = float((dH_ord @ v̂).mean())
proj_ord_by_stage = [float((dH_ord[stages == s] @ v̂).mean()) for s in range(n_stages)]
proj_shuf, proj_shuf_by_stage_list = [], []
for lm_s in lms_shuf:
    H_s = encode_isolated(texts, lm_s)[:, -1]
    dH = H_s - R0l
    proj_shuf.append(float((dH @ v̂).mean()))
    proj_shuf_by_stage_list.append([float((dH[stages == s] @ v̂).mean()) for s in range(n_stages)])
proj_shuf_mean = float(np.mean(proj_shuf))
#  sanity: 权重空间夹角 (应与 C42 同读数 ≈88.5°, 证明两读法的可比性)
def angle_deg(v1, v2):
    c = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, c)))))
ang_ord_shuf = [angle_deg(dw_ord, d) for d in dws_shuf]

# ---------- 图 ----------
print(f"[7/7] 图 …", flush=True)
# K-2 时间机器: 平均预测阶随 α
fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
ax.plot(ALPHA_GRID, rank_curve, "o-", color="#21918c", lw=1.8, ms=5)
ax.axhline(true_rank_mean, color="#666666", lw=0.9, ls="--", label=f"真实平均阶 {true_rank_mean:.1f}")
ax.set_xlabel("干预强度 α (H_new = H_C0 + α·v̂)")
ax.set_ylabel("平均预测阶 (centroid argmax)")
ax.set_title("时间机器: 沿时序切线拨动表征")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_tdv_timemachine.png")); plt.close(fig)

# K-1 探针: 打分散点 (按阶着色)
STAGE_COLORS = ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"]
fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=150)
for s in range(n_stages):
    msk = stages == s
    ax.scatter(score_t[msk], np.zeros(msk.sum()) + np.random.default_rng(s).normal(0, 0.02, msk.sum()),
               c=STAGE_COLORS[s], s=46, alpha=0.9, label=stage_names[s], edgecolors="white", linewidths=0.5)
ax.set_xlabel("score = H_C0 · v̂_temporal")
ax.set_yticks([])
ax.set_title(f"K-1 探针: v̂ 打分恢复时序轴 (ρ = {rho_probe_t:+.3f})")
ax.legend(title="年龄阶", frameon=False, fontsize=9, title_fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_tdv_probe.png")); plt.close(fig)

# 逐阶投影 (切线方向性)
fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=150)
x = np.arange(n_stages)
w = 0.38
ax.bar(x - w/2, proj_by_stage, w, color="#21918c", label="C1−C0 (时序切线)")
ax.bar(x + w/2, proj_shuf_by_stage, w, color="#b8860b", label="C2−C0 (打乱对照)")
ax.axhline(0, color="#666666", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(stage_names, fontsize=9)
ax.set_ylabel("mean((H_cond − H_C0) · v̂)")
ax.set_title("切线投影的逐阶结构")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_tdv_stageproj.png")); plt.close(fig)

# K-3: LoRA 表示偏移投影 ord vs shuf
fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=150)
labels = ["ord\n(有序课程)"] + [f"shuf\n(seed {s})" for s in SHUFFLE_SEEDS]
vals = [proj_ord] + proj_shuf
ax.bar(labels, vals, color=["#21918c", "#b8860b", "#b8860b", "#b8860b"], edgecolor="#333333", linewidth=0.6)
ax.axhline(0, color="#666666", lw=0.8)
ax.set_ylabel("mean(ΔH · v̂_temporal)")
ax.set_title("K-3 破 C42: LoRA 表示偏移在时序切线上的投影")
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_tdv_lora_proj.png")); plt.close(fig)

# ---------- 判据裁决 ----------
k1 = bool(abs(rho_probe_t) >= 0.55 and abs(rho_probe_t) > abs(rho_probe_s))
k2 = bool(mono and abs(span) >= 1.0)
k3 = bool(proj_ord > proj_shuf_mean)

metrics = {
    "minted": time.strftime("%Y-%m-%d"),
    "experiment": "E-TDV temporal direction vector",
    "source": "Qwen 帖 chat-export-1791896688253.json (Transformer时序信息引入)",
    "corpus": {"deck": "D5_baby", "n": N, "stages": stage_names,
               "per_stage": [int((stages == s).sum()) for s in range(n_stages)]},
    "model": {"arch": "Qwen2ForCausalLM", "layers": L, "hidden": HID, "device": dev},
    "knobs": {"seed_fixed": SEED_FIXED, "shuffle_seeds": SHUFFLE_SEEDS,
              "alpha_grid": ALPHA_GRID, "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
              "epochs": EPOCHS, "lr": LR, "adam_beta2": ADAM_BETA2,
              "block": BLOCK, "targets": TARGETS},
    "vector": {
        "cos_vtemporal_vshuf": cos_vv,
        "proj_by_stage_temporal": proj_by_stage,
        "proj_by_stage_shufctrl": proj_shuf_by_stage,
    },
    "K1_probe": {
        "rho_vtemporal": rho_probe_t, "rho_vshuf_ctrl": rho_probe_s,
        "rho_pca_pc1_ref": rho_pca0, "rho_C0_ref": -0.621314874173432,
        "verdict": "时序切线成立" if k1 else "v̂ 不优于打乱对照 (伪影嫌疑)",
        "pass": k1,
    },
    "K2_timemachine": {
        "alpha_grid": ALPHA_GRID, "mean_predicted_rank": rank_curve,
        "true_mean_rank": true_rank_mean, "monotone": mono, "span": span,
        "predicted_ranks": [r.tolist() for r in ranks],
        "verdict": "时间机器成立" if k2 else "干预非单调或跨度不足",
        "pass": k2,
    },
    "K3_lora_projection": {
        "proj_ord": proj_ord, "proj_ord_by_stage": proj_ord_by_stage,
        "proj_shuf": proj_shuf, "proj_shuf_mean": proj_shuf_mean,
        "proj_shuf_by_stage": proj_shuf_by_stage_list,
        "losses_ord": loss_ord, "losses_shuf": losses_shuf,
        "angle_dw_ord_shuf": ang_ord_shuf,
        "angle_dw_note": "权重空间夹角复现 C42 读数 (≈88.5°), 与表示投影两读法并置",
        "verdict": "时序课程在表示层可辨" if k3 else "ord≈shuf: 此预算下不可辨 (与 C42 一致)",
        "pass": k3,
    },
    "elapsed_s": time.time() - t0,
}
json.dump(metrics, open(os.path.join(HERE, "metrics_temporal_direction.json"), "w"),
          ensure_ascii=False, indent=1)
print(json.dumps({
    "K1": metrics["K1_probe"], "K2": {"ranks": rank_curve, "mono": mono, "span": span},
    "K3": {"proj_ord": proj_ord, "proj_shuf_mean": proj_shuf_mean,
           "dw_angles": [round(a, 2) for a in ang_ord_shuf]},
    "elapsed_s": round(time.time() - t0),
}, ensure_ascii=False, indent=1))
print(f"DONE — metrics_temporal_direction.json + figs_tdv/")
