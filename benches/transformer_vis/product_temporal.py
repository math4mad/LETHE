#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
商品时序结构实验 —— 婴幼儿商品名语料（D5_baby）

问题: 把界限分明的商品卡按年龄阶排成流, transformer 的表征里读出时序结构吗?
      打乱顺序之后, 结构还显影吗? （dag06 C42 非交换律 · 编码级轻量版）

三条件:
  C0 孤立  — 每张卡单独编码 (基线: 世界知识里的时序)
  C1 有序  — 50 卡按 0-3月→2-3岁 拼成流, 因果上下文携带时序
  C2 打乱  — 同 50 卡, 三个 shuffle seed (14/15/16) 各跑一遍

读法: 末层 hidden states → 按卡切 span(mean-pool) → PCA + 逐卡 PC1 vs 年龄阶 Spearman ρ
      + 阶内/阶间相似度比 + 顺序效应 (与 C0 的 cosine 距离)
产物: figs/*.png + metrics_stage.json （供 report_stage.qmd 引用）

正本: 本脚本 (源of truth); 派生: figs/ + metrics_stage.json。
复现: /opt/miniconda3/envs/default/bin/python product_temporal.py  (退出码 0 即过)
"""

import json, os, sys, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.stats import spearmanr

# ---------- 旋钮区 ----------
HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "corpus_baby.json")
MODEL = os.path.expanduser("~/Programming/code-2026/first-lora/model")
SEEDS_SHUFFLE = [14, 15, 16]   # 打乱条件的三个 seed
SEED_FIXED = 13
corp = json.load(open(CORPUS))
DECK = corp["deck"]
FIGS = os.path.join(HERE, f"figs_{DECK}"); os.makedirs(FIGS, exist_ok=True)
# ------------------------------

torch.manual_seed(SEED_FIXED); np.random.seed(SEED_FIXED)

# 中文字体
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

# ---------- 语料 ----------
items = corp["items"]
stages = np.array([it["stage"] for it in items])
n_stages = len(corp["stages"])
stage_names = corp["stages"]
texts = [f"{it['name']}：{it['desc']}。" for it in items]
N = len(texts)
STAGE_COLORS = ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"]

# ---------- 模型 ----------
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager")
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(dev).eval()
L = model.config.num_hidden_layers

def all_hidden(ids):
    with torch.no_grad():
        out = model(**ids, output_hidden_states=True)
    return out.hidden_states                 # tuple (L+1, (1, T, d))

def encode_isolated(texts):
    """C0: 逐卡孤立编码, 全层 mean-pool → (n, L+1, d)"""
    reps = []
    for t in texts:
        ids = tok(t, return_tensors="pt", add_special_tokens=False).to(dev)
        hs = all_hidden(ids)
        reps.append(np.stack([h[0].float().mean(0).cpu().numpy() for h in hs]))
    return np.stack(reps)

def encode_stream(texts, order):
    """C1/C2: 按 order 拼流, 用 offset mapping 把 token 切回每张卡, 全层 mean-pool → (n, L+1, d)"""
    stream = "\n".join(texts[i] for i in order)
    # 每张卡在流中的字符区间
    spans, pos = {}, 0
    for i in order:
        t = texts[i]
        spans[i] = (pos, pos + len(t))
        pos += len(t) + 1                    # +1 = '\n'
    enc = tok(stream, return_tensors="pt", return_offsets_mapping=True,
              add_special_tokens=False).to(dev)
    offs = enc.pop("offset_mapping")[0].tolist()
    hs = all_hidden(enc)
    reps = {}
    for i in order:
        lo, hi = spans[i]
        sel = [k for k, (s, e) in enumerate(offs) if s >= lo and e <= hi and e > s]
        if not sel:                          # 退化为邻域, 不该发生
            sel = [k for k, (s, e) in enumerate(offs) if lo <= s < hi]
        reps[i] = np.stack([h[0][sel].float().mean(0).cpu().numpy() for h in hs])
    return np.stack([reps[i] for i in range(N)])

t0 = time.time()
print(f"[1/6] C0 孤立编码 {N} 卡 …", flush=True)
R0 = encode_isolated(texts)                 # (n, L+1, d)
print(f"[2/6] C1 有序流 …", flush=True)
R1 = encode_stream(texts, list(range(N)))   # (n, L+1, d)
R2s = []
for sd in SEEDS_SHUFFLE:
    print(f"[3/6] C2 打乱流 seed={sd} …", flush=True)
    rng = np.random.default_rng(sd)
    R2s.append(encode_stream(texts, rng.permutation(N).tolist()))
print(f"      编码完成 {time.time()-t0:.0f}s", flush=True)

# ---------- 读数 ----------
def pc1_and_rho(R):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=SEED_FIXED).fit_transform(R)
    rho, p = spearmanr(pca[:, 0], stages)
    return pca, float(rho), float(p)

def stage_geometry(R):
    """阶内/阶间 相似度比 (末层 cosine)"""
    S = R @ R.T / np.outer(np.linalg.norm(R, axis=1), np.linalg.norm(R, axis=1))
    iw, ib, nw, nb = 0.0, 0.0, 0, 0
    for a in range(N):
        for b in range(a + 1, N):
            if stages[a] == stages[b]:
                iw += S[a, b]; nw += 1
            else:
                ib += S[a, b]; nb += 1
    return S, float(iw / nw), float(ib / nb)

print("[4/6] 读数 …", flush=True)
# 末层视图 (n, d) —— 投影/几何/顺序效应全在末层
R0l, R1l = R0[:, -1], R1[:, -1]
R2l = [R2[:, -1] for R2 in R2s]
pca0, rho0, p0 = pc1_and_rho(R0l)
pca1, rho1, p1 = pc1_and_rho(R1l)
pca2s, rho2s = [], []
for R2l_ in R2l:
    pca2, rho2, _ = pc1_and_rho(R2l_)
    pca2s.append(pca2); rho2s.append(rho2)
rho2_mean, rho2_sd = float(np.mean(rho2s)), float(np.std(rho2s))

S1, iw1, ib1 = stage_geometry(R1l)
S0, iw0, ib0 = stage_geometry(R0l)

def order_effect(R):
    """与 C0 的 cosine 距离, 按阶"""
    d = 1 - (R * R0l).sum(1) / (np.linalg.norm(R, axis=1) * np.linalg.norm(R0l, axis=1))
    return d

oe1 = order_effect(R1l)
oe2 = np.mean([order_effect(R2l_) for R2l_ in R2l], axis=0)

# 逐层 ρ (射击 2): 时序轴在网络哪一层显影
NL = R0.shape[1]
rho0_per = [pc1_and_rho(R0[:, li])[1] for li in range(NL)]
rho1_per = [pc1_and_rho(R1[:, li])[1] for li in range(NL)]
rho2_per = np.array([[pc1_and_rho(R2[:, li])[1] for li in range(NL)] for R2 in R2s])

# ---------- 图 ----------
print("[5/6] 图 …", flush=True)

# 逐层 ρ 曲线 (射击 2)
fig, ax = plt.subplots(figsize=(6.8, 4.6), dpi=150)
lay = np.arange(NL)
ax.plot(lay, -np.array(rho0_per), "o-", color="#3b528b", lw=1.6, ms=3, label="C0 孤立")
ax.plot(lay, -np.array(rho1_per), "s-", color="#21918c", lw=1.6, ms=3, label="C1 有序流")
ax.plot(lay, -rho2_per.mean(0), "^-", color="#b8860b", lw=1.6, ms=3, label="C2 打乱流 (3 seeds)")
ax.fill_between(lay, -(rho2_per.mean(0) + rho2_per.std(0)), -(rho2_per.mean(0) - rho2_per.std(0)), color="#b8860b", alpha=0.18)
ax.axhline(0, color="#666666", lw=0.8)
ax.set_xlabel("layer (0 = embedding, L = last)")
ax.set_ylabel("−Spearman ρ（PC1 vs 年龄阶）")
ax.set_title("时序轴的逐层显影（ρ 取反仅 PCA 轴符号约定）")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig7_layer_rho.png")); plt.close(fig)

def scatter_stage(xy, fname, title, subtitle=""):
    fig, ax = plt.subplots(figsize=(6.6, 5.4), dpi=150)
    for s in range(n_stages):
        m = stages == s
        ax.scatter(xy[m, 0], xy[m, 1], c=STAGE_COLORS[s], s=46, alpha=0.9,
                   label=stage_names[s], edgecolors="white", linewidths=0.5)
    ax.set_title(title + (f"\n{subtitle}" if subtitle else ""), fontsize=11)
    ax.legend(title="年龄阶", frameon=False, fontsize=9, title_fontsize=9)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    for sp in ax.spines.values(): sp.set_color("#cccccc")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, fname)); plt.close(fig)

scatter_stage(pca0, "fig1_isolated.png", "C0 孤立编码 — 商品时序知识")
scatter_stage(pca1, "fig2_ordered.png", "C1 有序流 — 时序轴",
              f"Spearman ρ(PC1, 年龄阶) = {rho1:+.2f}")
scatter_stage(pca2s[0], "fig3_shuffled.png", "C2 打乱流 (seed=14)",
              f"Spearman ρ = {rho2s[0]:+.2f}")

# ρ 对比条形图
fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=150)
labels = ["C0 孤立", "C1 有序流", "C2 打乱流\n(3 seeds 均值)"]
vals = [-rho0, -rho1, -rho2_mean]          # 取反: 读作「年龄越大 PC1 越大」
errs = [0, 0, rho2_sd]
bars = ax.bar(labels, vals, yerr=errs, capsize=5,
              color=["#3b528b", "#21918c", "#fde725"], edgecolor="#333333", linewidth=0.6)
ax.axhline(0, color="#666666", lw=0.8)
ax.set_ylabel("−Spearman ρ（PC1 vs 年龄阶）")
ax.set_ylim(-0.05, 0.95)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:+.2f}", ha="center", fontsize=10)
ax.set_title("时序轴显影强度（ρ 取反仅正 PCA 轴符号约定）")
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig4_rho.png")); plt.close(fig)

# 顺序效应
fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=150)
for s in range(n_stages):
    m = stages == s
    ax.scatter(oe1[m], oe2[m], c=STAGE_COLORS[s], s=46, alpha=0.9, label=stage_names[s],
               edgecolors="white", linewidths=0.5)
lim = [0, max(oe1.max(), oe2.max()) * 1.1]
ax.plot(lim, lim, "--", color="#666666", lw=0.9)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("cos 距离: 有序流 vs 孤立 (C1−C0)")
ax.set_ylabel("cos 距离: 打乱流 vs 孤立 (C2−C0)")
ax.set_title("顺序效应：上下文如何移动每张卡")
ax.legend(title="年龄阶", frameon=False, fontsize=9, title_fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig5_order_effect.png")); plt.close(fig)

# 阶×阶相似度矩阵 (有序流)
fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=150)
M = np.zeros((n_stages, n_stages))
for a in range(n_stages):
    for b in range(n_stages):
        ia, ib = stages == a, stages == b
        M[a, b] = S1[np.ix_(ia, ib)].mean()
im = ax.imshow(M, cmap="viridis")
ax.set_xticks(range(n_stages)); ax.set_xticklabels(stage_names, fontsize=9)
ax.set_yticks(range(n_stages)); ax.set_yticklabels(stage_names, fontsize=9)
for a in range(n_stages):
    for b in range(n_stages):
        ax.text(b, a, f"{M[a,b]:.2f}", ha="center", va="center",
                color="white" if M[a, b] < 0.6 else "black", fontsize=9)
ax.set_title("有序流 · 阶×阶 相似度矩阵")
fig.colorbar(im, shrink=0.85)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig6_stage_matrix.png")); plt.close(fig)

# ---------- 落账 ----------
metrics = {
    "minted": time.strftime("%Y-%m-%d"),
    "corpus": {"deck": "D5_baby", "n": N, "stages": stage_names,
               "per_stage": [int((stages == s).sum()) for s in range(n_stages)],
               "note": "描述不含年龄词; 时序只能靠模型世界知识"},
    "model": {"arch": "Qwen2ForCausalLM", "layers": L, "hidden": model.config.hidden_size,
              "device": dev},
    "knobs": {"seed_fixed": SEED_FIXED, "shuffle_seeds": SEEDS_SHUFFLE},
    "rho": {"C0_isolated": rho0, "C0_p": p0, "C1_ordered": rho1, "C1_p": p1,
            "C2_shuffled": rho2s, "C2_mean": rho2_mean, "C2_sd": rho2_sd},
    "rho_per_layer": {"C0": rho0_per, "C1": rho1_per,
                        "C2": rho2_per.tolist(), "C2_mean": rho2_per.mean(0).tolist()},
    "geometry": {"C0_within": iw0, "C0_between": ib0,
                 "C1_within": iw1, "C1_between": ib1},
    "order_effect_mean": {"C1": float(oe1.mean()), "C2": float(oe2.mean())},
    "order_effect_by_stage": {"C1": [float(oe1[stages == s].mean()) for s in range(n_stages)],
                              "C2": [float(oe2[stages == s].mean()) for s in range(n_stages)]},
    "elapsed_s": time.time() - t0,
}
json.dump(metrics, open(os.path.join(HERE, f"metrics_{DECK}.json"), "w"),
          ensure_ascii=False, indent=1)
print(f"[6/6] DONE in {time.time()-t0:.0f}s — figs_{DECK}/ + metrics_{DECK}.json")
print(json.dumps({"rho_C0": rho0, "rho_C1": rho1, "rho_C2_mean": rho2_mean,
                  "geom_C1": [iw1, ib1], "oe": [float(oe1.mean()), float(oe2.mean())]},
                 ensure_ascii=False))
