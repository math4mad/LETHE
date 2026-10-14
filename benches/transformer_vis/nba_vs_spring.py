#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NBA vs 春节 —— transformer 表征结构差异可视化（P-B19 四舱语料之 D1_nba × D4_spring）

语料: chora/experiments/pb19-mrd/pack.b64 → corps.D1_nba (选秀编年卡) / corps.D4_spring (春节舱)
模型: first-lora/model (Qwen2.5-0.5B-Instruct, 本地盘, 免下载)
读法: 全层 hidden states 抽取 → PCA/t-SNE/UMAP 投影 + 逐层逻辑回归探针 + 逐层注意力熵
产物: figs/*.png + metrics.json （供 report.qmd 引用）

园规: 每个数一行意愿（见下方旋钮区）。跑法:
  /opt/miniconda3/envs/default/bin/python nba_vs_spring.py
"""

import base64, json, os, sys, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- 旋钮区（园规·参数意愿律: 每个数一行意愿, 不许"都这么设"） ----------
PACK   = os.path.expanduser("~/Programming/code-2026/chora/experiments/pb19-mrd/pack.b64")
MODEL  = os.path.expanduser("~/Programming/code-2026/first-lora/model")
DECK_A = "D1_nba"        # 舱A: NBA 选秀编年卡（内容×顺序 的二维编年结构）
DECK_B = "D4_spring"     # 舱B: 春节舱（习俗—吉祥话语义网）
SEED   = 13              # 园子第一号 seed
TSNE_PERPLEXITY = 15     # t-SNE 困惑度
PROBE_FOLDS = 5          # 探针交叉验证折数
UMAP_N = 15              # UMAP 近邻数
# ---------------------------------------------------------------------------

torch.manual_seed(SEED); np.random.seed(SEED)
HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs"); os.makedirs(FIGS, exist_ok=True)

# ---------- 语料 ----------
pack = json.loads(base64.b64decode(open(PACK, "rb").read()))
def deck_texts(name):
    return [json.loads(ln)["messages"][1]["content"] for ln in pack["corps"][name]]
A_texts, B_texts = deck_texts(DECK_A), deck_texts(DECK_B)

# ---------- 中文字体探测（无 CJK 则图内一律英文, 免豆腐块） ----------
from matplotlib import font_manager
CJK = None
for cand in ("PingFang SC", "Arial Unicode MS", "STHeiti", "Songti SC",
             "Heiti TC", "Noto Sans CJK SC", "Hiragino Sans GB"):
    try:
        f = font_manager.findfont(cand, fallback_to_default=False)
        if "DejaVu" not in f:
            CJK = cand; plt.rcParams["font.family"] = cand
            plt.rcParams["axes.unicode_minus"] = False
            break
    except Exception:
        pass
LBL_A = "NBA draft cards" + ("（选秀编年卡）" if CJK else "")
LBL_B = "Spring Festival" + ("（春节）" if CJK else "")

# ---------- 模型 ----------
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager")
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(dev).eval()
L = model.config.num_hidden_layers

def encode(texts):
    """逐条前向: 返回 (n, L+1, d) 全层 mean-pool 表征 与 (n, L) 逐层注意力熵"""
    pooled, ents = [], []
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt").to(dev)
            out = model(**ids, output_hidden_states=True, output_attentions=True)
            pooled.append(np.stack([h[0].mean(0).float().cpu().numpy() for h in out.hidden_states]))
            layer_ent = []
            for att in out.attentions:                      # (1, H, T, T)
                a = att[0].float().cpu()
                layer_ent.append(float((-(a * (a + 1e-12).log()).sum(-1)).mean()))
            ents.append(layer_ent)
    return np.stack(pooled), np.stack(ents)

t0 = time.time()
print(f"[1/5] 编码 {DECK_A}={len(A_texts)} 条 / {DECK_B}={len(B_texts)} 条 …", flush=True)
Xa, Ea = encode(A_texts)
Xb, Eb = encode(B_texts)
print(f"      编码完成 {time.time()-t0:.0f}s · 层数 L+1={Xa.shape[1]} · 隐维 d={Xa.shape[2]}", flush=True)

# ---------- 投影（末层表征） ----------
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
X = np.vstack([Xa[:, -1], Xb[:, -1]])
y = np.array([0]*len(A_texts) + [1]*len(B_texts))
pca = PCA(n_components=2, random_state=SEED).fit_transform(X)
tsne = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, random_state=SEED,
            init="pca", learning_rate="auto").fit_transform(X)
umap_xy = None
try:
    import umap
    umap_xy = umap.UMAP(n_neighbors=UMAP_N, random_state=SEED).fit_transform(X)
except ImportError:
    print("      umap-learn 未装 — 跳过 UMAP 图", flush=True)

def scatter(xy, fname, title):
    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=150)
    for xy_, c, m, lbl in ((xy[y == 0], "#d1495b", "o", LBL_A), (xy[y == 1], "#2e86ab", "s", LBL_B)):
        ax.scatter(xy_[:, 0], xy_[:, 1], c=c, marker=m, s=42, alpha=0.85, label=lbl, edgecolors="white", linewidths=0.4)
    ax.set_title(title, fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color("#cccccc")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, fname)); plt.close(fig)

print("[2/5] 投影图 …", flush=True)
scatter(pca, "fig1_pca.png", "Last-layer representations — PCA")
scatter(tsne, "fig2_tsne.png", f"Last-layer representations — t-SNE (perplexity={TSNE_PERPLEXITY})")
if umap_xy is not None:
    scatter(umap_xy, "fig3_umap.png", f"Last-layer representations — UMAP (k={UMAP_N})")

# ---------- 逐层探针 + 质心几何 ----------
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics.pairwise import cosine_similarity
print("[3/5] 逐层探针 …", flush=True)
probe_acc, cent_dist, wb_ratio = [], [], []
for li in range(Xa.shape[1]):
    Xl = np.vstack([Xa[:, li], Xb[:, li]])
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    acc = cross_val_score(clf, Xl, y, cv=PROBE_FOLDS).mean()
    probe_acc.append(float(acc))
    ca, cb = Xa[:, li].mean(0), Xb[:, li].mean(0)
    cent_dist.append(float(1 - np.dot(ca, cb) / (np.linalg.norm(ca) * np.linalg.norm(cb))))
    Wa = cosine_similarity(Xa[:, li]); Wb = cosine_similarity(Xb[:, li])
    Wab = cosine_similarity(Xa[:, li], Xb[:, li])
    iw = (Wa.sum() - len(A_texts) + Wb.sum() - len(B_texts)) / (len(A_texts)*(len(A_texts)-1) + len(B_texts)*(len(B_texts)-1))
    ib = Wab.mean()
    wb_ratio.append(float(iw / (ib + 1e-12)))

fig, ax1 = plt.subplots(figsize=(6.6, 4.4), dpi=150)
layers = np.arange(Xa.shape[1])
ax1.plot(layers, probe_acc, "o-", color="#d1495b", lw=1.8, ms=3.5, label="probe accuracy (5-fold)")
ax1.axhline(0.5, color="#999999", ls="--", lw=0.8)
ax1.set_xlabel("layer (0 = embedding, L = last)"); ax1.set_ylabel("probe accuracy", color="#d1495b")
ax1.tick_params(axis="y", labelcolor="#d1495b"); ax1.set_ylim(0.35, 1.02)
ax2 = ax1.twinx()
ax2.plot(layers, cent_dist, "s-", color="#2e86ab", lw=1.8, ms=3.5, label="centroid cosine distance")
ax2.set_ylabel("centroid cosine distance", color="#2e86ab"); ax2.tick_params(axis="y", labelcolor="#2e86ab")
ax1.set_title("Where does the network separate the two cabins?")
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8.5, loc="center right")
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig4_layer_probe.png")); plt.close(fig)

# ---------- 逐层注意力熵 ----------
print("[4/5] 注意力熵 …", flush=True)
fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=150)
la = np.arange(Ea.shape[1])
ax.plot(la, Ea.mean(0), "o-", color="#d1495b", lw=1.8, ms=3.5, label=LBL_A)
ax.fill_between(la, Ea.mean(0) - Ea.std(0), Ea.mean(0) + Ea.std(0), color="#d1495b", alpha=0.15)
ax.plot(la, Eb.mean(0), "s-", color="#2e86ab", lw=1.8, ms=3.5, label=LBL_B)
ax.fill_between(la, Eb.mean(0) - Eb.std(0), Eb.mean(0) + Eb.std(0), color="#2e86ab", alpha=0.15)
ax.set_xlabel("layer"); ax.set_ylabel("mean attention entropy (nats)")
ax.set_title("Attention sharpness per layer")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig5_attn_entropy.png")); plt.close(fig)

# ---------- 末层相似度分布 ----------
Wa = cosine_similarity(Xa[:, -1]); Wb = cosine_similarity(Xb[:, -1]); Wab = cosine_similarity(Xa[:, -1], Xb[:, -1])
iu = lambda M: M[np.triu_indices(len(M), 1)]
sim_a, sim_b, sim_ab = iu(Wa), iu(Wb), Wab.ravel()
fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=150)
bins = np.linspace(0, 1, 41)
ax.hist(sim_a, bins=bins, color="#d1495b", alpha=0.65, label=f"{LBL_A} within", density=True)
ax.hist(sim_b, bins=bins, color="#2e86ab", alpha=0.65, label=f"{LBL_B} within", density=True)
ax.hist(sim_ab, bins=bins, color="#555555", alpha=0.45, label="NBA↔春节 between", density=True, histtype="step", lw=2)
ax.axvline(sim_a.mean(), color="#d1495b", ls="--", lw=1)
ax.axvline(sim_b.mean(), color="#2e86ab", ls="--", lw=1)
ax.axvline(sim_ab.mean(), color="#555555", ls="--", lw=1)
ax.set_xlabel("cosine similarity (last layer)"); ax.set_ylabel("density")
ax.set_title("Within-cabin vs between-cabin geometry")
ax.legend(frameon=False, fontsize=8.5)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig6_sim.png")); plt.close(fig)

# ---------- 落账 ----------
metrics = {
    "minted": time.strftime("%Y-%m-%d"),
    "model": {"path": MODEL, "arch": model.config.architectures[0],
              "layers": L, "hidden": model.config.hidden_size,
              "heads": model.config.num_attention_heads, "dtype": "bf16-on-disk",
              "device": dev},
    "corpora": {DECK_A: {"n": len(A_texts),
                         "chars_mean": float(np.mean([len(t) for t in A_texts])),
                         "tokens_mean": float(np.mean([len(tok(t)["input_ids"]) for t in A_texts]))},
                DECK_B: {"n": len(B_texts),
                         "chars_mean": float(np.mean([len(t) for t in B_texts])),
                         "tokens_mean": float(np.mean([len(tok(t)["input_ids"]) for t in B_texts]))}},
    "knobs": {"seed": SEED, "tsne_perplexity": TSNE_PERPLEXITY, "probe_folds": PROBE_FOLDS, "umap_n": UMAP_N},
    "cjk_font": CJK,
    "probe_acc_per_layer": probe_acc,
    "probe_acc_max_layer": int(np.argmax(probe_acc)),
    "probe_acc_max": float(max(probe_acc)),
    "centroid_distance_per_layer": cent_dist,
    "within_over_between_per_layer": wb_ratio,
    "attn_entropy_mean_per_layer": {"A": Ea.mean(0).tolist(), "B": Eb.mean(0).tolist()},
    "sim_last_layer": {"within_A_mean": float(sim_a.mean()), "within_B_mean": float(sim_b.mean()),
                       "between_mean": float(sim_ab.mean())},
    "elapsed_s": time.time() - t0,
}
json.dump(metrics, open(os.path.join(HERE, "metrics.json"), "w"), ensure_ascii=False, indent=1)
print(f"[5/5] DONE in {time.time()-t0:.0f}s — figs/ + metrics.json")
print(json.dumps({k: metrics[k] for k in ("probe_acc_max_layer", "probe_acc_max", "sim_last_layer")}, ensure_ascii=False))
