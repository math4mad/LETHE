#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Friends 季节演化 —— Rachel 的特色是否在 S03/S04 才凝聚？

主人观感: "到了第三季和第四季, Rachel 才表现出特色"
假设: Rachel 在 S03/S04 的台词表示更凝聚(舱内相似度↑)、与其余角色更分离(质心距↑)
对照: 若六角色同指标随季齐升 = 季间语料漂移, 非 Rachel 特异

语料: corpus_friends_s134.json (S01/S03/S04 × 六角色 × 50 条)
      S02 缺失 — fangj 站该季页面残缺(多集为占位页/404), 如实标注
模型: first-lora/model (Qwen2.5-0.5B-Instruct)

量法 (沿用 friends_attractor.py):
  [1] 孤立编码 900 条台词
  [2] 逐季角色几何: 舱内相似度 / 质心距离 / 舱内跨角色比
  [3] Rachel 时间线: 舱内相似 + 质心均距 (S01→S03→S04), 对照其余五角色
  [4] 逐季探针 (6 类, sanity check)
  [5] 对话流: S04 开头流 (对照 S01 流)
  [6] 投影: t-SNE 季着色

产物: figs_friends_s134/*.png + metrics_friends_s134.json (供 report_friends_s134.qmd)

园规: 每个数一行意愿 (旋钮区)。跑法:
  /opt/miniconda3/envs/default/bin/python friends_seasonal.py
"""

import json, os, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- 旋钮区（园规·参数意愿律: 每个数一行愿望, 不许"都这么设"） ----------
CORPUS = "corpus_friends_s1234.json"  # S01-S04 四季正本
RAW    = "friends_s1234_raw.json"     # 对话流用
MODEL  = os.path.expanduser("~/Programming/code-2026/first-lora/model")
SEED   = 13              # 园子第一号 seed
TSNE_PERPLEXITY = 15     # t-SNE 困惑度
PROBE_FOLDS = 5          # 探针交叉验证折数
FLOW_MAX_TOKENS = 1500   # 流编码 token 上限 (内存保护)
LEAD = "Rachel"          # 主人点名的角色
# ---------------------------------------------------------------------------

torch.manual_seed(SEED); np.random.seed(SEED)
HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs_friends_s1234"); os.makedirs(FIGS, exist_ok=True)

# ---------- 语料 ----------
corpus = json.load(open(os.path.join(HERE, CORPUS)))
ROLES = corpus["roles"]                      # [Rachel, Chandler, Ross, Monica, Joey, Phoebe]
SEASONS = corpus["seasons"]                  # [1, 3, 4]
texts = [it["desc"] for it in corpus["items"]]
role_of = [it["stage"] for it in corpus["items"]]
season_of = [it["season"] for it in corpus["items"]]
n = len(texts)

# ---------- 中文字体探测 ----------
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
SEASON_COLORS = {1: "#2e86ab", 2: "#6a994e", 3: "#f4a259", 4: "#d1495b"}
ROLE_COLORS = ["#d1495b", "#2e86ab", "#f4a259", "#8ac926", "#9b5de5", "#00b4d8"]

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
            for att in out.attentions:
                a = att[0].float().cpu()
                layer_ent.append(float((-(a * (a + 1e-12).log()).sum(-1)).mean()))
            ents.append(layer_ent)
    return np.stack(pooled), np.stack(ents)

t0 = time.time()
print(f"[1/6] 孤立编码 {n} 条 ({len(SEASONS)} 季 × 6 角色 × 50) …", flush=True)
X, E = encode(texts)
print(f"      编码完成 {time.time()-t0:.0f}s · 层数 L+1={X.shape[1]}", flush=True)

# ---------- 逐季角色几何 ----------
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

print("[2/6] 逐季角色几何 …", flush=True)
season_metrics = {}
for si, season in enumerate(SEASONS):
    m_season = np.array(season_of) == season
    Xs = X[m_season]
    Es = E[m_season]
    roles_s = np.array(role_of)[m_season]
    within = {r: [] for r in ROLES}
    ratio = {r: [] for r in ROLES}
    cents_last = []
    probe_acc = []
    for li in range(X.shape[1]):
        Xl = Xs[:, li]
        W = cosine_similarity(Xl)
        for ri, role in enumerate(ROLES):
            m = roles_s == ri
            iw = W[np.ix_(m, m)]
            win = iw[np.triu_indices(m.sum(), 1)].mean()
            ib = W[np.ix_(m, ~m)].mean()
            within[role].append(float(win))
            ratio[role].append(float(win / (ib + 1e-12)))
        if li == X.shape[1] - 1:
            cents_last = np.stack([Xl[roles_s == ri].mean(0) for ri in range(6)])
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        probe_acc.append(float(cross_val_score(
            clf, Xl, roles_s, cv=PROBE_FOLDS).mean()))
    cent_dist = 1 - cosine_similarity(cents_last)   # 余弦距离矩阵
    # Rachel 与其余角色质心均距 (逐层)
    rachel_cent = Xs[:, -1][roles_s == 0].mean(0)
    other_cents = np.stack([Xs[:, -1][roles_s == ri].mean(0)
                            for ri in range(1, 6)])
    rachel_cent_dist_last = float(np.mean([
        1 - np.dot(rachel_cent, c) / (np.linalg.norm(rachel_cent) * np.linalg.norm(c))
        for c in other_cents]))
    season_metrics[season] = {
        "within": within, "ratio": ratio,
        "probe_acc_per_layer": probe_acc,
        "probe_acc_max": float(max(probe_acc)),
        "probe_acc_max_layer": int(np.argmax(probe_acc)),
        "centroid_distance_matrix": cent_dist.tolist(),
        "rachel_centroid_dist_last": rachel_cent_dist_last,
        "attn_entropy": {r: Es[roles_s == ri].mean(0).tolist()
                         for ri, r in enumerate(ROLES)},
        "attn_entropy_last": {r: float(Es[roles_s == ri][:, -1].mean())
                              for ri, r in enumerate(ROLES)},
    }

# ---------- Rachel 时间线 ----------
print("[3/6] Rachel 时间线 …", flush=True)
lead_within = [season_metrics[s]["within"][LEAD][-1] for s in SEASONS]
lead_centdist = [season_metrics[s]["rachel_centroid_dist_last"] for s in SEASONS]
others_within_mean = [float(np.mean([season_metrics[s]["within"][r][-1]
                                     for r in ROLES if r != LEAD])) for s in SEASONS]
others_centdist_mean = [float(np.mean([
    1 - season_metrics[s]["centroid_distance_matrix"][0][ri]
    for ri in range(1, 6)])) for s in SEASONS]
# 探针: Rachel 单独 vs 其余 (二类, 逐季末层)
probe_lead_vs_rest = []
for season in SEASONS:
    m_season = np.array(season_of) == season
    Xs = X[m_season][:, -1]
    roles_s = np.array(role_of)[m_season]
    y = (roles_s == 0).astype(int)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    probe_lead_vs_rest.append(float(cross_val_score(
        clf, Xs, y, cv=PROBE_FOLDS).mean()))

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), dpi=150)
xs = np.arange(len(SEASONS))
axes[0].plot(xs, lead_within, "o-", color="#d1495b", lw=2, ms=7,
             label="Rachel 舱内相似")
axes[0].plot(xs, others_within_mean, "s--", color="#2e86ab", lw=1.6, ms=6,
             label="其余五角色均值")
axes[0].set_xticks(xs); axes[0].set_xticklabels([f"S{s:02d}" for s in SEASONS])
axes[0].set_ylabel("within-role cosine (last layer)")
axes[0].set_title("Rachel 台词凝聚力随季变化")
axes[0].legend(frameon=False, fontsize=9)
for i, v in enumerate(lead_within):
    axes[0].annotate(f"{v:.4f}", (xs[i], v), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8)
axes[1].plot(xs, lead_centdist, "o-", color="#d1495b", lw=2, ms=7,
             label="Rachel ↔ 其余质心距")
axes[1].plot(xs, others_centdist_mean, "s--", color="#2e86ab", lw=1.6, ms=6,
             label="其余角色间均距")
axes[1].set_xticks(xs); axes[1].set_xticklabels([f"S{s:02d}" for s in SEASONS])
axes[1].set_ylabel("centroid cosine distance")
axes[1].set_title("Rachel 与其余角色分离度随季变化")
axes[1].legend(frameon=False, fontsize=9)
for i, v in enumerate(lead_centdist):
    axes[1].annotate(f"{v:.4f}", (xs[i], v), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_s1_rachel_timeline.png"))
plt.close(fig)

# ---------- 逐季质心热图 + 探针 ----------
print("[4/6] 质心热图 + 探针 …", flush=True)
fig, axes = plt.subplots(1, len(SEASONS), figsize=(4.6 * len(SEASONS), 4.2), dpi=150)
for si, season in enumerate(SEASONS):
    ax = axes[si]
    cm = np.array(season_metrics[season]["centroid_distance_matrix"])
    im = ax.imshow(cm, cmap="viridis", vmin=0, vmax=cm.max())
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_xticklabels([r[:6] for r in ROLES], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([r[:6] for r in ROLES], fontsize=8)
    ax.set_title(f"S{season:02d} 质心距离", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_s2_heatmaps.png")); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=150)
la = np.arange(X.shape[1])
for season in SEASONS:
    ax.plot(la, season_metrics[season]["probe_acc_per_layer"], "o-",
            color=SEASON_COLORS[season], lw=1.8, ms=3.5, label=f"S{season:02d}")
ax.axhline(1 / 6, color="#999999", ls="--", lw=0.8)
ax.set_xlabel("layer (0 = embedding, L = last)")
ax.set_ylabel("6-class probe accuracy (5-fold)")
ax.set_title("角色探针逐层（sanity check — 词汇域不暴露说话人）")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_s3_probe.png")); plt.close(fig)

# ---------- t-SNE 季着色 ----------
print("[5/6] 投影 …", flush=True)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
Xl = X[:, -1]
tsne = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, random_state=SEED,
            init="pca", learning_rate="auto").fit_transform(Xl)
fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=150)
for season in SEASONS:
    m = np.array(season_of) == season
    ax.scatter(tsne[m, 0], tsne[m, 1], c=SEASON_COLORS[season], s=38,
               alpha=0.8, label=f"S{season:02d}", edgecolors="white", linewidths=0.4)
ax.set_title("末层表征 — t-SNE (季着色)")
ax.legend(frameon=False, fontsize=9)
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_color("#cccccc")
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_s4_tsne.png")); plt.close(fig)

# ---------- 对话流: S04 开头 ----------
print(f"[6/6] 对话流编码 (S04 开头) …", flush=True)
raw = json.load(open(os.path.join(HERE, RAW)))
s04_lines = raw["data"]["S04"]["lines"]
line_ids, line_meta = [], []
for sp_line in s04_lines:
    sp = sp_line["speaker"]
    if sp not in ROLES:
        continue
    ids = tok(sp_line["line"], add_special_tokens=False)["input_ids"]
    if len(line_ids) + len(ids) > FLOW_MAX_TOKENS:
        break
    line_ids.extend(ids)
    line_meta.append((sp, len(line_ids) - 1))
flow_ids = torch.tensor([line_ids], device=dev)
with torch.no_grad():
    out = model(input_ids=flow_ids, output_hidden_states=True, output_attentions=True)
flow_sent = []
for li in range(X.shape[1]):
    h = out.hidden_states[li][0]
    ends = [e for _, e in line_meta]
    flow_sent.append(h[ends].float().cpu().numpy())
flow_sent = np.stack(flow_sent)
flow_sp = [sp for sp, _ in line_meta]
fs = flow_sent[-1]
adj_same, adj_diff = [], []
for i in range(len(line_meta) - 1):
    sim = float(np.dot(fs[i], fs[i + 1]) / (np.linalg.norm(fs[i]) * np.linalg.norm(fs[i + 1])))
    (adj_same if flow_sp[i] == flow_sp[i + 1] else adj_diff).append(sim)
flow_ent = []
for att in out.attentions:
    a = att[0].float().cpu()
    flow_ent.append(float((-(a * (a + 1e-12).log()).sum(-1)).mean()))
flow_stats = {
    "episode": "S04 开头 (E01 起)",
    "n_sentences": len(line_meta), "n_tokens": len(line_ids),
    "adj_same_role_sim": float(np.mean(adj_same)) if adj_same else None,
    "adj_diff_role_sim": float(np.mean(adj_diff)) if adj_diff else None,
    "n_adj_same": len(adj_same), "n_adj_diff": len(adj_diff),
    "attn_entropy_per_layer": flow_ent,
    "speaker_counts": {r: flow_sp.count(r) for r in ROLES},
}
fig, ax = plt.subplots(figsize=(6.6, 3.9), dpi=150)
groups = {"同角色相邻": adj_same, "跨角色相邻": adj_diff}
ax.hist([g for g in groups.values() if g], bins=np.linspace(0.5, 1.0, 26),
        label=[k for k, g in groups.items() if g],
        color=["#d1495b", "#2e86ab"], alpha=0.6, density=True)
ax.set_xlabel("cosine similarity (last layer, adjacent lines)")
ax.set_ylabel("density")
ax.set_title("S04 对话流: 相邻台词相似度")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_s5_flow.png")); plt.close(fig)

# ---------- 落账 ----------
metrics = {
    "minted": time.strftime("%Y-%m-%d"),
    "question": "Rachel 的特色是否在 S03/S04 才凝聚 (主人观感)",
    "hypotheses": {
        "H-S1": "Rachel 舱内相似度 S03/S04 > S01 (凝聚力上升)",
        "H-S2": "Rachel 与其余角色质心距 S03/S04 > S01 (分离度上升)",
        "H-S3": "对照: 其余五角色同指标随季平坦 (否则 = 季间漂移)"},
    "model": {"path": MODEL, "arch": model.config.architectures[0],
              "layers": L, "hidden": model.config.hidden_size,
              "heads": model.config.num_attention_heads, "dtype": "bf16-on-disk",
              "device": dev},
    "corpus": {"deck": corpus["deck"], "n": n, "roles": ROLES,
                 "seasons": SEASONS, "per_role_per_season": 50,
                 "S02": "fangj 站 S02 为整集单块格式, 已兼容解析"},
    "knobs": {"seed": SEED, "tsne_perplexity": TSNE_PERPLEXITY,
                "probe_folds": PROBE_FOLDS, "flow_max_tokens": FLOW_MAX_TOKENS},
    "cjk_font": CJK,
    "season_metrics": {str(s): {
        "within_last": {r: season_metrics[s]["within"][r][-1] for r in ROLES},
        "ratio_last": {r: season_metrics[s]["ratio"][r][-1] for r in ROLES},
        "probe_acc_max": season_metrics[s]["probe_acc_max"],
        "probe_acc_max_layer": season_metrics[s]["probe_acc_max_layer"],
        "rachel_centroid_dist_last": season_metrics[s]["rachel_centroid_dist_last"],
        "attn_entropy_last": season_metrics[s]["attn_entropy_last"],
    } for s in SEASONS},
    "rachel_timeline": {
        "seasons": SEASONS,
        "rachel_within_last": lead_within,
        "others_within_mean": others_within_mean,
        "rachel_centroid_dist": lead_centdist,
        "others_centroid_dist_mean": others_centdist_mean,
        "probe_lead_vs_rest": probe_lead_vs_rest,
    },
    "flow_stats": flow_stats,
    "elapsed_s": time.time() - t0,
}
json.dump(metrics, open(os.path.join(HERE, "metrics_friends_s1234.json"), "w"),
          ensure_ascii=False, indent=1)
print(f"DONE in {time.time()-t0:.0f}s — figs_friends_s134/ + metrics_friends_s134.json")
print(json.dumps({
    "rachel_within_S": lead_within,
    "others_within_S": others_within_mean,
    "rachel_centdist_S": lead_centdist,
    "probe_lead_vs_rest": probe_lead_vs_rest,
}, ensure_ascii=False))
