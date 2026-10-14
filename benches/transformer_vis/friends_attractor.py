#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Friends 台词语料 —— Rachel/Chandler 是否为注意力尖峰（吸引子对照）

语料: corpus_friends.json (D5_friends, S01 六主角各 50 条, fangj.github.io 说话人标注正本)
模型: first-lora/model (Qwen2.5-0.5B-Instruct, 本地盘)
背景: 实验一发现 NBA 舱 = 模板驱动紧簇(舱内 0.973) + 低熵(1.048 nats) —— 「尖峰」= 紧簇+低熵。
      本实验问: Friends 群像对话剧里, Rachel/Chandler 是否构成同类尖峰?

量法 (沿用 nba_vs_spring.py + product_temporal.py):
  [1] 孤立编码 300 条台词 → (n, L+1, d) 全层 mean-pool + (n, L) 注意力熵
  [2] 投影 PCA/t-SNE 角色着色
  [3] 逐层探针 (6 类) + 质心余弦距离矩阵 —— sanity check (台词词汇域本就不同, 探针只作标注)
  [4] 角色相似度几何: 舱内/跨角色比 (逐层, 每角色 + 主角对)
  [5] 注意力熵逐层 (6 角色)
  [6] 尖峰指数: 每角色末层舱内相似度 (紧度) × 注意力熵 —— 主角 vs 其余四角色
  [7] 对话流编码: S01E01 完整台词流 → 相邻台词表示相似度 (对话网结构) + 流注意力熵

判读框架 (诚实账):
  - 尖峰 = 紧簇 + 低熵; 群像剧预期 = 松散「对话网」(对照 NBA 紧簇/春节松网)
  - 「吸引子」是比喻: 检验台词表示是否被角色质心吸引, 非动力系统吸引子
  - 探针高 = 词汇域分离, 非结构证据 (同实验一纪律)
  - 语料仅 S01 一季; 季间差异未验

产物: figs_friends/*.png + metrics_friends.json (供 report_friends.qmd 引用)

园规: 每个数一行意愿 (旋钮区)。跑法:
  /opt/miniconda3/envs/default/bin/python friends_attractor.py
"""

import json, os, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- 旋钮区（园规·参数意愿律: 每个数一行意愿, 不许"都这么设"） ----------
CORPUS = "corpus_friends.json"   # D5_friends 正本
RAW    = "friends_s01_raw.json"  # E01 对话流用 (说话人标注原始序列)
MODEL  = os.path.expanduser("~/Programming/code-2026/first-lora/model")
SEED   = 13              # 园子第一号 seed
TSNE_PERPLEXITY = 15     # t-SNE 困惑度
PROBE_FOLDS = 5          # 探针交叉验证折数
UMAP_N = 15              # UMAP 近邻数
FLOW_EPISODE = "E01"     # 对话流编码用集
FLOW_MAX_TOKENS = 1500   # 流编码 token 上限 (内存保护, ~70 句)
LEAD_PAIR = ("Rachel", "Chandler")  # 主人点名的尖峰候选 (主角对)
# ---------------------------------------------------------------------------

torch.manual_seed(SEED); np.random.seed(SEED)
HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs_friends"); os.makedirs(FIGS, exist_ok=True)

# ---------- 语料 ----------
corpus = json.load(open(os.path.join(HERE, CORPUS)))
ROLES = corpus["roles"]                      # [Rachel, Chandler, Ross, Monica, Joey, Phoebe]
texts = [it["desc"] for it in corpus["items"]]
stage_of = [it["stage"] for it in corpus["items"]]
n = len(texts)

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
def LBL(role):
    return role + ("·" if CJK else "")

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
print(f"[1/7] 孤立编码 {n} 条台词 (六角色各 {n//6}) …", flush=True)
X, E = encode(texts)
print(f"      编码完成 {time.time()-t0:.0f}s · 层数 L+1={X.shape[1]} · 隐维 d={X.shape[2]}", flush=True)

# ---------- 投影（末层表征, 角色着色） ----------
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
Xl = X[:, -1]
pca = PCA(n_components=2, random_state=SEED).fit_transform(Xl)
tsne = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, random_state=SEED,
            init="pca", learning_rate="auto").fit_transform(Xl)
ROLE_COLORS = ["#d1495b", "#2e86ab", "#f4a259", "#8ac926", "#9b5de5", "#00b4d8"]

def scatter_roles(xy, fname, title):
    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=150)
    for ri, role in enumerate(ROLES):
        m = np.array(stage_of) == ri
        ax.scatter(xy[m, 0], xy[m, 1], c=ROLE_COLORS[ri], s=40, alpha=0.85,
                   label=LBL(role), edgecolors="white", linewidths=0.4)
    ax.set_title(title, fontsize=11)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color("#cccccc")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, fname)); plt.close(fig)

print("[2/7] 投影图 …", flush=True)
scatter_roles(pca, "fig_f1_pca.png", "Last-layer representations — PCA")
scatter_roles(tsne, "fig_f2_tsne.png", f"Last-layer representations — t-SNE (perplexity={TSNE_PERPLEXITY})")

# ---------- 逐层探针 + 质心几何 ----------
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics.pairwise import cosine_similarity

print("[3/7] 逐层探针 + 质心几何 …", flush=True)
probe_acc, cent_mat = [], None
for li in range(X.shape[1]):
    Xl2 = X[:, li]
    y = np.array(stage_of)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    probe_acc.append(float(cross_val_score(clf, Xl2, y, cv=PROBE_FOLDS).mean()))
    if li == X.shape[1] - 1:
        cents = np.stack([Xl2[np.array(stage_of) == ri].mean(0) for ri in range(6)])
        cent_mat = 1 - cosine_similarity(cents)   # 余弦距离矩阵

# 质心热图
fig, ax = plt.subplots(figsize=(5.6, 4.8), dpi=150)
im = ax.imshow(cent_mat, cmap="viridis", vmin=0, vmax=cent_mat.max())
ax.set_xticks(range(6)); ax.set_yticks(range(6))
ax.set_xticklabels([LBL(r) for r in ROLES], rotation=45, ha="right", fontsize=9)
ax.set_yticklabels([LBL(r) for r in ROLES], fontsize=9)
for i in range(6):
    for j in range(6):
        ax.text(j, i, f"{cent_mat[i,j]:.2f}", ha="center", va="center",
                color="white" if cent_mat[i, j] > cent_mat.max() * 0.6 else "black", fontsize=8)
ax.set_title("Centroid cosine distance (last layer)")
fig.colorbar(im, shrink=0.85)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_f3_centroid.png")); plt.close(fig)

# ---------- 角色相似度几何: 舱内/跨角色比 (逐层) ----------
print("[4/7] 角色相似度几何 …", flush=True)
within_per_role = {r: [] for r in ROLES}     # 每角色舱内相似度 (逐层)
ratio_per_role = {r: [] for r in ROLES}      # 舱内/跨角色比 (逐层)
lead_within, lead_between = [], []            # 主角对舱内 / 主角对↔其余
for li in range(X.shape[1]):
    Xl2 = X[:, li]
    W = cosine_similarity(Xl2)
    for ri, role in enumerate(ROLES):
        m = np.array(stage_of) == ri
        iw = W[np.ix_(m, m)]
        within = iw[np.triu_indices(m.sum(), 1)].mean()
        ib = W[np.ix_(m, ~m)].mean()
        within_per_role[role].append(float(within))
        ratio_per_role[role].append(float(within / (ib + 1e-12)))
    lm = np.isin(np.array(stage_of), [0, 1])  # Rachel/Chandler
    lwi = W[np.ix_(lm, lm)]
    lead_within.append(float(lwi[np.triu_indices(lm.sum(), 1)].mean()))
    lead_between.append(float(W[np.ix_(lm, ~lm)].mean()))

# 逐层舱内相似度曲线 (6 角色)
fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=150)
la = np.arange(X.shape[1])
for ri, role in enumerate(ROLES):
    ax.plot(la, within_per_role[role], "o-", color=ROLE_COLORS[ri], lw=1.6, ms=3,
            label=LBL(role) + (" (主角对)" if role in LEAD_PAIR else ""))
ax.set_xlabel("layer (0 = embedding, L = last)")
ax.set_ylabel("within-role cosine similarity")
ax.set_title("Within-role similarity per layer")
ax.legend(frameon=False, fontsize=8, ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_f4_layer_ratio.png")); plt.close(fig)

# ---------- 逐层注意力熵 (6 角色) ----------
print("[5/7] 注意力熵 …", flush=True)
fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=150)
le = np.arange(E.shape[1])   # 注意力熵只有 L 层 (hidden 是 L+1)
for ri, role in enumerate(ROLES):
    m = np.array(stage_of) == ri
    mu, sd = E[m].mean(0), E[m].std(0)
    ax.plot(le, mu, "o-", color=ROLE_COLORS[ri], lw=1.6, ms=3, label=LBL(role))
    ax.fill_between(le, mu - sd, mu + sd, color=ROLE_COLORS[ri], alpha=0.12)
ax.set_xlabel("layer")
ax.set_ylabel("mean attention entropy (nats)")
ax.set_title("Attention sharpness per layer, by role")
ax.legend(frameon=False, fontsize=8, ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_f5_attn_entropy.png")); plt.close(fig)

# ---------- 尖峰指数: 紧度 × 熵 ----------
print("[6/7] 尖峰指数 …", flush=True)
peak = {}
for ri, role in enumerate(ROLES):
    m = np.array(stage_of) == ri
    peak[role] = {
        "within_last": float(within_per_role[role][-1]),
        "ratio_last": float(ratio_per_role[role][-1]),
        "attn_entropy_last": float(E[m][:, -1].mean()),
        "attn_entropy_mean": float(E[m].mean()),
        "n": int(m.sum()),
    }
# 主角对 vs 其余四角色
others = [r for r in ROLES if r not in LEAD_PAIR]
lead_stats = {
    "lead_pair": list(LEAD_PAIR),
    "lead_within_last": float(lead_within[-1]),
    "lead_ratio_last": float(lead_within[-1] / (lead_between[-1] + 1e-12)),
    "others_within_last_mean": float(np.mean([peak[r]["within_last"] for r in others])),
    "others_entropy_last_mean": float(np.mean([peak[r]["attn_entropy_last"] for r in others])),
    "lead_entropy_last_mean": float(np.mean([peak[r]["attn_entropy_last"] for r in LEAD_PAIR])),
}
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9), dpi=150)
xs = np.arange(6)
vals = [peak[r]["within_last"] for r in ROLES]
cols = [ROLE_COLORS[ri] for ri in range(6)]
axes[0].bar(xs, vals, color=cols, alpha=0.85)
axes[0].set_xticks(xs); axes[0].set_xticklabels([LBL(r) for r in ROLES], rotation=45, ha="right", fontsize=8)
axes[0].set_title("Tightness: within-role cosine (last layer)")
for i, v in enumerate(vals):
    axes[0].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=7.5)
vals2 = [peak[r]["attn_entropy_last"] for r in ROLES]
axes[1].bar(xs, vals2, color=cols, alpha=0.85)
axes[1].set_xticks(xs); axes[1].set_xticklabels([LBL(r) for r in ROLES], rotation=45, ha="right", fontsize=8)
axes[1].set_title("Sharpness: attention entropy (last layer, nats)")
for i, v in enumerate(vals2):
    axes[1].text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=7.5)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_f6_peak_index.png")); plt.close(fig)

# ---------- 对话流编码: S01E01 台词流 ----------
print(f"[7/7] 对话流编码 ({FLOW_EPISODE}) …", flush=True)
raw = json.load(open(os.path.join(HERE, RAW)))
e01 = [(l["speaker"], l["line"]) for l in raw["lines"]]
# 按集序取六主角台词, 记录边界
line_ids, line_meta = [], []
for sp, line in e01:
    if sp not in ROLES:
        continue
    ids = tok(line, add_special_tokens=False)["input_ids"]
    if len(line_ids) + len(ids) > FLOW_MAX_TOKENS:
        break
    line_ids.extend(ids)
    line_meta.append((sp, len(line_ids) - 1))   # (说话人, 句末 token 实际索引)
flow_ids = torch.tensor([line_ids], device=dev)
with torch.no_grad():
    out = model(input_ids=flow_ids, output_hidden_states=True, output_attentions=True)
# 句表示: 每句末 token 的 hidden (各层)
flow_sent = []   # (L+1, n_sent, d)
for li in range(X.shape[1]):
    h = out.hidden_states[li][0]      # (T, d)
    ends = [e for _, e in line_meta]
    flow_sent.append(h[ends].float().cpu().numpy())
flow_sent = np.stack(flow_sent)       # (L+1, n_sent, d)
flow_sp = [sp for sp, _ in line_meta]
# 相邻台词表示相似度: 同角色相邻 vs 跨角色相邻 (末层)
fs = flow_sent[-1]
adj_same, adj_diff = [], []
for i in range(len(line_meta) - 1):
    sim = float(np.dot(fs[i], fs[i + 1]) / (np.linalg.norm(fs[i]) * np.linalg.norm(fs[i + 1])))
    if flow_sp[i] == flow_sp[i + 1]:
        adj_same.append(sim)
    else:
        adj_diff.append(sim)
# 流注意力熵 (逐层, 全体 token 均值)
flow_ent = []
for att in out.attentions:
    a = att[0].float().cpu()
    flow_ent.append(float((-(a * (a + 1e-12).log()).sum(-1)).mean()))
flow_stats = {
    "episode": FLOW_EPISODE,
    "n_sentences": len(line_meta),
    "n_tokens": len(line_ids),
    "adj_same_role_sim": float(np.mean(adj_same)) if adj_same else None,
    "adj_diff_role_sim": float(np.mean(adj_diff)) if adj_diff else None,
    "n_adj_same": len(adj_same), "n_adj_diff": len(adj_diff),
    "attn_entropy_per_layer": flow_ent,
    "speaker_counts": {r: flow_sp.count(r) for r in ROLES},
}
# 流图: 相邻相似度按说话人对
fig, ax = plt.subplots(figsize=(6.6, 3.9), dpi=150)
groups = {"同角色相邻": adj_same, "跨角色相邻": adj_diff}
ax.hist([g for g in groups.values() if g], bins=np.linspace(0.5, 1.0, 26),
        label=[k for k, g in groups.items() if g],
        color=["#d1495b", "#2e86ab"], alpha=0.6, density=True)
ax.set_xlabel("cosine similarity (last layer, adjacent lines)")
ax.set_ylabel("density")
ax.set_title(f"{FLOW_EPISODE} dialogue flow: adjacent-line similarity")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_f7_flow.png")); plt.close(fig)

# ---------- 落账 ----------
metrics = {
    "minted": time.strftime("%Y-%m-%d"),
    "question": "Rachel/Chandler 是否为台词语料里的注意力尖峰(吸引子对照)",
    "model": {"path": MODEL, "arch": model.config.architectures[0],
              "layers": L, "hidden": model.config.hidden_size,
              "heads": model.config.num_attention_heads, "dtype": "bf16-on-disk",
              "device": dev},
    "corpus": {"deck": corpus["deck"], "n": n, "roles": ROLES,
                 "per_role": n // 6,
                 "source": corpus.get("sampling", {})},
    "knobs": {"seed": SEED, "tsne_perplexity": TSNE_PERPLEXITY,
                "probe_folds": PROBE_FOLDS, "umap_n": UMAP_N,
                "flow_episode": FLOW_EPISODE, "flow_max_tokens": FLOW_MAX_TOKENS},
    "cjk_font": CJK,
    "probe_acc_per_layer": probe_acc,
    "probe_acc_max_layer": int(np.argmax(probe_acc)),
    "probe_acc_max": float(max(probe_acc)),
    "centroid_distance_matrix_last": cent_mat.tolist(),
    "within_role_per_layer": within_per_role,
    "within_over_between_per_layer": ratio_per_role,
    "attn_entropy_mean_per_layer": {r: E[np.array(stage_of) == ri].mean(0).tolist()
                                     for ri, r in enumerate(ROLES)},
    "peak_index": peak,
    "lead_pair_stats": lead_stats,
    "flow_stats": flow_stats,
    "elapsed_s": time.time() - t0,
}
json.dump(metrics, open(os.path.join(HERE, "metrics_friends.json"), "w"),
          ensure_ascii=False, indent=1)
print(f"DONE in {time.time()-t0:.0f}s — figs_friends/ + metrics_friends.json")
print(json.dumps({"peak_index": {r: round(peak[r]["within_last"], 4) for r in ROLES},
                  "lead_pair": {k: (round(v, 4) if isinstance(v, float) else v)
                                for k, v in lead_stats.items()}},
                 ensure_ascii=False))
