#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Friends 季节演化 — Kaggle 夜航臂 (#__ARM_TAG__#)

正本声明 (三样交付律):
  正本  = benches/transformer_vis/friends_seasonal.py (本地 MPS, Qwen2.5-0.5B)
          + corpus_friends_s1234.json / friends_s1234_raw.json (语料正本)
  本臂  = 正本的 Kaggle 投影 (模型: #__ARM_MODEL__#)
          语料内嵌 b64 + sha256 自校验, 与正本逐位同源
  对表  = 本地 metrics_friends_s1234.json (臂 A 自检基准, 容差见 PREREG.md)

夜航规矩 (kaggle-night-flight skill):
  疫苗二代行 / 语料内嵌 / 即算即落盘 / REPORT_LINE 哨兵 / 错误臂带 traceback
判据先冻: PREREG.md (K-1..K-4)

量法 (与正本逐一致):
  [1] 孤立编码 1200 条台词 (4 季 × 6 角色 × 50, 逐条前向, 全层 mean-pool + 逐层注意力熵)
  [2] 逐季角色几何: 舱内相似度 / 质心距离 / 舱内跨角色比 / 6 类探针
  [3] Rachel 时间线: 舱内相似 + 质心均距 (S01→S04), 对照其余五角色; Rachel-vs-rest 二类探针
  [4] 逐季质心热图 + 探针逐层曲线
  [5] 投影: t-SNE 季着色
  [6] 对话流: S04 开头流 (同角色相邻 vs 跨角色相邻)

产物: /kaggle/working/figs_friends_s1234_kaggle/*.png + metrics_friends_s1234_kaggle.json
"""
import os as _osx
_osx.system("python -m pip uninstall -y -q torchao 2>/dev/null")  # 疫苗二代: torchao 0.10 与 transformers 不相容 (0919 矩阵全灭真死因)

import gzip
import json, os, time, base64, hashlib, traceback
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- 语料内嵌 (gz+b64 + sha256 自校验; Kaggle 源 1MB 上限, 05b 首射实测) ----------
# sha 均取解压后字节的 sha256 —— 与本地正本文件逐位同源
_CORPUS_B64 = "#__CORPUS_B64__#"
_CORPUS_SHA = "#__CORPUS_SHA__#"
_RAW_B64 = "#__RAW_B64__#"
_RAW_SHA = "#__RAW_SHA__#"

def _load_emb(b64s, sha, name):
    data = gzip.decompress(base64.b64decode(b64s))
    got = hashlib.sha256(data).hexdigest()
    assert got == sha, f"{name} sha256 mismatch: got {got} want {sha}"
    return json.loads(data.decode("utf-8"))

# ---------- 旋钮区（园规·参数意愿律: 每个数一行愿望, 不许"都这么设"） ----------
SEED = 13              # 园子第一号 seed
TSNE_PERPLEXITY = 15   # t-SNE 困惑度
PROBE_FOLDS = 5        # 探针交叉验证折数
FLOW_MAX_TOKENS = 1500 # 流编码 token 上限 (内存保护)
LEAD = "Rachel"        # 主人点名的角色
MODEL = "#__MOUNT__#"  # Kaggle 模型挂载
QUANT = "#__QUANT__#"  # "none" (bf16) | "4bit" (bnb)
if QUANT == "4bit":  # 7b 臂: Kaggle 镜像不预装 bitsandbytes (10-13 两射 ERROR 实测)
    # 注: 本臂 metadata enable_internet=true (仅为此安装; PREREG 原设 false, 偏离记战报)
    _osx.system("python -m pip install -q -U 'bitsandbytes>=0.46.1'")  # bnb 疫苗
    _osx.system("python -m pip show bitsandbytes | grep -i version")  # 版本留痕
# ---------------------------------------------------------------------------

HERE = "/kaggle/working"
FIGS = os.path.join(HERE, "figs_friends_s1234_kaggle")
OUT_METRICS = os.path.join(HERE, "metrics_friends_s1234_kaggle.json")
os.makedirs(FIGS, exist_ok=True)

PARTIAL = {"arm": "#__ARM_TAG__#", "model_mount": MODEL, "quant": QUANT}

def fl(stage, **kv):
    """即算即落盘: 每阶段算完即 dump + REPORT_LINE 哨兵 (日志即幸存者)"""
    PARTIAL.update({"stage": stage, **kv})
    with open(OUT_METRICS, "w") as f:
        json.dump(PARTIAL, f, ensure_ascii=False, indent=1)
    print("REPORT_LINE::" + base64.b64encode(
        json.dumps(PARTIAL, ensure_ascii=False).encode()).decode()[:3800], flush=True)

def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    t0 = time.time()

    corpus = _load_emb(_CORPUS_B64, _CORPUS_SHA, "corpus_friends_s1234.json")
    raw = _load_emb(_RAW_B64, _RAW_SHA, "friends_s1234_raw.json")

    ROLES = corpus["roles"]                      # [Rachel, Chandler, Ross, Monica, Joey, Phoebe]
    SEASONS = corpus["seasons"]                  # [1, 2, 3, 4]
    texts = [it["desc"] for it in corpus["items"]]
    role_of = [it["stage"] for it in corpus["items"]]
    season_of = [it["season"] for it in corpus["items"]]
    n = len(texts)
    assert raw["data"]["S04"]["lines"], "raw S04 lines missing"

    # ---------- 模型 ----------
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(MODEL)
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    if QUANT == "4bit":
        from transformers import BitsAndBytesConfig
        qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, quantization_config=qc, device_map="auto",
            attn_implementation="eager")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.bfloat16, attn_implementation="eager").to(DEV)
    model = model.eval()
    L = model.config.num_hidden_layers
    print(f"model ready: {MODEL} quant={QUANT} dev={DEV} L={L}", flush=True)

    # ---------- [1] 孤立编码 ----------
    def encode(texts):
        """逐条前向: 返回 (n, L+1, d) 全层 mean-pool 表征 与 (n, L) 逐层注意力熵"""
        pooled, ents = [], []
        with torch.no_grad():
            for i, t in enumerate(texts):
                ids = tok(t, return_tensors="pt").to(DEV)
                out = model(**ids, output_hidden_states=True, output_attentions=True)
                pooled.append(np.stack(
                    [h[0].mean(0).float().cpu().numpy() for h in out.hidden_states]))
                layer_ent = []
                for att in out.attentions:
                    a = att[0].float().cpu()
                    layer_ent.append(float((-(a * (a + 1e-12).log()).sum(-1)).mean()))
                ents.append(layer_ent)
                if (i + 1) % 200 == 0:
                    print(f"      encoded {i+1}/{len(texts)}", flush=True)
        return np.stack(pooled), np.stack(ents)

    print(f"[1/6] 孤立编码 {n} 条 ({len(SEASONS)} 季 × 6 角色 × 50) …", flush=True)
    X, E = encode(texts)
    print(f"      编码完成 {time.time()-t0:.0f}s · 层数 L+1={X.shape[1]}", flush=True)
    fl("encode", n=n, layers=L + 1, hidden=int(X.shape[2]), elapsed_s=round(time.time() - t0, 1))

    # ---------- [2] 逐季角色几何 ----------
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
        # Rachel 与其余角色质心均距 (末层)
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
        print(f"      S{season:02d} done · Rachel within={within[LEAD][-1]:.4f}", flush=True)

    within_last = {str(s): {r: season_metrics[s]["within"][r][-1] for r in ROLES}
                   for s in SEASONS}
    fl("season_geometry", within_last=within_last,
       rachel_centroid_dist={str(s): season_metrics[s]["rachel_centroid_dist_last"]
                             for s in SEASONS})

    # ---------- [3] Rachel 时间线 ----------
    print("[3/6] Rachel timeline …", flush=True)
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
    print(f"      Rachel within: {[round(v, 4) for v in lead_within]}", flush=True)
    print(f"      others  within: {[round(v, 4) for v in others_within_mean]}", flush=True)
    fl("rachel_timeline", rachel_within_last=lead_within, others_within_mean=others_within_mean,
       rachel_centroid_dist=lead_centdist, others_centroid_dist_mean=others_centdist_mean,
       probe_lead_vs_rest=probe_lead_vs_rest)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), dpi=150)
    xs = np.arange(len(SEASONS))
    axes[0].plot(xs, lead_within, "o-", color="#d1495b", lw=2, ms=7,
                 label="Rachel within-role")
    axes[0].plot(xs, others_within_mean, "s--", color="#2e86ab", lw=1.6, ms=6,
                 label="other five (mean)")
    axes[0].set_xticks(xs); axes[0].set_xticklabels([f"S{s:02d}" for s in SEASONS])
    axes[0].set_ylabel("within-role cosine (last layer)")
    axes[0].set_title("Rachel cohesion by season")
    axes[0].legend(frameon=False, fontsize=9)
    for i, v in enumerate(lead_within):
        axes[0].annotate(f"{v:.4f}", (xs[i], v), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=8)
    axes[1].plot(xs, lead_centdist, "o-", color="#d1495b", lw=2, ms=7,
                 label="Rachel ↔ others centroid dist")
    axes[1].plot(xs, others_centdist_mean, "s--", color="#2e86ab", lw=1.6, ms=6,
                 label="among others (mean)")
    axes[1].set_xticks(xs); axes[1].set_xticklabels([f"S{s:02d}" for s in SEASONS])
    axes[1].set_ylabel("centroid cosine distance")
    axes[1].set_title("Rachel separation by season")
    axes[1].legend(frameon=False, fontsize=9)
    for i, v in enumerate(lead_centdist):
        axes[1].annotate(f"{v:.4f}", (xs[i], v), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_s1_rachel_timeline.png"))
    plt.close(fig)

    # ---------- [4] 逐季质心热图 + 探针 ----------
    print("[4/6] centroid heatmaps + probe …", flush=True)
    fig, axes = plt.subplots(1, len(SEASONS), figsize=(4.6 * len(SEASONS), 4.2), dpi=150)
    for si, season in enumerate(SEASONS):
        ax = axes[si]
        cm = np.array(season_metrics[season]["centroid_distance_matrix"])
        im = ax.imshow(cm, cmap="viridis", vmin=0, vmax=cm.max())
        ax.set_xticks(range(6)); ax.set_yticks(range(6))
        ax.set_xticklabels([r[:6] for r in ROLES], rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels([r[:6] for r in ROLES], fontsize=8)
        ax.set_title(f"S{season:02d} centroid distance", fontsize=10)
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
    ax.set_title("Role probe per layer (sanity check — lexicon does not leak speaker)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_s3_probe.png")); plt.close(fig)

    # ---------- [5] t-SNE 季着色 ----------
    print("[5/6] projection …", flush=True)
    from sklearn.manifold import TSNE
    Xl = X[:, -1]
    tsne = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, random_state=SEED,
                init="pca", learning_rate="auto").fit_transform(Xl)
    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=150)
    for season in SEASONS:
        m = np.array(season_of) == season
        ax.scatter(tsne[m, 0], tsne[m, 1], c=SEASON_COLORS[season], s=38,
                   alpha=0.8, label=f"S{season:02d}", edgecolors="white", linewidths=0.4)
    ax.set_title("Last-layer representations — t-SNE (colored by season)")
    ax.legend(frameon=False, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#cccccc")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_s4_tsne.png")); plt.close(fig)
    fl("projection", tsne="fig_s4_tsne.png saved")

    # ---------- [6] 对话流: S04 开头 ----------
    print("[6/6] dialogue flow (S04 opening) …", flush=True)
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
    flow_ids = torch.tensor([line_ids], device=DEV)
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
        "episode": "S04 opening (from E01)",
        "n_sentences": len(line_meta), "n_tokens": len(line_ids),
        "adj_same_role_sim": float(np.mean(adj_same)) if adj_same else None,
        "adj_diff_role_sim": float(np.mean(adj_diff)) if adj_diff else None,
        "n_adj_same": len(adj_same), "n_adj_diff": len(adj_diff),
        "attn_entropy_per_layer": flow_ent,
        "speaker_counts": {r: flow_sp.count(r) for r in ROLES},
    }
    print(f"      flow: {len(line_meta)} sentences / {len(line_ids)} tokens · "
          f"same={flow_stats['adj_same_role_sim']:.4f} diff={flow_stats['adj_diff_role_sim']:.4f}",
          flush=True)

    fig, ax = plt.subplots(figsize=(6.6, 3.9), dpi=150)
    groups = {"same-role adjacent": adj_same, "cross-role adjacent": adj_diff}
    ax.hist([g for g in groups.values() if g], bins=np.linspace(0.5, 1.0, 26),
            label=[k for k, g in groups.items() if g],
            color=["#d1495b", "#2e86ab"], alpha=0.6, density=True)
    ax.set_xlabel("cosine similarity (last layer, adjacent lines)")
    ax.set_ylabel("density")
    ax.set_title("S04 dialogue flow: adjacent-line similarity")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_s5_flow.png")); plt.close(fig)
    fl("flow", **{k: v for k, v in flow_stats.items() if k != "attn_entropy_per_layer"})

    # ---------- 落账 ----------
    metrics = {
        "minted": time.strftime("%Y-%m-%d"),
        "arm": "#__ARM_TAG__#",
        "platform": "kaggle",
        "question": "Does Rachel's distinctiveness only coalesce in S03/S04 (owner's impression)?",
        "hypotheses": {
            "H-S1": "Rachel within-role similarity S03/S04 > S01 (cohesion rises)",
            "H-S2": "Rachel-to-others centroid distance S03/S04 > S01 (separation rises)",
            "H-S3": "control: the other five roles move the same way across seasons "
                    "(otherwise = season drift, not Rachel-specific)"},
        "model": {"mount": MODEL, "quant": QUANT, "arch": model.config.architectures[0],
                  "layers": L, "hidden": model.config.hidden_size,
                  "heads": model.config.num_attention_heads, "device": DEV},
        "corpus": {"deck": corpus["deck"], "n": n, "roles": ROLES,
                   "seasons": SEASONS, "per_role_per_season": 50,
                   "S02": "fangj site S02 is single-block-per-episode format; parser handles it"},
        "knobs": {"seed": SEED, "tsne_perplexity": TSNE_PERPLEXITY,
                  "probe_folds": PROBE_FOLDS, "flow_max_tokens": FLOW_MAX_TOKENS},
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
    with open(OUT_METRICS, "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=1)
    print("SEASONAL_DONE::" + base64.b64encode(json.dumps({
        "arm": "#__ARM_TAG__#",
        "rachel_within": [round(v, 4) for v in lead_within],
        "others_within": [round(v, 4) for v in others_within_mean],
        "rachel_centdist": [round(v, 4) for v in lead_centdist],
        "probe_lead_vs_rest": [round(v, 4) for v in probe_lead_vs_rest],
        "flow_same": round(flow_stats["adj_same_role_sim"], 4),
        "flow_diff": round(flow_stats["adj_diff_role_sim"], 4),
        "elapsed_s": round(time.time() - t0),
    }, ensure_ascii=False).encode()).decode(), flush=True)

SEASON_COLORS = {1: "#2e86ab", 2: "#6a994e", 3: "#f4a259", 4: "#d1495b"}
ROLE_COLORS = ["#d1495b", "#2e86ab", "#f4a259", "#8ac926", "#9b5de5", "#00b4d8"]

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("SEASONAL_ERROR::" + traceback.format_exc(), flush=True)
        raise
