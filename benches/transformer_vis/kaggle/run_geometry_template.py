#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geometry of Meaning — Kaggle 夜航臂 (#__ARM_TAG__#)

正本声明 (三样交付律):
  正本  = /Users/mac/CodeBuddy/Geometry-Of-Meaning/
            extract_representations.py (逐层提取, 三策略)
            analyze_geometry.py        (逐层几何量, G1-G4 判据)
            corpus_paraphrase.json     (语料正本, 50 概念 × 4 变体)
            metrics_geometry.json      (0.5B 本地对表基准)
  本臂  = 正本的 Kaggle 投影 (模型: #__ARM_MODEL__#)
          语料 gz+b64 内嵌 + sha256 自校验, 与正本逐位同源
  对表  = 本地 metrics_geometry.json (0.5B; 跨模型只比形态不比绝对值)

夜航规矩 (kaggle-night-flight):
  疫苗二代行 / 语料内嵌 / 即算即落盘 / REPORT_LINE 哨兵 / 错误臂带 traceback
判据先冻: geometry-meaning-7b/PREREG.md (K-1..K-4)

量法 (与本地 analyze_geometry.py 逐同口径):
  [1] 逐句一次前向 (output_hidden_states), 层采样 LAYER_SKIP=2 (7B 28L → 15 层)
  [2] 三策略: last-token / mean-pool / entity-centric (实体 token 滑动匹配定位)
  [3] 逐层: pos (同概念 4 变体两两余弦) / neg (rng(42) 采 20 概念各取变体 0)
          / margin / cka (= cosine(v0,v1)^2, 与本地同口径)
  [4] 末 3 层裁决: G-1 (pos>0.75 且 >neg) / G-3 (cka>0.85); G-2 UMAP 出图+辅助读数
  [5] 图: cosine+margin 曲线 / 热力图 / UMAP / CKA 曲线 (对齐本地 figs_geometry)

产物: /kaggle/working/figs_geometry_kaggle/*.png + metrics_geometry_kaggle.json
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

# ---------- 语料内嵌 (gz+b64 + sha256 自校验; Kaggle 源 1MB 上限) ----------
# sha 取解压后字节的 sha256 —— 与本地正本文件逐位同源
_CORPUS_B64 = "#__CORPUS_B64__#"
_CORPUS_SHA = "#__CORPUS_SHA__#"

def _load_corpus():
    data = gzip.decompress(base64.b64decode(_CORPUS_B64))
    got = hashlib.sha256(data).hexdigest()
    assert got == _CORPUS_SHA, f"corpus sha256 mismatch: got {got} want {_CORPUS_SHA}"
    return json.loads(data.decode("utf-8"))

# ---------- 旋钮区（园规·参数意愿律: 每个数一行, 不许"都这么设"） ----------
SEED = 42          # 与本地 analyze_geometry.py 的 neg 采样 rng 同号
LAYER_SKIP = 2     # 层采样步长 (7B 28L → 15 层, 含 0 与末层)
UMAP_NEIGHBORS = 15
NEG_SAMPLE = 20    # 负样本采样概念数 (与本地同)
G1_THRESHOLD = 0.75
G3_THRESHOLD = 0.85
MOUNT = "#__MOUNT__#"   # Kaggle 模型挂载
QUANT = "#__QUANT__#"   # "4bit" (bnb, compute fp16)
if QUANT == "4bit":  # Kaggle 镜像不预装 bitsandbytes (friends 7b 臂 10-13 实测)
    # 注: 本臂 metadata enable_internet=true (仅为此安装; PREREG 记此偏离)
    _osx.system("python -m pip install -q -U 'bitsandbytes>=0.46.1'")  # bnb 疫苗
    _osx.system("python -m pip show bitsandbytes | grep -i version")    # 版本留痕
# ---------------------------------------------------------------------------

HERE = "/kaggle/working"
FIGS = os.path.join(HERE, "figs_geometry_kaggle")
OUT_METRICS = os.path.join(HERE, "metrics_geometry_kaggle.json")
os.makedirs(FIGS, exist_ok=True)

PARTIAL = {"arm": "#__ARM_TAG__#", "model_mount": MOUNT, "quant": QUANT}

def fl(stage, **kv):
    """即算即落盘: 每阶段算完即 dump + REPORT_LINE 哨兵 (日志即幸存者)"""
    PARTIAL.update({"stage": stage, **kv})
    with open(OUT_METRICS, "w") as f:
        json.dump(PARTIAL, f, ensure_ascii=False, indent=1)
    print("REPORT_LINE::" + base64.b64encode(
        json.dumps(PARTIAL, ensure_ascii=False).encode()).decode()[:3800], flush=True)

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def cosine_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / (norms + 1e-8)
    return Xn @ Xn.T

def main():
    t0 = time.time()
    corpus = _load_corpus()
    # 展平语料 (口径与正本 extract_representations.py 同)
    sentences, labels, stages, variant_indices, concept_names = [], [], [], [], []
    for cid, data in corpus["paraphrases"].items():
        for vi, s in enumerate(data["sentences"]):
            sentences.append(s)
            labels.append(cid)
            stages.append(data["stage"])
            variant_indices.append(vi)
            concept_names.append(data["name"])
    N = len(sentences)
    fl("corpus", n_sentences=N, n_concepts=len(corpus["paraphrases"]),
       corpus_sha=_CORPUS_SHA[:16])

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers import BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(MOUNT)
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    if QUANT == "4bit":
        qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(
            MOUNT, quantization_config=qc, device_map="auto", attn_implementation="eager")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MOUNT, torch_dtype=torch.float16, device_map="auto", attn_implementation="eager")
    model.eval()
    L = int(model.config.num_hidden_layers)
    H = int(model.config.hidden_size)
    fl("model", arch=model.config.architectures[0], layers=L, hidden=H, device=DEV)

    # 层采样: 每 LAYER_SKIP 层取一层, 必含末层
    total = L + 1
    layer_indices = list(range(0, total, LAYER_SKIP))
    if layer_indices[-1] != total - 1:
        layer_indices.append(total - 1)
    n_ext = len(layer_indices)
    fl("layers", indices=layer_indices, n_sampled=n_ext)

    # ── [1] 逐句一次前向, 三策略同出 ──────────────────────────────
    lt = np.zeros((N, n_ext, H), dtype=np.float32)
    mp = np.zeros((N, n_ext, H), dtype=np.float32)
    ent = np.zeros((N, n_ext, H), dtype=np.float32)
    ent_ids_cache = {}
    t1 = time.time()
    for i in range(N):
        ids = tok(sentences[i], return_tensors="pt", add_special_tokens=False).to(DEV)
        with torch.no_grad():
            out = model(**ids, output_hidden_states=True)
        hs = out.hidden_states  # tuple, len=total, hs[0]=embedding
        inp = ids.input_ids[0].tolist()
        cname = concept_names[i]
        if cname not in ent_ids_cache:
            ent_ids_cache[cname] = tok(cname, add_special_tokens=False).input_ids
        eids = ent_ids_cache[cname]
        epos = -1
        for s0 in range(len(inp) - len(eids) + 1):
            if inp[s0:s0 + len(eids)] == eids:
                epos = s0
                break
        if epos == -1:  # fallback: 中间 token (与正本同)
            epos = len(inp) // 2
        for j in range(n_ext):
            h = hs[layer_indices[j]][0]
            lt[i, j] = h[-1].float().cpu().numpy()
            mp[i, j] = h.float().mean(0).cpu().numpy()
            ent[i, j] = h[epos].float().cpu().numpy()
        if (i + 1) % 20 == 0 or i == N - 1:
            print(f"  [{i+1}/{N}] {time.time()-t1:.0f}s", flush=True)
            fl("extract", done=i + 1, elapsed_s=round(time.time() - t1, 1))
    np.savez_compressed(
        os.path.join(HERE, "reps_geometry_kaggle.npz"),
        last_token=lt, mean_pool=mp, entity=ent,
        labels=np.array(labels), stages=np.array(stages),
        variant_idx=np.array(variant_indices),
        sentences=np.array(sentences, dtype=object),
        concept_names=np.array(concept_names, dtype=object),
        layer_indices=np.array(layer_indices))
    fl("encode_done", n_sentences=N, elapsed_s=round(time.time() - t0, 1))

    # ── [2][3] 逐层几何量 (口径与本地 analyze_geometry.py 逐同) ────
    concept_groups = {}
    for i, cid in enumerate(labels):
        concept_groups.setdefault(cid, []).append(i)
    concept_ids = sorted(concept_groups.keys())

    results = {}
    for strat, reps in [("last_token", lt), ("mean_pool", mp), ("entity", ent)]:
        per_pos, per_neg, per_margin, per_cka = [], [], [], []
        for j in range(n_ext):
            lr = reps[:, j, :]
            pos_sims = []
            for cid in concept_ids:
                idx = concept_groups[cid]
                for a in range(len(idx)):
                    for b in range(a + 1, len(idx)):
                        pos_sims.append(cosine_sim(lr[idx[a]], lr[idx[b]]))
            rng = np.random.default_rng(SEED)
            samp = rng.choice(len(concept_ids), size=min(NEG_SAMPLE, len(concept_ids)), replace=False)
            neg_sims = []
            for ia in range(len(samp)):
                for ib in range(ia + 1, len(samp)):
                    ca = concept_ids[int(samp[ia])]
                    cb = concept_ids[int(samp[ib])]
                    neg_sims.append(cosine_sim(
                        lr[concept_groups[ca][0]], lr[concept_groups[cb][0]]))
            pm = float(np.mean(pos_sims))
            nm = float(np.mean(neg_sims))
            per_pos.append(pm)
            per_neg.append(nm)
            per_margin.append(pm - nm)
            cka_vals = []
            for cid in concept_ids:
                idx = concept_groups[cid]
                cka_vals.append(cosine_sim(lr[idx[0]], lr[idx[1]]) ** 2)
            per_cka.append(float(np.mean(cka_vals)))
        v = {
            "last3_pos_mean": float(np.mean(per_pos[-3:])),
            "last3_neg_mean": float(np.mean(per_neg[-3:])),
            "last3_margin": float(np.mean(per_pos[-3:]) - np.mean(per_neg[-3:])),
            "G3_cka": float(np.mean(per_cka[-3:])),
        }
        v["G1_pass"] = bool(v["last3_pos_mean"] > G1_THRESHOLD
                            and v["last3_pos_mean"] > v["last3_neg_mean"])
        v["G3_pass"] = bool(v["G3_cka"] > G3_THRESHOLD)
        results[strat] = {"pos": per_pos, "neg": per_neg, "margin": per_margin,
                          "cka": per_cka, "verdict": v}
        print(f"{strat}: pos={v['last3_pos_mean']:.3f} neg={v['last3_neg_mean']:.3f} "
              f"margin={v['last3_margin']:.3f} cka={v['G3_cka']:.3f} "
              f"G1={'PASS' if v['G1_pass'] else 'FAIL'} G3={'PASS' if v['G3_pass'] else 'FAIL'}",
              flush=True)
        fl("geometry", strategy=strat, **v)

    # ── [4] G-2 辅助读数: 末层 UMAP 同概念簇内距 vs 异概念距 ──────
    import umap
    g2 = {}
    for strat, reps in [("last_token", lt), ("mean_pool", mp)]:
        emb = umap.UMAP(n_components=2, random_state=SEED,
                        n_neighbors=UMAP_NEIGHBORS, min_dist=0.1).fit_transform(reps[:, -1, :])
        intra = []
        for cid in concept_ids:
            idx = concept_groups[cid]
            for a in range(len(idx)):
                for b in range(a + 1, len(idx)):
                    intra.append(float(np.linalg.norm(emb[idx[a]] - emb[idx[b]])))
        rng = np.random.default_rng(SEED)
        samp = rng.choice(len(concept_ids), size=min(NEG_SAMPLE, len(concept_ids)), replace=False)
        inter = []
        for ia in range(len(samp)):
            for ib in range(ia + 1, len(samp)):
                ca = concept_ids[int(samp[ia])]
                cb = concept_ids[int(samp[ib])]
                inter.append(float(np.linalg.norm(
                    emb[concept_groups[ca][0]] - emb[concept_groups[cb][0]])))
        ratio = float(np.mean(intra) / (np.mean(inter) + 1e-8))
        g2[strat] = {"umap_intra_mean": float(np.mean(intra)),
                     "umap_inter_mean": float(np.mean(inter)),
                     "intra_over_inter": ratio}
        print(f"G2-aux {strat}: intra={g2[strat]['umap_intra_mean']:.3f} "
              f"inter={g2[strat]['umap_inter_mean']:.3f} ratio={ratio:.3f} (<1 = 同概念更紧)",
              flush=True)
        fl("g2_aux", strategy=strat, **g2[strat])

    # ── [5] 图 (对齐本地 figs_geometry 四件) ────────────────────────
    LAYERS_VIZ = [layer_indices[0], layer_indices[n_ext // 2], layer_indices[-1]]
    for strat, reps in [("last_token", lt), ("mean_pool", mp), ("entity", ent)]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
        ax = axes[0]
        ax.plot(range(n_ext), results[strat]["pos"], "s-", color="#21918c", lw=1.6, ms=4,
                label="Same concept (pos)")
        ax.plot(range(n_ext), results[strat]["neg"], "^-", color="#b8860b", lw=1.6, ms=4,
                label="Diff concept (neg)")
        ax.axhline(G1_THRESHOLD, color="#d62728", ls="--", lw=1.0,
                   label=f"G-1 threshold {G1_THRESHOLD}")
        ax.set_xlabel(f"Layer index (sampled x{LAYER_SKIP}; 0=embedding, {layer_indices[-1]}=last)")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title(f"{strat}: Same vs Diff Concept Cosine (per layer)")
        ax.legend(frameon=False, fontsize=9)
        ax.set_ylim(-0.1, 1.05)
        ax = axes[1]
        ax.plot(range(n_ext), results[strat]["margin"], "o-", color="#440154", lw=1.6, ms=4)
        ax.axhline(0, color="#666666", lw=0.8)
        ax.set_xlabel("Layer index")
        ax.set_ylabel("Margin (pos - neg)")
        ax.set_title(f"{strat}: Geometric Margin")
        ax.set_ylim(min(-0.1, min(results[strat]["margin"]) - 0.05),
                    max(0.8, max(results[strat]["margin"]) + 0.05))
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS, f"fig_{strat}_cosine_margin.png"))
        plt.close(fig)

    for strat, reps in [("last_token", lt), ("mean_pool", mp), ("entity", ent)]:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=150)
        sub_indices, sub_labels = [], []
        for cid in concept_ids[:10]:
            sub_indices.extend(concept_groups[cid])
            sub_labels.extend([cid] * 4)
        for ax_idx, layer in enumerate(LAYERS_VIZ):
            sim_matrix = cosine_matrix(reps[sub_indices, layer, :])
            ax = axes[ax_idx]
            im = ax.imshow(sim_matrix, cmap="RdYlGn", vmin=-0.5, vmax=1.0, aspect="auto")
            ax.set_title(f"Layer {layer} ({'embedding' if layer == 0 else 'last' if layer == layer_indices[-1] else 'mid'})")
            ax.set_xlabel("Sentence Index")
            ax.set_ylabel("Sentence Index")
            for b in range(4, 40, 4):
                ax.axhline(b - 0.5, color="white", lw=0.5, alpha=0.5)
                ax.axvline(b - 0.5, color="white", lw=0.5, alpha=0.5)
            plt.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"{strat}: Cosine Similarity Heatmap (10 concepts x 4 variants)", fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS, f"fig_{strat}_heatmap.png"))
        plt.close(fig)

    colors = plt.cm.tab20(np.linspace(0, 1, len(concept_ids)))
    for strat, reps in [("last_token", lt), ("mean_pool", mp)]:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=150)
        for ax_idx, layer in enumerate(LAYERS_VIZ):
            emb = umap.UMAP(n_components=2, random_state=SEED,
                            n_neighbors=UMAP_NEIGHBORS, min_dist=0.1).fit_transform(
                                reps[:, layer, :])
            ax = axes[ax_idx]
            for ci, cid in enumerate(concept_ids):
                indices = concept_groups[cid]
                ax.scatter(emb[indices, 0], emb[indices, 1], c=[colors[ci]], s=30, alpha=0.7)
                for a in indices:
                    for b in indices:
                        if a < b:
                            ax.plot([emb[a, 0], emb[b, 0]], [emb[a, 1], emb[b, 1]],
                                    color=colors[ci], alpha=0.15, lw=0.5)
            ax.set_title(f"Layer {layer} ({'embedding' if layer == 0 else 'last' if layer == layer_indices[-1] else 'mid'})")
            ax.set_xlabel("Dim 1")
            ax.set_ylabel("Dim 2")
        fig.suptitle(f"{strat}: UMAP Projection (same-concept linked)", fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS, f"fig_{strat}_umap.png"))
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for strat in ["last_token", "mean_pool", "entity"]:
        ax.plot(range(n_ext), results[strat]["cka"], "o-", lw=1.6, ms=4, label=strat)
    ax.axhline(G3_THRESHOLD, color="#d62728", ls="--", lw=1.0,
               label=f"G-3 threshold {G3_THRESHOLD}")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("CKA (cosine^2)")
    ax.set_title("CKA Alignment per Layer (same concept, diff variants)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_cka_per_layer.png"))
    plt.close(fig)
    fl("figs_done", n_figs=len(os.listdir(FIGS)))

    # ── 判据汇总 + 落盘 ──────────────────────────────────────────────
    metrics = {
        "arm": "#__ARM_TAG__#",
        "platform": "kaggle",
        "minted": time.strftime("%Y-%m-%d"),
        "question": "同概念不同表达的深层几何坍缩在 7B 规模是否成立 (G-1..G-3)?",
        "hypotheses": {
            "H1": "7B 容量下同概念不同表达在深层坍缩更显著 (正样本余弦形态高于 0.5B)",
            "H2": "几何边距 (pos-neg) 随层加深, 末层达峰",
            "H3": "三策略排序与 0.5B 同形 (last_token ≈ mean_pool > entity)",
        },
        "model": {"mount": MOUNT, "quant": QUANT,
                  "arch": model.config.architectures[0],
                  "layers": L, "hidden_dim": H, "device": DEV},
        "corpus": {"deck": corpus.get("deck"),
                   "n_sentences": N, "n_concepts": len(concept_ids),
                   "source": corpus.get("source"), "sha256_16": _CORPUS_SHA[:16]},
        "layer_indices": layer_indices,
        "knobs": {"SEED": SEED, "LAYER_SKIP": LAYER_SKIP,
                  "UMAP_NEIGHBORS": UMAP_NEIGHBORS, "NEG_SAMPLE": NEG_SAMPLE,
                  "G1_THRESHOLD": G1_THRESHOLD, "G3_THRESHOLD": G3_THRESHOLD},
        "strategies": results,
        "g2_aux": g2,
        "vs_05b_local": {
            "last_token": {"last3_pos_mean": 0.9560886737373139, "last3_neg_mean": 0.9255487206735108,
                           "last3_margin": 0.030539953063803083, "G3_cka": 0.9016327373654262},
            "mean_pool": {"last3_pos_mean": 0.964401524066251, "last3_neg_mean": 0.9155182492314724,
                          "last3_margin": 0.04888327483545274, "G3_cka": 0.899818049920687},
            "entity": {"last3_pos_mean": 0.58360329462456, "last3_neg_mean": 0.8638554916047213,
                       "last3_margin": -0.2802521993584757, "G3_cka": 0.1572047174156259},
            "note": "0.5B 本地 metrics_geometry.json; 跨模型只比形态 (曲线走势/策略排序), "
                    "不比绝对值 (bnb-4bit 量化容差)",
        },
        "g4_note": "G-4 (几何记忆 vs 联想记忆训练消融) 不在本推理臂裁决; "
                   "0.5B 本地 G-4 FAIL 已在案 (高 LR+WD 反破坏 margin), 训练臂后续",
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(OUT_METRICS, "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=1)
    verdicts = {k: results[k]["verdict"] for k in results}
    print("GEOM_DONE::" + base64.b64encode(json.dumps(
        {"arm": "#__ARM_TAG__#", "elapsed_s": metrics["elapsed_s"],
         "verdicts": verdicts, "g2_aux": g2}).encode()).decode(), flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("GEOM_ERROR::" + traceback.format_exc(), flush=True)
        raise
