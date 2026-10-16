#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geometry of Meaning — Kaggle 夜航臂 (geometry-meaning-7b)

正本声明 (三样交付律):
  正本  = /Users/mac/CodeBuddy/Geometry-Of-Meaning/
            extract_representations.py (逐层提取, 三策略)
            analyze_geometry.py        (逐层几何量, G1-G4 判据)
            corpus_paraphrase.json     (语料正本, 50 概念 × 4 变体)
            metrics_geometry.json      (0.5B 本地对表基准)
  本臂  = 正本的 Kaggle 投影 (模型: Qwen2.5-7B-Instruct (bnb-4bit, 挂载 qwen-lm))
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
  [5] 图: cosine+margin 曲线 / 热力图 / UMAP / CKA 曲线 (对齐本地 figs_geometry); figs 全包 try/except（单图失败不杀进程）

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
_CORPUS_B64 = "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"
_CORPUS_SHA = "5d9dd063d27f15a90aa11df54259bf066f27f658271ac34837e673b01bc52a31"

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
MOUNT = "/kaggle/input/models/qwen-lm/qwen2.5/transformers/7b-instruct/1"   # Kaggle 模型挂载
QUANT = "4bit"   # "4bit" (bnb, compute fp16)
if QUANT == "4bit":  # Kaggle 镜像不预装 bitsandbytes (friends 7b 臂 10-13 实测)
    # 注: 本臂 metadata enable_internet=true (仅为此安装; PREREG 记此偏离)
    # 钉 0.50.2 —— friends-seasonal-7b 臂 10-13 实测通版; >=0.46.1 的 -U 在
    # 10-15 首射 ERROR (版本漂移疑为死因: 新版 bnb 与镜像 torch/CUDA 不匹配)
    _osx.system("python -m pip install -q 'bitsandbytes==0.50.2'")  # bnb 疫苗(钉版)
    _osx.system("python -m pip show bitsandbytes | grep -i version")    # 版本留痕
# ---------------------------------------------------------------------------

HERE = "/kaggle/working"
FIGS = os.path.join(HERE, "figs_geometry_kaggle")
OUT_METRICS = os.path.join(HERE, "metrics_geometry_kaggle.json")
os.makedirs(FIGS, exist_ok=True)

PARTIAL = {"arm": "geometry-meaning-7b", "model_mount": MOUNT, "quant": QUANT}

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
        PARTIAL.setdefault("verdicts", {})[strat] = v  # 分策略存, 防覆盖
        print(f"{strat}: pos={v['last3_pos_mean']:.3f} neg={v['last3_neg_mean']:.3f} "
              f"margin={v['last3_margin']:.3f} cka={v['G3_cka']:.3f} "
              f"G1={'PASS' if v['G1_pass'] else 'FAIL'} G3={'PASS' if v['G3_pass'] else 'FAIL'}",
              flush=True)
        fl("geometry", strategy=strat, **v)

    # ── [4] G-2 辅助读数: 末层 UMAP 同概念簇内距 vs 异概念距 ──────
    # 注: UMAP 回迁（v4/v5 TSNE 在 Kaggle 镜像 segfault; UMAP 在 v2 figs 成功）
    import umap
    g2 = {}
    for strat, reps in [("last_token", lt), ("mean_pool", mp)]:
        emb = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=UMAP_NEIGHBORS).fit_transform(reps[:, -1, :])
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
    LAYERS_VIZ_POS = [0, n_ext // 2, n_ext - 1]            # 采样轴位置 (reps 轴 1)
    LAYERS_VIZ = [layer_indices[p] for p in LAYERS_VIZ_POS]  # 绝对层号 (标题用)
    # ── figs 全部包 try/except（单图失败不杀进程；日志即幸存者）───
    def _save_fig(fig, name):
        try:
            fig.savefig(os.path.join(FIGS, name))
            plt.close(fig)
            fl("fig", name=name)
        except Exception as e:
            print(f"FIG_FAIL {name}: {e}", flush=True)
            plt.close(fig)

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
        _save_fig(fig, f"fig_{strat}_cosine_margin.png")

    for strat, reps in [("last_token", lt), ("mean_pool", mp), ("entity", ent)]:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=150)
        sub_indices, sub_labels = [], []
        for cid in concept_ids[:10]:
            sub_indices.extend(concept_groups[cid])
            sub_labels.extend([cid] * 4)
        for ax_idx, (pos, layer) in enumerate(zip(LAYERS_VIZ_POS, LAYERS_VIZ)):
            sim_matrix = cosine_matrix(reps[sub_indices, pos, :])
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
        _save_fig(fig, f"fig_{strat}_heatmap.png")

    colors = plt.cm.tab20(np.linspace(0, 1, len(concept_ids)))
    for strat, reps in [("last_token", lt), ("mean_pool", mp)]:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=150)
        for ax_idx, (pos, layer) in enumerate(zip(LAYERS_VIZ_POS, LAYERS_VIZ)):
            try:
                emb = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=UMAP_NEIGHBORS).fit_transform(
                                    reps[:, pos, :])
            except Exception as e:
                print(f"UMAP_FAIL {strat} layer={layer}: {e}", flush=True)
                continue
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
        _save_fig(fig, f"fig_{strat}_umap.png")

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
    _save_fig(fig, "fig_cka_per_layer.png")
    fl("figs_done", n_figs=len(os.listdir(FIGS)))

    # ── 判据汇总 + 落盘 ──────────────────────────────────────────────
    metrics = {
        "arm": "geometry-meaning-7b",
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
        {"arm": "geometry-meaning-7b", "elapsed_s": metrics["elapsed_s"],
         "verdicts": verdicts, "g2_aux": g2}).encode()).decode(), flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        print("GEOM_ERROR::" + tb, flush=True)
        # 错误落盘: output 可拉 (日志即幸存者协议——哨兵不仅 print)
        try:
            PARTIAL["stage"] = "ERROR"
            PARTIAL["error_tail"] = tb[-2000:]
            with open(OUT_METRICS, "w") as f:
                json.dump(PARTIAL, f, ensure_ascii=False, indent=1)
            with open(os.path.join(HERE, "geom_error.txt"), "w") as f:
                f.write(tb)
        except Exception:
            pass
        raise
