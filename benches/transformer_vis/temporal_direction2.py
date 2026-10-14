#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E-TDV-2 残差切线 —— 扣掉上下文伪影后的时序切线

E-TDV 判明 v̂_temporal 是「上下文在场」伪影 (cos(v̂, v̂_shuf)=0.994)。
本射正面拆解: 把每卡 C1−C0 差对 v̂_shuf 正交投影, 残差 PCA 主轴作去伪影切线 ṽ,
验 D5_baby 是否存在**去上下文的全局时序切线**。

判据 (PREREG 附录 E-TDV-2 先冻):
  K-1' |ρ(res_i·ṽ, 阶)| ≥ 0.55 且 > |ρ_v̂|=0.418
  K-2' 时间机器 span ≥ 1.0 阶且单调
  失败条款: ρ 不显著 > 0.418 → 无全局去伪影切线

正本: 本脚本。产物: metrics_temporal_direction2.json + figs_tdv2/ (派生)
"""
import json, os, time
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "corpus_baby.json")
MODEL = os.path.expanduser("~/Programming/code-2026/first-lora/model")
SEED_FIXED = 13
SHUFFLE_SEEDS = [14, 15, 16]
ALPHA_GRID = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
RHO_VHAT_REF = 0.4184565477477216     # E-TDV K-1 v̂ 探针读数
RHO_PCA_REF = 0.621314874173432       # E-TDV PCA-PC1 对照
FIGS = os.path.join(HERE, "figs_tdv2"); os.makedirs(FIGS, exist_ok=True)

for cand in ("PingFang SC", "Arial Unicode MS", "STHeiti", "Songti SC", "Heiti TC"):
    try:
        f = font_manager.findfont(cand, fallback_to_default=False)
        if "DejaVu" not in f:
            plt.rcParams["font.family"] = cand; plt.rcParams["axes.unicode_minus"] = False; break
    except Exception:
        pass

from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(MODEL)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager").to(dev).eval()

corp = json.load(open(CORPUS))
items = corp["items"]
stage_names = corp["stages"]
texts = [f"{it['name']}：{it['desc']}。" for it in items]
N = len(texts); n_stages = len(stage_names)
stages = np.array([it["stage"] for it in items])

def all_hidden(ids, m):
    with torch.no_grad():
        return m(**ids, output_hidden_states=True).hidden_states

def enc_iso(texts_, m):
    return np.stack([
        np.stack([h[0].float().mean(0).cpu().numpy() for h in all_hidden(
            tok(t, return_tensors="pt", add_special_tokens=False).to(dev), m)])
        for t in texts_])

def enc_stream(order, m):
    stream = "\n".join(texts[i] for i in order)
    spans, pos = {}, 0
    for i in order:
        t = texts[i]; spans[i] = (pos, pos + len(t)); pos += len(t) + 1
    enc = tok(stream, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False).to(dev)
    offs = enc.pop("offset_mapping")[0].tolist()
    hs = all_hidden(enc, m)
    reps = {}
    for i in order:
        lo, hi = spans[i]
        sel = [k for k, (s, e) in enumerate(offs) if s >= lo and e <= hi and e > s]
        reps[i] = np.stack([h[0][sel].float().mean(0).cpu().numpy() for h in hs])
    return np.stack([reps[i] for i in range(N)])

t0 = time.time()
print("编码 C0/C1/C2 …", flush=True)
R0 = enc_iso(texts, model)
R1 = enc_stream(list(range(N)), model)
R2 = np.stack([enc_stream(np.random.default_rng(sd).permutation(N).tolist(), model) for sd in SHUFFLE_SEEDS])
print(f"编码完成 {time.time()-t0:.0f}s", flush=True)
R0l, R1l = R0[:, -1], R1[:, -1]
R2l = np.stack([R2[i][:, -1] for i in range(3)])

DELTA = R1l - R0l                                           # (n, d) C1−C0
vals_ctl = np.mean([R2l[i] - R0l for i in range(3)], axis=0)  # (n, d) 上下文伪影分量
vshuf = vals_ctl.mean(0); vshuf = vshuf / np.linalg.norm(vshuf)

# 正交化: 扣掉每卡差里 v̂_shuf 方向的分量
res = DELTA - np.outer(DELTA @ vshuf, vshuf)                 # (n, d)

# 残差 PCA 主轴 (符号规约到与阶正相关)
pca = PCA(n_components=2, random_state=SEED_FIXED)
score2 = pca.fit_transform(res)
pc1 = pca.components_[0]
if spearmanr(score2[:, 0], stages).statistic < 0:
    pc1 = -pc1; score2[:, 0] = -score2[:, 0]
vtilde = pc1 / np.linalg.norm(pc1)

def spearman(x, y):
    return float(spearmanr(x, y).statistic)

# K-1' 探针
scores_res = res @ vtilde
rho_res = spearman(scores_res, stages)
# 对照: 全差 PCA 主轴 (不扣伪影) 的探针 —— 与 v̂ 同族, 看正交化是否净增益
pca_full = PCA(n_components=2, random_state=SEED_FIXED)
sf = pca_full.fit_transform(DELTA)
if spearmanr(sf[:, 0], stages).statistic < 0:
    pca_full.components_[0] = -pca_full.components_[0]; sf[:, 0] = -sf[:, 0]
rho_full_pca = spearman(sf[:, 0], stages)
# 残差探针 vs 打乱残差基线 (控制: 打乱也该没时序, 但扣法不得凭空造出 ρ)
res_shuf = R2l[0] - R0l - np.outer((R2l[0] - R0l) @ vshuf, vshuf)
rho_res_shuf_ctrl = spearman(res_shuf @ vtilde, stages)

# K-2' 时间机器 (用 ṽ)
centroids = np.stack([R0l[stages == s].mean(0) for s in range(n_stages)])
centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
rank_curve, spans_by_alpha = [], []
for a in ALPHA_GRID:
    q = R0l + a * vtilde
    qn = q / np.linalg.norm(q, axis=1, keepdims=True)
    pred = (qn @ centroids.T).argmax(1)
    rank_curve.append(float(pred.mean()))
rank_curve = [float(r) for r in rank_curve]
mono = all(rank_curve[i] <= rank_curve[i+1] + 1e-9 for i in range(len(ALPHA_GRID)-1))
span = rank_curve[-1] - rank_curve[0]
true_mean_rank = float(stages.mean())

# 图
fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=150)
for s in range(n_stages):
    msk = stages == s
    ax.scatter(scores_res[msk], np.zeros(msk.sum()) + np.random.default_rng(s).normal(0, 0.02, msk.sum()),
               c=plt.cm.viridis(s / (n_stages - 1)), s=46, alpha=0.9, label=stage_names[s],
               edgecolors="white", linewidths=0.5)
ax.set_xlabel("score = res_i · ṽ (去伪影残差切线)")
ax.set_yticks([])
ax.set_title(f"K-1' 去伪影切线探针 (ρ = {rho_res:+.3f})")
ax.legend(title="年龄阶", frameon=False, fontsize=8, title_fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_tdv2_probe.png")); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=150)
ax.plot(ALPHA_GRID, rank_curve, "o-", color="#21918c", lw=1.8, ms=5)
ax.axhline(true_mean_rank, color="#666666", lw=0.9, ls="--", label=f"真实平均阶 {true_mean_rank:.1f}")
ax.set_xlabel("干预强度 α (H_new = H_C0 + α·ṽ)")
ax.set_ylabel("平均预测阶")
ax.set_title("时间机器(去伪影切线): span = %+.2f 阶" % span)
ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_tdv2_timemachine.png")); plt.close(fig)

# 判据
k1 = bool(abs(rho_res) >= 0.55 and abs(rho_res) > RHO_VHAT_REF)
k1_strict = bool(abs(rho_res) - RHO_VHAT_REF >= 0.05)
k2 = bool(abs(span) >= 1.0 and mono)

metrics = {
    "minted": time.strftime("%Y-%m-%d"),
    "experiment": "E-TDV-2 residual tangent (de-artifact)",
    "parent": "E-TDV",
    "corpus": {"deck": "D5_baby", "n": N, "stages": stage_names},
    "knobs": {"seed_fixed": SEED_FIXED, "shuffle_seeds": SHUFFLE_SEEDS,
              "alpha_grid": ALPHA_GRID, "orth_target": "v̂_shuf(3seeds平均)",
              "component": "PCA主轴 on 残差", "sign": "规约到与阶正相关"},
    "artifact": {
        "cos_vtemporal_vshuf": 0.9942976832389832,
        "orth_gain_rho": {"rho_res": rho_res, "rho_full_pca_none_orth": rho_full_pca,
                          "rho_vhat_ref": RHO_VHAT_REF, "rho_pca_pc1_ref": RHO_PCA_REF,
                          "rho_res_shuf_ctrl": rho_res_shuf_ctrl},
        "orth_note": "PDF: 正交化让探针从 mean偏移族的0.418→残差, 目标是与PCA-PC1的0.621同量级再超"
    },
    "K1_residual_probe": {
        "rho_residual_tangent": rho_res, "rho_plain_diff_pca": rho_full_pca,
        "rho_vhat_ref_E-TDV": RHO_VHAT_REF, "rho_pca_pc1_ref": RHO_PCA_REF,
        "rho_residual_shuf_ctrl": rho_res_shuf_ctrl,
        "gain_vs_vhat": rho_res - RHO_VHAT_REF,
        "verdict": ("去伪影切线成立" if k1 else
                    ("无全局去伪影切线 (与v̂同族, 无净增益)" if not k1_strict
                     else "部分增益但未达0.55门槛")),
        "pass": k1,
    },
    "K2_timemachine": {
        "alpha_grid": ALPHA_GRID, "mean_predicted_rank": rank_curve,
        "true_mean_rank": true_mean_rank, "monotone": mono, "span": span,
        "verdict": "拨得动" if k2 else "拨不动",
        "pass": k2,
    },
    "elapsed_s": time.time() - t0,
}
json.dump(metrics, open(os.path.join(HERE, "metrics_temporal_direction2.json"), "w"),
          ensure_ascii=False, indent=1)
print(json.dumps({
    "K1'": {"rho_res": rho_res, "rho_plain_pca": rho_full_pca,
            "rho_vhat_ref": RHO_VHAT_REF, "rho_pca_ref": RHO_PCA_REF,
            "shuf_ctrl": rho_res_shuf_ctrl},
    "K2'": {"curve": [round(r, 2) for r in rank_curve], "mono": mono, "span": round(span, 2)},
    "elapsed_s": round(time.time() - t0),
}, ensure_ascii=False, indent=1))
print("DONE — metrics_temporal_direction2.json + figs_tdv2/")