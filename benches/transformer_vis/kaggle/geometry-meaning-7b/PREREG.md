# PREREG — Geometry of Meaning · 7B Kaggle 夜航臂 (geometry-meaning-7b)

## 案由
主人与 Qwen 对话「几何记忆 vs 联想记忆」→ Geometry-Of-Meaning 项目（buzz 座席 2026-10-15 立）
→ 0.5B 本地全流程（G-1/2/3 PASS，G-4 FAIL）→ 主人 10-15 点「大模型 Kaggle 实验」
→ 移交 lola 排队执行（排期权在主人，10-15 已点）。

## 正本（三样交付律）
- 正本: `~/CodeBuddy/Geometry-Of-Meaning/`
  `{extract_representations.py, analyze_geometry.py, corpus_paraphrase.json}`
  + 本地 `metrics_geometry.json`（0.5B 对表基准）
- 本臂: `benches/transformer_vis/kaggle/geometry-meaning-7b/`
  （正本的 Kaggle 投影；语料 gz+b64 内嵌 + sha256 对撞正本）
- 生成器: `../make_geometry_arm.py`（臂的唯一来源，禁手改 run.py）
- verify: `python3 verify_geometry_kaggle.py <kaggle-output-dir>`（退出码即判据）

## 假设
- H1: 7B 容量下同概念不同表达的深层坍缩更显著（正样本余弦形态高于 0.5B）
- H2: 几何边距 (pos−neg) 随层加深、末层达峰
- H3: 三策略排序与 0.5B 同形（last_token ≈ mean_pool > entity）

## 冻结判据（先于数据，事后不改）
- **K-1 (G-1)**: 末 3 层正样本余弦 > 0.75 且 > 负样本（last_token / mean_pool / entity 各自裁决）
- **K-2 (G-2)**: 末层 UMAP 中同概念 4 变体聚为紧密簇（图判 + 辅助读数 intra/inter < 1）
- **K-3 (G-3)**: 末 3 层 CKA（= cosine(v0,v1)² 口径，与本地同）> 0.85
- **K-4 (形态对表)**: 与 0.5B 本地 metrics_geometry.json 比曲线走势与策略排序；
  跨模型**只比形态不比绝对值**（bnb-4bit 量化容差）
- G-4 注记: 训练消融（几何记忆 vs 联想记忆）不在本推理臂裁决；
  0.5B 本地 G-4 FAIL 已在案（高 LR+WD 反破坏 margin），训练臂后续

## 旋钮（园规·参数意愿律）
- SEED = 42（与本地 analyze_geometry.py 的 neg 采样同号）
- LAYER_SKIP = 2（7B 28L → 15 层采样，含 0 与末层）
- t-SNE perplexity = 15（**v3 起 t-SNE 替代 UMAP**——v2 死于 UMAP 段，Kaggle 免费档 umap-learn 不稳；与 friends-seasonal-7b 臂同法，PREREG 偏离注记）
- NEG_SAMPLE = 20（与本地同）
- G1_THRESHOLD = 0.75 / G3_THRESHOLD = 0.85
- 模型: Qwen2.5-7B-Instruct bnb-4bit（compute fp16, device_map=auto, attn eager）
- 挂载: `/kaggle/input/models/qwen-lm/qwen2.5/transformers/7b-instruct/1`（免 15GB 下载）

## 战况
- v1 (17:03) ERROR @8min：bnb `-U '>=0.46.1'` 版本漂移疑 → v2 钉 `bitsandbytes==0.50.2`（friends 7b 臂 10-13 实测通版）
- v2 (17:16) ERROR @8min：**已过编码关**（200 句/355s，reps.npz 落盘），死于 UMAP→heatmap 段。v2 半程信号：mean_pool 末 3 层 pos=0.504 / neg=0.965 / margin=−0.461 / cka=0.139 —— **与 0.5B 完全反向**（朋友们 7b 4bit 舱内 0.867/0.825 基线健康，非量化噪声）；v3 存分策略 verdicts 验证全貌
- v3 (17:31) ERROR @9min：**全量 verdicts 已得**（geom_error.txt 落盘成功）：死因 = heatmap 段用绝对层号 28 索引采样轴（size 15）→ IndexError（v3 修复：位置/绝对层号双轨）。**v3 裁决**：last_token G1✅(0.886>0.835, margin+0.051) G3❌(cka 0.768)；mean_pool G1✅(0.978) G3✅(cka **0.940**，超 0.5B 的 0.900)；entity 双❌（与 0.5B entity 双 FAIL 同形——跨规模一致）。形态对表：G-1 在 7B 成立（双策略 PASS），G-3 依赖 pooling 策略（mean_pool 过、last_token 跌）
- v4 (17:46) ERROR @9min：**全量 verdicts + 6/10 figs 落盘**（heatmap ✓ cosine_margin ✓，t-SNE figs 段 segfault）。死因 = sklearn TSNE segfault（Kaggle Python 3.13 + sklearn 版本兼容问题；g2_aux 的 t-SNE 已成功算完）。g2_aux 数据完整：last_token intra=26.09/inter=18.39/ratio=1.42（同概念不更紧 ❌），mean_pool intra=12.70/inter=22.56/ratio=0.56（同概念更紧 ✓）
- v5 (17:55) ERROR @9min：同 v4 死因——sklearn TSNE segfault（Kaggle 镜像 C 层崩溃，try/except 截获不了）。g2_aux UMAP 辅助读数完整落盘（last_token intra=26.09/inter=18.39/ratio=1.42；mean_pool intra=12.70/inter=22.56/ratio=0.56），6/10 figs 成功（heatmap ✓ cosine_margin ✓，TSNE figs ✗），reps.npz 0 字节（进程被杀时文件未刷盘）
- v6 (18:01) **COMPLETE** @305s：10/10 figs ✓（9 PNG + log），reps.npz ✓，verify PASS。裁决与 v3/v4/v5 一致：last_token G1✅ G3❌(cka 0.768) · mean_pool G1✅ G3✅(cka **0.940**) · entity 双❌。g2_aux UMAP 辅助读数：last_token intra/inter=1.46（同概念不更紧 ❌），mean_pool intra/inter=0.74（同概念更紧 ✓）。**7B 结论**：G-1 在 7B 成立（双策略 PASS），G-3 依赖 pooling 策略（mean_pool 过、last_token 跌），entity 策略跨规模一致失效。
- 累计：v1 bnb 漂移 → v2 heatmap IndexError → v3/v4/v5 TSNE segfault → v6 UMAP 回迁 ✓

## 失败条款
- OOM → 降 LAYER_SKIP=4 重推（判据不变）
- bnb 缺失/版本漂移 → internet 安装，**钉 `bitsandbytes==0.50.2`**（friends-seasonal-7b 臂 10-13 实测通版；10-15 首射用 `-U '>=0.46.1'` 致 ERROR——版本漂移疑为死因，已钉版修复重推 v2）
- 挂载 404 → 诊断 `!ls /kaggle/input/models/*/*/*/`（friends 7b 臂 10-13 战法）
- 席位满（Maximum batch GPU session count）→ 排队等位，禁高频重推（夜航律）
- 源 1MB 上限 → 语料 gz+b64 内嵌（250KB → ~70KB）

## 产出
`/kaggle/working/`: `figs_geometry_kaggle/*.png` + `metrics_geometry_kaggle.json`
+ `reps_geometry_kaggle.npz`（逐层表示, 供后续分析）
回收: `kaggle kernels output math4amd/geometry-meaning-7b -p <dir>`
