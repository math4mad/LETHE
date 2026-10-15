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
- UMAP_NEIGHBORS = 15
- NEG_SAMPLE = 20（与本地同）
- G1_THRESHOLD = 0.75 / G3_THRESHOLD = 0.85
- 模型: Qwen2.5-7B-Instruct bnb-4bit（compute fp16, device_map=auto, attn eager）
- 挂载: `/kaggle/input/models/qwen-lm/qwen2.5/transformers/7b-instruct/1`（免 15GB 下载）

## 失败条款
- OOM → 降 LAYER_SKIP=4 重推（判据不变）
- bnb 缺失 → internet 安装（metadata enable_internet=true，PREREG 记此偏离）
- 挂载 404 → 诊断 `!ls /kaggle/input/models/*/*/*/`（friends 7b 臂 10-13 战法）
- 席位满（Maximum batch GPU session count）→ 排队等位，禁高频重推（夜航律）
- 源 1MB 上限 → 语料 gz+b64 内嵌（250KB → ~70KB）

## 产出
`/kaggle/working/`: `figs_geometry_kaggle/*.png` + `metrics_geometry_kaggle.json`
+ `reps_geometry_kaggle.npz`（逐层表示, 供后续分析）
回收: `kaggle kernels output math4amd/geometry-meaning-7b -p <dir>`
