# PREREG — Friends 季节演化 · Kaggle 夜航 (判据先冻)

## 案
主人观感: "到了第三季和第四季, Rachel 才表现出特色"
假设: Rachel 在 S03/S04 的台词表示更凝聚 (舱内相似度↑)、与其余角色更分离 (质心距↑)
对照 H-S3: 若六角色同指标随季齐变 = 季间语料漂移, 非 Rachel 特异
背景: 本地四季版 (S01–S04, 0.5B) 读数 = 观感**未获支持且方向相反** (Rachel 舱内相似 S01 0.8937 → S02 0.9065 → S03 0.8634 → S04 0.8640; S03/S04 反降)。但主人质疑本地 0.5B 容量, 点菜 7B 复跑; 且本地为 MPS 单点, 须跨平台对表。

## 正本声明 (三样交付律)
- **正本**: `benches/transformer_vis/friends_seasonal.py` + `corpus_friends_s1234.json` + `friends_s1234_raw.json` + 本地 `metrics_friends_s1234.json` (对表基准)
- **投影**: 本目录两臂 `run.py` (由 `run_template.py` 生成; 语料内嵌 b64+sha256 自校验, 与正本逐位同源)
- **verify 命令**: `python3 verify_friends_kaggle.py <kaggle_metrics.json>` — 退出码 0 = 臂 A 全指标在容差内 (K-1 通过); 1 = 有超差项

## 臂
- **A `friends-seasonal-0-5b`** — Qwen2.5-0.5B-Instruct bf16 (挂载 `qwen-lm/qwen2.5/transformers/0.5b-instruct/1`), 自检臂: 复册本地数字
- **B `friends-seasonal-7b`** — Qwen2.5-7B-Instruct bnb-4bit (compute fp16, 挂载 `qwen-lm/qwen2.5/transformers/7b-instruct/1`), 主人点菜「Qwen2.5-7B 复跑」
  - K-4 注记: 本地无 7B bf16 基线, bnb-4bit 量化偏差如实注记; 跨模型读数**只比形态 (时间线斜率/对照 H-S3), 不比绝对值**

## 冻结容差 (臂 A 自检 vs 本地 metrics_friends_s1234.json)
- 余弦类指标 (within / centroid dist / flow sims): |Δ| ≤ 1e-2
- 探针精度 (probe_lead_vs_rest / 6-class probe): |Δ| ≤ 0.02
- 流 n_sentences / n_tokens: 逐位相等 (tokenizer 同源同版)
- 参考数 (本地, 全精度):
  - rachel_within_last: S01 0.893710 / S02 0.906500 / S03 0.863380 / S04 0.863977
  - others_within_mean: 0.888314 / 0.910756 / 0.883183 / 0.879113
  - rachel_centroid_dist: 0.002572 / 0.002015 / 0.003414 / 0.002552
  - probe_lead_vs_rest: 0.756667 / 0.766667 / 0.776667 / 0.786667
  - flow: adj_same 0.912933 (n=12) / adj_diff 0.908055 (n=73) / 86 句 / 1471 token

## 判读条款
- **K-1**: 臂 A 全部指标在容差内 → 平台迁移成立, 本地四季读数为真 (S02 修复后的四季版结论成立)
- **K-2**: 臂 B Rachel within 时间线形态 (升/平/降) 记录; 7B 容量下观感是否翻案
- **K-3**: 对照 H-S3 — 其余五角色均值随季变化与 Rachel 同形 → 季间漂移; 异形 → Rachel 特异
- **K-4**: 臂 B 4bit 注记, 不裁决

## 失败条款
- OOM → FLOW_MAX_TOKENS 减半 (750) 重推
- bnb 不可用 → 落 SEASONAL_ERROR 日志, 回收后改 bf16 重推
- 挂载 404 → 列 /kaggle/input/models 诊断
- 席位满 ('Maximum batch GPU session count of 2 reached') → 排队等位, 勿高频重推 (GitHub 式节流同理)

## 旋钮 (园规·参数意愿律: 每个数一行)
- SEED = 13
- TSNE_PERPLEXITY = 15
- PROBE_FOLDS = 5
- FLOW_MAX_TOKENS = 1500
- 编码批大小 = 1 (逐条前向, 与本地正本一致)
- 臂 A dtype = bfloat16 / 臂 B = bnb-4bit (compute fp16)
- attn_implementation = eager (output_attentions 所需, 与正本一致)
