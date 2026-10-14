# PREREG — E-TDV 时序方向向量 (判据先冻)

## 案
出处: 主人 Qwen 帖「Transformer时序信息引入」(chat-export-1791896688253.json, 2026-10-13)
提案: v_temporal = mean(H_C1) − mean(H_C0)（末层, 逐卡差再取均值）——模型内部「语义演化曲线」的切线方向。
假设: 该向量是真实时序轴的方向导数, 非上下文长度伪影。三条判据线:
  K-1 探针: 用 v̂ 打分 C0 孤立表征, 应恢复时序轴 (基线 ρ_C0 = −0.621)
  K-2 时间机器: H_new = H_C0 + α·v̂ 应沿年龄阶单调漂移 (零样本干预)
  K-3 破 C42: LoRA ΔW 的绝对夹角已归零 (88.5°≈随机); 改量**表示偏移** ΔH 在 v̂ 上的投影 —— ord 训练应大于 shuf

## 正本声明 (三样交付律)
- **正本**: `benches/transformer_vis/temporal_direction.py` (源 of truth; 编码协议与 product_temporal.py 逐位同源)
- **派生**: `metrics_temporal_direction.json` + `figs_tdv/` + `report_temporal_direction.qmd`
- **verify 命令**: `python3 verify_temporal_direction.py metrics_temporal_direction.json` — 退出码 0 = 三判据全裁决且落账完整; 1 = 有缺项

## 臂与对照
- 语料: D5_baby (50 卡 × 5 阶, 描述不含年龄词) — 正本 corpus_baby.json
- 模型: Qwen2.5-0.5B-Instruct, 本地 MPS, attn eager
- C0 孤立 / C1 有序流 / C2 打乱 (seeds 14/15/16) — 与 D5_baby 正本同旋钮
- 对照向量 v̂_shuf = mean(H_C2) − mean(H_C0) (3 seeds 平均) —— 若 v̂_temporal 的探针 ρ 不显著强于 v̂_shuf, 则「切线」只是「有上下文」的泛化伪影

## 冻结判据 (先于实验落盘)
- **K-1**: |ρ(score = R0l·v̂_temporal, 年龄阶)| ≥ 0.55 **且** |ρ_temporal| > |ρ_shuf| → 时序切线成立
- **K-2**: 平均预测阶 rank(α) 在 α ∈ {−2,−1,−0.5,0,0.5,1,2} 上单调; 且 |rank(+2) − rank(−2)| ≥ 1.0 阶 → 时间机器成立 (方向符号如实记录, 不预设正负)
- **K-3**: proj = mean(ΔH_arm · v̂_temporal), ΔH_arm = H^C0_{LoRA} − H^C0_base; 判 proj_ord > mean(proj_shuf) → 时序课程在表示层可辨 (C42 的 ΔW 夹角归零的破局读法)
  - 注: ΔW 在权重空间、v̂ 在表示空间, 维度不同; 故 K-3 量「LoRA 引起的表示偏移在时序切线上的投影」, 不量 ΔW 本身 (此为对 Qwen 帖公式的维度修正, 落账注明)

## 失败条款
- K-1 不过 → v̂ 只是上下文伪影, 判「时序方向向量不成立」, 落账即止
- K-2 不单调 → 干预非线性, 记录 α 曲线形状, 不强行裁决
- K-3 ord ≈ shuf → 与 C42 一致: 此预算下时序课程不可辨 (两读法皆零)

## 旋钮 (园规·参数意愿律: 每个数一行)
- SEED_FIXED = 13; SHUFFLE_SEEDS = [14, 15, 16]
- LORA_R = 16; LORA_ALPHA = 32; EPOCHS = 6; LR = 1e-4; ADAM_BETA2 = 0.95; BLOCK = 128
- TARGETS = [q_proj, k_proj, v_proj, o_proj]
- ALPHA_GRID = [-2, -1, -0.5, 0, 0.5, 1, 2]
- 读数层 = 末层 (L); PCA 仅用于对照显示, 判据全在 v̂ 投影

## 范围
- 本实验为**本地臂** (MPS, 0.5B); Kaggle 跨平台复跑 (7B) 候补, 待主人点菜
- 不动 instacart_cognitive_space 两臂 (其另案处理: 挂载缺失 + bnb/py3.13 双死因已定位)

---

# 附录 E-TDV-2 残差切线 (主人 10-13 点菜「上候选下一射」; 判据先冻)

## 案
E-TDV 判明 v̂ 是上下文在场伪影 (cos(v̂, v̂_shuf)=0.994)。本射正面拆解:
扣掉伪影方向后再取切线, 验 D5_baby 是否存在**去上下文的全局时序切线**。

## 步骤 (编码协议与 E-TDV 逐位同源)
1. Δ_i = H_C1_i − H_C0_i (末层逐卡)  —— 沿用 E-TDV 已算 R0l/R1l
2. 正交化: res_i = Δ_i − (Δ_i · v̂_shuf) v̂_shuf
3. 切线 ṽ = res 矩阵对年龄阶的 PCA 主轴 (符号规约到与阶正相关)
4. K-2 时间机器用 ṽ 重做
5. 不重训 LoRA (K-3 已定论); 不重编码 (复用 E-TDV 潜伏的 R0l/R1l/R2l)

## 冻结判据
- **K-1' 探针**: |ρ(res_i·ṽ, 阶)| ≥ 0.55 且 > |ρ_v̂|=0.418 → 去伪影切线成立
- **K-2' 时间机器**: span ≥ 1.0 阶且单调 → ṽ 是可用以拨动的结构梯度
- **失败条款**: ρ 不显著 > 0.418 (Δ < 0.05) → 判「该语料无全局去伪影切线」,落账即止

## 旋钮
- 仅新增: 正交化目标 = v̂_shuf (3 seeds 平均); PCA 主轴 on 残差, SEED_FIXED=13
- 其余 (SEED/SHUFFLE/ALPHA_GRID/LoRA 旋钮) 全部沿用 E-TDV, 无改动
