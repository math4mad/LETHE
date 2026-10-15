# 度量 spec — 三件套 + 混淆控制

> 来历：llm-geometry-analysis 的 Metric spec（What / Why it's hard / How to compute）
> 与 Architecture Spec（Choice / Why，含「因混淆而拒」写法）。

## 度量（每个三件套）

### primary_acc
- **What**：<!-- 是什么 -->
- **Why it's hard（区分度在哪）**：<!-- 什么情况下假阳性 / 假阴性 -->
- **How to compute**：<!-- 命令 + 容差理由（compare.py 的 TOL 旁也要注） -->

### 非退化度量（必填，对应 gates.json 的 spread / stable_rank 门限）
- **maxsim_by_space 极差**：各概念空间 MaxSim 的 max−min；≈0 = 引擎恒报同值（1008 病）。
- **stable_rank**：成员矩阵 ‖E‖_F²/‖E‖₂²；坍到 ~2 = 几何退化。他们实测 C2 坍至 ~2.5D
  仍能解题（几何对能力非必需），故门限只作退化报警，不作能力判据。

## 设计选择与混淆控制

| 选择 | 为什么 | 排除了什么混淆 |
|------|--------|----------------|
| <!-- 例：one-hot 义素基 | 承重件 Required | 嵌入训练噪声 --> |

> 写法范例（他们原文）：
> - 「RoPE 被拒 —— adds coupling that confounds geometry probes」（因混淆而拒）
> - 「Tied embeddings: **Required.** Geometric memory lives in this matrix」（承重件标注）
