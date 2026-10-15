# 度量 spec — FSSSS

> 来历：llm-geometry-analysis 的 Metric spec 三件套（What / Why it's hard / How to compute）。

## bridge_rank_agreement
- **What**：8 个探针上 FSSSS 隶属 argmax 与园引擎后验 argmax 的一致率（`report_c_bridge.json` 之 `rank_agreement`）。
- **Why it's hard**：一致不代表同构（可能两机器错得一样）；不一致也不代表不同构（可能仅先验差异）。故只作**排序同构**的弱证据，不作等价证明。
- **How to compute**：`make_metrics.py` 解析 report 的 `"7/8"` 字段。比例为精确计数，无容差。

## bridge_anchor_selfconsistency
- **What**：10 个锚点（超市 5 + 路边摊 5）各自归回本空间的比率。
- **Why it's hard**：锚点若不自洽，整个对表的参照系就垮了；它是「定义没写错」的自检，不是科学主张。
- **How to compute**：report 的 `"10/10"` 字段。

## pcs_norm_rel_err
- **What**：2D 上解析 size 与 Monte-Carlo 积分的相对误差（Bechberger 那步除法的正确性自检）。
- **Why it's hard**：MC 误差随样本数波动；0.0031 是当前样本量下的正常波动，>0.01 才指向实现 bug。
- **How to compute**：`report_pcs_prototype.json` → `normalization_2d.rel_err`。

## pcs_single_step_agreement / pcs_sequential_agreement
- **What**：PCS 后验 argmax 与园引擎后验 argmax 在单步（8 探针）/ 序贯轨迹终态上的一致率。
- **Why it's hard**：两者数学不同（密度归一 vs 截断余弦），一致说明「排序同构」在概率化后仍成立；分歧点即两语义真正分岔处。
- **How to compute**：`make_metrics.py` 按 argmax 逐条比对（序贯按词对齐）。

## pcs_posterior_spread_mean（非退化门限）
- **What**：序贯后验跨 5 概念空间的极差（max−min）的均值。
- **Why it's hard**：≈0 = 五空间平报同值（1008 病），此时一切「一致」都是平凡一致，门限全过也无意义。
- **How to compute**：`make_metrics.py`；实测 0.952。

## pcs_model_selection_margin_nats
- **What**：留出边际似然扫描 c∈{2,4,8,12,16}，**留出 test** 证据最优与最差之差（nats）。
- **Why it's hard**：c 不可辨识（各 c 证据接近）时，「PCS 能选 c」的补白主张不成立；这是 PCS 相对园引擎的核心增量。口径必须只用 test —— 选 c 与评 c 分开，用 total 会混入 train 对大 c 的单调偏好（total 口径 argmax 会漂到 c=16，与按 test 选出的 best_c=12 不一致，差为 117.6 nats，仅作辅助信息 `pcs_model_selection_total_margin_nats`，不进门限）。
- **How to compute**：`make_metrics.py` 自 `c_scan[*].test` 计算；实测 20.6 nats（−34.7 − (−55.2)），best_c=12。

## 设计选择与混淆控制

| 选择 | 为什么 | 排除了什么混淆 |
|------|--------|----------------|
| 先验 1/3, 1/3, 1/9, 1/9, 1/9 | 园引擎原 prior，原样搬 | 先验差异伪装成似然差异 |
| 锚点同池（同园词汇） | 两机器同一批锚点 | 词汇池差异 |
| c_scan 留出（train/test 分列） | 防用同一批数据选 c 又评 c | 过拟合混淆 |
| 确定性实验（无随机源） | FSSSS 积分与园引擎均确定性 | 种子方差不适用；故 gates 不设种子门限 |
