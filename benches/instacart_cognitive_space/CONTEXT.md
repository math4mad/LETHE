# Instacart 认知空间实验 (Instacart Cognitive Space)

## 1. 核心目标
验证 Transformer 在处理消费行为数据时，是否在隐藏层中自发构建了 $\text{Product} \rightarrow \text{Aisle} \rightarrow \text{Department}$ 的三级抽象金字塔，并分析“购物意图” (Intent) 在空间中的分布。

## 2. 概念空间架构 (Concept Schema)

### A. 抽象层级 (Hierarchical Abstraction)
- **L0 (叶子层): Product** $\rightarrow$ 具体商品（如 "Organic Bananas"）。
  - *关注点*: 细粒度属性（Organic vs Conventional）。
- **L1 (中层): Aisle** $\rightarrow$ 商品类目（如 "Fresh Produce"）。
  - *关注点*: 类别聚合，将相似商品归类。
- **L2 (根层): Department** $\rightarrow$ 大类部门（如 "Produce"）。
  - *关注点*: 顶级语义抽象。

### B. 意图维度 (Intent Dimensions)
将订单 (Order) 视为一个整体，通过其成员商品的分布定义意图：
- **$\text{Intent}_{\text{Healthy}}$**: 重点在 `Organic` 标记或 `Fresh Produce` 类目。
- **$\text{Intent}_{\text{Bulk}}$**: 重点在 `Household` / `Frozen` / `Paper Products` 类目，且单笔订单量大。
- **$\text{Intent}_{\text{Convenience}}$**: 重点在 `Snacks` / `Beverages` / `Frozen` 中快速消费品。

## 3. 实验链路 (Pipeline)

1. **语料提取 (Data Extraction)** $\rightarrow$ 在 Kaggle 上运行 `extract_corpus.py` $\rightarrow$ 生成抽样 `corpus_instacart.json`。
2. **夜航部署 (Night-Flight)** $\rightarrow$ 使用 `make_arms.py` 将语料 gz+b64 内嵌至 `run.py` $\rightarrow$ 推送两臂 (0.5B vs 7B)。
3. **读数分析 (Readout)** $\rightarrow$ 逐层探针 (Layer-wise Probe) $\rightarrow$ 质心漂移 (Centroid Drift) $\rightarrow$ 意图聚类 (t-SNE)。

## 4. 判据定义 (PREREG)
- **K-1 (层级递增)**: $\text{Aisle}$ 探针在 $L_{mid}$ 达峰，$\text{Department}$ 探针在 $L_{high}$ 达峰。
- **K-2 (意图可分)**: 在 7B 模型中，$\text{Healthy}$ vs $\text{Bulk}$ 意图的质心距离 $\text{dist} > \text{threshold}$。
- **K-3 (容量效应)**: 7B 模型在 $\text{Department}$ 层的探针精度显著高于 0.5B。
