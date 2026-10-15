# 度量 spec — seme-hypercube

> 来历：llm-geometry-analysis 的 Metric spec 三件套（What / Why it's hard / How to compute）。

## 第一批：装置不变量（确定性，今可跑）

### n_nodes / n_edges / node_degree_mean / node_degree_min
- **What**：超立方体的节点数（V^D）、边数（D·V^D/2）、节点度（正则 = D）。
- **Why it's hard**：图构造一旦偏斜（漏边/重边/自环），后续所有「几何性质」都在测一个坏图；不变量是装置自检，不是科学主张。
- **How to compute**：`seme_graph.py` 构造时统计；D=4, V=4 → 256 节点 / 1536 边 / 度 12（V 进制超立方体度 = D·(V-1)，V=2 二进制时才退化为 D）。

### hamming1_edge_fraction
- **What**：边上汉明距离恰为 1 的比例。
- **Why it's hard**：边定义即「只差一个义素」；若混入其他距离，「邻居=共享一个义素」的语义解读垮掉。
- **How to compute**：逐边验证 `count_nonzero(seme_a != seme_b) == 1`；应为 1.0（构造保证，属自检）。

### train_test_overlap / coverage / test_block_size
- **What**：训练/测试划分的不相交性、完备性、测试块大小（V^(D-1)=64）。
- **Why it's hard**：原文的教训是「未见必须真未见」；测试块若与训练路径共享节点，组合泛化结论即被污染。整维隔离比原文的随机叶划分更狠（未见的是整个语义维度的新取值）。
- **How to compute**：`build_seme_hypercube` 的划分逻辑直接派生。

## 第二批：训练期诊断（训练接入后冻结门限，复用原仓库 src/metrics.py）

### hardest_token_acc / final_auc / alignment_delta
- **What**：最难 token 准确率、嵌入 AUC、邻接对齐差（原仓库 `eval_hardest_token_acc` / `eval_embedding_auc` / `eval_alignment_delta`）。
- **Why it's hard**：三者分别测「记住」「排序」「几何对齐」，单独一个都会被另一种成功模式伪装；必须联合判读（原 SUCCESS_GATES 之所以是三元组）。
- **How to compute**：原仓库函数原样调用，graph 参数换为义素超立方体。

### hop1_align / hop26_align
- **What**：嵌入对 hop-1 / hop 2–6 邻接矩阵主特征向量的对齐度。
- **Why it's hard**：原文核心发现「几何是浅的」（hop-1=0.883，hop 2–6=0.10–0.18）；超立方体的组合结构可能让深层也组织起来 —— 这是本实验最有价值的可证伪预言，也最容易被「对齐指标本身偏向局部」混淆。
- **How to compute**：原仓库 `compute_eigenspace_gap` / `compute_principal_angles` 族。

### steps_to_90
- **What**：准确率到 90% 所需训练步数（`steps_to_pct`）。
- **Why it's hard**：预对齐的 ~50× 加速（780 vs 19k 步）是原文 C3 的关键优势；步数受机器与 seed 影响，故 3 seeds 取中位。
- **How to compute**：训练日志 + `steps_to_pct(log, 0.90)`。

### stable_rank / eigenspace_gap（仅记录）
- **What**：表示的有效秩与特征谱间隙（`compute_stable_rank` / `compute_eigenspace_gap`）。
- **Why it's hard**：原文 C2 坍至 ~2.5D 仍能解题 —— 退化≠无能，故只记录不作判据，防「低秩=失败」的误读。
- **How to compute**：原仓库函数原样调用。

## 设计选择与混淆控制

| 选择 | 为什么 | 排除了什么混淆 |
|------|--------|----------------|
| D=4, V=4（256 节点） | 规模够产生组合结构，又能在 CPU 复现 | 规模混淆 |
| 整维隔离划分（第 3 维==3） | 测试节点是「整个语义维度的新取值」 | 原文式随机划分可能留出语义近邻 |
| 特殊 token 排词表外（EDGE=V^D, SEP=V^D+1） | token 0 是合法义素组合 [0,0,0,0] | 特殊标记与合法 token 碰撞 |
| 3 seeds（seed=42 起） | 原文条件矩阵惯例 | 种子方差伪装成条件效应 |
| 训练/诊断全复用原仓库 | 只换图来源 | 实现差异伪装成图拓扑效应 |
