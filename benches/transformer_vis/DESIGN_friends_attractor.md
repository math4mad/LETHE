# DESIGN · Friends 台词语料：Rachel / Chandler 是否为注意力尖峰（吸引子对照）

> 状态：**已完成**（2026-10-13 射击 3 交付后执行）→ `report_friends.html`
> 数据源: fangj.github.io/friends（S01 全季 4,305 句，说话人标注正本）
> 结论: Rachel/Chandler **不是**尖峰 —— 六角色均匀对话网（舱内≈0.90, 比≈1.00, 探针 0.24）

## 主人原话

> 根据 NBA 的尖峰，我想，美剧 friends 里 Rachel 和 Chandler。对照吸引子学说。这两个演员是否是台词语料里的注意力尖峰？

## 概念澄清（先记账）

- 「Rachel 和 Chandler」是**角色**（演员：Jennifer Aniston / Matthew Perry）。
  台词语料里是角色台词；若要按「演员」检验，需演员访谈/出镜语料——另一条线，先不混。
- 「NBA 的尖峰」指实验一读数：NBA 舱 = 模板驱动的**紧簇**（舱内相似度 0.973），
  注意力熵低（1.048 nats）——「尖峰」= 紧簇 + 低熵的合称。
- 「吸引子学说」：概念是状态空间中的吸引子，附近状态被吸向它。
  **台词语料里没有动力系统轨迹**，「吸引子」只能作比喻检验：
  非主角台词的表示是否被拉向主角质心（吸引域）。如实标注，不宣称为物理吸引子。

## 待检验命题

- H1（尖峰）：Rachel / Chandler 台词在嵌入空间成**紧簇**（如 NBA 舱 0.973），
  且注意力熵低——主角台词 = 高模板度。
- H2（吸引域）：流编码中，配角（Ross/Monica/Phoebe/Joey）台词的表示
  被拉向 Rachel/Chandler 质心（cos 距离 < 随机基线）。
- H3（对照）：若 H1/H2 不成立，则「尖峰」是编年语料（NBA 选秀模板）特有，
  台词语料无此结构——这本身是结构差异证据。

## 语料（待建）

- 主角：Rachel、Chandler 各 ~30 条代表性台词；
- 对照：Ross、Monica、Phoebe、Joey 各 ~30 条；
- 描述不含角色名（避免字面泄漏），只留台词正文；
- 来源：待定（本地语料 or 手动摘录；⚠ 版权——仅分析用，不入库外部/）。

## 方法（沿用已有正本）

1. `nba_vs_spring.py` 量法：逐层嵌入几何（舱内/跨舱相似度比）+ 逐层注意力熵 + 相似度矩阵。
2. `product_temporal.py` 量法：流编码（有序/打乱）+ 质心吸引度（非主角台词 → 主角质心 cos 距离，打乱流作基线）。
3. 探针：角色分类（ sanity check，台词词汇域本就不同——诚实标注，不作结构证据）。

## 旋钮（园规：参数意愿律，逐条列）

- SEED=13 / 打乱 seeds=14,15,16 / TSNE_PERPLEXITY=15 / PROBE_FOLDS=5 / UMAP_N=15
- 模型：Qwen2.5-0.5B-Instruct（`first-lora/model`），MPS，attn eager

## 交付（三样交付律）

- 正本：`friends_attractor.py` + `corpus_friends.json`
- verify：退出码 0 且 metrics_friends.json 重生成一致
- 可读投影：`report_friends.qmd` → html
