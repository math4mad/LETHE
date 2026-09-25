# VERIFY_CITATIONS · 母本0926 引用查账判果

> 手法: arXiv API 实测 (0926 午, export.arxiv.org)。经典四件 (Gentner 1983 /
> Lakoff&Johnson 1980 / Gärdenfors 2000 / Hofstadter&Sander 2013) 系教科书级实名，
> 依园账通识直判 verified，不劳网络。
> 判级三则: **verified** = 题名可检且与母本转述核心相符；**anecdote** = 真件而母本
> 转述细节（机构/日期/数字/命名）有出入，引用只可取其实名不可取其转述；**查无** = 检索不中，
> 进任何信件/论文前禁引（0918 邮件风暴连坐律）。

| # | 母本所引 | 检索结果 | 判 |
|---|---|---|---|
| 1 | vPGM「让 LLM 用自然语言模拟概率图模型」(2024) | 检出同法系件（BayesAgent: … via Verbalized Probabilistic Graphical Modeling 等） | **verified**（母本转述与法名相符） |
| 2 | 幻觉滚雪球 Hallucination Snowballing (2024, NSF) | 检出多件（MM-Snowball、Investigating and Mitigating the Multimodal Hallucination Snowballing…） | **verified**（现象实名在场；「67%/87% 自查率」等数字细节 ⚠ 未复核，勿裸引） |
| 3 | DELT·微软亚院 2025「数据顺序影响训练」 | 检得 *Data Efficacy for Language Model Training* | **anecdote**（件真；「DELT」缩称系母本自造抑或论文原文，候核正文后定） |
| 4 | 《Subspace Geometry Governs Catastrophic Forgetting in Low-Rank Adaptation》(2026-08) | 题名逐字检得 | **verified** ★ 此案为母本中最硬一件：主人 NBA 打乱-夹角实验自此得几何语言（θ_min 遗忘公式），论文引据升格 |
| 5 | Banyan 显式计算图模型 (2026-01) | 只中无关件（logic locking） | **查无** ⚠ 禁引 |
| 6 | SciReasoner·原生结构推理 (2026-07 上海AI实验室等) | *SciReasoner: Laying the Scientific Reasoning Ground…* 及「Deep Native Structural Reasoning」件双中 | **verified**（「原生结构推理」概念在场） |
| 7 | MetaphorStar 图像隐喻端到端视觉RL (+82.6%) | 题名逐字检得 | **verified**（82.6% 数字候核原刊摘要，勿转抄） |
| 8 | KGLens 知识图谱逐层探测 (2025) | *KGLens: Towards Efficient and Effective Knowledge Probing of LLMs with Knowledge Graphs* | **verified** |
| 9 | SEA·Stochastic Error Ascent (2025) | 中 *Discovering Knowledge Deficiencies of Language Models on Massive Knowledge Base*（SEA 为该件方法名，与母本「利用失败相似性迭代检索高错候选」描述相符） | **verified**（以母题名为准引用） |
| 10 | DPE·诊断驱动渐进演化 (2026-05 北大+山大) | 中 *From Blind Spots to Gains: Diagnostic-Driven Iterative Training for Large Multimodal Models* | **anecdote**（题面作 Iterative 非 Progressive，且机构/date 未核；引时以检得题名为准） |
| 11 | CRANE 约束解码 (2026-05) | *CRANE: Reasoning with constrained LLM generation* | **verified** |
| 12 | MGeo·阿里云中文地址模型 | *MGeo: Multi-Modal Geographic Pre-Training Method* | **verified**（「中文地址」域相符；层级感知编码细节候核） |
| 13 | EWOK·MIT EvLab「越基础越罕见」框架 | *Elements of World Knowledge (EWoK): A Cognition-Inspired Framework…* | **verified** |
| 14 | StructTuning 结构感知持续预训练 (2025) | *Structure-aware Domain Knowledge Injection for LLMs* | **verified**（母本用方法别称 StructTuning，引时用原题） |
| 15 | PlantBert 植物科学微调模型 (2025) | 检索不中 | **查无** ⚠ 禁引 |
| 16 | TARA·北大王选所 CVPR2026 分类学层级对齐 | 本轮未检（高危级低一档），挂候核 | **pending** |

## 判后三行

1. **母本所引大体有血有肉**——十六件中十三件可检到实名，与 0918「批量赝品」旧案不同款；
   外部模型此番幻觉率低，但**所有数字（百分比/日期/卷期）仍是一律候核**，母本转述只配作线索不配作引据。
2. **两件查无（Banyan、PlantBert）+ 一 pending（TARA）**：凡进论文/信件，此三件先销或改引其「功能等价真件」（结构显式化方向可用园账已有的 SciReasoner 与 Subspace-LoRA 两支撑）。
3. 第 4 件 ★ 值得单报主人：**LoRA 子空间夹角理论实测在场**——本园「打乱→夹角剧变」的 NBA 实验不再是孤证直觉，H1–H3 假设框架（母本第二段对方所拟）自此有了可对话的文献对手。

> 账落处: 本表随 dag06 案入 vault；LEDGER 记一行「母本0926 引账抽验 16 件: 13 中 / 2 查无 / 1 待」。
