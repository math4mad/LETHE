# AGE-LADDER 年阶语料梯 —— 设计裁定书

> 主人裁定 (2026-09-24, 本会话): 「0-5-10-15 的语料阶梯，才是该喂进模型的世界。」
> 本项目从"幼儿语料实验"升格为"发生学语料梯"——训练对象不是通用 AI，
> 而是认知引擎（义素基 + MaxSim 锚点 + 序贯贝叶斯）的**成长轨迹本身**。

## 0. 核心论点：两种缺失，靶心是第二种

| | 稀有域缺失 (neutralino.js 型) | 发生序缺失 (本项目的靶) |
|---|---|---|
| 症状 | 某领域模型不会 | 所有模型都不知道概念是"长"出来的 |
| 成因 | 语料量小 | 互联网 = 成人写给成年人的终态快照 |
| 修法 | 堆数据 | 只有按年龄层排序的课程语料能修 |
| 架构 | Transformer 可吃 | Transformer 吃不下 —— 序贯贝叶斯天生吃这个 |

引擎对应物:
- **时间轴指针** (external/《时间轴的指针》手稿) = 逐层喂入时先验的连续演化;
- **MaxSim 锚点机制** ≡ 皮亚杰"集中化"(centration) —— 引擎天生是幼儿心智;
- **球生命线** (每颗概念球的出生层、撑大事件) = 本项目的核心观测量。

## 1. 语料缺席的四种机制 (评测必须分层归因)

1. **门禁型**: CHILDES/TalkBank —— 需注册登录，爬虫结构性到不了 → 大概率完全不在任何 web 抓取语料 (C4/FineWeb) 内。
2. **非文本型**: 0-5 纸板书 = 图片为主、全文 50-150 词 —— 没有文本层可爬，缺失与版权无关。
3. **截断型**: DK 452 本中 2025-26 新品 —— 在一切模型知识截止之后 (可实测: 盲区探针)。
4. **序缺失型**: Gutenberg 公版绘本 **反而不缺** (The Pile 有 Gutenberg-PGBooks 子集) ——
   缺的是"它被当成第几周、第几层喂进来"。此条是最反直觉的预言, 论文卖点。

## 2. 梯子 (公版优先, 每层标授权与获取路径)

| 层 | 认知阶段 | 语料源 | 授权状态 |
|---|---|---|---|
| **L0 0-2** | 感知运动 | DK Peekaboo/Touch&Feel 层 (86本 已目检);  Nursery rhymes: Real Mother Goose (PG#10641) | 书目=事实免版权;  rhyme=公版 |
| **L1 2-3** | 命名期 | DK first-words/bilingual 层 (66本); Winnie-the-Pooh 1926 (美区公版); Beatrix Potter (部分公版) | 公版 + 书目 |
| **L2 3-5** | 分类/前运算 | DK 色数形/动物层 (72本); Alice(PG#11), Wind in the Willows(PG#863), Wizard of Oz(PG#2600), Jungle Book(PG#400), Andersen/Grimm; **真实幼儿语言: CHILDES Brown(Bretherton)/Bernstein/NewBrunson** | 全公版; CHILDES 需 TalkBank 账号 (人工一步) |
| **L3 6-9** | 具体运算 | **McGuffey Readers 全级** (公版历史课程梯, 天然的 6→15 分级!), Treasure Island, Black Beauty, Swiss Family Robinson; 公版自然启蒙读本 (natural philosophy primers) | 公版 |
| **L4 10-14** | 分类推理 | 公版少年科普/哈佛经典选段/Search-and-find 式百科公版前身 | 公版 |
| **L5 15+** | 形式运算 (对照) | 即现代 LLM 的默认世界: 成人 web 文本 —— 作为"终态快照"对照组 | — |
| 探针 | 假信念/守恒 | Piaget 经典任务文本化 (高瘦杯/月亮在走/巧克力搬家) | 自制 |

设计约束:
- 逐层喂入, 后验不重置 → 记录球生命线;
- 每层同时跑"一次性混喂"对照臂 —— 证明**序本身**有效应 (论文主判决);
- 三层配方 (DK骨架/Gutenberg正文/CHILDES对话) = L0-L2 的实装, 不变。

## 3. 可测预言 (写作时逐条判)

- P1 CHILDES 系不在主流开放语料 (可用稀有 n-gram 探针实测);
- P2 公版绘本在 (PGBooks 血脉), 但"按序发生"信息在所有模型中为零 → 守恒/泛灵探针上成人模型**不会**表现出幼儿特征, 而逐层引擎**应该**表现;
- P3 引擎逐层训练后, "月亮"先入动物球后迁出 (泛灵→去泛灵) —— 发展可逆性事件;
- P4 452 本书名以 ≤12 维义素基张成 (幼儿语言的天然低维性)。

## 4. 文献锚 (诚实定位, 防夸大)

TinyStories (Eldan & Li 2023): 小规模儿童级合成语料已足以训出连贯语言 ——
说明"0-5 语料能教说话"不是新闻; **新闻是"教次序"**。发展机器人学
(cognitive robotics,babbling literature) 与 McGuffey 历史课程是先行脉络, 方法节必引。

## 5. 目录约定

```
corpus/ladder05/
  layers/dk452/     L0-L2 骨架层 (书目已入库, sha 见 manifest)
  layers/gutenberg/ L1-L3 正文层 (fetch 脚本 + 公版核对)
  layers/childes/   L2 对话层 (门禁说明 + 登录后 fetch 脚本)
  layers/mcguffey/  L3-L5 历史课程梯 (待拉)
  analysis/         分层/义素统计、探针判分、球生命线
```
