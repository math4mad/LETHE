# seme-hypercube bench — 义素语义图复现「几何记忆」

把 llm-geometry-analysis 的**人造路径-星图**换成**义素超立方体**
（节点 = 义素组合，边 = 只差一个义素），对照检验原文的「几何记忆」
性质（2D 坍缩、hop-1 强/深层弱的浅层组织、接口瓶颈）是否普遍成立。
设计动机与对话：`cora-atlas/external/idea-20261014.md`（本地 lola × ima）；
对方仓库 rsharma441/llm-geometry-analysis（Noroozizadeh et al. 2025）。

## 件

- `seme_graph.py` — **装置正本**：义素超立方体图构造 + 词表 + 边编码
  + 结构自检（清理自 idea 文件骨架，修五处，见文末「骨架清理」）
- `seme_adapter.py` — **接驳正本**：义素超立方体 → 原仓库 path-star 管线
  适配器（graph 形制 / 词表 / 超立方体感知几何度量）
- `run_smoke.py` — 冒烟驱动（B0 基线 vs C1 义素组，同模型同训练循环）
- `prereg.md` — **真预注册**（冻稿 2026-10-14，训练实验开跑前）
- `metrics-spec.md` — 度量三件套（What / Why it's hard / How to compute）
- `gates.json` — 门限正本（第一批：装置不变量；第二批：训练期门限，
  2026-10-14 训练接入后、首跑前冻结）
- `report_structure.json` — 装置自检产物（正本，确定性可重生成）
- `report_smoke.json` — 冒烟产物（派生面；训练日志在 /tmp/seme-smoke/）
- `make_metrics.py` — 聚合器（report_structure.json → metrics.json）
- `make_training_metrics.py` — 训练判读面聚合器（夜航回收
  `experiments/seme-hypercube/results/<臂>/<COND>_summary.json`
  → `metrics_training.json`；门限别名按原仓库口径）
- `metrics_training.json` — 训练期判读面（派生面，第二批门限输入）

## 运行（装置 + verify）

```bash
python3 seme_graph.py && python3 make_metrics.py \
  && python3 ../_TEMPLATE/check_gates.py --bench .
```

退出码即判据（0 = 第一批门限全过）。需 numpy。

训练期门限（第二批）判读：`python3 make_training_metrics.py &&
python3 ../_TEMPLATE/check_gates.py --bench . --stage training --metrics metrics_training.json`
（夜航回收 `<臂>/<COND>_summary.json` → `metrics_training.json` → 判第二批门限。
**必须显式指 `--metrics metrics_training.json`** —— 缺省读装置 `metrics.json`，
训练门限会全报「缺失」按不过计。缺数据同按不过计。）。

## 夜航正式对照（2026-10-14 完成，3 seeds × 40k 步）

双臂 Kaggle kernel（stage1_local_core 形制：8 层 d256，lr 2e-4，
wd 0.01，batch 8 × grad_accum 4，40k 步；种子 42/43/44）：

- B0 `math4amd/seme-hypercube-night-flight-b0-star-baseline`（v5）
- C1 `math4amd/seme-hypercube-night-flight-c1-seme-hypercube`（v4）

臂目录 `experiments/seme-hypercube/{B0,C1}/`：
`run_template.py`（逻辑正本）+ `vendor/`（原仓库 src 与
`seme_graph`/`seme_adapter` 的 vendored 副本）+ `build_arm.py`
（构建器：vendor 以 b64+sha256 自校验内嵌进 `run.py` ——
Kaggle kernels push 只执行 code_file、不携带子目录，
1014 实测 ModuleNotFoundError 后改此形制）；
`run.py` 为**派生部署面**（可随时重生成对表）。
回收纪律：训练后清 ckpt（output 包 28MB 时本机下载反复
IncompleteRead；ckpt 非判读依据，度量全在 REPORT_LINE /
汇总 json）。回收正本：`experiments/seme-hypercube/results/`。

### 正式结果（3-seed 均值；判读口径见 prereg §5）

| 指标 | B0 | C1 |
|------|-----|-----|
| hardest_token_acc（best-ckpt） | **0.9844** ✓≥0.95 | **0.5833** ✓≥0.40 |
| hardest_token_acc（终态） | 0.9766 | 0.3177 |
| steps_to_90 | 22535.7 | 未达（3/3 seeds） |
| final_auc（arm_cluster_acc） | **0.9985** ✓≥0.90 | **0.3320** ✗<0.80 |
| hop_level_cluster_acc | — | **0.9961**（随机基线 1/255=0.004） |
| hop-1 对齐 | **0.8628** ✓≥0.70 | **0.8231** ✓≥0.70 |
| hop 2–6 对齐 | **0.1580** ✓≤0.30 | **0.3415**（>0.30 未复现，<0.40 未推翻） |
| alignment_delta | 0.2755 | **0.4619** ✓≥0.05 |
| stable_rank（仅记录） | 34.67 | 27.81 |

**训练期门限 7/8 过**（`check_gates.py --stage training`，退出码 1）。
核心发现：C1 深层对齐为 B0 之 2.16×、远超无差异带（差 0.184 > 0.05），
但未达 0.40 推翻线 —— 「几何是浅的」在语义图下被显著削弱、未被推翻；
且 C1 嵌入几何按**组合深度**（跳数聚类 0.996）而非项目身份
（臂身份聚类 0.332）组织。详见 `prereg.md` §5 判读。

### 第二飞（2026-10-15，C2 消融 / C3 DeepWalk 预对齐，同图同形制）

| 指标 | C2 随机冻结 | C3 DeepWalk 预对齐 |
|------|------------|-------------------|
| hardest_token_acc（best-ckpt） | 0.3073（较 C1 跌 47%） | **0.9948** |
| steps_to_90 | 未达 | 2086（C1 3/3 seeds 未达 → 加速 ≥19×） |
| hop-1 对齐 | 0.7689 | 0.7819 |
| hop 2–6 对齐 | 0.2347 | **0.6578 ≥ 0.40 推翻线** |
| 按跳数聚类 | 0.7409 | 0.9063 |
| stable_rank | 65.00（满秩） | 7.67 |

第二飞判读（prereg §4 证伪条件）：
- **C3 推翻「几何是浅的」**：hop 2–6 对齐 0.6578 ≥ 0.40 推翻线。
  与 C1 0.3415（削弱未推翻）合读：深层几何组织程度取决于几何来源 ——
  训练自组织时弱、DeepWalk 预对齐注入时强。
- **DeepWalk 编码义素组合结构**（真 vs 随机置换对照，3-seed 均值）：
  臂身份聚类 0.3320 vs 0.1115（3.0×）、按跳数聚类 0.8815 vs 0.3411
  （2.6×）、hop 2–6 对齐 0.6258 vs 0.2068（3.0×）；训练贡献仅
  +0.032 —— 几何基本由预对齐注入，Transformer 训练小幅精修。
- **嵌入学习是组合泛化必要条件**：C2 冻结嵌入 hardest 0.3073
  较 C1 0.5833 跌 47%。
- DeepWalk 探针原始数据（`dw_*` / `dw_perm_ctrl_*`）在
  `experiments/seme-hypercube/results/C3/C3_summary.json`。

### 第三飞（2026-10-15 立臂待发射）

- **D0 界面实验臂**（prereg §6 冻结）：方向性格式
  `[EDGE] prev curr [SEP] next [EOS]`（预测位 logits[3]），
  界面为唯一变量；判据 Δhardest = D0 − C1（≥0.50 接口
  瓶颈普遍成立 / 0.05–0.50 部分 / <0.05 证伪）。第 12
  判据 `D0_hardest_token_acc ≥ 0.40` 已入 gates.json（首跑
  前冻结）；回收后按 12 判据重判。
- **E 臂 α 七档插值扫描**（prereg §7 冻结）：
  `emb(α) = α·DW + (1−α)·rnd_s`（α = i/6 七档；α=1 ≡ C3
  恒等端点；rnd_s 行范数匹配 DW 尺度，α 纯控结构比例）。
  判据（§4 已冻）：hardest(α) 曲线单调增 / 倒 U（存在
  α*）/ 单调减 三选一；门限 E0..E6 各档 hardest ≥ 0.40
  （§7 冻结，构建入 gates.json 时入册）。承科学追问
  「什么注入量最优」（Proust-Squid 倒 U 形 vs 夜航全注入
  近满分：两图定义是调节变量，Hamming 图须独立验证）。
- Proust-Squid 次飞修复清单已冻（`Proust-Squid-Land/
  results/REPORT.md`「次飞修复清单」节：EVAL_EVERY 250→50、
  GEO_THRESHOLD 5.0→3.0、scale 对照修为同范数随机嵌入、
  多种子 [42,43,44]）。

## 训练阶段（已接入原仓库，冒烟通过；正式对照已完成）

接原仓库（本机克隆 `~/Documents/GitHub/llm-geometry-analysis`，
venv `~/Programming/code-2026/llm-venv`，torch CPU）：

- `seme_adapter.to_orig_graph` 把义素超立方体包装成原仓库 graph dict
  形制（root / paths / edges_forward / edges_reverse），
  `build_dataset` / `MixedBatchSampler` / `train` / `eval_hardest_token_acc`
  原样接驳；词表按原仓库形制（[PAD]=0 …，节点名在后）
- 诊断全复用 `src/metrics.py`；共享中间节点使原 `_build_graph_node_index`
  的「paths 划分节点」假设失效，故几何度量附超立方体感知版（`*_h`），
  已在 B0 基线上对原实现 selftest 对表（确定性指标精确一致；eigsh 指标
  在原实现两跑噪声地板内——ARPACK 随机起始向量，原实现自身不可精确复现）
- 夜航臂：`experiments/seme-hypercube/{B0,C1}/`（stage1 形制，
  3 seeds，40k 步；Kaggle GPU）

### 冒烟记录（2026-10-14，3000 步截短，stage0 形制模型，seed=42）

装置 selftest 5/5 过。B0（路径-星图 128×4）vs C1（义素超立方体 D=4,V=4）：

| 指标 | B0 | C1 |
|------|-----|-----|
| hardest_token_acc final / best | 1.0000 / 1.0000 | 0.3750 / 0.3906 |
| steps_to_50 / steps_to_90 | 954 / 1816 | 未达 / 未达 |
| arm_cluster_acc | 1.0000 | 0.3320（共享中间节点，机械类比） |
| hop_level_cluster_acc | — | 0.8594（随机基线 0.2） |
| per-hop alignment | [0.86, 0.24, 0.20, 0.28] | [0.83, 0.46, 0.32, 0.36] |
| alignment_delta_mean | 0.393 | 0.492 |
| stable_rank | 29.79 | 16.49 |
| eigenspace_gap | 0.827 | 0.581 |
| principal_angle_mean | 48.9° | 24.7° |

初探信号（**不作门限判读**，步数/模型均截短）：
① B0 复现原文「几何是浅的」模式（hop-1 高、深层低）；
② C1 组合泛化弱得多，但几何对齐全面更优（gap 0.58 vs 0.83、
角度 25° vs 49°）、深层 hop 一致性更高（hop-2 = 0.458 已过预注册
§3 的 0.40 推翻线——单 seed 冒烟，正式夜航 3 seeds 验证）。
张力「几何对齐好 ≠ 组合泛化强」正是本实验要检验的核心。

## 三样交付律

- **正本声明**：门限正本 `gates.json`；装置正本 `seme_graph.py`；
  接驳正本 `seme_adapter.py`；实测正本 `report_structure.json`（装置）
  与夜航训练日志（训练）；解释框架正本 `prereg.md`（真预注册）；
  度量定义正本 `metrics-spec.md`。`metrics.json`、`report_smoke.json`、
  `verdict.json` 为**派生面**。
- **一条 verify 命令**：见上「运行」（装置）；`--stage training` 判第二批。
- **一份可读投影**：本 README + `prereg.md` + `metrics-spec.md` + `verdict.json`。
- **最近判词**（`verdict.json` / `verdict-training.json` 均为派生面，
  可由上两命令重生成；check_gates 写 verdict.json，故训练判词快照于
  `verdict-training.json`）：
  - 第一批（装置不变量）**PASS 8/8**（退出码 0）；
    gates@d3e7b20d4c464f2a × metrics@e3b05d105408d618（见 `verdict.json`）。
  - 第二批（训练期门限，含 C2/C3 第二飞）**9/11 过**（退出码 1；
    `verdict-training.json`，gates@198e4c2f × metrics@b4ceac280cc6725a
    —— 2026-10-15 主人裁定已落实为门限形式替换：绝对步数门限
    C3_steps_to_90 ≤ 2000 由相对加速比 C3_accel_ratio_min ≥ 10 取代）。
    **D0 界面门限（D0_hardest_token_acc ≥ 0.40）按 prereg §6 于
    D0 首跑前增补入册（第 12 判据）**：D0 首跑回收、指标并入
    metrics_training.json 后按 12 判据重判（当前 9/11 判读对应
    gates@198e4c2f 快照，D0 数据未到不计）：
    - ✗ C1_final_auc = 0.332 < 0.80、C3_final_auc = 0.332 < 0.90 ——
      门限所测「臂身份可分性」在 255 臂语义图上被结构钉死
      （训练前即 0.332，训练后不变），不构成能力判据，属适用性局限
      （prereg §5）；语义图自然的几何量是按跳数聚类（C1 0.996 / C3 0.906）。
    - ✓ C3_accel_ratio_min = 19.18 ≥ 10 —— 加速 ≥19×（原文约 24×）；
      旧绝对门限实测 2086 > 2000（门限余量偏紧：夜航 255 臂图 vs 原文
      512 臂图），按裁定以相对形式重判过（prereg §5 审阅者注）。
  - 装置 verify 复跑：`python3 seme_graph.py && python3 make_metrics.py
    && python3 ../_TEMPLATE/check_gates.py --bench .`
  - 训练期判读复跑：`python3 make_training_metrics.py &&
    python3 ../_TEMPLATE/check_gates.py --bench . --stage training --metrics metrics_training.json`

## 骨架清理（相对 `idea-20261014.md`）

1. 删 `encode_edge_seme` 的 `if False else` 死分支与 `_lookup` 线性扫描
   —— 直接用 `seme2token` 字典（O(1) 查表）。
2. **修特殊 token 碰撞**：原骨架用 token 0 作 `[EDGE]`/`[SEP]`，
   但 0 是合法义素组合 `[0,0,0,0]` —— 特殊 token 排词表外
   （`EDGE = V^D`, `SEP = V^D+1`）。
3. 邻接构造 O(n²·D) → O(n·D·V)（逐维变值生成邻居），天然无自环/重边，
   并断言无自环、邻接对称、每边汉明距离恰为 1。
4. 接口对齐原仓库：返回 dict 同 `build_path_star_graph` 形制；
   `encode_edge_example(u, v, vocab)` 对齐原仓库同名函数。
5. 纠算术错误：V 进制超立方体节点度 = D·(V-1) = 12、边数 1536
   （原骨架注释按二进制的 D/512 想当然）。
