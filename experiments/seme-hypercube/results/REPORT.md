# 夜航战报 — seme-hypercube B0 / C1 / C2 / C3（B0/C1 2026-10-14 收；C2/C3 2026-10-15 收）

## 发射

| 臂 | kernel | 版本 | 设置 |
|----|--------|------|------|
| B0 路径-星图基线 | `math4amd/seme-hypercube-night-flight-b0-star-baseline` | v5 | stage1_local_core 形制：8 层 d256 / n_head 8 / lr 2e-4 / wd 0.01 / batch 8 × grad_accum 4 / 40k 步 / max_seq_len 12 / edge_path_mix 1.0；种子 42/43/44；EVAL_EVERY 500；CKPT_EVERY 5000 |
| C1 义素超立方体 | `math4amd/seme-hypercube-night-flight-c1-seme-hypercube` | v4 | 同上 |
| C2 义素+随机冻结 | `math4amd/seme-hypercube-night-flight-c2-random-frozen` | v1 | 同上；嵌入层 `requires_grad=False`（权重共享输出层随之冻结），测「无嵌入学习」时注意力结构能否解题 |
| C3 义素+DeepWalk 预对齐 | `math4amd/seme-hypercube-night-flight-c3-deepwalk-prealign` | v1 | 同上；舱内 `pip install gensim`（enable_internet）；DeepWalk 在 Hamming 边集（`to_orig_graph`）上初始化嵌入（walk_length 10 / walks_per_node 40 / window 5 / seed=训练种子） |
| D0 界面实验（方向性格式） | `math4amd/seme-hypercube-night-flight-d0-directional` | v1 | 同 C1；唯数据集换为方向性格式三元组 `[EDGE] prev curr [SEP] next [EOS]`（`vendor/directional.py`，prereg §6）：游走三元组 36,864（全边集 × 前驱语境）+ 训练路径双向三元组 1,120；训练期评估与 best-ckpt 口径同 C1；专有度量 traversal_full_acc / traversal_per_hop（teacher forcing 叶→原点） |

版本史：B0 v1–2 / C1 v1 `ModuleNotFoundError: No module named 'src'`
（Kaggle kernels push 不携带子目录）→ vendor 以 b64+sha256 自校验
内嵌进 `run.py`（v3/v2）；B0 v3 / C1 v2 forward 报模型未上设备 →
`model.to(device)`（原仓库约定：调用者搬设备）（v4/v3）；v4/v3
output 包 28MB（ckpt 占位）致本机下载反复 IncompleteRead → 训练后
清 `*.pt`（v5/v4，度量全在 REPORT_LINE / 汇总 json，ckpt 非判读
依据）。v5/v4 双臂 COMPLETE。
C2/C3（v1）直接过：vendor b64 内嵌成熟路径 + 模型上设备 +
训练后清 ckpt；C3 舱内自动 `pip install gensim`（DeepWalk 依赖，
enable_internet=true）。双臂完赛 elapsed 1258.8s。

## 回收四验（全过）

1. 完赛：双臂 `KernelWorkerStatus.COMPLETE`；
2. 格数齐：每臂 `{<COND>_seeds.json, <COND>_summary.json,
   metrics.json, kernel.log}`；
3. 哨兵专属词：日志 `NIGHT-DONE` ×1、`REPORT_LINE` ×4
   （3 seeds + 汇总，b64 JSON）；
4. 无损坏壳：仅 `vendor_runtime/src/__init__.py` 合法空文件
   （vendor 包标记）；无 `*.pt`（清 ckpt 纪律生效）。

正本落位：本目录 `results/{B0,C1,C2,C3}/`（C2/C3 于 2026-10-15 回收，`kaggle kernels output` EXIT=0，四验同 B0/C1；C2/C3 之 `metrics.json` + kernel.log 于同日复取补齐，seeds/summary 与首回收视 sha 一致 —— aad70bd0d230686b / 41bd07d94b978d2d / 518815d5b2d09d21 / 3b0669753944d835）。

## 训练期门限判读（C1 首跑 7/8；C2/C3 第二飞后合计 9/11，退出码 1 —— 绝对步数门限已按 2026-10-15 主人裁定替换为相对加速比 C3_accel_ratio_min = 19.18× ✓）

`python3 ../_TEMPLATE/check_gates.py --stage training --metrics metrics_training.json`
（gates@39619c19210c5869 × metrics@9a6a03d49076e504；C2 无门限，为消融对照）：

- ✓ B0_hardest_token_acc = 0.9844 ≥ 0.95（best-ckpt 口径）
- ✓ B0_final_auc = 0.9985 ≥ 0.90
- ✓ B0_hop1_alignment = 0.8628 ≥ 0.70
- ✓ B0_hop2to6_alignment = 0.1580 ≤ 0.30
- ✓ C1_hardest_token_acc = 0.5833 ≥ 0.40（best-ckpt 口径）
- ✗ **C1_final_auc = 0.3320 < 0.80**（唯一未过）
- ✓ C1_alignment_delta = 0.4619 ≥ 0.05
- ✓ C1_hop1_alignment = 0.8231 ≥ 0.70
- ✓ **C3_hardest_token_acc = 0.9948 ≥ 0.95**（best-ckpt 口径；终态 0.750）
- ✗ **C3_steps_to_90 = 2086 > 2000**（差 86 步 / 4.3%；判读见下）
- ✗ **C3_final_auc = 0.3320 < 0.90**（同 C1 适用性局限）

## 解释规则判读（prereg §3，不回改）

- **核心可证伪预言未达线**：C1 hop 2–6 对齐 0.3415 ——
  为 B0 之 2.16×、超无差异带（差 0.184 ≫ 0.05），但 < 0.40
  推翻线。「几何是浅的」在义素语义图上**被显著削弱、未被推翻**。
- acc 无差异带：|C1−B0| = 0.401 ≫ 5bp（有信号，C1 更难）。
- **几何编码组合深度而非项目身份**：C1 按跳数聚类 0.9961
  （近完美；随机基线 1/255 = 0.004）vs 臂身份聚类 0.3320。
- stable_rank：B0 34.67 / C1 27.81 / C2 65.00 / C3 7.67（仅记录，不作判据）。

**C2/C3 第二飞判读（2026-10-15，同图 Hamming H(4,4)，hop 量纲与 C1 一致可横比）**：
- **证伪条件触发（核心翻案）**：C3 hop 2–6 对齐 = **0.6578 ≥ 0.40 推翻线** —— 「几何是浅的」在 DeepWalk 预对齐的义素图上**被推翻**（prereg §4 判读规则）。C1（训练自组织）0.3415 为中间态（削弱未推翻），C3（预对齐注入）0.6578 翻案：**深层几何的组织程度取决于几何来源 —— 训练自组织时弱、预对齐注入时强**。
- **C3_steps_to_90 = 2086 超门限 86 步（4.3%）**：加速性质保留 —— C3 2086 步达 90%，C1 3/3 seeds 均未达（>40k 步）→ 加速 ≥19×（原文 C3 = 780 vs 19k = 24×）。门限锚 780 步 × 余量 2.5× = 2000，实测为锚之 2.7× —— 判为**门限余量设计偏紧**（夜航 255 路径图 vs 原文 512 臂图规模差异），非性质失败。
- **C3_final_auc = 0.332 未过**：同 C1 适用性局限 —— arm_cluster_acc 在 255 臂共享节点图上被结构钉死（C3 训练前 DeepWalk 嵌入的 arm_cluster 已 = 0.3320，训练后不变），非能力问题。
- **C2 消融**：hardest（best-ckpt）0.3073，较 C1 0.5833 **跌 47%** —— **嵌入学习是组合泛化的必要条件**。随机嵌入浅层 hop-1 对齐 0.7689 不低（probe 结构性量），深层 hop 2–6 仅 0.2347、stable_rank 65.00（满秩）。
- **DeepWalk 探针（ima 移交摘要待办③，真 vs 随机置换对照，3-seed 均值）**：臂身份聚类 0.3320 vs 0.1115（**3.0×**）、按跳数聚类 0.8815 vs 0.3411（**2.6×**）、hop 2–6 对齐 0.6258 vs 0.2068（**3.0×**）→ **DeepWalk 嵌入编码义素组合结构，非纯格子拓扑机械注入**（若纯机械注入，置换对照不应跌 3 倍）。训练贡献：hop 2–6 0.6258 → 0.6578（+0.032）、按跳数聚类 0.8815 → 0.9063（+0.025）—— **几何基本由预对齐注入，Transformer 训练仅小幅精修**。hop-1 反常：真 0.6485 < 置换 0.7617 —— hop-1 对齐对结构不敏感（短距离易对齐），判别性在深层量。

## 结论（每条声明带数字 + 产出命令）

1. B0 在义素实验装置上完整复现原文「几何记忆」：
   hop-1 0.8628 / hop 2–6 0.1580 的浅层组织
   （`check_gates.py --stage training` 四条 B0 门限全过）。
2. C1 组合泛化可学：hardest（best-ckpt）0.5833 ≥ 0.40；
   训练终态 0.3177，3 seeds 均未达 90% 线。
3. 深层对齐被语义组合结构显著抬升但未推翻「几何是浅的」
   （0.3415，介于 0.30 复现线与 0.40 推翻线之间）。
4. C1_final_auc 门限 FAIL 属门限适用性局限：其所测臂身份
   可分性在 255 臂语义图上不构成能力判据（按跳数聚类
   0.9961 才是语义图自然的几何量）。装置无问题
   （第一批不变量 8/8 过）。
5. 口径注明：hardest 门限按原仓库 notebooks/02 cell 38
   语义取 best-checkpoint 值；final_auc 为训练终态模型上的
   arm_cluster_acc（原仓库在 best ckpt 上评估，ckpt 已按回收
   纪律清除，口径差诚实标注）。
6. **C3 推翻「几何是浅的」**：hop 2–6 对齐 0.6578 ≥ 0.40
   推翻线（prereg §4 证伪条件触发）。与 C1 0.3415（削弱
   未推翻）合读：深层几何组织程度取决于几何来源 ——
   训练自组织时弱、DeepWalk 预对齐注入时强。
7. **DeepWalk 编码义素组合结构**（非纯格子机械注入）：
   真 vs 置换对照臂聚类 3.0× / 按跳数聚类 2.6× / hop 2–6
   对齐 3.0×；训练贡献仅 +0.032 —— 几何基本由预对齐注入，
   Transformer 训练小幅精修。
8. **嵌入学习是组合泛化必要条件**：C2 冻结嵌入 hardest
   0.3073 较 C1 0.5833 跌 47%。

## 后续

- C2/C3 第二飞已回收（2026-10-15，见上）。
- **D0 界面实验第三飞已回收（2026-10-15，判败＝科学证伪）**：方向性格式 `[EDGE] prev curr [SEP] next [EOS]`，3 seeds × 40k 步，航程 739s（kernel `math4amd/seme-hypercube-night-flight-d0-directional` v1，回收 `results/D0/`）。**hardest_token_acc = 0.2500（best-ckpt）vs C1 0.5833 → Δ = −0.333 < 0.05 证伪线触发**：「接口瓶颈」主张在语义图下不成立——方向性格式较对称格式劣化 57%，40k 步 3/3 seeds 未达 50% 线。部分结构学习仍在：hop-1 对齐 0.8165（≈C1 .823）、hop 2–6 0.3966（>C1 .342，未达 .40 推翻线）、teacher-forcing 遍历逐跳 [.17, .35, .85, 1.0]——根端几何已组织、叶端组合预测失败。与原文 1.8b（0→1.0）之张力：语义图上的瓶颈不在界面（对称格式信号已充分）；path 三元组仅占 3.0%（37,984 游走三元组 vs 1,120 路径三元组，均匀采样稀释路径结构信号）。判据/判读正本见 `benches/seme-hypercube/prereg.md` §5「D0 判读」+ §6。
- **Proust-Squid 次飞修复清单已冻**（2026-10-15，`Proust-Squid-Land/results/REPORT.md`「次飞修复清单」节）：EVAL_EVERY 250→50（steps_to_0.5 分辨率 8.3%→1.7%）、GEO_THRESHOLD 5.0→3.0、scale 对照修为同范数随机嵌入（首飞 scale_ctrl 与 α=1 完全等价之病灶）、多种子 [0]→[42,43,44]；C2/C3 Hamming 重跑项已由夜航第二飞完成销账。
- **E 臂 α 七档插值扫描已预注册**（2026-10-15，prereg §7）：`emb(α) = α·DW + (1−α)·rnd_s`（α = i/6 七档；DW 与 C3 逐 seed 同源 → α=1 ≡ C3 恒等端点；rnd_s 为行范数匹配到 DW 尺度的 N(0,1) 随机——尺度钉在 DW 轴上，α 纯控结构比例）。判据三选一（单调增 / 倒 U 存在 α* / 单调减，§4 已冻）；门限 E0..E6 各档 hardest ≥ 0.40（§7 冻结，构建入 gates.json 时入册）。承科学追问「什么注入量最优」（Proust-Squid 倒 U 形 α=0.5 峰值 .886 vs 夜航 C3 全注入 hardest .995：两图定义是调节变量，Hamming 图须独立验证）。
