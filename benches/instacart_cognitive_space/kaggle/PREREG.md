# PREREG · Instacart 认知空间夜航（2026-10-13）

> 实验目标：验证 Transformer 隐藏层在处理购物篮数据时，是否呈现 $\text{Product} \rightarrow \text{Aisle} \rightarrow \text{Department}$ 的抽象层级递增结构。

## 1. 实验配置 (Knobs)
- **语料**: `corpus_instacart.json` (抽样 $N$ 个订单，含 Aisle/Dept 标注)
- **模型**: 
  - Arm A: Qwen2.5-0.5B-Instruct (bf16)
  - Arm B: Qwen2.5-7B-Instruct (bnb-4bit)
- **探针方法**: 质心探测 (Centroid Probing)
  - $\text{Acc}(L, \text{Target}) = \text{Count}(\text{argmax}_{C \in \text{Centroids}} \cos(v_L, C) == \text{Label}) / N$
- **意图定义**:
  - $\text{Healthy}$: `Organic` keyword OR `Fresh Produce` aisle.
  - $\text{Bulk}$: `Household` OR `Paper Products` aisle.
  - $\text{Convenience}$: `Snacks` OR `Beverages` aisle.

## 2. 判据 (Judgment Clauses)

### K-1: 层级递增 (Hierarchical Ascent)
- **定义**: 设 $\text{Acc}_{\text{Aisle}}(L)$ 和 $\text{Acc}_{\text{Dept}}(L)$ 为各层对类目和部门的预测准确率。
- **判定**: 
  - $\text{Acc}_{\text{Aisle}}$ 的峰值层 $L_{\text{peak\_A}}$ 必须早于 $\text{Acc}_{\text{Dept}}$ 的峰值层 $L_{\text{peak\_D}}$。
  - 且 $L_{\text{peak\_D}} > L_{\text{peak\_A}} + 2$ (显著层级差)。

### K-2: 意图可分 (Intent Separability)
- **定义**: 在末层 $\text{L}_{last}$，计算不同意图订单的平均向量 (Mean Pool) 的质心距离。
- **判定**: $\text{dist}(\text{Healthy}, \text{Bulk}) > \text{dist}(\text{General}, \text{Bulk}) \times 1.2$。

### K-3: 容量增益 (Capacity Gain)
- **判定**: Arm B (7B) 的 $\text{Acc}_{\text{Dept}}$ 峰值 $\text{Acc}_{\text{max\_B}} > \text{Acc}_{\text{max\_A}}$ (0.5B) $\times 1.1$。

## 3. 失败分支 (Failure Clauses)
- **OOM**: $\text{FLOW\_MAX\_TOKENS} \rightarrow 500$ 重推。
- **bnb 不可用**: 切换为 $\text{fp16}$ (Kaggle T4x2) $\rightarrow$ 调整 $\text{device\_map="auto"}$。
- **数据对撞失败**: $\text{sha256}$ 不匹配 $\rightarrow$ 重新生成 `run.py`。

---

## 4. 运行日志 (Execution Log)

### 2026-10-13 夜航重射 (两臂修复三因)

**前置诊断 (首射两臂均"没动"的根因)**, 双因叠加:
1. **metadata 无 `model_sources`** — kaggle push 时不挂任何外部输入 → `/kaggle/input` 空 → 模型无从装载。
2. **生成 run.py 是旧模板** — 硬编码绝对路径 `/kaggle/input/models/...`, 无 `find_model_path` 动态发现; 本地找不到即回退 HF Hub, 0-5b 臂离线 (enable_internet=False) 直接 `OSError: Repo id must be in the form...`。
3. **7b 臂旧产物缺 bnb install 行** — `ImportError: bitsandbytes 4-bit requires >=0.46.1` (t≈57s)。

**修复 (均落地验证)**:
- `make_arms.py` 注入 `model_sources` (`0.5b-instruct/1` / `7b-instruct/1`); 修 `id` slug 为 `instacart-cognitive-space-{arm}` 以与 title 推导 slug 一致 (否则 push 409 "title does not resolve to specified id")。
- 重生成 run.py 获得模板 `find_model_path` 动态发现 + bnb install 行。

**重推后新暴露的 dtype 缺陷 (2026-10-13 夜, 已修)**:
- 0-5b 臂新版本 RUNNING 后于 `get_centroid_acc` 报 `RuntimeError: expected mat1 and mat2 to have the same dtype, but got: c10::BFloat16 != float` — 模型 bf16 输出 vs 质心 f32 累加, `torch.mm` 违规。
- 修复: `run_template.py` 的 `get_centroid_acc` 入口将 hidden_states 上浮到 float32 (bf16→f32 无损), 标签转 long 索引, `preds == labels` 直接比较。
- 两臂已重新生成 + 重推, 0-5b RUNNING (验落版 3 处 dtype 修复行在远端), 7b QUEUED。

**回收哨 (queue.py)**: `instacart-0-5b-collect` / `instacart-7b-collect` 已入队 (6 tries @5min), 完赛即收 `out-0-5b` / `out-7b`。明早看结果四验。
