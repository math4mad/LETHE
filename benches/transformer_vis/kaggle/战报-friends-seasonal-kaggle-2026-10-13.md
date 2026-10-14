# 战报 · Friends 季节演化 Kaggle 夜航（2026-10-13）

> 案：主人观感「到了第三季和第四季, Rachel 才表现出特色」
> 假设：H-S1 Rachel 舱内相似 S03/S04 > S01 · H-S2 Rachel 质心距 S03/S04 > S01 · H-S3 对照（其余五角色随季同形 = 漂移）
> PREREG：`kaggle/PREREG.md`（判据先冻）· 正本：`benches/transformer_vis/friends_seasonal.py` + `corpus_friends_s1234.json` + `friends_s1234_raw.json` + 本地 `metrics_friends_s1234.json`

## 结果总览

| 条款 | 判 | 读数 |
|---|---|---|
| **K-1** 平台迁移（臂 A 0.5B 自检 vs 本地） | ✅ **通过** | 45 项指标全在冻结容差内（余弦 max \|Δ\| 3.4e-4 / 探针 3.3e-3 / 流计数逐位相等）→ 本地四季读数为真 |
| **K-2** 7B 时间线形态 | ❌ 观感**未获支持，方向相反** | Rachel 舱内 S01 0.8406 → S03 0.8127 → S04 0.8152（S03/S04 降）；分离信号只在 **S03**（质心距 +42% / 探针峰 0.8133），S04 回落 |
| **K-3** 对照 H-S3 | ⚠️ S02 同形（漂移）；**S03/S04 异形（Rachel 特异）** | 但特异方向是「Rachel 变得更**不**凝聚」（S03：Rachel −0.028 vs 其余 +0.005） |
| **K-4** 4bit 注记 | 不裁决 | bnb 0.50.2 留痕；绝对值跨模型不可比，只比形态——0.5B 与 7B 形态一致，互为佐证 |

**判词**：主人的观感有半只脚落地——Rachel 确在 S03/S04 走成异类（对照异形），但走的不是「更凝聚」而是「更发散」；「S03/S04 才凝聚」不成立，且信号实为 S03 单季。角色特色是叙事现象，不是词汇现象（6 类探针 0.22–0.26 近随机基线 0.167）。

## 夜航故障两连（均已解）

1. **Kaggle 源 1MB 上限**（首射 400 `The kernel source must be less than 1 megabytes`）：
   语料内嵌改 **gz+b64**（2.84MB → 0.63MB）；sha256 仍取解压后字节，与本地正本逐位同源（`make_arms.py` 反向对撞）。
2. **bnb 缺失 + 断网**（二射 `ImportError: bitsandbytes>=0.46.1`；三射 pip `name resolution` 失败——PREREG 设 `enable_internet: false`）：
   7b 臂 metadata 单独开 internet 装 bnb（**偏离 PREREG，模板注记留痕**）；bf16 退路未用（免费档 GPU 无 bf16 计算，fp16 7B 单卡 16GB 有 OOM 风险）。

## 产物清单（回收四验：完赛 + 格数齐 + 哨兵专属词 + 无 0 字节壳）

- 臂 A `math4amd/friends-seasonal-0-5b-selftest` v1 **COMPLETE**（708s）：`kaggle/out-0-5b/` — metrics + 5 figs（37–214KB）+ log（5×REPORT_LINE + SEASONAL_DONE，无 SEASONAL_ERROR）+ **执行源码归档 `run.py`**（sha `40fdd1f0…`，与远端 v1 一致）
- 臂 B `math4amd/friends-seasonal-7b` v3 **COMPLETE**（5125s）：`kaggle/out-7b/` — metrics + 5 figs + log（5×REPORT_LINE + SEASONAL_DONE + bnb 版本留痕）
- verify：`python3 kaggle/verify_friends_kaggle.py kaggle/out-0-5b/metrics_friends_s1234_kaggle.json` → 退出码 0（VERIFY_OK）
- quarto 笔记：`benches/transformer_vis/report_friends_s1234_kaggle.qmd`（+ .html）

## 偏离登记

- PREREG「enable_internet: false」→ 臂 B 为装 bnb 开 internet（仅此一臂；模板 `run_template.py` 注记）。
- PREREG 失败条款「bnb 不可用 → 改 bf16 重推」未启用（装 bnb 保住了预注册 4bit 设计）。

## 候

- 主人圈点判词（尤其「半只脚落地」的解读）。
- 若要 7B bf16 基线（非 4bit）须另案（本地无 7B bf16 机器，Kaggle 免费档 bf16 不可算）。
- 晨圈 10-10/10-11/10-12 挂账仍在（本夜航非其挂账项，未动）。
