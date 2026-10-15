# benches/_TEMPLATE — 新 bench 标准骨架

自检式模板：本目录自己也守三样交付律（AGENTS.md「Agent 面约定 · 三样交付律」）。

## 正本声明（source of truth）

| 面 | 正本 | 派生面 |
|----|------|--------|
| 门限 | `gates.json`（数值只此一处，别处不许另写一套） | 本 README 之引用 |
| 解释框架 | `prereg.md`（冻稿即唯一解释框架） | 跑后结论 |
| 度量定义 | `metrics-spec.md` | — |
| 裁判 | `check_gates.py` | `verdict.json`（可重生成） |
| 实测 | `metrics.json`（bench 产物；本模板只带 `example/`） | — |

## 一条 verify 命令

```
python3 check_gates.py            # 退出码即判据：0 = 全过，1 = 有门限未过
```

## 一份可读投影

`README.md` + `prereg.md` + `metrics-spec.md` + `verdict.json`（人可读 JSON，附输入 sha）。

## 用法

1. `cp -r benches/_TEMPLATE benches/<新bench>`
2. **跑之前**：填 `gates.json` / `prereg.md` / `metrics-spec.md`（prereg 冻稿，sha 记 LEDGER.md）
3. 跑 bench → 产出 `metrics.json`
4. `python3 check_gates.py` → 退出码 0 才可写结论；`verdict.json` 之 sha 记 LEDGER.md

## 门限语法速查

见 `check_gates.py` 头部 docstring：`min` / `max` / `spread_min`（非退化门限）/ `expect_drop_vs`（对照条件须显著更弱）。

## 来历

提炼自 llm-geometry-analysis（Noroozizadeh et al. 2025《Deep Sequence Models Tend to
Memorize Geometrically》配套仓）的方法论：`src/config.py` 的 SUCCESS_GATES、notebook 里的
「A Priori Interpretation Rules (pre-registered)」、度量三件套 spec（What / Why it's hard /
How to compute）、混淆控制表（Choice / Why）、条件矩阵 + 种子。

注意：该仓 README 承诺的 `tasks/lessons.md` 与 `tasks/research_plan.md` **并不存在**
（raw 404、git 全历史无记录）——文档超前于实物，本身即「文档漂移」病例，与本园
「手册话 vs 引擎实测」（0.89/0.45 vs 0.7374，1008 案）同病。本模板即解药之一。
