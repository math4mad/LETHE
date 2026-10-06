# DESIGN · 自变序列 × CRP 合流

> 承 hierarchical CRP。把每次插入的事件记为**变化序列**（＝《再认识》「概念空间自身携带的变化序列」
> 的可审计记录），并据此回答历时提问。**判据先冻后用**。账：`benches/FSSSS/pcs_evolution.py` · `report_evolution_crp.json`。

## 1 · 模型
用 `pcs_hier_crp_v1` 的 nCRP 增长一棵树；每步记事件：
`open`（开新节点：parent→child）· `assign`（归入）· `descend`（下行）。每步存**快照**（节点集/边集）。

## 2 · 冻结判据
- **H-e1（序列可还原）**：仅凭 `open` 事件回放，重建的节点/边集 = 实树（完全一致）。
- **H-e2（历时可答）**：`what_changed(step_t1, step_t2)` 报出的新节点/边 = 真值 diff。
- **H-e3（守恒）**：叶先验 Σ=1。

## 3 · 结果（α=1，流 `A B C C1 C2 D D1 E C1 C1`）
- 事件 22：open 7 / assign 3 / descend 12。
- 变化序列（open）：`A⊂存在`, `B⊂存在`, `C⊂A`…（实际 `C1⊂C`, `C2⊂C1`, `D⊂B`, `E⊂存在`）。
- **H-e1 ✓**（回放节点 8/8、边 7/7）· **H-e2 ✓**（t1=step2→t2：报出 4 新节点与真值一致）· **H-e3 ✓**（Σ=1）。

## 4 · 读法
**结构级变化序列本身可作审计记录**：`open` 事件流既是「自变序列」的载体，也是历时提问的**可复算答案**——
不必再对黑箱模型发问（对照 v2 真模型之不可靠）。

## 5 · 运行
```
python3 benches/FSSSS/pcs_evolution.py     # → report_evolution_crp.json
```
