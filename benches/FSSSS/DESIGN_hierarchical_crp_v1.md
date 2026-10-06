# DESIGN · hierarchical CRP v1（概率版 nCRP ＋ α 扫描）

> 承 v0（阈值版 `pcs_hier_crp.py`）。把「开新」从阈值改为 **CRP 概率**，并以 α 扫描检验
> 「α 控开列率」。**判据先冻后用**（园律 5）。账：`benches/FSSSS/pcs_hier_crp_v1.py` · `report_pcs_hier_crp_v1.json`。

## 1 · 模型（nCRP，混合式）
节点 `v` 内（子 `{c_i}`，计数 `n_i`，总 `N_v`）：
```
P(c_i | v) = n_i / (N_v + α)
P(new | v) = α    / (N_v + α)
```
乘似然 `exp(β·cos(x, p_{c_i}))` / `exp(β·novelty)`（`novelty = 1 − max_i cos`）→ 取 **MAP**：
- 若 `new` 胜 → 在 `v` 下开新子（`p_new := x`），停；
- 否则取最优已有子 `k`：`cos ≥ τ_match` → **归入** `k`（停）；否则**下行**至 `k` 再试。
- 每节点权重 `w_i = n_i / N_v`（归一）；叶先验 = 根先验 × 沿路径累乘。

## 2 · 观测流
`A B C C1 C2 D D1 E C1 C1`（A/B/C/… 为 3 维合成向量，方向即语义；C1 重复出现以验「归入」）。

## 3 · 冻结判据
- **H-n1（α 控开列率）**：`根直子数` 随 α 非减，且末 α 严格增。
- **H-n2（递归保留）**：α=1 时树深度 ≥2 且各内部节点 Σw=1。
- **H-n3（守恒）**：全 α 下叶先验 Σ=1。

## 4 · 失败条款（照登）
- H-n1 败（α 不影响结构）⇒「α 控开列率」不成立；
- H-n3 败 ⇒ 权重守恒破（须修；本轮曾因**同级同名子节点**致字典塌陷 Σ<1，已去重修复）。

## 5 · 非此
- 非「真 CRP 采样」（取 **MAP** 确定性版，为可复算；采样版候 v2）；
- 非真语料（合成向量，为判据可控）。

## 6 · 运行
```
python3 benches/FSSSS/pcs_hier_crp_v1.py     # → report_pcs_hier_crp_v1.json
```
