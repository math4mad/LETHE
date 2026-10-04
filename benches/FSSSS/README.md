# FSSSS bench — Bechberger 概念空间形式化 与本园引擎的对表

**FSSSS** = Fuzzy Simple Star-Shaped Set（Bechberger & Kühnberger 对 Gärdenfors 概念空间的形式化：
核心＝若干凸体之并，隶属 μ = μ₀·exp(−c·d(x, core))）。

## 件

- `bridge_maxsim_fssss.py` — 任务 C 对表：把园引擎 `cognitive_engine.py` 的 16 义素空间
  搬进 FSSSS（v1.3.2），用「超市 / 路边摊」锚点定义凸体概念，
  逐探针比较 **FSSSS 模糊隶属** vs **园 余弦似然×贝叶斯后验**。
- `report_c_bridge.json` — 运行产物（排序一致 7/8，锚点自洽 10/10）。

## 运行

```bash
# 需 external/ConceptualSpaces-1.3.2-py3 + numpy/scipy/shapely/numdifftools
/path/to/python benches/FSSSS/bridge_maxsim_fssss.py
```

## 结论一句话

两套机器在**排序**上同构 —— 都在算**几何相似度**，**都没有概率语义**
（FSSSS 的隶属是模糊/可能性，园引擎的「似然」是截断余弦）。
这正是本园的补白处：**概率概念空间（PCS）**。
详见 `notes/DRAFT-概率概念空间-Bechberger缺口诊断-1004.md`。
