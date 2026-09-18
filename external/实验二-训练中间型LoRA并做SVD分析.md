---
title: "实验二：训练中间型 LoRA 并做 SVD 分析"
date: 2026-09-18
status: 待执行
---

# 实验二：训练中间型 LoRA 并做 SVD 分析

## 实验目标

通过训练一个混合数据的"中间型 LoRA"，并利用 SVD（奇异值分解）分析不同 adapter 的内部子空间结构，验证"中间型 adapter 与两个离散 adapter 的子空间相似性均低于两个离散 adapter 之间的相似性"这一假设。

## 实验步骤

### Step 1：训练中间型 LoRA

使用春节和夏季的混合数据训练一个"中间型 LoRA"adapter。

**训练配置：**

- 数据来源：春节数据 + 夏季数据，按一定比例混合（建议 1:1）
- LoRA rank（r）：与春节/夏季 LoRA 保持一致
- 其他超参数（learning rate、epochs、batch size 等）：与已有 LoRA 保持一致

**控制要点：**

- 确保训练轮数和总步数与离散 LoRA 可比
- 记录训练 loss 曲线，确认收敛正常

### Step 2：对三个 LoRA 做 SVD 分解

对以下三个 adapter 的 A 矩阵和 B 矩阵分别进行奇异值分解：

1. **春节 LoRA** — A_spring, B_spring
2. **夏季 LoRA** — A_summer, B_summer
3. **中间型 LoRA** — A_mid, B_mid

**SVD 分解：**

对每个矩阵 M 做分解：M = U · Σ · V^T

提取：
- U：左奇异向量矩阵（列空间基）
- Σ：奇异值对角矩阵（各方向的重要性）
- V：右奇异向量矩阵（行空间基）

### Step 3：提取前 k 个奇异向量，计算子空间相似度

**操作：**

- 选取前 k 个奇异值对应的奇异向量（k 的选择可参考奇异值累积贡献率 ≥ 90%）
- 对每对 adapter 的对应矩阵（A 或 B），计算前 k 奇异向量子空间的相似度

**相似度度量：**

使用 Frobenius 范数计算子空间距离：

```
sim(U_i, U_j) = ||U_i^T · U_j||_F
```

值越大，子空间越相似。

**需要计算的相似度矩阵（A 矩阵和 B 矩阵各一组）：**

| 对比 | A 矩阵相似度 | B 矩阵相似度 |
| --- | --- | --- |
| 春节 vs 夏季 | sim(U_spring, U_summer) | sim(V_spring, V_summer) |
| 春节 vs 中间型 | sim(U_spring, U_mid) | sim(V_spring, V_mid) |
| 夏季 vs 中间型 | sim(U_summer, U_mid) | sim(V_summer, V_mid) |

### Step 4：验证相似性减弱假设

**核心假设：**

```
sim(U_mid, U_spring) < sim(U_spring, U_summer)
sim(U_mid, U_summer) < sim(U_spring, U_summer)
```

即：中间型 LoRA 与任一离散 LoRA 的子空间相似度，低于两个离散 LoRA 之间的相似度。

**验证方式：**

- 直接比较数值大小
- 在不同 k 值下（k=1, 5, 10, 20, ...）重复验证，确认结论的稳健性
- 若假设成立，说明混合训练产生的 adapter 并非两个离散 adapter 的简单插值

### Step 5：分析奇异值分布和入侵维度

**奇异值分布分析：**

- 绘制三个 LoRA 的奇异值衰减曲线（横轴：维度序号，纵轴：奇异值大小）
- 对比有效秩（effective rank）：奇异值衰减越快，有效秩越低，说明 adapter 利用了更少的方向

**入侵维度分析：**

- 识别"入侵维度"：在基座模型中不显著、但被 LoRA 激活的方向
- 对比三个 LoRA 各激活了哪些不同的方向
- 分析中间型 LoRA 是否在入侵维度上呈现两个离散 LoRA 的混合特征

## 关键产出

- [ ] 中间型 LoRA 训练日志（loss 曲线、超参数记录）
- [ ] 三个 LoRA 的 A/B 矩阵 SVD 分解结果
- [ ] 子空间相似度矩阵（不同 k 值下）
- [ ] 假设验证结论（相似性减弱是否成立）
- [ ] 奇异值衰减曲线图
- [ ] 入侵维度分析报告

## 注意事项

- SVD 计算注意数值稳定性，建议使用 double precision
- 对比时注意 A 矩阵和 B 矩阵要分开分析，它们的子空间含义不同
- 如果 LoRA 作用于模型的多个层，需逐层分析或汇总（明确说明处理方式）
- k 值的选择对结论有影响，建议做敏感性分析
