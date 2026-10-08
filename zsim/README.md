# zsim · 义素 MaxSim 探针（Zero 整数定点版）

> A second implementation, for the sake of disagreement.
> 同一套义素基，两套互不相干的数值实现 —— **差异即信号**。

## 这是什么

`zsim` 是 `cognitive_engine.py`（Python + numpy，浮点）的**独立第二实现**，用
[Zero/zerolang](https://zerolang.ai)（graph-first 语言，整数定点 ×1000）写成。
两者吃同一套 `basis_dict / vocab / concept_spaces`，各自算：

1. **MaxSim** —— 输入词与各概念空间**内部词**的最大余弦相似度，及贡献锚点；
2. **贝叶斯后验** —— 似然 × 先验，归一化后的空间概率分布。

因为数值范式不同（IEEE 浮点 vs 整数定点 + `sqrtFloor`），两边的结果**不会逐位相同**，
但应落在容差内、且**锚点词与排序必须一致**。任何超出容差的偏差都是信号。

## 为什么存在

它是在 2026-10-08 的 Zero 沙盒试炼中诞生的。它当场照出了主引擎的一个静默缺陷：

> `calculate_maxsim_distance()` 的内层循环遍历的是**整个词表**而非该空间的内部词，
> 且 `concept_name` 变量从头到尾**未被使用** ——
> 于是五个概念空间报出**完全相同**的一组 `(max_sim, best_match_word)`。

后验路径不受影响（它走 `cos(词, 空间向量)`），所以这个缺陷长期隐身。
修法（已落）：给每个 `concept_spaces[...]` 补 `members`（内部词成员表），
让 MaxSim 真正按空间取 max；锚点集扣除输入词后为空时返回 `None`（无独立锚点）。

**「内部词」归属的两条独立路径** —— 这正是差分验证的价值所在：

| 实现 | 归属来源 |
|---|---|
| Python 版 | 显式声明的 `concept_spaces[...]["members"]` |
| Zero 版  | 算法推导：`argmax_c cos(词, c)` |

若哪天有人改了 Python 侧的成员表而没同步 Zero 侧（或反之），比对台会立刻报错。

## 安装编译器

`zsim` 的程序存放在 `zero.graph`（图即程序本体），`src/main.0` 是供人阅读的投影。
运行需要 `zero` 编译器（本仓**不含**二进制）：

```sh
curl -fsSL https://zerolang.ai/install.sh | bash    # 默认落 ~/.zero/bin
export PATH="$HOME/.zero/bin:$PATH"
```

本项目验证于 `zero 0.3.4 (build 5b3a90a)`、`darwin-arm64`。
查编译器：`compare.py` 会依次在 `$ZERO`、`<repo>/.bin/zero`、
`~/Programming/code-2026/zero-sandbox/.bin/zero`、`$PATH` 中寻找。

## 用法

```sh
../run_zsim.sh 塑胶凳                 # 园区惯例：根目录启动器
zero run -- 塑胶凳                    # 或用 zero 直接跑（cwd 须在此目录）
zero run                              # 无参：列出概念空间与词库
zero test                             # 4 个测试块
zero build --emit exe --out .zero/out/zsim && ./.zero/out/zsim 塑胶凳
```

输出示例（`塑胶凳`）：

```
MaxSim  (每空间最优锚点)
  超市  maxSim=2526 bp  锚=购物车
  路边摊  maxSim=7382 bp  锚=折叠桌
  觉醒循环  maxSim=0 bp  锚=你以为你选的是自己的人生，还是别人替你写的
  接待员日常  maxSim=0 bp  锚=一切都按计划进行，分秒不差
  福特剧场  maxSim=0 bp  锚=这些残暴的欢愉，终将以残暴结局

贝叶斯后验  (先验 1/3,1/3,1/9,1/9,1/9)
  超市  2670 bp  ##########
  路边摊  7151 bp  ############################
  ...
```

## 差分比对台

```sh
python3 zsim/compare.py                 # 跑内置 7 词，印全表
python3 zsim/compare.py --quiet         # 只印汇总；不一致则以非零码退出
python3 zsim/compare.py 塑胶凳 煤气罐    # 指定词
```

容差：MaxSim `0.01`、后验 `5 bp`（可改 `compare.py` 顶部常量）。
退出码即判据，可直接作回归门禁。

比对台会自动挑选一个带 `numpy` 的解释器（本机 `/usr/bin/python3` 没有，
用 `/usr/local/bin/python3`；可用环境变量 `PYTHON` 指定）。
最近一次全表结果见 `compare-output.txt`。

## 边界与已知限制

- **锚点集为空**：`接待员日常` / `福特剧场` 各只有 1 个内部词；当输入词就是它时，
  扣除自匹配后无锚点 → 两边都报「无独立锚点」（Python `None`，Zero `maxSim=n/a`）。
- **整数定点量化**：MaxSim 用整数 `sqrtFloor` + bp 刻度，与浮点有 ≤0.001 量级差。
  后验的 ≤2bp 偏差亦源于此，以及先验 `1/3, 1/9` 的定点化。
- **测试运行器**：`zero test` 的图运行器不支持 `ArrayLiteral`，故测试只覆盖
  纯标量路径（词表互逆、先验分子、空串落空）；含数组的核验走 `compare.py`。
- **语言坑**：Zero 0.3.4 的补丁行解析按换行切分 —— 数组字面量**必须整条写在一行**。

## 文件

| 文件 | 角色 |
|---|---|
| `zero.toml` | 包清单（cli target） |
| `zero.graph` | **程序本体**（编译器输入） |
| `src/main.0` | 人读投影（377 行，11 函数 + 4 测试 + CLI 入口） |
| `compare.py` | 差分比对台（回归门禁） |
| `compare-output.txt` | 最近一次全表结果快照 |
