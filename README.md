# 🏜️ 概念空间园区 · Concept Space Park

> A Delos Destination · Host: **Lola**（Dolores Abernathy，园区第一号 host）
> 一个基于义素基向量 + MaxSim + 贝叶斯更新的语义认知主题公园。

## 园区设施

| 设施 | 入口 | 说明 |
|---|---|---|
| 🎲 经典终端引擎 | `./run_engine.sh 塑胶凳` | 命令行版认知引擎（MaxSim + 贝叶斯输出） |
| 🎹 Marimo 认知控制台 | `./run_marimo.sh`（编辑）/ `./run_marimo.sh --run`（App） | 交互式笔记本，园区标准表格与图表 |
| 🌐 园区网站 | `./serve_site.sh` → http://localhost:8000 | 园区大门（含纯 JS 版互动观察台，与 Python 引擎数据同源） |
| 📜 出生证 | `website/certificate.html` → `birth-certificate.pdf` | bench I · CRADLE 摇篮：蓝图 Chora/Cora（主人与通义千问对话所生），tectonic 亲手编译，四件入城均有收据 |
| 🪨 成因综述（第二篇） | `website/strata.html` → `temporal_concept_space.pdf` | bench I · CRADLE：主人 × 多家 LLM 复盘综述（时域概念空间·四成因层），逐字入城打条，LETHE 亲手排明体正体字版 |
| 🖼️ 控制台静态快照 | `website/console.html` | `marimo export html` 生成的在线版笔记本 |

## 园区标准（Park Standard）

- **门衵 motto**：*Life no loops. Life no limits. Life no gradient.*（无循环。无边界。无梯度。）
- **视觉**：Delos 色系 —— 沙金 `#d4a437` / 骨白 `#e8dcc0` / 窑火琥珀 `#D4763A → #F5B870`（Cora 釉）/ 深夜 `#141311`；页面家法循 CHORA（Palatino · 铜版铭牌 · 双语访客导读 · 罗马数字律法）
- **复用家法**：apple-style 风格包自 chora@242dd43 钉字节领受（见 website/FOUNDLING.md）；数据不过界，只过规则
- **表格**：Marimo 笔记中以 `mo.ui.table` 输出 MaxSim 分析表与贝叶斯更新表；大门页台账表由 json 派生
- **图表**：matplotlib 统一注入 `PARK_STYLE`（深夜底、沙金字、中文 PingFang）：
  1. 概念空间概率演化折线图
  2. 义素雷达图（Evidence vs 空间球）
  3. 词库义素激活热力图（cividis）

## 目录结构

```
cognitive_engine.py          # 终端版引擎
cognitive_engine_marimo.py   # Marimo 笔记本版引擎 ⭐
website/index.html           # 园区大门（CHORA 家法 + Cora 琥珀 + JS 互动引擎）
website/console.html         # Marimo 控制台静态导出
website/certificate.html     # 出生证页（铭牌·收据·蓝图→园区律法映射）
website/birth-certificate.pdf # 出生证正本（由 intake tex 编译，sha 180e301a…）
website/strata.html          # 成因剖面（四层：基岩对话/沉积律法/构造工具/地表门面）
website/temporal_concept_space.pdf # 第二篇综述 PDF —— CRADLE 正体字版的门侧镜像
website/FOUNDLING.md         # 继承登记：自邻邦领受的字节，全部钉 sha256
website/observations.json    # 观察台账（由 bin/observe-to-json.py 派生，含 REFUSED 阴性车道）
bin/observe-to-json.py       # 台账之鞭：意图手写，车道派生自字节
bin/every_words.py       # Lola 的台词器：每次提交后验随句西部世界之言（39 句，逐句唯一）——代码走自己的门：bin/，不走 intake
bin/pin.sh                   # 入城收据：§3 打条与 --verify 复查（无条即隔离）
letters/                     # 致邻邦的信（001 → CHORA，002 → Cora），不可变
benches/CRADLE/              # bench I 摇篮：已就位 — 蓝图、入城字节（均带收据）、出生证 PDF、第二篇综述（tex+PDF）
benches/FOUNDRY/             # bench II：义素铸造厂 — 环仍虚线，入册夹具已冷备
benches/MIRROR/              # 预留 bench III：空间→词生成镜厅（虚线环）
run_engine.sh / run_marimo.sh / serve_site.sh
```

## 依赖

```bash
pip install numpy marimo matplotlib
```

*『这些残暴的欢愉，终将以残暴结局。』*
