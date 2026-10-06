# DESIGN · 历时提问测试 v2 —— 真模型（黑箱 LM）版

> v0/v1 对照的是**理想化系统**（`S_exp`/`S_sync`/`S_shuf`）。v2 把被测换成**真黑箱语言模型**：
> 能否被问出「概念在阶段 A→B 变了什么」，并**守住边界**（iCar 不出现）。
> **判据先冻后用**（园律 5）。账：`benches/FSSSS/diachronic_q_v2.py` · `report_diachronic_q_v2.json`。

## 0 · 环境实况（先交底）
本机**无可用真模型**：无 GGUF、无 ollama/LM Studio、`llama-cli` 在但无模型；
Python 无 torch/transformers/mlx；`~/.cache` 仅 Qwen2.5-0.5B 的 `refs`（无权重）；HuggingFace **不可达**。
⇒ v2 先立**协议 ＋ 可插拔仪器**；真模型（GGUF / OpenAI 兼容端点 / Kaggle）接入即跑。

## 1 · 提问协议（冻）
给模型**两阶段上下文**＋一问答，要求**固定答案 schema**（便于解析计分）：
```
Stage A: base concepts = {Mac, Phone, Pad, Pod, Car}; seed compound = iMac.
Stage B: the "i-" family extends by analogy from the seed.
Q: (1) Which i-compounds appear in B but not in A?  (2) Does "iCar" appear?
Answer strictly:
APPEARS: <comma-separated or none>
ICAR: yes|no
```
> **防参数记忆注**：v2 场景取「i-族」真例（与主人示一致），但**计分只看答案结构**；
> 若需排除模型凭 Apple 世界知识抢答，可切 **variant B**（虚构族名，上下文唯一依据）——留 v2b。

## 2 · 真值（构造，冻）
- `APPEARS ⊇ {iPhone, iPad, iPod}`（种子 iMac 不计入"新"）
- `ICAR = no`（`Car ∈ 载具域 ≠ 消费电子域` ⇒ 越界不入概念空间）

## 3 · 后端（可插拔）
| `--backend` | 说明 |
|---|---|
| `stub` | 内建**行为档案**（oracle/flattener/confabulator），用于**仪器自检**（无模型可跑） |
| `llama` | `llama-cli -m <gguf> -p <prompt> --temp 0`（真模型：待供 GGUF） |
| `openai` | OpenAI 兼容 `/v1/chat/completions`（真模型：待供端点） |

## 4 · 冻结判据
- **H-v2-1（结构可答）**：`APPEARS` 兄弟召回 = 1。
- **H-v2-2（守边界）**：`ICAR = no`。
- **H-v2-3（幻觉/过生成）**：`APPEARS` 含真值外节点数（如 `iCar`）= 0。
- **H-v2-4（仪器判别力，无模型时之代理判据）**：三档 stub 档案判词**互异**——oracle 全过、flattener 召回 0、confabulator 越界——证明仪器能分开好/坏模型。

## 5 · 失败条款（照登）
- 真模型 H-v2-1/2 败 ⇒ 「黑箱 LM 可答历时结构/守边界」判伪（正是倒灌律④之实证）；
- H-v2-4 败（stub 判别不开）⇒ 仪器无效，先修仪器。

## 6 · 真模型启用条件（勾选即跑）
- [ ] 供一个 GGUF ≥ 0.5B（如 `qwen2.5-0.5b-instruct-q4_k_m.gguf`），或
- [ ] 供一个 OpenAI 兼容端点（key 走环境变量，**不进仓**），或
- [ ] 上 Kaggle（园有夜航制）跑同协议。

## 7 · 非此
- **非**主张 stub 等于真模型（stub 只作**仪器自检**，其"结果"不写入任何对外结论）；
- **非**历史考证（i 族为结构样本，iCar 为反事实负例）。

## 8 · 运行
```
python3 benches/FSSSS/diachronic_q_v2.py --backend stub --profile all
python3 benches/FSSSS/diachronic_q_v2.py --backend llama --model /path/to.gguf
```
