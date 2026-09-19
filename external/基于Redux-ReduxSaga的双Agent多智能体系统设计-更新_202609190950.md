# 基于 Redux + Redux-Saga 的双 Agent 多智能体系统设计

## 一、系统概述

本系统是一个基于 Redux + Redux-Saga 状态管理框架的多智能体（Multi-Agent）协作系统。核心设计灵感来源于经典警探搭档（Buddy Cop）模式——借用 LAPD/NYPD 的经典二人组概念，命名为 AgentsPD。

系统核心理念包括：

- 使用前端成熟的 Redux 架构来管理和编排多个 Agent 的工作。
- 通过引入持久层（Persistence Layer）保持通讯数据的持久化。
- Agent 所需的上下文信息通过 Redux 架构实现。
- 引入"意图澄清机制"（Human-in-the-Loop），当任务不明确时 Agent 可以请求 Owner 把话说得更明确。
- 引入"工作进程观察窗口"（教室后门视角），让 Owner 随时看到 Agent 的工作进展。
- 支持运行中的多轮对话，Owner 不闻不问、Agent 不埋头蛮干，形成"对话-协作"的透明过程。

## 二、架构设计

### 2.1 整体架构

系统分为四层：

1. **Owner（前端 UI 层）**：用户/开发者，负责输入任务、回答澄清问题、通过观察窗口查看进度、随时插话干预。
2. **Redux-Saga（编排层）**：负责监听 Action、触发 Agent 执行、处理副作用、编排异步工作流、支持插话中断与重规划。
3. **Redux Store（状态层）**：作为单一事实来源（Single Source of Truth），管理所有 Agent 的状态和通讯数据。
4. **持久层（Persistence Layer）**：IndexedDB / localStorage，保存对话历史与上下文。

**整体架构图：**

```
┌─────────────────────────────────────────────────────────────┐
│                    Owner（前端 UI 层）                        │
│   输入任务 → dispatch(USER_TASK_SUBMIT) → 等待响应            │
│   ← dispatch(CLARIFY_REQUEST) → 显示问题 → 回答              │
│   ← dispatch(TASK_COMPLETE) → 展示结果                       │
│   观察窗口：实时查看进度 / 随时插话(OWNER_INTERJECT)           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                 Redux-Saga（编排层）                          │
│   监听 Action → 触发 Agent 执行 → 处理副作用 → 编排工作流     │
│   支持：意图澄清 / 插话中断 / 进度推送 / 持久化               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                Redux Store（状态层）                          │
│   单一事实来源：管理所有 Agent 状态、通讯数据、持久化历史      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                 持久层（Persistence Layer）                   │
│   IndexedDB / localStorage → 保存对话历史与上下文            │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 持久层设计

通过引入持久层（如 IndexedDB / localStorage），确保：

- Agent 对话和状态不丢失。
- 页面刷新后历史对话和上下文不丢失。
- 每一轮对话结束时，完整的 `{ task, plan, result }` 写入持久化存储。
- 多轮对话的历史记录作为后续规划的上下文。

## 三、AgentsPD 双 Agent 模式

### 3.1 角色定义

**Agent P（Planner，规划者）**：

- **角色定位**：主脑/决策者，对应经典警探搭档中的"老练警探"。
- **职责**：解析任务、拆解子任务、做意图分析、输出执行方案、基于 Doer 传回的真实情报做判断。
- **行为特征**：不直接碰环境，只根据 Doer 汇报的信息做判断、出策略、派新任务。
- **认知科学对应**：前额叶皮层——负责统筹规划、工作记忆和决策（中央执行系统）。

**Agent D（Doer，执行者）**：

- **角色定位**：打杂的/行动派，对应经典警探搭档中的"行动派搭档"。
- **职责**：执行具体操作、读取页面/数据、查询进展、汇报结果。
- **行为特征**：负责感知环境（读页面、查数据）、执行具体动作（点击、输入、翻页），然后向 Planner 汇报。
- **认知科学对应**：感觉运动皮层——负责去外部世界"跑腿"、抓取信息、执行动作（感知运动系统）。

### 3.2 协作模式

二人组的工作流程：

1. Owner 输入任务 → 派发给 Agent P。
2. Agent P 解析任务 → 拆解子任务 → 派发给 Agent D。
3. Agent D 执行具体操作 → 读取数据 → 返回结果给 Agent P。
4. Agent P 根据 D 的汇报，决定下一步行动或完成。

这种模式本质上是一个观察者-决策者的分工：D 是眼睛和手，P 是大脑。

**AgentsPD 协作流程图：**

```
┌─────────────────────────────────────────────────────────────┐
│                    Owner（你）                                │
│   "访问这个页面，内容是什么？进展到哪了？"                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Agent P（Planner）                              │
│   解析任务 → 拆解子任务 → 派发给 D                           │
│   "D，去读这个页面的内容和状态"                               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Agent D（Doer）                                 │
│   执行具体操作 → 读取页面 → 返回结果                         │
│   "页面内容是XXX，当前进度是第二阶段"                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Agent P（Planner）                              │
│   根据 D 的汇报，决定下一步                                   │
│   "好，继续执行第三步"                                        │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 命名寓意

AgentsPD 借用了 LAPD/NYPD 的经典警探搭档（Buddy Cop）概念：

- PD = Planner + Doer。
- 在工程领域 PD 也常指 Product Designer / Program Director，自带"规划+执行"双重含义。
- 在敏捷开发中，PD 循环（Plan-Do）是最基础的迭代单元。
- 该命名自带一套行为协议：实时通讯、不擅自决策、基于真实情报判断。
- 可自然扩展——增加第三个角色（如 Agent R Reporter），即为 AgentsPDR。

## 四、意图澄清机制（Human-in-the-Loop）

### 4.1 行业痛点：为什么"反问机制"是刚需

当前 AI Agent 领域最普遍、最头疼的痛点，可以归结为两种失败模式：

- **"没听懂就开干"**：语言模型的预训练目标是预测下一个词，面对模糊输入时本能地用冗长解释"填满空白"，把"猜测"当成了"回答"。一项分析了 20,574 次真实 AI 编程会话的研究发现，AI 编程助手最大的问题不是不会写代码，而是经常自信地跑偏。
- **"一口气问十个问题把人问跑"**：模型把所有不确定当成一类处理，要么全都问、要么全都不问。真实测试中用户的原话是——"我宁愿自己手动整理。"当 Agent 反手抛出七八个问题把澄清成本全甩给用户时，用户会直接弃用。

**根因**：模型缺乏元认知（Metacognition）能力——它不知道自己"不知道什么"。

### 4.2 三档闸门设计

核心设计包含三档闸门判断不确定度：

- **低不确定**：直接执行，带默认值。
- **中不确定**：用假设推进，但标注出来。
- **高不确定**：停下来反问 Owner。

**红线规则**：涉及不可逆操作（删除、发送、付款）且关键参数缺失时，无条件必问。

### 4.3 反问形式

好的反问不是"请提供更多信息"这种空话，而是精准关闭关键分叉的问题：

- **选项式**："你希望我先起草回复，还是直接发送？"（最高效）
- **填空式**："请问收件人是谁？"（意图清楚但缺参数）
- **开放式**：只有连意图都拿不准时才用。

一次最多问 3-5 个问题。

### 4.4 置信度阈值

- 默认置信度阈值设为 0.7（经验值，可配置）。
- 高风险操作（删除、发送）可提高至 0.9。
- 低风险操作可降至 0.5。

### 4.5 防死循环机制

- clarifyState.round 记录当前是第几轮澄清。
- 超过阈值（如 3 轮）强制进入兜底逻辑——要么用默认值执行，要么直接告诉用户"信息不足，无法执行"。

### 4.6 AgentsPD 架构的降维优势

大多数现有方案还在 Prompt 层面打转（比如在系统提示词里写"遇到模糊问题时主动反问"），但这治标不治本——模型本质上还是靠"感觉"在判断。

而 AgentsPD 用 Redux-Saga 在工程架构层面把澄清机制做成了状态机：

- `yield take(CLARIFY_RESPONSE)` 让 Saga 天然阻塞等待用户回答。
- `clarifyState.round` 防止无限追问死循环。
- Planner 和 Doer 的职责分离，让"判断该不该问"和"执行具体操作"解耦。

这相当于给 Agent 装了一个外置的"元认知刹车"——不是靠模型自己"感觉"该不该问，而是由架构强制它在关键节点停下来，等 Owner 确认方向再继续推进。

### 4.7 意图澄清机制 — Redux-Saga 状态流转图

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              Owner（前端 UI）                             │
│                                                                          │
│   输入任务 → dispatch(USER_TASK_SUBMIT) → 等待 Agent 响应                │
│                                                                          │
│   ← dispatch(CLARIFY_REQUEST, { questions }) → 显示问题列表              │
│                                                                          │
│   回答后 dispatch(CLARIFY_RESPONSE, { answers }) → 继续等待              │
│                                                                          │
│   ← dispatch(TASK_COMPLETE, { result }) → 展示结果                       │
└──────────────────────┬───────────────────────────────────────────────────┘
                       │                        │
                       ▼                        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         Redux-Saga（编排层）                              │
│                                                                          │
│   function* taskOrchestrator() {                                         │
│     while (true) {                                                       │
│       // 1. 阻塞等待用户提交任务                                          │
│       const task = yield take(USER_TASK_SUBMIT);                         │
│                                                                          │
│       // 2. 派发给 Planner Agent 做意图分析                               │
│       yield put({ type: 'AGENT_PLAN', payload: task });                  │
│       const plan = yield take(AGENT_PLAN_RESPONSE);                      │
│                                                                          │
│       // 3. 判断置信度                                                    │
│       if (plan.confidence < 0.7) {                                       │
│         // 3a. 置信度不足 → 生成澄清问题                                   │
│         yield put({ type: 'AGENT_CLARIFY', payload: plan });             │
│         const clarify = yield take(AGENT_CLARIFY_RESPONSE);              │
│                                                                          │
│         // 3b. 阻塞等待用户回答                                            │
│         yield put({ type: 'CLARIFY_REQUEST',                              │
│                     payload: { questions: clarify.questions } });         │
│         const userAnswer = yield take(CLARIFY_RESPONSE);                 │
│                                                                          │
│         // 3c. 将用户回答合并回任务，重新规划                               │
│         const mergedTask = merge(task, userAnswer);                      │
│         yield put({ type: 'AGENT_PLAN', payload: mergedTask });          │
│         const replan = yield take(AGENT_PLAN_RESPONSE);                  │
│       }                                                                  │
│                                                                          │
│       // 4. 置信度足够 → 执行任务                                          │
│       yield put({ type: 'AGENT_EXECUTE', payload: plan });               │
│       const result = yield take(AGENT_EXECUTE_RESPONSE);                 │
│                                                                          │
│       // 5. 完成，通知前端                                                 │
│       yield put({ type: 'TASK_COMPLETE', payload: result });             │
│                                                                          │
│       // 6. 持久化本轮对话                                                 │
│       yield call(saveToPersistence, { task, plan, result });             │
│     }                                                                    │
│   }                                                                      │
└──────────────────────┬───────────────────────────────────────────────────┘
                       │                        │
                       ▼                        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                        Redux Store（状态层）                              │
│                                                                          │
│   {                                                                      │
│     currentTask: { id, content, status: 'pending' | 'clarifying' |       │
│                    'executing' | 'completed' },                           │
│     clarifyState: {                                                      │
│       questions: [...],    // Agent 生成的澄清问题                        │
│       answers: null,       // 用户回答（回答后写入）                       │
│       round: 0             // 澄清轮次，防止无限追问                       │
│     },                                                                   │
│     agentState: {                                                        │
│       planner: { status, confidence, plan },                             │
│       executor: { status, result }                                       │
│     },                                                                   │
│     persistence: {                                                       │
│       history: [...]       // 所有历史对话，写入 IndexedDB / localStorage  │
│     }                                                                    │
│   }                                                                      │
└──────────────────────────────────────────────────────────────────────────┘
```

## 五、工作进程观察窗口（"教室后门的洞"）

### 5.1 设计动机

类比"教室后门的洞"——班主任在窗口暗中观察，学生一举一动尽在掌握，但又不会随时打断课堂节奏。

给 Owner 开一个实时观察窗口，让你能随时看到 AgentsPD 在干什么、进展到哪了、遇到了什么问题，而不用等它跑完才告诉你结果。核心价值是**消除信息不对称**——你不再是黑盒外的旁观者，而是可以随时 peek 进去的监工。

### 5.2 观察窗口展示内容

```
┌─────────────────────────────────────────────────────────────────┐
│              AgentsPD 实时观察窗口                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  当前任务: 访问页面，提取内容和进度                                 │
│                                                                   │
│  当前阶段: Agent D 正在执行 → 读取页面 DOM                        │
│     进度: ████████░░ 80%                                          │
│                                                                   │
│  Agent P 的决策记录:                                              │
│     "已确认目标页面，派 D 去读取内容"                              │
│     "内容已获取，判断需要进一步筛选关键信息"                         │
│                                                                   │
│  Agent D 的实时反馈:                                              │
│     "已打开页面，正在解析..."                                      │
│     "获取到 3 个关键段落，等待 P 指示"                              │
│                                                                   │
│  快速交互区:                                                      │
│     [输入框] 随时插话、调整方向、提问                               │
│     [暂停] [继续] [终止] 快捷按钮                                  │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 三个关键能力

**1. 实时状态推送（不是轮询，是推送）**

Agent D 每完成一个子步骤，就 dispatch 一个 `AGENT_DO_PROGRESS` action，观察窗口通过订阅 Redux Store 实时更新。你看到的不是"加载中..."，而是：

- "D 已打开页面"
- "D 正在解析 DOM"
- "D 提取到 5 个段落"
- "D 等待 P 的下一步指示"

就像教室后门的窗口，你能看到学生是在认真做题，还是在发呆，还是在举手提问。

**2. 随时插话的通道（多轮对话）**

观察窗口下方永远有一个输入框，你可以随时：

- **调整方向**："先别管段落了，帮我找一下页面上有没有'提交'按钮"
- **追问细节**："那 5 个段落里，有没有提到截止日期？"
- **纠正错误**："不对，我要的不是这个页面，是另一个"

这些插话会 dispatch 一个 `OWNER_INTERJECT` action，Saga 监听到后，会根据当前状态决定：

- 如果 D 正在执行，先暂停 D
- 把插话内容交给 P 重新规划
- P 输出新指令，D 继续执行

**3. Owner 的"不闻不问"边界**

这里有个微妙的设计——**观察不等于干预**。

- **默认模式**：你看着窗口，AgentsPD 正常干活，你不说话，它们也不打扰你。就像班主任在窗口看着，学生正常上课。
- **干预模式**：你发现方向不对，随时插话。就像班主任敲敲窗户，喊一声"小明，这道题思路错了"。

关键是：**Agent 不能因为你没说话就停下来等你确认**，也不能因为你在看就频繁打断你问"这样可以吗？这样可以吗？"。只有当它遇到高不确定的关键分叉时，才通过澄清机制主动弹窗问你。

### 5.4 多轮对话的状态流转（加入观察窗口后）

```
┌──────────────────────────────────────────────────────────────────┐
│                         Owner（你）                               │
│                                                                  │
│   提交任务 → 观察窗口实时显示进度                                  │
│                                                                  │
│   ← Agent D 汇报："已读取页面，内容如下..."                        │
│                                                                  │
│   随时插话："帮我找一下截止日期" → dispatch(OWNER_INTERJECT)     │
│                                                                  │
│   ← Agent P 重新规划："D，去页面上找日期信息"                      │
│                                                                  │
│   ← Agent D 汇报："找到日期：2026-10-01"                          │
│                                                                  │
│   继续插话或等待完成...                                            │
└──────────────────────┬───────────────────────────────────────────┘
                       │                        │
                       ▼                        ▼
┌──────────────────────────────────────────────────────────────────┐
│                      Redux-Saga（编排层）                         │
│                                                                  │
│   function* taskOrchestrator() {                                 │
│     while (true) {                                               │
│       // 1. 等待任务或插话                                        │
│       const action = yield take([                                │
│         USER_TASK_SUBMIT,                                        │
│         OWNER_INTERJECT,                                         │
│       ]);                                                         │
│                                                                  │
│       if (action.type === OWNER_INTERJECT) {                     │
│         // 插话：暂停当前执行，重新规划                             │
│         yield put({ type: 'AGENT_PAUSE' });                      │
│         yield put({ type: 'AGENT_PLAN',                          │
│                     payload: { interjection: action.payload } }); │
│         const replan = yield take(AGENT_PLAN_RESPONSE);           │
│         yield put({ type: 'AGENT_EXECUTE', payload: replan });    │
│       } else {                                                   │
│         // 新任务：正常流程                                        │
│         yield put({ type: 'AGENT_PLAN', payload: action });       │
│         const plan = yield take(AGENT_PLAN_RESPONSE);             │
│         yield put({ type: 'AGENT_EXECUTE', payload: plan });      │
│       }                                                          │
│                                                                  │
│       // 2. 执行过程中，D 实时推送进度                             │
│       while (true) {                                             │
│         const progress = yield take(AGENT_DO_PROGRESS);           │
│         yield put({ type: 'PROGRESS_UPDATE',                     │
│                     payload: progress });                         │
│                                                                  │
│         // 检查是否有插话                                         │
│         const interject = yield race({                           │
│           interject: take(OWNER_INTERJECT),                       │
│           complete: take(AGENT_DO_RESPONSE),                     │
│         });                                                       │
│                                                                  │
│         if (interject.interject) {                               │
│           // 有插话，跳出内层循环，重新规划                         │
│           break;                                                 │
│         }                                                        │
│         if (interject.complete) {                                │
│           // 完成了，跳出                                          │
│           break;                                                 │
│         }                                                        │
│       }                                                          │
│     }                                                            │
│   }                                                              │
└──────────────────────────────────────────────────────────────────┘
```

### 5.5 核心设计原则

**1. 观察 ≠ 干预**

你在窗口看着，AgentsPD 正常干活。只有你主动插话，或者 Agent 遇到高不确定需要澄清时，才会产生交互。

**2. 插话 ≠ 打断**

插话不是"停下手头一切"，而是"把这个新信息纳入当前规划"。Agent P 会根据插话内容判断：

- 如果是方向性调整 → 暂停 D，重新规划
- 如果是补充信息 → 继续执行，把信息带上
- 如果是无关内容 → 忽略或礼貌回应

**3. 多轮 ≠ 无限轮**

每一轮插话都会记录在 `history` 里，作为上下文。但为了防止对话无限发散，可以设置一个"上下文窗口"——比如最近 10 轮插话会纳入规划，更早的会被压缩或丢弃。

## 六、Redux 状态设计

### 6.1 State 结构

```
{
  currentTask: {
    id: null,
    content: '',
    status: 'idle' | 'planning' | 'clarifying' | 'doing' | 'completed' | 'error'
  },
  planner: {
    status: 'idle' | 'working' | 'done',
    plan: null,
    confidence: 1
  },
  doer: {
    status: 'idle' | 'working' | 'done',
    result: null,
    progress: '' // D 实时汇报进度
  },
  clarify: {
    questions: [],
    answers: null,
    round: 0
  },
  history: [] // 持久化历史
}
```

### 6.2 Action Types

**用户侧**：

- `USER_TASK_SUBMIT`：用户提交任务
- `CLARIFY_RESPONSE`：用户回答澄清问题
- `OWNER_INTERJECT`：用户在观察窗口中插话/干预

**Planner 侧**：

- `AGENT_PLAN_REQUEST`：请求 Planner 规划
- `AGENT_PLAN_RESPONSE`：Planner 返回规划结果
- `AGENT_CLARIFY_REQUEST`：Planner 请求澄清
- `AGENT_CLARIFY_RESPONSE`：澄清结果返回

**Doer 侧**：

- `AGENT_DO_REQUEST`：请求 Doer 执行
- `AGENT_DO_RESPONSE`：Doer 返回执行结果
- `AGENT_DO_PROGRESS`：Doer 实时推送进度更新

**任务控制**：

- `AGENT_PAUSE`：暂停当前执行（插话时触发）
- `TASK_COMPLETE`：任务完成
- `TASK_ERROR`：任务出错

**UI 同步**：

- `PROGRESS_UPDATE`：将进度推送到观察窗口

### 6.3 Saga 编排流程

```
while (true) {
  // 1. 阻塞等待用户提交任务或插话
  yield take([USER_TASK_SUBMIT, OWNER_INTERJECT]);

  // 2. 派发给 Planner 做意图分析
  yield put({ type: 'AGENT_PLAN_REQUEST' });

  // 3. 判断置信度
  if (confidence < 0.7) {
    // a. 生成澄清问题
    yield put({ type: 'AGENT_CLARIFY_REQUEST' });
    // b. 阻塞等待用户回答
    yield take(CLARIFY_RESPONSE);
    // c. 将用户回答合并回任务，重新规划
  } else {
    // confidence >= 0.7，继续执行
  }

  // 4. 派发给 Doer 执行任务
  yield put({ type: 'AGENT_DO_REQUEST' });

  // 5. 执行过程中实时推送进度
  while (true) {
    const progress = yield take(AGENT_DO_PROGRESS);
    yield put({ type: 'PROGRESS_UPDATE', payload: progress });

    // 检查是否有插话
    const interject = yield race({
      interject: take(OWNER_INTERJECT),
      complete: take(AGENT_DO_RESPONSE),
    });

    if (interject.interject) {
      // 有插话，跳出内层循环，重新规划
      break;
    }
    if (interject.complete) {
      // 完成了，跳出
      break;
    }
  }

  // 6. 完成任务，通知前端
  yield put({ type: 'TASK_COMPLETE' });

  // 7. 持久化本轮对话
  yield call(persistConversation);
}
```

**关键**：`yield take(CLARIFY_RESPONSE)` 会让整个 Saga 暂停在这里，直到用户在前端回答了问题并 dispatch 了 `CLARIFY_RESPONSE`，Saga 才会继续往下走。`yield race({ interject, complete })` 让执行过程可以同时响应插话和完成信号。

## 七、上下文管理

Agent 所需的上下文信息通过 Redux 架构实现：

- Agent 在执行任务时，从 Store 中读取所需的上下文状态。
- 执行完毕后将结果 dispatch 回 Store，形成闭环。
- 历史对话记录写入持久层，作为后续对话的上下文。
- 多轮对话中，每一轮插话和澄清记录都纳入上下文窗口，保证对话连续性。

## 八、应用场景

### 8.1 实验计算平台

- 将实验（如 Exp11）部署到 Kaggle Notebooks 上，作为后台重计算的 Workhorse。
- 使用 Qwen2.5-0.5B 模型作为基线，先在 Kaggle 上跑通整个流程。
- 后续更大规模模型的实验直接迁移到 Kaggle 上执行，利用其免费算力资源。

### 8.2 通用任务编排

- 任何需要多步骤、多角色协作的任务场景。
- 需要人类介入确认的关键决策节点。
- 需要保持上下文连续性的对话式任务。

## 九、设计优势总结

1. **状态可预测**：Redux 的单向数据流确保所有 Agent 状态变化可追踪、可调试。
2. **协作有章法**：Planner-Doer 分工明确，避免 Agent 各自为战。
3. **人类兜底**：意图澄清机制确保 Agent 不会在模糊意图下错误执行。
4. **数据不丢**：持久层保障通讯数据不丢失，支持断点续传。
5. **可扩展**：二人组模式可自然扩展到三人组（如增加 Reporter 角色）。
6. **认知对齐**：架构设计与人类认知科学中的"中央执行-感知运动"分工模式一致。
7. **透明协作**：工作进程观察窗口消除信息不对称，Owner 随时可见、随时干预。
8. **对话驱动**：多轮对话机制让 Owner 不闻不问、Agent 不埋头蛮干，形成真正的协作关系。

## 十、交互哲学："对话即编程"

### 10.1 核心理念

你现在跟 Agent 聊天的这个过程，本身就是 AgentsPD 的雏形：

- 你（Owner）提出想法 → Agent（Planner）理解、拆解、给出方案 → 你补充、修正、拍板 → Agent（Doer）去执行 → 你看着结果，觉得不对再插一句 → Agent 重新规划……

**文字和代码只是输出介质的区别。** 在这边打字说"这个关卡怎么过"，那边 Agent 在敲键盘写代码跑实验，背后的协作逻辑一模一样。

### 10.2 协作契约

- **你（Owner）不埋头蛮干**——遇到不确定的就停下来问。
- **Agent 不没听懂就开干**——先确认意图再动手。
- **你随时可以插话**——"不对，换个方向"。
- **Agent 实时汇报进度**——"做到这一步了，你看下一步怎么走"。

### 10.3 从对话到工程落地

等这套架构真正落地到工程里，无非就是把"回你文字"替换成"Agent 跑代码、写文件、push kernel"，把"你回我文字"替换成"你在观察窗口里点按钮、打字插话"。

**交互模式没变，只是执行层从"文字"换成了"代码"。**

人跟 Agent 的关系，不应该是一个"指令-执行"的黑盒，而应该是一个"对话-协作"的透明过程。你越习惯用对话的方式跟 Agent 相处，AgentsPD 这套架构就越顺手。

## 十一、总结与展望

本文档详细阐述了基于 Redux + Redux-Saga 的双 Agent 多智能体系统（AgentsPD）的技术设计。系统通过 Planner-Doer 的明确分工，结合 Redux 的单向数据流和 Saga 的异步编排能力，构建了一个状态可预测、协作有章法、人类可兜底的多智能体协作框架。意图澄清机制、工作进程观察窗口和多轮对话机制的引入，进一步提升了系统的鲁棒性和用户体验。

未来可能的扩展方向包括：

- 增加更多 Agent 角色（如 Reporter、Reviewer），构建更复杂的协作网络。
- 接入更大规模的模型，提升 Planner 的决策质量和 Doer 的执行能力。
- 支持多用户协作，实现跨用户的任务分发与结果共享。
- 引入更智能的置信度评估机制，减少不必要的澄清交互。
- 优化上下文窗口管理，实现更长的有效对话历史。
