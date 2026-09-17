# 架构设计

## 1. 分层

```
┌─────────────────────────────────────────────────┐
│ 接入层 (Gateway)                                 │
│ REST / WebSocket / 健康检查                       │
├─────────────────────────────────────────────────┤
│ 编排层 (Orchestration)                           │
│ LangGraph StateGraph：Pipeline + Fan-out/Fan-in   │
├─────────────────────────────────────────────────┤
│ Agent 层                                         │
│ Transcription / Summary / Action / Insight /     │
│ Follow-up                                        │
├─────────────────────────────────────────────────┤
│ 集成层 (Integration)                             │
│ MiniMax LLM / WhisperX + pyannote / Jira / 飞书    │
├─────────────────────────────────────────────────┤
│ 数据层 (Storage)                                 │
│ 进程内会议结果 / SQLite 同步台账 / Markdown 报告     │
└─────────────────────────────────────────────────┘
```

## 2. 编排模式

```
START → [Transcription] → Fan-out → [Summary | Action | Insight] → Fan-in → [Follow-up] → END
```

| 阶段 | 模式 | 为什么 |
|------|------|--------|
| 音频 → 转写 | Pipeline（串行） | 后续所有分析都依赖转写文本，无法并行 |
| 转写 → 纪要/待办/洞察 | Fan-out（并行） | 三者输入相同、输出互不依赖，串行只会叠加延迟 |
| 纪要+待办+洞察 → 跟进 | Fan-in（汇聚） | 跟进要汇总三路结果，必须等全部完成 |

并行收益由 `scripts/evaluate.py` 现场测量（模拟 0.4s/次 LLM 延迟时加速比约 2.9x），不写成文档里的固定数字。

## 3. 状态模型

所有 Agent 共享一个状态对象（`src/models/schemas.py::MeetingState` / `src/graph/meeting_graph.py::GraphState`）：

```
MeetingState = {
  meeting_id,      ← 全局标识
  status,          ← 处理阶段
  audio_data,      ← 输入
  transcript,      ← TranscriptionAgent 写
  transcript_text, ← TranscriptionAgent 写（扁平文本，供三个分析 Agent 消费）
  summary,         ← SummaryAgent 写（并行）
  actions,         ← ActionAgent 写（并行）
  insights,        ← InsightAgent 写（并行）
  followup,        ← FollowUpAgent 写
  errors,          ← 所有节点都可追加（Annotated[list[str], operator.add] 归约）
}
```

三条约定：

1. 每个 Agent 只读自己需要的字段、只写自己负责的字段 —— 并行节点写不同字段，天然无冲突；
2. 节点返回值是**状态增量**（只含自己写的字段），由 LangGraph 合并，而不是原地改整个 state；
3. 任何失败都追加到 `errors` 并继续，绝不抛出中断 Pipeline —— 并行阶段一个节点挂掉，其余两个照常产出。

## 4. 容错设计

### 4.1 Agent 级

```python
try:
    ...核心逻辑...
except Exception as e:
    errors_delta.append(f"{AgentName}: {e}")   # 进 state["errors"]
    state[...] = 降级结果                       # 保证下游有东西可用
```

### 4.2 降级策略

| Agent | 降级方案 | 下游可见性 |
|-------|----------|------------|
| Transcription | WhisperX 不可用 → 内置演示转写（8 段示例会议） | 报告照常生成 |
| Summary | LLM 失败 → 规则提取说话人 + 占位议题 | 报告标注「自动摘要降级模式」 |
| Action | LLM 失败 → 空待办清单，错误入 `errors` | 报告显示「无待办事项」 |
| Insight | LLM 失败 → 保留规则统计（发言占比/效率分），语义字段留空 | 报告仍有发言统计 |
| Follow-up | 飞书不可用 → 只落盘本地报告 | `sync_status` 标明 disabled |

### 4.3 重试

- LLM 与外部 API 调用用 `tenacity` 指数退避（1s → 2s → 4s，最多 3 次）；
- 只对可重试错误重试，参数类错误直接抛出（MiniMax 的 `base_resp.status_code != 0` 会转成异常）。

## 5. 幂等同步

会议可能被重复处理（WebSocket 重连、重投、手工重跑），外部系统写入必须可重放：

```
业务键 = sha256(归一化(meeting_id) | 归一化(assignee) | 归一化(task))[:32]
```

- 台账是 SQLite 表 `sync_ledger(item_key, target, external_id, meeting_id, created_at)`，主键 `(item_key, target)`；
- 写入前先查：命中则复用已有外部 ID，跳过 API 调用，并计入 `ActionResult.duplicates_skipped`；
- 只有外部写入成功才登记 —— 失败不占坑，下轮重试仍会尝试创建；
- 台账落盘，进程重启后依然生效；并行阶段多 Agent 同时写，用线程锁串行化 sqlite 写入。

## 6. 可扩展性

- **新增 Agent**：实现 `async def process(state) -> dict`，在 `build_meeting_graph` 注册节点并连边即可，不需要改其它 Agent；
- **换 LLM**：`MiniMaxClient` 只暴露 `chat` / `chat_json` 两个方法，替换实现即可（各 Agent 构造函数均支持注入 `llm_client`）；
- **换外部系统**：`JiraClient` / `FeishuClient` 同样支持注入，测试里就是用假客户端替换的；
- **横向扩展**：Agent 无状态，会议状态目前存进程内，多实例需要先把 `meeting_results` 换成共享存储。

## 7. 可观测性

- 结构化日志：`[AgentName] action: meeting_id, detail`，loguru 输出到 stderr；
- 关键计数：`ActionAgent` 每次同步都输出 `created / skipped / failed`，落进 `ActionResult.sync_status`；
- 健康检查：`GET /healthz` 返回版本、各集成是否配置就绪、活跃会议数，只做配置探测、不发外部请求。
