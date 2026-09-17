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
│ Transcription / Context(RAG) / Summary / Action / │
│ Insight / Follow-up                              │
├─────────────────────────────────────────────────┤
│ 检索层 (RAG)                                     │
│ 分块 → 混合检索(向量+BM25+RRF) → 精排 → 引用        │
├─────────────────────────────────────────────────┤
│ 集成层 (Integration)                             │
│ MiniMax LLM / WhisperX + pyannote / Jira / 飞书    │
├─────────────────────────────────────────────────┤
│ 数据层 (Storage)                                 │
│ 进程内会议结果 / SQLite 台账 / 向量索引 / 报告        │
└─────────────────────────────────────────────────┘
```

## 2. 编排模式

```
START → [Transcription] → [Context/RAG] → Fan-out → [Summary | Action | Insight] → Fan-in → [Follow-up] → END
```

| 阶段 | 模式 | 为什么 |
|------|------|--------|
| 音频 → 转写 | Pipeline（串行） | 后续所有分析都依赖转写文本，无法并行 |
| 转写 → 检索上下文 | Pipeline（串行） | 检索结果要喂给三个分析 Agent，必须先行完成 |
| 检索 → 纪要/待办/洞察 | Fan-out（并行） | 三者输入相同、输出互不依赖，串行只会叠加延迟 |
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
- 健康检查：`GET /healthz` 返回版本、各集成是否配置就绪、知识库索引状态、活跃会议数，只做配置探测、不发外部请求。

## 8. 检索增强（RAG）

### 8.1 模块与数据流

```
data/knowledge(内部文档) ─┐
data/meetings(会议纪要)  ─┼─► 分块(标题感知+滑窗) ─► 向量索引(numpy) ┐
config/glossary.json    ─┘                                          ├─► 混合检索 ─► 精排 ─► Top-K ─► 引用
                                            ─► BM25 倒排 ────────────┘
```

| 模块 | 职责 |
|------|------|
| `rag/chunking.py` | Markdown 标题感知分块 + 超长滑窗重叠，产出带 doc/section 元数据的 chunk |
| `rag/tokenize.py` | 中文二元切分 + 西文词，查询额外补单字（无 jieba 依赖） |
| `rag/bm25.py` | BM25 倒排（精确命中：编号、人名、数字门限） |
| `rag/embedding.py` | 可插拔向量后端：本地 BGE / 离线哈希 / OpenAI 兼容，失败自动降级 |
| `rag/vector_store.py` | numpy 余弦索引 + 落盘 |
| `rag/terminology.py` | 术语表：匹配、查询扩展、加权、prompt 约束、术语表自身入库 |
| `rag/retriever.py` | 四路召回 + RRF 融合 + 引用生成 |
| `rag/rerank.py` | 确定性精排：查询词覆盖度、连续短语、小节标题、术语密度 |
| `rag/qa.py` | 带引用的问答：资料约束 + 编号引用 + 术语约束，无命中不调用 LLM |
| `rag/ingest.py` | 索引构建/加载/落盘（`data/index`） |

### 8.2 与主链路的关系

- **Context 节点**（`agents/context_agent.py`）：转写 → 构造查询（开场片段 + 命中术语）→ 检索历史会议 → 写 `state["context"]`；
- Summary 的 user prompt 前置「历史背景 + 公司术语」块，并明确「背景不得当成本次会议内容」；
- Follow-up 报告新增「相关历史决议」章节；
- **失败不阻塞**：没有索引、检索异常、无命中，一律写空上下文并记一条 error，主流程照常完成（有 `test_retrieval_failure_is_recorded_not_raised` 覆盖）。

### 8.3 为什么扩展词走独立通道

第一版把「原查询 + 术语标准名/别名」拼成一个查询去检索，实测 Recall@1 反而从 0.625 掉到 0.417：扩展词稀释了原查询的语义中心，术语表 chunk 还靠术语密度抢占首位。现在向量通道永远只用原查询，扩展只作为**独立低权重通道**参与 RRF，术语表 chunk 在查询不含术语时降权 0.6。完整消融数据见 [evaluation.md](evaluation.md#五rag-检索评测scriptssevaluate_ragpy)。

## 9. 工具协议与自主工具调用（MCP）

### 9.1 模块与数据流

```
MCP 客户端（Claude Desktop / Cursor / 自研）
        │  stdio: 换行分隔 JSON-RPC     │  HTTP: POST /mcp   或  GET /mcp/sse + POST /mcp/messages
        ▼                                ▼
   mcp/server.py  ── 方法分发 ──► mcp/registry.py ──► mcp/tools.py ──► 既有能力
   （initialize / tools / resources / prompts / ping）      │            ├ rag/retriever.py
                                                          │            ├ reports/*.md
   mcp/policy.py   只读默认、写工具需授权、白名单            │            ├ integrations/{jira,feishu,idempotency}
   mcp/audit.py    每次调用一行 JSONL（参数摘要）            │            └ config/glossary.json
                                                          └── llm_catalog() ──► agents/tool_agent.py（ReAct）
```

| 文件 | 职责 |
|------|------|
| `mcp/protocol.py` | JSON-RPC 2.0 报文解析/编码、错误码、协议版本协商 |
| `mcp/registry.py` | `ToolSpec` 注册表、JSON Schema 参数校验、调用与计时 |
| `mcp/policy.py` | 权限策略：只读默认、写工具开关、白名单 |
| `mcp/audit.py` | 审计日志（JSONL，只记参数 SHA-256 摘要） |
| `mcp/tools.py` | 四个业务工具 |
| `mcp/catalog.py` | Resources（会议报告）与 Prompts（`summarize_meeting`） |
| `mcp/server.py` | 方法分发 + stdio 传输（`python -m src.mcp.server`） |
| `mcp/client.py` | 最小客户端（进程内 + 子进程 stdio），测试与评测用 |
| `agents/tool_agent.py` | 自主工具调用循环（非确定性编排） |

### 9.2 两种编排的边界

主流水线是**确定性编排**：节点与边写死在 `meeting_graph.py`，可预测、可测试、适合「每场会议都要做同样几件事」。工具调用循环是**非确定性编排**：走几步、调什么由模型决定，适合「用户随口一问」。两者共用同一批底层能力与同一份工具定义，但**不互相调用**——把非确定性循环塞进状态图会让主流水线的行为不可复现。

### 9.3 状态与副作用

- MCP 层**不写** `MeetingState`：工具只读既有产物（索引、报告文件、术语表、台账），写操作仅限「建单」，且复用 `SyncLedger` 幂等键，重复调用返回已有外部 ID；
- 会话状态（SSE session → queue）只在进程内，属接入层状态，不进入业务状态模型。

### 9.4 失败与熔断

| 层 | 失败行为 |
|----|---------|
| 协议层 | 解析失败 `-32700`、未知方法 `-32601`、参数非法 `-32602`、内部错误 `-32603`；**工具执行失败走 `isError` 而不是 JSON-RPC 错误**（协议层说「调用姿势对不对」，结果层说「工具成没成」） |
| 工具层 | 参数校验先于执行；异常统一捕获为 `isError` 并写审计；错误信息不回显服务器路径 |
| 循环层 | 最大步数 / 单次超时 / 相同 `(tool, args)` 重复熔断 / 连续失败上限；熔断后回落成总结，不返回假答案 |
