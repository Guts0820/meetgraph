# MCP 分步实施计划（工具协议 + 自主工具调用）

## 目标

把 MeetGraph 已有的能力（会议历史检索、会议报告、待办建单、术语查询）从「进程内硬编码调用」升级为两件面试必考的东西：

1. **MCP Server**：用标准协议暴露工具与资源，Claude Desktop / Cursor / 自研 Agent 都能直接接入；
2. **自主工具调用循环**：让 LLM 自己决定调哪个工具、传什么参数，并处理参数非法、工具报错、重复调用这些真实问题。

同时补齐「工具调用」专题的追问面：Function Calling vs MCP、协议组成、通信方式、权限与审计、死循环防护。

## 设计

```
                    ┌───────────────────────────────┐
Claude Desktop ──┐  │  src/mcp/                     │
Cursor ──────────┼─►│  protocol   JSON-RPC 2.0      │──► registry ──► tools ──► 既有能力
自研 Agent ──────┘  │  server     initialize / list │      │            │      ├ 会议历史检索(RAG)
（stdio / SSE）      │  policy     白名单 + 只读/写入   │      │            │      ├ 会议报告(文件)
                    │  audit      JSONL 审计          │      │            │      ├ 待办建单(幂等台账)
                    └───────────────────────────────┘      │            │      └ 术语查询(glossary)
                                                           └── 同一份 Schema ──► ToolCallingAgent（ReAct 循环）
```

**关键设计：一份工具定义，两处消费。** `registry` 里的 `ToolSpec`（名称/描述/JSON Schema/只读标记）既是 MCP `tools/list` 的返回内容，也是喂给 LLM 的工具目录——避免「MCP 暴露的工具」和「模型能调的工具」两套定义漂移。

### 协议实现范围（自己实现，不引官方 SDK）

| 方法 | 说明 |
|------|------|
| `initialize` | 协议版本协商 + 能力声明（tools / resources / prompts） |
| `notifications/initialized` | 客户端就绪通知（无响应） |
| `tools/list` / `tools/call` | 工具发现与调用，返回 `content[]` + `isError` |
| `resources/list` / `resources/read` | 会议报告作为资源（`meeting://report/{id}`） |
| `prompts/list` / `prompts/get` | 提供 `summarize_meeting` 提示模板 |
| `ping` | 连通性 |

传输：**stdio**（换行分隔的 JSON-RPC，MCP 标准做法）+ **HTTP**（`GET /mcp/sse` 事件流 + `POST /mcp/messages`，另提供 `POST /mcp` 简单直返）。

## 步骤、产出与验收

| # | 步骤 | 产出 | 验收标准 | 状态 |
|---|------|------|----------|------|
| 1 | 协议内核 | `src/mcp/{protocol,registry,audit,policy}.py` | JSON-RPC 解析/编码/错误码有单测；工具注册与白名单可测；审计落 JSONL | ✅ 完成 |
| 2 | 工具实现 | `src/mcp/tools.py`：`search_meetings` / `get_meeting_report` / `create_action_item` / `lookup_glossary` | 四个工具各自的正常路径与错误路径有单测；写工具走幂等台账 | ✅ 完成 |
| 3 | 传输层 | `src/mcp/server.py`（stdio，可 `python -m src.mcp.server` 启动）+ `src/mcp/client.py`（最小客户端）+ HTTP/SSE 端点接入 `src/websocket/server.py` | 子进程真起 server，完成 initialize → tools/list → tools/call → 错误路径；SSE 端点用 TestClient 验证 | ✅ 完成（SSE 改为直测事件流生成器，原因见 docs/mcp.md 负结果 2） |
| 4 | 自主工具调用 | `src/agents/tool_agent.py`（ReAct 循环） | 最大步数、超时、参数校验、错误回传自修复、重复调用熔断都有测试 | ✅ 完成 |
| 5 | 评测 | `scripts/evaluate_tools.py` + `tests/fixtures/tool_tasks.jsonl`（12 条标注任务） | 输出工具选择准确率、参数正确率、循环熔断命中率、权限拒绝率、审计完整性；支持 `--live` 用真实 LLM | ✅ 完成（指标口径被实测修正过一次，见下） |
| 6 | 测试与文档 | `tests/test_mcp_*.py`、`tests/test_tool_agent.py`、`docs/mcp.md`、README/architecture/development 更新 | `pytest` 全绿；文档含 Claude Desktop / Cursor 的接入配置样例 | ✅ 完成 |

### 执行中的偏差（实测推翻的设计）

| 原设计 | 实测问题 | 改成 |
|--------|---------|------|
| 工具选择用「工具序列完全相等」、参数用「精确相等」评分 | 真实 LLM 拿到 0.182 / 0.364，但翻轨迹发现扣分多是「多查一次确认」和「用自己的措辞检索」，不是错误 | 集合级召回/精确率 + 结构化参数精确、自由文本参数非空即可 |
| SSE 用 `TestClient` / `httpx.ASGITransport` 写集成测试 | 流未关闭时 ASGITransport 不支持并发请求（挂死测试套件），TestClient 关闭流又不取消服务端生成器 | 事件流抽成 `sse_event_stream` 直接单测（帧/心跳/断开清理），HTTP 层只覆盖 `POST /mcp` 与会话投递 |
| 写工具「暴露但不允许」 | 模型会去调一个注定失败的工具 | 未授权时干脆不出现在 `tools/list` 与 LLM 目录里（`test_disabled_write_tool_is_not_advertised`） |
| SSE 端点没有断开检查 | 客户端断开后会话永远留在 `_mcp_sessions`（长连接泄漏） | 每个心跳周期 `request.is_disconnected()`，断开即回收会话 |

## 安全与可观测（面试高频追问）

| 关注点 | 做法 |
|--------|------|
| 权限最小化 | 工具声明 `readonly`；写工具默认关闭，需 `MCP_ALLOW_WRITE=1`；支持 `MCP_TOOL_ALLOWLIST` 白名单 |
| 审计 | 每次调用落 JSONL：时间、工具、调用方、状态、耗时、**参数摘要（SHA-256，不落明文）** |
| 参数校验 | 调用前按 JSON Schema 做类型/枚举/必填校验，校验失败返回 `INVALID_PARAMS` 而不是抛异常 |
| 工具投毒 / 越权 | 工具描述固定、不接受客户端注入；写工具禁止删除类操作；错误信息不回显内部路径 |
| 死循环 | 循环层限制最大步数、单次工具超时、相同 (tool, args) 重复调用熔断、连续错误上限 |
| 幂等 | 写工具复用 SyncLedger：重复调用返回已有外部 ID 而不是重复建单 |

## 与既有模块的边界

- **不引入官方 `mcp` SDK**：协议只有 7 个方法，自己实现能讲清协商/传输/错误码细节，且测试完全不联网；未来要接远端 Server 时再引 SDK 不冲突；
- **不引入原生 tool_calls**：当前 LLM 客户端无该能力，用严格 JSON 协议（`{"action": ..., "args": {...}}`）实现同等效果，代码里预留 native tool_calls 分支；
- **不改动主流水线**：MCP 与 ToolAgent 是并列的对外能力，会议 Pipeline 的 6 个节点保持不变。
