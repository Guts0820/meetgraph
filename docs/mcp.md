# MCP 工具调用与协议实现

> 目标：让 MeetGraph 的能力不只是「本进程里能调的函数」，而是**任何 MCP 客户端都能接的工具与资源**，同时让 LLM 能在受控前提下自己决定调用哪个工具。

## 1. 为什么要有这一层

主流水线（`src/graph/meeting_graph.py`）是**确定性编排**：转写 → 检索 → 摘要 → 待办 → 同步，顺序写死在图里。它解决的是「会议处理」，不解决「用户随口问一句、需要现场决定查什么」的问题。

这一层补两件事：

| 能力 | 解决的问题 | 实现位置 |
|------|-----------|----------|
| MCP Server | 工具/资源的**标准化对外暴露**（跨客户端复用，不写 N×M 适配代码） | `src/mcp/` |
| 自主工具调用循环 | 让 LLM **自己决定**调哪个工具、传什么参数，并处理失败 | `src/agents/tool_agent.py` |

## 2. 架构

```
                            ┌──────────────────────────────────────┐
Claude Desktop ──┐          │  src/mcp/                            │
Cursor ──────────┼── stdio ─►│  protocol.py   JSON-RPC 2.0 报文/错误码│
自研 Agent ──────┤          │  server.py     initialize / dispatch  │
（HTTP/SSE）─────┴─ HTTP ──►│  policy.py     只读默认 / 写需授权     │──┐
                            │  audit.py      JSONL 审计（参数摘要）  │  │
                            │  registry.py   工具注册 + 参数校验     │  │
                            └──────────────────────────────────────┘  │
                                                                       ▼
      ┌──────────────────────── 一份 ToolSpec（名称/描述/JSON Schema/只读标记）────────────────────────┐
      │                                                                                             │
      ▼ tools/list（MCP 客户端看到的）                          ▼ llm_catalog()（模型看到的）           │
 Claude Desktop / Cursor                                ToolCallingAgent（ReAct 循环）              │
      │                                                                  │                          │
      ▼ tools/call                                                       ▼ 调用                       │
   ┌──────────────────────────────── 业务工具 ────────────────────────────────┐                    │
   │ search_meetings  → src/rag/retriever.py（只取会议纪要来源）                │◄───────────────────┘
   │ get_meeting_report → reports/meeting-report-<id>.md                      │
   │ create_action_item → SyncLedger 幂等 + Jira/飞书客户端                     │
   │ lookup_glossary   → config/glossary.json                                │
   └──────────────────────────────────────────────────────────────────────────┘
```

**关键设计：一份工具定义，两处消费。** `ToolSpec` 既是 MCP `tools/list` 的返回内容，也是喂给 LLM 的工具目录。两套定义分开写，迟早出现「MCP 暴露了但模型不知道」或反之的漂移。

## 3. 协议实现范围

自己实现，**不引官方 `mcp` SDK**（理由见第 8 节）。已实现的方法：

| 方法 | 是否响应 | 说明 |
|------|---------|------|
| `initialize` | ✅ | 版本协商（客户端版本在支持列表里则回显，否则回服务端版本）+ 能力声明 |
| `notifications/initialized` | ❌ | 客户端就绪通知，按协议不得回复 |
| `tools/list` | ✅ | 只列出当前策略允许的工具（写工具未授权时直接不出现在列表里） |
| `tools/call` | ✅ | 返回 `content[]` + `isError`；工具执行失败走 `isError`，**调用方式错误**走 JSON-RPC 错误 |
| `resources/list` / `resources/read` | ✅ | 会议报告作为资源：`meeting://report/{meeting_id}` |
| `prompts/list` / `prompts/get` | ✅ | `summarize_meeting` 提示模板（把「读报告 → 出纪要」固化成模板） |
| `ping` | ✅ | 连通性 |
| 未知方法 | ✅（错误） | `-32601 METHOD_NOT_FOUND` |

错误码：`-32700` 解析失败 / `-32600` 非法请求 / `-32601` 方法不存在 / `-32602` 参数非法 / `-32603` 内部错误。

### 传输方式

| 传输 | 路径 | 适用 |
|------|------|------|
| stdio | `python -m src.mcp.server` | 本地客户端（Claude Desktop / Cursor）。**换行分隔 JSON-RPC**，消息内不得含裸换行 |
| HTTP + SSE | `GET /mcp/sse` 拿投递端点 → `POST /mcp/messages?session_id=…` | 远程客户端；响应经事件流回推，15s 心跳，断连即回收会话 |
| HTTP 简化 | `POST /mcp` | 一次请求一次响应，适合脚本/curl 与不需流的客户端 |

stdio 为什么用换行而不是 `Content-Length` 帧：MCP 标准就是这么定的（消息必须单行），也省掉一层帧解析。

## 4. 四个工具

| 工具 | 只读 | 参数（要点） | 复用模块 |
|------|------|-------------|----------|
| `search_meetings` | ✅ | `query`（必填）、`top_k`（1~20，默认 5） | `src/rag/retriever.py`，只保留 `source_type=meeting` 的片段 |
| `get_meeting_report` | ✅ | `meeting_id`（必填，字符白名单净化） | `reports/meeting-report-<id>.md` |
| `create_action_item` | ❌ 写 | `task`、`task_assignee`（必填）、`deadline`（`YYYY-MM-DD`）、`priority`（枚举）、`meeting_id` | `SyncLedger` 幂等键 + `JiraClient` + `FeishuClient` |
| `lookup_glossary` | ✅ | `term`（必填，支持别名/大小写） | `config/glossary.json`（20 条术语） |

写工具的幂等：业务键 = `sha256(meeting_id | assignee | task)`，命中的话返回**已有**的 Jira/飞书 ID 并标 `duplicate=true`，不会重复建单；失败不占坑。

## 5. 权限与审计

| 关注点 | 做法 |
|--------|------|
| 最小权限 | 工具在注册时声明 `readonly`；写工具默认关闭（`MCP_ALLOW_WRITE=1` 才出现）；`MCP_TOOL_ALLOWLIST` 可再收窄 |
| 参数校验 | 调用前按 JSON Schema 校验（有 `jsonschema` 就用，没有退回内置校验器）：必填、类型、枚举、范围、数组元素类型 |
| 审计 | 每次调用落一行 JSONL：`ts / tool / actor / status / duration_ms / args_digest`；**只记参数摘要（SHA-256），不记明文** |
| 越权与注入 | 工具描述固定不接受客户端注入；写工具仅支持「建单」不支持删除；错误信息不回显服务器路径 |
| 死循环 | 最大步数、单工具超时、相同 `(tool, args)` 重复熔断、连续错误上限 |

## 6. 自主工具调用循环

`src/agents/tool_agent.py`。每一步把「工具目录 + 已完成步骤」重新拼成一个 prompt 交给 LLM（无状态重建，便于复现与测试），要求模型回一段严格 JSON：

```json
{"action": "search_meetings", "arguments": {"query": "数据接入频率"}}
{"final_answer": "根据 2026-08-19 会议纪要……"}
```

停止条件是显式的，不靠「模型自己乖」：

| 停止原因 | 触发条件 | 用户拿到什么 |
|---------|---------|-------------|
| `final_answer` | 模型给出最终答案 | 答案 + 完整调用轨迹 |
| `max_steps` | 达到步数上限（默认 5） | 回落总结：把已调工具的结果拼成答复，并注明未收敛 |
| `repeat_detected` | 相同 `(tool, 参数摘要)` 重复调用 | 同上，注明重复调用被熔断 |
| `tool_errors` | 连续失败达到上限（默认 3） | 同上，附最后一条错误 |
| `llm_error` | LLM 调用本身失败 | 错误信息，不返回假答案 |

失败路径的处理原则：**参数非法/工具报错都回传给模型做自修复**（observation 里写清哪里错了），而不是直接抛给用户；但自修复次数受限。

## 7. 接入方式

### Claude Desktop / Cursor（stdio）

```json
{
  "mcpServers": {
    "meetgraph": {
      "command": "D:\\multi-agent-meeting-assistant\\.venv\\Scripts\\python.exe",
      "args": ["-m", "src.mcp.server"],
      "cwd": "D:\\multi-agent-meeting-assistant",
      "env": {
        "PYTHONPATH": "D:\\multi-agent-meeting-assistant",
        "MCP_ALLOW_WRITE": "0"
      }
    }
  }
}
```

其他客户端同理（Cursor 用 `~/.cursor/mcp.json`）。服务端日志走 stderr，stdout 只走协议报文——接错流会把 stdout 污染成非法 JSON，客户端直接断连。

### 本地冒烟

```bash
# 手喂一条 initialize（stdout 只应出现一行 JSON）
printf '{"jsonrpc":"2.0","id":1,"method":"tools/list"}\n' \
  | ./.venv/Scripts/python.exe -m src.mcp.server

# 走 HTTP 简化传输
curl -s http://127.0.0.1:8000/mcp -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
```

### 评测

```bash
python scripts/evaluate_tools.py          # 离线：协议一致性 + 熔断 + oracle 自检
python scripts/evaluate_tools.py --live   # 真实 LLM 工具选择
```

## 8. 实测结果

`python scripts/evaluate_tools.py`（离线）与 `--live`（真实 LLM）的结果：

| 指标 | 数值 |
|------|------|
| MCP 协议一致性 | **11/11**（真实子进程 stdio，握手到 ping 0.22s） |
| 熔断与权限（5 个构造场景） | **5/5** |
| 首个工具准确率（只读任务 9 条，`abab6.5s-chat`） | **0.889** |
| 首个工具准确率（全部 12 条） | 0.727 |
| 工具集合召回 / 精确率 | 0.773 / 0.591 |
| 参数正确率 | 0.727 |
| 正常收敛率（`final_answer`） | 0.667 |
| 触发步数上限后回落总结 | 0.083 |
| 工具序列完全一致（参考值） | 0.182 |

完整逐题明细见 `docs/evaluation.md` 第六节，原始 JSON 在 `reports/tools-eval-*.json`。

## 9. 设计取舍与已知问题

| 取舍 | 原因 | 代价 |
|------|------|------|
| 不引官方 `mcp` SDK，自己实现 7 个方法 | 能讲清版本协商/传输/错误码细节；测试完全不联网；不引入依赖 | 未来接远端 Server 时仍需引 SDK（不冲突，客户端侧才需要） |
| 用严格 JSON 协议而不是原生 `tool_calls` | 当前 LLM 客户端（MiniMax `chatcompletion_v2`）不返回原生 tool_calls | 模型偶尔输出非 JSON，需要解析容错 + 重试；已在代码里预留 native 分支 |
| 工具目录按策略过滤 | 未授权的写工具干脆不出现，模型不会去调一个注定失败的工具 | 模型无法主动告知用户「有这个工具但被关闭了」，只能回「无法执行」 |
| SSE 走 `GET /sse` + `POST /messages` | 就是 MCP 的 SSE 传输约定；实现简单、可 curl 调试 | 无状态扩展（多实例）需要把会话表搬到 Redis，当前是进程内 dict |

**已知问题 / 负结果（如实记录）：**

1. **评测指标口径被实测推翻过**：最初用「工具序列完全相等」和「参数精确相等」评分，真实 LLM 拿到 0.182 / 0.364——但翻轨迹发现多数扣分是「多查了一次确认」和「用自己措辞检索」，不是错误。改成集合级召回/精确率 + 结构化参数精确、自由文本非空后才有意义。
2. **SSE 的 HTTP 集成测试没写成**：`httpx.ASGITransport` 在流未关闭时不支持并发请求，`TestClient` 关闭流又不会取消服务端生成器。改为直接单测事件流生成器（帧格式/心跳/断开清理），HTTP 层只覆盖 `POST /mcp` 与会话投递——**这是测试台架的限制，不是接口没验**。
3. **评测时写工具被隐藏**：`--live` 不带 `MCP_ALLOW_WRITE`，避免评测脚本在真实 Jira/飞书里建单。因此写任务只统计「模型是否尝试调用写工具」，不测真实建单质量（幂等与建单本身由单测覆盖）。
