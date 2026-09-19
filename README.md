# MeetGraph 智能会议助手

> 基于 **LangGraph** 的多智能体会议纪要系统：一场会议开完，自动得到结构化纪要、可跟踪的待办、会议洞察和会后跟进。

```
会议音频 ──► 转写 ──► 检索历史决议与术语 ──► ┌ 纪要 ┐
                                            ├ 待办 ┤ ──► 跟进（推送 / 建单 / 落盘报告）
                                            └ 洞察 ┘
              Pipeline        RAG 增强        Fan-out（并行）      Fan-in（汇聚）
```

五个各司其职的 Agent 共享一个状态对象，由 LangGraph 状态图编排：**Transcription → Context（RAG）→ (Summary | Action | Insight) → Follow-up**。
转写是串行的前提，检索为三个分析 Agent 提供历史背景与公司术语，三个分析 Agent 互不依赖因此并行，跟进需要三者结果因此汇聚。

---

## 这个项目解决什么

| 痛点 | 本项目的做法 |
|------|--------------|
| 一场 1 小时会议，整理纪要要花 20–30 分钟 | Summary Agent 直接产出「议题 / 讨论要点 / 结论 / 决策 / 下一步」五段结构 |
| 会上分配的任务口头说完就散 | Action Agent 抽取「谁 / 做什么 / 何时」，带幂等地同步到 Jira 与飞书任务 |
| 不知道会议开得怎么样 | Insight Agent 用确定性规则算发言占比与效率评分，用 LLM 判情绪、提关键词 |
| 纪要发完没人跟 | Follow-up Agent 汇总三路结果，推送飞书、生成 Markdown 报告落盘 |

---

## 实测数据

下面每个数字都由 [`scripts/evaluate.py`](scripts/evaluate.py) 现场测量、可复现，不是设计目标：

| 指标 | 实测 | 测量方式 | 复现命令 |
|------|------|----------|----------|
| 并行编排加速比 | **2.95x** | 同一批三个分析 Agent（Fan-out 图 vs 串行 await），模拟单次 LLM 延迟 0.4s，3 次取中位：1.218s → 0.412s | `python scripts/evaluate.py` |
| 并行编排固有开销 | 0.012s | 并行总耗时 − 单次 LLM 延迟（图调度+状态归约的净开销） | 同上 |
| LLM 全挂时的完成率 | 3/3 | 注入必然失败的 LLM，Pipeline 仍跑完并落盘报告 | 同上 |
| 失败可观测率 | 3/3 | 上述失败如实进入 `state["errors"]`，不静默吞掉 | 同上 |
| 报告完整率 | 100% | 报告三个章节非空且不含占位符 | 同上 |
| 待办抽取 P/R/F1 | **1.00 / 1.00 / 1.00** | 真实 LLM（abab6.5s-chat）在 3 条人工标注上 | `python scripts/evaluate.py --live` |
| 待办截止时间填充率 | 100% | 抽取结果中带合法 `YYYY-MM-DD` 的比例 | 同上 |
| **知识库检索 Recall@1 / @5 / MRR** | **0.875 / 1.000 / 0.938** | 24 条标注问答，向量(BGE-small-zh) + BM25 混合 + 确定性精排 | `python scripts/evaluate_rag.py` |
| **引用可核验率** | **1.000** | 返回的引用全部能在索引里定位到原文 | 同上 |
| **术语层增益（无向量模型时）** | Recall@1 **+4.1pt**（向量通道 +8.4pt） | 关掉语义通道后术语扩展的作用；有语义模型时增益被掩盖 | 同上（自动跑的消融矩阵） |
| 答案级术语一致性（真实 LLM） | 0.625 vs 0.500 | 术语定义注入 prompt 前后，答案使用公司标准术语的比例 | `--live` |
| **MCP 协议一致性** | **11/11 项** | 真实子进程 stdio：握手 → 工具发现 → 调用 → 权限拒绝 → 资源/提示 → 错误码 → ping | `python scripts/evaluate_tools.py` |
| **工具调用熔断与权限** | **5/5 项** | 重复调用 / 步数上限 / 连续失败 / 参数非法自修复 / 审计覆盖 | 同上 |
| **工具选择准确率（真实 LLM）** | 首个工具 **0.889**（只读任务，9 条）/ 0.727（全部 12 条） | MiniMax `abab6.5s-chat` 自主决定调哪个工具；集合召回 0.773、参数正确 0.727 | `python scripts/evaluate_tools.py --live` |
| 单元测试 | 195 passed | 不联网、不写外部系统、不加载向量模型 • 含真实子进程跑 MCP stdio 传输 | `python -m pytest` |

> **口径说明**：待办抽取样本仅 3 条、知识库问答标注 24 条，都属于链路联调用的最小标注集，只能说明「链路与评分可用」，不是模型能力的结论；报告与幂等相关的断言有测试覆盖，可信度更高。**术语层在语义通道存在时几乎没有检索增益（实测 ≈0）**——这个负结果与原因分析都记录在 [docs/evaluation.md](docs/evaluation.md)，没有粉饰。**加速比的测量口径先后修正过两次**（一次拿完整流水线比三个 Agent、一次扣错了 LLM 延迟次数），修正记录也留在文档里——数字是否自洽（加速比不能小于 1、开销不能为负）比数字本身更重要。

---

## 三个工程重点

### 1. 并行编排的收益是可测量的

三条分析链路的 LLM 调用互相独立，串行 await 会把延迟叠加，Fan-out 后总耗时收敛到「最长的一条 + 调度开销」：

```python
# src/graph/meeting_graph.py
graph.add_edge(START, "transcription")           # Pipeline：必须先转写
graph.add_edge("transcription", "summary")       # Fan-out：三条出边
graph.add_edge("transcription", "action")
graph.add_edge("transcription", "insight")
graph.add_edge("summary", "followup")            # Fan-in：三条入边汇聚
graph.add_edge("action", "followup")
graph.add_edge("insight", "followup")
```

串行 vs 并行的对照由评测脚本现场跑，而不是写在文档里的「优化了 60%」。

### 2. 幂等同步：重复触发不会重复建单

外部系统写入必须可重放 —— WebSocket 重连、消息重投、手工重跑，都可能让同一个会议被处理多次。做法是本地 SQLite 台账（`src/integrations/idempotency.py`）：

- 业务键 = `sha256(meeting_id | assignee | task)`，内容先做归一化（压缩空白、统一大小写），避免「同一件事差一个空格」算两条；
- 写入前查台账，命中就复用已有的 `Jira key / 飞书 task id`，**不调用外部 API**；
- 只有外部写入成功才登记台账 —— 失败不占坑，下轮会重试，不会把失败伪装成「已同步」。

`tests/test_graph.py::test_pipeline_sync_is_idempotent` 断言：同一会议跑两遍，Jira 与飞书各只创建 3 条，第二遍 6 次全部命中台账。

### 3. 降级要静默但不能无声

- **Agent 级**：每个 `process()` 自带 try/except，失败写降级结果并追加到 `state["errors"]`（`errors` 用 `Annotated[list[str], operator.add]` 归约，并行节点各自追加不会互相覆盖）；
- **同步级**：Jira 建单失败不影响飞书，反之亦然，失败计数与原因都进 `sync_status`；
- **解析级**：LLM 返回的 JSON 带前后废话时，用首尾括号裁剪兜底；截止时间不是 `YYYY-MM-DD` 就丢弃（Jira 对日期格式过敏，宁可留空）；
- **人员映射**：显示名解析不到 Jira 账号时返回 `None` 并告警，建一张无负责人的单，而不是猜一个可能错的人。

### 4. 检索增强：内部文档 + 会议历史 + 公司术语

主流水线里有一个 **Context 节点**（`src/agents/context_agent.py`）：转写完成后，用开场内容 + 命中的公司术语构造查询，从历史会议纪要里捞出相关决议，写进 `state["context"]`；Summary 的 prompt 会带上这段背景与术语定义，报告里多出「相关历史决议」一节。

检索本身是**四路召回 + RRF 融合 + 确定性精排**（`src/rag/`）：

| 通道 | 查询 | 权重 | 作用 |
|------|------|------|------|
| 向量 + 原始查询 | 用户原话 | 1.0 | 语义近似，**不被扩展词稀释** |
| BM25 + 原始查询 | 用户原话 | 1.0 | 精确命中编号、人名、数字门限 |
| BM25 + 扩展查询 | 原话 + 术语标准名/别名 | 0.6 | 缩写与文档用语不一致时的召回 |
| 向量 + 扩展查询 | 原话 + 术语标准名/别名 | 0.4 | 语义层的术语桥接 |

精排特征（`src/rag/rerank.py`）：查询词覆盖度、连续短语命中、小节标题命中、非术语表的术语密度；术语表 chunk 在查询不含术语时降权（它只是「定义证据」，不是「事实证据」）。

**公司术语的四个作用点**（术语表是唯一权威来源：正文只写标准术语，缩写统一登记在 `config/glossary.json`）：

1. 查询扩展 —— 「MRD 什么时候评审」自动补上「市场需求文档」；
2. 检索加权 —— 命中术语的片段加分；
3. 生成约束 —— 把标准定义注入 prompt，要求使用公司术语、不得自造说法；
4. 评测口径 —— 术语类问题单独统计，并做开关 A/B（结论见 docs/evaluation.md）。

```bash
# 建索引（内部文档 data/knowledge + 历史纪要 data/meetings + 术语表 config/glossary.json）
python scripts/rag_cli.py reindex

# 命令行问答（带引用）
python scripts/rag_cli.py ask "DT 数据多久接入一次？"

# 只看检索结果与召回通道
python scripts/rag_cli.py search "版本冻结之后能改什么"

# 评测：Recall@K / MRR / 术语消融 / 引用可核验率
python scripts/evaluate_rag.py            # 离线（含消融矩阵）
python scripts/evaluate_rag.py --live     # 追加真实 LLM 的答案级评测
```

换成本公司的语料不需要改代码：把文档放进 `data/knowledge/`（或设 `RAG_CORPUS_DIRS`），术语写进 `config/glossary.json`，重建索引即可。向量后端可插拔：默认本地 `BAAI/bge-small-zh-v1.5`（约 95MB，无 GPU 也能跑），也支持 OpenAI 兼容的 embeddings 接口；模型不可用时自动降级为 BM25 + 术语扩展（离线哈希向量），服务照常可用。

### 5. 工具协议与自主工具调用（MCP）

RAG 解决「查得到」，MCP 解决「**别的客户端也能用、模型能自己决定怎么用**」。

`src/mcp/` 是自己实现的 MCP Server（JSON-RPC 2.0，不引官方 SDK），把四个能力暴露成标准工具：

| 工具 | 只读 | 复用模块 |
|------|------|----------|
| `search_meetings` | ✅ | `src/rag/retriever.py`（只要会议纪要来源） |
| `get_meeting_report` | ✅ | `reports/meeting-report-<id>.md` |
| `create_action_item` | ❌ | `SyncLedger` 幂等键 + Jira/飞书 |
| `lookup_glossary` | ✅ | `config/glossary.json` |

协议层实现 `initialize`（版本协商 + 能力声明）/ `tools/list` / `tools/call` / `resources/*` / `prompts/*` / `ping`，传输支持 **stdio**（`python -m src.mcp.server`，Claude Desktop / Cursor 直接接）与 **HTTP + SSE**（`GET /mcp/sse` + `POST /mcp/messages`，另有 `POST /mcp` 简化直返）。

**一份工具定义，两处消费**：`ToolSpec`（名称/描述/JSON Schema/只读标记）既是 `tools/list` 的返回，也是喂给 LLM 的工具目录——避免「MCP 暴露了但模型不知道」这类漂移。

安全姿态是最小权限 + 可审计：写工具默认关闭（`MCP_ALLOW_WRITE=1` 才出现，注意是**不出现**而不是调了再拒），支持白名单，每次调用落一行 JSONL 审计（只记参数摘要 SHA-256，不记明文）。

`src/agents/tool_agent.py` 是自主工具调用循环（ReAct 风格）：模型每步回一段严格 JSON（`{"action": …}` 或 `{"final_answer": …}`），循环层负责参数校验、把工具报错回传给模型自修复、以及四种熔断——重复调用、步数上限、连续失败、LLM 自身报错。熔断后回落成「已查到什么」的总结，不返回假答案。

接入配置、协议细节与设计取舍见 [docs/mcp.md](docs/mcp.md)。

---

## 架构

```
┌──────────────────────────────────────────────────────┐
│ 接入层   REST(FastAPI) / WebSocket / MCP(stdio+SSE)    │
├──────────────────────────────────────────────────────┤
│ 编排层   LangGraph StateGraph：Pipeline + Fan-out/Fan-in │
│          + 自主工具调用循环（ReAct，非确定性编排）        │
├──────────────────────────────────────────────────────┤
│ Agent层  Transcription / Context(RAG) / Summary /      │
│          Action / Insight / Follow-up                 │
├──────────────────────────────────────────────────────┤
│ 集成层   MiniMax LLM / WhisperX+pyannote / Jira / 飞书  │
│          / MCP 工具注册表（权限 + 审计）                 │
├──────────────────────────────────────────────────────┤
│ 数据层   内存会议结果 / SQLite 同步台账 / Markdown 报告   │
│          / RAG 索引 / MCP 审计日志(JSONL)               │
└──────────────────────────────────────────────────────┘
```

**共享状态是节点之间唯一的通信通道**：每个 Agent 只读自己要的字段、只写自己负责的字段，并行节点写不同字段因此无冲突。

```python
MeetingState = {
  meeting_id, status, audio_data,
  transcript, transcript_text,      # TranscriptionAgent 写
  summary, actions, insights,       # 三个并行 Agent 各写一个
  followup,                         # Follow-upAgent 写
  errors,                           # 所有节点都可追加
}
```

目录结构：

```
meetgraph/
├── src/
│   ├── agents/          # 6 个节点（transcription / context(RAG) / summary / action / insight / followup）
│   │                    # + tool_agent.py 自主工具调用循环
│   ├── graph/           # LangGraph 编排与主入口 run_meeting_pipeline
│   ├── integrations/    # LLM、Jira、飞书、幂等台账
│   ├── mcp/             # MCP Server：协议内核 / 工具注册表 / 权限 / 审计 / 客户端
│   ├── models/          # pydantic 数据契约
│   ├── rag/             # 分块、BM25、向量、术语、精排、检索、问答
│   ├── websocket/       # FastAPI 应用（REST + WebSocket + MCP HTTP/SSE + 健康检查）
│   └── main.py          # 服务入口
├── config/              # jira_users.json（人员映射）/ glossary.json（术语表）
├── data/                # knowledge/（内部文档）、meetings/（历史纪要）、index/（索引）
├── docs/                # 架构 / 接口 / 开发 / 评测 / MCP / 实施计划
├── scripts/             # evaluate.py、evaluate_rag.py、evaluate_tools.py、rag_cli.py
├── tests/               # pytest（195 个用例，含假 LLM 与假外部系统）
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 快速开始

```bash
# 1. 安装依赖
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. 配置（只填 LLM Key 就能跑通全流程；Jira/飞书留空则自动降级）
cp .env.example .env

# 3. 启动
python -m src.main                 # http://localhost:8000 ，接口文档 /docs
```

不想装 WhisperX（体积大）也能跑：无音频输入时走内置演示转写，完整的 5-Agent 链路照常执行。

```bash
# 跑一次演示（不传音频，走演示转写 + LLM 生成纪要/待办/洞察）
curl -X POST http://localhost:8000/api/v1/meeting/demo-1/demo

# 查看落盘的报告（默认 <仓库>/reports/meeting-report-demo-1.md）
curl http://localhost:8000/api/v1/meeting/demo-1/report

# 健康检查：报告各外部集成是否就绪
curl http://localhost:8000/healthz

# 上传真实音频（需要 WhisperX）
curl -X POST -F "file=@meeting.wav" \
     http://localhost:8000/api/v1/meeting/meeting-1/upload
```

Docker：

```bash
cp .env.example .env      # 镜像会读取根目录 .env
docker compose up -d
```

---

## 接口一览

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/healthz` | 健康检查：版本、各集成配置状态、知识库索引状态、活跃会议数 |
| POST | `/api/v1/ask` | 知识库问答（内部文档 + 会议历史 + 术语表），返回带引用的答案 |
| POST | `/api/v1/knowledge/reindex` | 重建知识库索引 |
| POST | `/api/v1/meeting/start` | 创建会议，返回 WebSocket 地址 |
| POST | `/api/v1/meeting/{id}/demo` | 用演示转写跑完整链路 |
| POST | `/api/v1/meeting/{id}/upload` | 上传音频文件并处理 |
| GET | `/api/v1/meeting/{id}/{transcript\|summary\|actions\|insights\|report}` | 查询结果 |
| WS | `/ws/meeting/{id}` | 实时音频流：发送二进制帧，`{"type":"stop"}` 触发处理 |
| POST | `/mcp` | MCP 协议直返（JSON-RPC，一次请求一次响应） |
| GET | `/mcp/sse` | MCP SSE 传输：下发投递端点，响应经事件流回推 |
| POST | `/mcp/messages?session_id=` | MCP SSE 传输的请求入口 |
| GET | `/mcp/info` | 已暴露的工具、只读/写标记与当前策略（排障用） |

完整字段说明见 [docs/api-reference.md](docs/api-reference.md)，MCP 协议细节见 [docs/mcp.md](docs/mcp.md)。

---

## 配置

| 变量 | 作用 | 缺失时的行为 |
|------|------|--------------|
| `MINIMAX_API_KEY` / `MINIMAX_GROUP_ID` / `MINIMAX_MODEL` | 纪要/待办/洞察的 LLM | 三个 Agent 全部降级：纪要退化为规则摘要，洞察只剩发言统计，错误进 `errors` |
| `WHISPER_MODEL_SIZE` / `WHISPER_DEVICE` / `WHISPER_LANGUAGE` / `HF_TOKEN` | 本地转写与说话人分离 | 走内置演示转写（8 段带说话人标注的示例会议） |
| `JIRA_SERVER` / `JIRA_EMAIL` / `JIRA_API_TOKEN` / `JIRA_PROJECT_KEY` | 待办同步到 Jira | 跳过 Jira，只同步飞书 |
| `FEISHU_APP_ID` / `FEISHU_APP_SECRET` / `FEISHU_WEBHOOK_URL` | 待办同步到飞书任务 + 纪要推群 | 跳过飞书，只落盘报告 |
| `JIRA_USER_MAP` / `JIRA_USER_MAP_FILE` | 显示名 → Jira 账号（内联 JSON 优先，其次文件，最后 `config/jira_users.json`） | 建单不指派负责人并告警 |
| `REPORTS_DIR` / `SYNC_LEDGER_DB` | 报告目录 / 幂等台账路径 | 分别落在仓库的 `reports/`、`data/` |
| `RAG_EMBEDDER` | 向量后端：`auto`（默认，先试本地 BGE）/ `hash`（离线）/ `openai` | 模型不可用时自动降级为离线哈希向量（主要靠 BM25） |
| `RAG_INDEX_DIR` / `RAG_CORPUS_DIRS` / `RAG_GLOSSARY` | 索引目录 / 语料目录（逗号分隔）/ 术语表路径 | 分别用 `data/index`、`data/knowledge`+`data/meetings`、`config/glossary.json` |
| `RAG_HF_MIRROR` | 设为 `0` 可关闭 HF 镜像自动切换（默认走 `hf-mirror.com`） | — |
| `MCP_ALLOW_WRITE` | 设为 `1` 才把写工具（`create_action_item`）暴露给 MCP 客户端与 LLM | 写工具不出现在工具列表里（不是调了再拒） |
| `MCP_TOOL_ALLOWLIST` | 逗号分隔的工具白名单，进一步收窄可调用范围 | 不额外限制（仍受只读/写规则约束） |
| `MCP_AUDIT_LOG` | 工具调用审计日志路径（JSONL，只记参数摘要） | 落在仓库 `data/mcp-audit.jsonl` |
| `MCP_AGENT_MAX_STEPS` / `MCP_AGENT_TOOL_TIMEOUT` | 自主工具调用循环的最大步数 / 单次工具超时（秒） | 默认 5 步 / 20 秒 |

人员映射文件支持别名，减少「张总 / 张总（主持人）」被拆成两个人的概率：

```json
{
  "张总": { "account": "zhang.zong", "aliases": ["Zhang", "张总（主持人）"] },
  "李明": "li.ming"
}
```

---

## 测试与评测

```bash
python -m pytest                      # 195 个用例：不联网、不写真实 Jira/飞书、不加载向量模型
python -m pytest --cov=src            # 覆盖率
python scripts/evaluate.py            # 会议流水线评测（编排收益 / 降级行为）
python scripts/evaluate.py --live     # 追加真实 LLM 的抽取质量评测
python scripts/evaluate_rag.py        # RAG 检索评测（Recall@K / MRR / 术语消融 / 引用可核验）
python scripts/evaluate_rag.py --live # 追加真实 LLM 的答案级评测
python scripts/evaluate_tools.py      # MCP 协议一致性 + 熔断/权限 + oracle 自检
python scripts/evaluate_tools.py --live  # 真实 LLM 的工具选择评测
```

测试策略：外部世界全部替换成假实现（`tests/fakes.py` 的 `FakeLLM / FakeJiraClient / FakeFeishuClient`），但 Agent、Graph、报告落盘这些被测逻辑一律走真实代码路径；`FakeLLM` 可注入延迟与失败，因此「并行收益」和「降级行为」都是可断言的。MCP 那条路径更进一步：**真的把 Server 作为子进程拉起来**跑一遍 stdio 握手与工具调用，因为「客户端能接上」这件事没法靠单测内部函数证明。

指标定义、计算口径与已知偏差见 [docs/evaluation.md](docs/evaluation.md)。

---

## 已知限制与路线图

诚实清单 —— 这些是当前版本确实没做的事：

- **会议结果只在内存里**：`meeting_results` 是进程内字典，重启即失；报告只落 Markdown 文件。要长期检索需要接数据库（`docker-compose.yml` 里刻意没有预置 Postgres/Redis，避免留下「配了但没用」的组件）。
- **提醒是计数占位**：`Follow-up` 只统计了带截止时间的待办条数，没有真正注册定时任务。
- **说话人识别依赖外部 Token**：`pyannote` 需要 HuggingFace Token，未配置时只做转写不做说话人分离。
- **长会议未分块**：单次 LLM 调用覆盖约 1 小时会议，更长的会议需要按议题分块再合并（Map-Reduce），目前保留单次路径。
- **转写链路缺少端到端测试**：CI 里不加载 WhisperX 模型（体积与耗时），语音路径目前靠 demo 转写覆盖，真实音频只在本地验证过。
- **评测样本小**：待办抽取 3 条、知识库问答 24 条，只够验证链路与评分口径，还谈不上准确率结论。
- **知识库语料是示例数据**：`data/knowledge`、`data/meetings` 是构造的示例（与项目业务场景一致），换成真实内部文档无需改代码，但当前数字只代表这套示例语料上的表现。
- **术语层的增益有前提**：语义通道存在时术语扩展几乎不带来检索增益（实测 ≈0），它的价值在无向量模型的降级路径与生成侧的术语一致性；详见 docs/evaluation.md 的负结果记录。
- **精排是确定性特征，不是 Cross-Encoder**：没有引入 reranker 模型（体积与推理成本），进一步优化空间在 `src/rag/rerank.py` 的权重与特征上。
- **MCP 是自己实现的，不是官方 SDK**：好处是能讲清版本协商/传输/错误码且测试不联网，代价是未来要接远端 Server 时客户端侧仍需引 SDK。
- **工具调用用 JSON 协议而非原生 `tool_calls`**：当前 LLM 客户端（MiniMax `chatcompletion_v2`）不返回原生 tool_calls，用严格 JSON 复刻同等效果；模型偶尔输出非 JSON 时靠解析容错兜底。
- **MCP 会话表在进程内**：`_mcp_sessions` 是 dict，多实例部署时 SSE 会话不跨实例，需要换 Redis；stdio 传输不受影响。
- **工具选择评测样本只有 12 条**：只读任务 9 条上的首个工具准确率 0.889 属于小样本观察，不能当模型能力结论；写任务在评测时被策略隐藏（避免真的建单），因此只测「是否尝试调用写工具」。
- **SSE 的 HTTP 集成测试缺位**：`httpx.ASGITransport` 在流未关闭时不支持并发请求，改成直接单测事件流生成器（帧格式/心跳/断开清理），HTTP 层覆盖 `POST /mcp` 与会话投递——补这块需要真实网络客户端。

---

## 许可

MIT License，见 [LICENSE](LICENSE)。

本项目的编排骨架源自一个 MIT 许可的开源多智能体会议助手演示工程（原工程含 Python/Java/Go 三语言版本与面试导向文档）。当前版本已完全移除 Java/Go 实现与相关文档，并对转写以外的全部链路做了重构：补齐数据契约、加入幂等同步台账与人员映射、内置演示转写、建立测试与评测体系，所有文档与指标均按当前代码实际情况重写。
