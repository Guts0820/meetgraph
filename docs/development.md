# 开发指南

## 环境准备

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # 只填 MINIMAX_API_KEY 就能跑通全流程
```

不需要 GPU、不需要 Jira/飞书账号也能开发：

- **没有 WhisperX 模型** → 无音频输入时自动走内置演示转写；
- **没有 Jira/飞书** → 对应同步跳过，报告照常落盘；
- **不想花钱调 LLM** → 单元测试与离线评测全部使用假 LLM。

## 目录约定

```
src/agents/         每个 Agent 一个文件，实现 async process(state) -> dict
src/graph/          LangGraph 组装与主入口 run_meeting_pipeline
src/integrations/   LLM / Jira / 飞书 / 幂等台账
src/models/         pydantic 数据契约（改字段先想清楚影响哪些 Agent）
src/websocket/      FastAPI 应用：REST + WebSocket + /healthz
config/             人员映射等配置
scripts/            评测与运维脚本
tests/              pytest 用例与 fixture（假 LLM、假外部系统、标注集）
docs/               架构 / 接口 / 开发 / 评测
```

## 日常命令

```bash
python -m src.main                     # 启动服务（reload 模式）
python -m pytest                       # 全量测试
python -m pytest tests/test_graph.py -k idempotent   # 只跑某个用例
python -m pytest --cov=src --cov-report=term-missing # 覆盖率
python scripts/evaluate.py             # 离线评测（编排收益 / 降级 / 报告完整率）
python scripts/evaluate.py --live      # 追加真实 LLM 抽取质量
```

## 改代码时的约定

1. **外部世界一律可注入**：Agent 与 Graph 的构造函数都接受 `llm_client` / `jira_client` / `feishu_client` / `ledger`，测试里全部换成假实现。新增依赖同理，不要在模块内直接 `new` 一个客户端。
2. **节点只返回状态增量**：`return {"summary": ...}`，不要原地改整个 `state` 再 return；并行节点写不同键是并集合并的前提。
3. **失败信息必须落到 `errors` 或 `sync_status`**：允许降级，但不允许静默 —— 这是 `/api/v1/meeting/{id}/report` 里 `errors` 字段存在的意义。
4. **外部写入走幂等台账**：任何会在外部系统产生记录的动作（建单、发消息）都应先查 `SyncLedger`，成功后登记；失败不登记。
5. **数据模型加 `extra="forbid"`**：`ActionResult` / `FollowUpResult` 已开启，字段拼错会当场报错，而不是被 pydantic 静默丢弃。
6. **路径来自外部输入时必须清洗**：`meeting_id` 落盘前经白名单过滤（见 `followup_agent._generate_report`）。

## 新增一个 Agent

以「会议风险识别」为例：

```python
# src/agents/risk_agent.py
class RiskAgent:
    def __init__(self, llm_client=None):
        self.llm = llm_client or MiniMaxClient()

    async def process(self, state: dict) -> dict:
        errors: list[str] = []
        try:
            state["risks"] = await self._analyze(state.get("transcript_text", ""))
        except Exception as e:
            errors.append(f"RiskAgent: {e}")
            state["risks"] = None
        updates = {"risks": state["risks"]}
        if errors:
            updates["errors"] = errors
        return updates
```

然后：

1. 在 `schemas.py` 定义 `RiskResult`，并在 `MeetingState` / `GraphState` 里加字段；
2. 在 `build_meeting_graph` 里 `add_node` 并连边（并行就多接一条 Fan-out/Fan-in 边）；
3. 在 `followup_agent` 的报告模板里加一节；
4. 在 `tests/fakes.py` 里给 `FakeLLM` 加该场景的返回，并补测试。

## 调试技巧

- **先跑 demo**：`POST /api/v1/meeting/demo-1/demo` 不需要音频，能覆盖除转写外的全部链路；
- **日志**：loguru 输出到 stderr，`LOG_LEVEL=DEBUG` 能看到台账命中/登记的细节；
- **看落盘报告**：`reports/meeting-report-<meeting_id>.md` 是最直观的端到端产物；
- **台账自检**：
  ```python
  from src.integrations.idempotency import SyncLedger
  print(SyncLedger().stats())   # {'db_path': ..., 'total': 3, 'by_target': {'jira': 3}}
  ```
- **数据库**：台账是 SQLite，直接用 `sqlite3 data/sync-ledger.db "select * from sync_ledger"` 看；
- **假实现**：需要构造特定失败场景时，用 `FakeLLM(delay=..., fail_times=...)`、`FakeJiraClient(fail=True)`，比打真实接口快得多也可控。

## 检索增强（RAG）开发

```bash
python scripts/rag_cli.py reindex                  # 重建索引（内部文档 + 会议纪要 + 术语表）
python scripts/rag_cli.py stats                    # 索引概览
python scripts/rag_cli.py search "版本冻结"          # 只看检索结果与召回通道
python scripts/rag_cli.py ask "DT 多久接入一次"     # 检索 + 生成（带引用）
python scripts/rag_cli.py ask "..." --no-terms      # 关掉术语层做对照
python scripts/evaluate_rag.py                     # 检索评测 + 术语消融矩阵
```

改语料/改术语后必须重建索引，否则检索还是旧内容：

| 想改什么 | 改哪里 | 要不要重建索引 |
|----------|--------|----------------|
| 语料文档 | `data/knowledge/*.md`、`data/meetings/*.md`（或设 `RAG_CORPUS_DIRS`） | 要 |
| 术语表 | `config/glossary.json`（term / canonical / aliases / definition） | 要（术语表自身也是可检索文档） |
| 分块大小 | `rag/chunking.py::chunk_markdown(max_chars, overlap_chars)` | 要 |
| 向量后端 | 环境变量 `RAG_EMBEDDER`（`auto` / `hash` / `openai`） | 要（维度变了） |
| 召回权重 / 熔断 | `HybridRetriever(vector_weight, bm25_weight, expansion_weight, rrf_k)` | 不要 |
| 精排特征权重 | `rag/rerank.py::RerankConfig` | 不要 |

**调参必须走评测**：改完 `RerankConfig` 或召回权重后跑一次 `python scripts/evaluate_rag.py`（几秒钟，离线），对比 `reports/rag-eval-*.json` 里的 Recall@K / MRR 差值再看要不要保留——本项目第一版就是靠这个发现「术语扩展把排序做坏了」的。

约定与坑：

- 术语表是**唯一权威来源**：语料正文只写标准术语，缩写和别名登记在 glossary，否则查询扩展无从下手，也测不出术语层的价值；
- 向量后端必须可降级：新增 embedder 时保证 `resolve_embedder()` 失败能退回 `HashingEmbedder`，否则无网络/无模型的环境直接不可用；
- 单元测试**不许加载真实模型**：用 `rag_index` fixture（离线哈希向量），否则 CI 会依赖网络与几百 MB 缓存；
- 检索失败绝不能抛给主流程：Context 节点的约定是「写空上下文 + 记 error」（见 `test_retrieval_failure_is_recorded_not_raised`）；
- 查询侧与文档侧必须用同一套分词（`rag/tokenize.py`），否则 BM25 打分不可比。

## MCP 与工具调用开发

```bash
python -m src.mcp.server                        # 直接把 MCP Server 跑在 stdio 上（给客户端接）
printf '{"jsonrpc":"2.0","id":1,"method":"tools/list"}\n' \
  | ./.venv/Scripts/python.exe -m src.mcp.server  # 手工冒烟：stdout 只应有一行 JSON
python scripts/evaluate_tools.py                # 协议一致性 + 熔断/权限 + oracle 自检（离线）
python scripts/evaluate_tools.py --live         # 真实 LLM 的工具选择评测
python -m pytest tests/test_mcp_protocol.py tests/test_mcp_tools.py -q   # 只跑 MCP 相关
```

改工具时的规矩：

- **新增工具只改一处**：在 `mcp/tools.py::build_default_registry()` 里 `register(ToolSpec(...))`，MCP 的 `tools/list` 与 LLM 的工具目录自动同步（`tests/test_mcp_tools.py::test_llm_catalog_matches_tools_list` 守着这条）；
- **写工具必须声明 `readonly=False`**，否则会被当成只读工具默认放行；
- **不要在工具里 print**：stdio 传输下 stdout 是协议通道，一个 print 就会让客户端解析失败（日志一律走 `logger`，即 stderr）；
- **错误信息别带服务器路径**：`get_meeting_report` 之类要返回「没找到」而不是绝对路径；
- **改 SSE 事件流后跑 `tests/test_mcp_transport.py`**：帧格式（`event:`/`data:`/空行）和断开清理都有断言。

## 提交前检查

```bash
python -m pytest                    # 必须全绿
python scripts/evaluate.py          # 编排收益 / 降级行为无退化
python scripts/evaluate_rag.py      # 检索指标无退化
python scripts/evaluate_tools.py    # MCP 协议一致性 11/11、熔断 5/5 不退化
```

改动涉及外部集成时，另外用真实凭据手工跑一次 demo 并检查 `errors` 与 `sync_status`。
