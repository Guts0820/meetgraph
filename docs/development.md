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

## 提交前检查

```bash
python -m pytest                    # 必须全绿
python scripts/evaluate.py          # 编排收益 / 降级行为无退化
```

改动涉及外部集成时，另外用真实凭据手工跑一次 demo 并检查 `errors` 与 `sync_status`。
