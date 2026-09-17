# MeetGraph 智能会议助手

> 基于 **LangGraph** 的多智能体会议纪要系统：一场会议开完，自动得到结构化纪要、可跟踪的待办、会议洞察和会后跟进。

```
会议音频 ──► 转写 ──► ┌ 纪要 ┐
                      ├ 待办 ┤ ──► 跟进（推送纪要 / 同步 Jira+飞书 / 落盘报告）
                      └ 洞察 ┘
        Pipeline        Fan-out（并行）      Fan-in（汇聚）
```

五个各司其职的 Agent 共享一个状态对象，由 LangGraph 状态图编排：**Transcription → (Summary | Action | Insight) → Follow-up**。
转写是串行的前提，三个分析 Agent 互不依赖因此并行，跟进需要三者结果因此汇聚。

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
| 并行编排加速比 | **2.9x** | 模拟单次 LLM 延迟 0.4s×3，Fan-out 并行取 3 次中位 0.42s，串行基线 1.22s | `python scripts/evaluate.py` |
| 并行编排固有开销 | 0.021s | 并行总耗时 − 单次 LLM 延迟（LangGraph 调度+状态合并的净开销） | 同上 |
| LLM 全挂时的完成率 | 3/3 | 注入必然失败的 LLM，Pipeline 仍跑完并落盘报告 | 同上 |
| 失败可观测率 | 3/3 | 上述失败如实进入 `state["errors"]`，不静默吞掉 | 同上 |
| 报告完整率 | 100% | 报告三个章节非空且不含占位符 | 同上 |
| 待办抽取 P/R/F1 | **1.00 / 1.00 / 1.00** | 真实 LLM（abab6.5s-chat）在 3 条人工标注上 | `python scripts/evaluate.py --live` |
| 待办截止时间填充率 | 100% | 抽取结果中带合法 `YYYY-MM-DD` 的比例 | 同上 |
| 单元测试 | 58 passed | 不联网、不写外部系统 | `python -m pytest` |

> **口径说明**：抽取质量样本只有 3 条，是链路联调用的最小标注集，只能说明「抽取与评分链路是通的」，不能当成模型准确率结论；报告与幂等相关的指标有测试覆盖，可信度更高。所有指标的定义、计算方式和限制见 [docs/evaluation.md](docs/evaluation.md)。

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

---

## 架构

```
┌──────────────────────────────────────────────────────┐
│ 接入层   REST(FastAPI) / WebSocket(实时音频流)         │
├──────────────────────────────────────────────────────┤
│ 编排层   LangGraph StateGraph：Pipeline + Fan-out/Fan-in │
├──────────────────────────────────────────────────────┤
│ Agent层  Transcription / Summary / Action / Insight /  │
│          Follow-up                                    │
├──────────────────────────────────────────────────────┤
│ 集成层   MiniMax LLM / WhisperX+pyannote / Jira / 飞书  │
├──────────────────────────────────────────────────────┤
│ 数据层   内存会议结果 / SQLite 同步台账 / Markdown 报告   │
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
│   ├── agents/          # 5 个 Agent（transcription / summary / action / insight / followup）
│   ├── graph/           # LangGraph 编排与主入口 run_meeting_pipeline
│   ├── integrations/    # LLM、Jira、飞书、幂等台账
│   ├── models/          # pydantic 数据契约
│   ├── websocket/       # FastAPI 应用（REST + WebSocket + 健康检查）
│   └── main.py          # 服务入口
├── config/jira_users.json   # 显示名 → Jira 账号映射
├── docs/                # 架构 / 接口 / 开发 / 评测文档
├── scripts/evaluate.py  # 评测脚本
├── tests/               # pytest（58 个用例，含假 LLM 与假外部系统）
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
| GET | `/healthz` | 健康检查：版本、各集成配置状态、活跃会议数 |
| POST | `/api/v1/meeting/start` | 创建会议，返回 WebSocket 地址 |
| POST | `/api/v1/meeting/{id}/demo` | 用演示转写跑完整链路 |
| POST | `/api/v1/meeting/{id}/upload` | 上传音频文件并处理 |
| GET | `/api/v1/meeting/{id}/{transcript\|summary\|actions\|insights\|report}` | 查询结果 |
| WS | `/ws/meeting/{id}` | 实时音频流：发送二进制帧，`{"type":"stop"}` 触发处理 |

完整字段说明见 [docs/api-reference.md](docs/api-reference.md)。

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
python -m pytest                      # 58 个用例：不联网、不写真实 Jira/飞书
python -m pytest --cov=src            # 覆盖率
python scripts/evaluate.py            # 离线评测（假 LLM，不产生 API 费用）
python scripts/evaluate.py --live     # 追加真实 LLM 的抽取质量评测
```

测试策略：外部世界全部替换成假实现（`tests/fakes.py` 的 `FakeLLM / FakeJiraClient / FakeFeishuClient`），但 Agent、Graph、报告落盘这些被测逻辑一律走真实代码路径；`FakeLLM` 可注入延迟与失败，因此「并行收益」和「降级行为」都是可断言的。

指标定义、计算口径与已知偏差见 [docs/evaluation.md](docs/evaluation.md)。

---

## 已知限制与路线图

诚实清单 —— 这些是当前版本确实没做的事：

- **会议结果只在内存里**：`meeting_results` 是进程内字典，重启即失；报告只落 Markdown 文件。要长期检索需要接数据库（`docker-compose.yml` 里刻意没有预置 Postgres/Redis，避免留下「配了但没用」的组件）。
- **提醒是计数占位**：`Follow-up` 只统计了带截止时间的待办条数，没有真正注册定时任务。
- **说话人识别依赖外部 Token**：`pyannote` 需要 HuggingFace Token，未配置时只做转写不做说话人分离。
- **长会议未分块**：单次 LLM 调用覆盖约 1 小时会议，更长的会议需要按议题分块再合并（Map-Reduce），目前保留单次路径。
- **转写链路缺少端到端测试**：CI 里不加载 WhisperX 模型（体积与耗时），语音路径目前靠 demo 转写覆盖，真实音频只在本地验证过。
- **抽取评测样本小**：3 条标注只够验证链路，扩到几十条才谈得上准确率。

---

## 许可

MIT License，见 [LICENSE](LICENSE)。

本项目的编排骨架源自一个 MIT 许可的开源多智能体会议助手演示工程（原工程含 Python/Java/Go 三语言版本与面试导向文档）。当前版本已完全移除 Java/Go 实现与相关文档，并对转写以外的全部链路做了重构：补齐数据契约、加入幂等同步台账与人员映射、内置演示转写、建立测试与评测体系，所有文档与指标均按当前代码实际情况重写。
