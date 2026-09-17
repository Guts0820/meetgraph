# API 参考

服务默认监听 `http://localhost:8000`，交互式文档在 `/docs`。

## 健康检查

```
GET /healthz
```

```json
{
  "status": "ok",
  "version": "2.0.0",
  "active_meetings": 0,
  "stored_meetings": 3,
  "integrations": {
    "llm": true,
    "jira": false,
    "feishu": true,
    "whisper": "large-v2"
  },
  "ledger": "D:\\meetgraph\\data\\sync-ledger.db",
  "rag": {
    "index": "ready",
    "chunks": 54,
    "docs": 7,
    "embedder": "BAAI/bge-small-zh-v1.5",
    "built_at": "2026-09-17T12:12:19"
  }
}
```

只做配置探测（RAG 部分只读 `data/index/meta.json`，不加载模型、不发起外部请求）——探活不应依赖第三方可用性。

## 知识库问答（RAG）

### 提问

```
POST /api/v1/ask
Content-Type: application/json

{"question": "DT 数据多久接入一次？", "top_k": 5}
```

```json
{
  "question": "DT 数据多久接入一次？",
  "answer": "路测数据每周一、周四各接入一次 [1]。",
  "answered": true,
  "citations": [
    {
      "index": 1,
      "chunk_id": "数据接入规范-话单与路测#1-0-9f2c1a",
      "title": "数据接入规范-话单与路测",
      "section": "1. 数据源与频率",
      "source_type": "knowledge",
      "citation": "数据接入规范-话单与路测 / 1. 数据源与频率"
    }
  ],
  "used_terms": ["DT", "路测"],
  "retrieved": [{"citation": "...", "score": 0.0412, "features": {"coverage": 0.8, "phrase": 1.0, "section": 1.0}}],
  "debug": {"expanded_query": "DT 数据多久接入一次？ 路测 Drive Test 路测数据", "channels": ["vector:original", "bm25:original", "bm25:expanded"]}
}
```

- 检索不到相关内容时返回 `answered: false`、`answer: "资料中未提及相关内容。"`，**不会调用 LLM**；
- `citations[].chunk_id` 可直接在 `data/index/chunks.jsonl` 里核验，越界或编造的编号会被过滤掉；
- `used_terms` 是本次注入 prompt 的公司术语，`debug.channels` 说明走了哪几路召回（排障用）。

### 重建索引

```
POST /api/v1/knowledge/reindex
```

```json
{"status": "ok", "chunks": 54, "docs": 7, "by_source": {"knowledge": 17, "meeting": 17, "glossary": 20}, "embedder": "BAAI/bge-small-zh-v1.5", "dim": 512, "built_at": "2026-09-17T12:20:01"}
```

语料来自 `data/knowledge`、`data/meetings` 与 `config/glossary.json`，构建是 CPU 密集的同步流程，服务端放在线程里执行以免阻塞事件循环。

## REST API

### 创建会议

```
POST /api/v1/meeting/start
```

```json
{
  "meeting_id": "abc123def456",
  "websocket_url": "ws://localhost:8000/ws/meeting/abc123def456",
  "status": "created"
}
```

### 运行演示（无需音频）

```
POST /api/v1/meeting/{meeting_id}/demo
```

不传音频，走内置演示转写，其余链路照常执行。返回完整结果：

```json
{
  "meeting_id": "demo-1",
  "status": "completed",
  "transcript": {"meeting_id": "demo-1", "segments": [{"speaker": "张总", "text": "...", "start": 0.0, "end": 8.5}]},
  "summary": {"title": "Q3 预算评审会议", "topics": [...], "decisions": [...], "next_steps": [...]},
  "actions": {
    "action_items": [{"assignee": "李明", "task": "整理Q3详细预算方案", "deadline": "2026-09-23", "priority": "medium", "jira_issue_key": null, "feishu_task_id": "t1004"}],
    "sync_status": {"jira": "disabled,created=0,skipped=0,failed=0", "feishu": "enabled,created=3,skipped=0,failed=0"},
    "duplicates_skipped": 0
  },
  "insights": {"overall_sentiment": "positive", "efficiency_score": 8.1, "speaker_stats": [...], "keywords": [...]},
  "followup": {"summary_sent": true, "feishu_tasks_created": ["t1004"], "duplicates_skipped": 0, "report_url": "<REPORTS_DIR>/meeting-report-demo-1.md"},
  "errors": []
}
```

### 上传音频

```
POST /api/v1/meeting/{meeting_id}/upload
Content-Type: multipart/form-data
Body: file=@meeting.wav
```

需要 WhisperX 环境；同步等待整条 Pipeline 跑完，返回 `meeting_id` / `status` / `errors`。

### 查询结果

| 路径 | 返回 |
|------|------|
| `GET /api/v1/meeting/{id}/transcript` | 转写结果（分段 + 说话人 + 时间戳） |
| `GET /api/v1/meeting/{id}/summary` | 结构化纪要 |
| `GET /api/v1/meeting/{id}/actions` | 待办清单与同步状态 |
| `GET /api/v1/meeting/{id}/insights` | 洞察结果 |
| `GET /api/v1/meeting/{id}/report` | 上述全部 + `errors` |

会议结果保存在进程内存，重启后失效；Markdown 报告落在 `REPORTS_DIR` 不丢。查不到的会议统一返回 `{"error": "Meeting not found"}`。

## WebSocket API

```
ws://localhost:8000/ws/meeting/{meeting_id}
```

### 客户端 → 服务端

| 消息 | 说明 |
|------|------|
| 二进制帧 | 音频数据，累积到缓冲区 |
| `{"type": "stop"}` | 停止录制，用缓冲区音频触发完整 Pipeline |
| `{"type": "demo"}` | 忽略音频，用演示转写触发 Pipeline |
| `{"type": "ping"}` | 心跳 |

### 服务端 → 客户端

```json
{"type": "connected", "meeting_id": "..."}
{"type": "recording", "buffer_size": 1024}
{"type": "processing", "message": "正在处理音频，请稍候..."}
{"type": "transcript", "data": {...}}
{"type": "summary", "data": {...}}
{"type": "actions", "data": {...}}
{"type": "insights", "data": {...}}
{"type": "followup", "data": {...}}
{"type": "completed", "meeting_id": "...", "status": "completed", "errors": []}
```

`completed` 里的 `errors` 是降级/失败的可观测出口：非空不代表请求失败，而是部分能力降级了。

## 错误处理约定

- HTTP 层不做业务校验，缺配置的能力在 Agent 内静默降级，原因统一出现在 `errors` 与 `sync_status`；
- `meeting_id` 来自 URL，落盘前会做白名单清洗（只保留 `[A-Za-z0-9_.-]`），避免路径穿越；
- 外部系统写入失败不会让接口返回 5xx，只会体现在 `errors` 与 `sync_status.failed` 计数里。
