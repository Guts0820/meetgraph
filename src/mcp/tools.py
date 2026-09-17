"""业务工具：把 MeetGraph 的既有能力包装成 MCP 工具。

| 工具 | 只读 | 复用模块 |
|------|------|----------|
| ``search_meetings`` | ✅ | `src/rag/retriever.py`（只取会议纪要来源） |
| ``get_meeting_report`` | ✅ | `reports/meeting-report-*.md` |
| ``lookup_glossary`` | ✅ | `config/glossary.json` |
| ``create_action_item`` | ❌ | `SyncLedger` + Jira/飞书客户端（幂等） |

写工具的设计原则：**先查幂等台账再写外部系统**，重复调用返回已有 ID 而不是重复
建单——这样 Agent 重试、客户端重放都不会污染 Jira/飞书。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from loguru import logger

from .registry import ToolInputError, ToolRegistry, ToolSpec

SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]")
REPO_ROOT = Path(__file__).resolve().parents[2]


def reports_dir() -> Path:
    return Path(os.getenv("REPORTS_DIR") or REPO_ROOT / "reports")


def _safe_meeting_id(meeting_id: str) -> str:
    """meeting_id 来自外部输入，落盘/拼路径前必须清洗（防路径穿越）。"""
    return SAFE_ID.sub("_", str(meeting_id or "").strip())[:64]


# ----------------------------------------------------------------------
# 工具实现
# ----------------------------------------------------------------------

async def search_meetings(arguments: dict[str, Any]) -> dict[str, Any]:
    """在历史会议纪要里做混合检索。"""
    from ..rag.service import get_retriever

    query = arguments["query"]
    top_k = int(arguments.get("top_k", 5))
    retriever = get_retriever()

    results = retriever.search(query, top_k=max(top_k * 3, 6))
    meetings = [r for r in results if r.chunk.source_type == "meeting"][:top_k]

    return {
        "query": query,
        "count": len(meetings),
        "results": [
            {
                "citation": r.citation,
                "meeting_id": r.chunk.doc_id,
                "section": r.chunk.section,
                "score": r.score,
                "text": r.chunk.text,
            }
            for r in meetings
        ],
    }


async def get_meeting_report(arguments: dict[str, Any]) -> dict[str, Any]:
    """读取某场会议的 Markdown 报告。"""
    meeting_id = _safe_meeting_id(arguments["meeting_id"])
    if not meeting_id:
        raise ToolInputError("meeting_id is required")

    path = reports_dir() / f"meeting-report-{meeting_id}.md"
    if not path.exists():
        available = sorted(p.stem.replace("meeting-report-", "") for p in reports_dir().glob("meeting-report-*.md"))
        raise ToolInputError(
            f"report not found for meeting_id={meeting_id!r}"
            + (f"; available: {available[:10]}" if available else "")
        )

    text = path.read_text(encoding="utf-8")
    limit = int(arguments.get("max_chars", 6000))
    return {
        "meeting_id": meeting_id,
        "path": str(path),
        "characters": len(text),
        "truncated": len(text) > limit,
        "markdown": text[:limit],
    }


async def lookup_glossary(arguments: dict[str, Any]) -> dict[str, Any]:
    """按缩写/别名/标准名查公司术语定义。"""
    from ..rag.terminology import Terminology

    term = str(arguments["term"]).strip()
    terminology = Terminology.from_file()
    if terminology.is_empty:
        raise ToolInputError("glossary is not configured")

    needle = term.lower()
    exact: list[Any] = []
    fuzzy: list[Any] = []
    for item in terminology.terms:
        forms = [f.lower() for f in item.surface_forms]
        if needle in forms:
            exact.append(item)
        elif any(needle in form or form in needle for form in forms):
            fuzzy.append(item)

    matches = exact or fuzzy
    if not matches:
        raise ToolInputError(
            f"no glossary entry for {term!r}; "
            f"known terms: {sorted(t.term for t in terminology.terms)[:12]}"
        )

    return {
        "query": term,
        "exact": bool(exact),
        "count": len(matches),
        "matches": [
            {
                "term": t.term,
                "canonical": t.canonical,
                "aliases": list(t.aliases),
                "definition": t.definition,
                "owner": t.owner,
            }
            for t in matches[:5]
        ],
    }


async def create_action_item(arguments: dict[str, Any]) -> dict[str, Any]:
    """把一条待办同步到 Jira / 飞书（幂等：重复调用不重复建单）。"""
    from datetime import datetime

    from ..integrations.feishu_client import FeishuClient
    from ..integrations.idempotency import SyncLedger
    from ..integrations.jira_client import JiraClient, map_priority

    meeting_id = _safe_meeting_id(arguments.get("meeting_id") or "ad-hoc")
    assignee = str(arguments["task_assignee"]).strip()
    task = str(arguments["task"]).strip()
    if not task:
        raise ToolInputError("task must not be empty")

    deadline = str(arguments.get("deadline") or "").strip()
    if deadline:
        try:
            deadline = datetime.strptime(deadline[:10], "%Y-%m-%d").strftime("%Y-%m-%d")
        except ValueError as e:
            raise ToolInputError(f"deadline must be YYYY-MM-DD, got {deadline!r}") from e

    priority = str(arguments.get("priority") or "medium").lower()
    if priority not in ("low", "medium", "high", "urgent"):
        raise ToolInputError(f"invalid priority: {priority}")

    jira = JiraClient()
    feishu = FeishuClient()
    ledger = SyncLedger()
    key = SyncLedger.item_key(meeting_id, assignee, task)

    outcome: dict[str, Any] = {
        "meeting_id": meeting_id,
        "assignee": assignee,
        "task": task,
        "deadline": deadline,
        "priority": priority,
        "duplicate": False,
        "jira_issue_key": None,
        "feishu_task_id": None,
        "targets": {},
    }

    try:
        # ---- Jira ----
        if jira.is_enabled:
            existing = ledger.get(key, "jira")
            if existing:
                outcome["jira_issue_key"] = existing
                outcome["duplicate"] = True
            else:
                created = jira.create_issue(
                    summary=f"[会议待办] {task}",
                    description=f"来源：{meeting_id}（经 MCP 工具创建）\n负责人：{assignee}",
                    assignee=jira.resolve_user(assignee),
                    due_date=deadline or None,
                    priority=map_priority(priority),
                    labels=["meeting-auto", "mcp"],
                )
                outcome["jira_issue_key"] = created["key"]
                ledger.record(key, "jira", created["key"], meeting_id)
            outcome["targets"]["jira"] = "created" if not existing else "duplicate"
        else:
            outcome["targets"]["jira"] = "disabled"

        # ---- 飞书 ----
        if feishu.is_enabled:
            existing = ledger.get(key, "feishu")
            if existing:
                outcome["feishu_task_id"] = existing
                outcome["duplicate"] = True
            else:
                due_ts = None
                if deadline:
                    due_ts = int(datetime.strptime(deadline, "%Y-%m-%d").timestamp())
                created = await feishu.create_task(
                    summary=f"[会议待办] {task}",
                    description=f"负责人：{assignee}\n来源：{meeting_id}（经 MCP 工具创建）",
                    due_timestamp=due_ts,
                )
                outcome["feishu_task_id"] = created.get("task_id")
                if outcome["feishu_task_id"]:
                    ledger.record(key, "feishu", outcome["feishu_task_id"], meeting_id)
            outcome["targets"]["feishu"] = "created" if not existing else "duplicate"
        else:
            outcome["targets"]["feishu"] = "disabled"

        if outcome["targets"] == {"jira": "disabled", "feishu": "disabled"}:
            logger.warning("[MCP] create_action_item called but no target is configured")
    finally:
        ledger.close()

    return outcome


# ----------------------------------------------------------------------
# 工具目录
# ----------------------------------------------------------------------

TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        name="search_meetings",
        description="在公司历史会议纪要中做混合检索（向量 + 关键词 + 术语扩展），返回带出处的片段。用于回答「上次谁定的这个 deadline」「某个议题讨论过几次」这类跨会议问题。",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "自然语言问题或关键词"},
                "top_k": {
                    "type": "integer",
                    "description": "返回条数，默认 5",
                    "minimum": 1,
                    "maximum": 20,
                },
            },
            "required": ["query"],
        },
        handler=search_meetings,
        readonly=True,
    ),
    ToolSpec(
        name="get_meeting_report",
        description="按 meeting_id 读取某场会议的 Markdown 报告（含会议纪要、待办、洞察、相关历史决议）。",
        input_schema={
            "type": "object",
            "properties": {
                "meeting_id": {"type": "string", "description": "会议 ID"},
                "max_chars": {
                    "type": "integer",
                    "description": "最多返回多少字符，默认 6000",
                    "minimum": 200,
                    "maximum": 20000,
                },
            },
            "required": ["meeting_id"],
        },
        handler=get_meeting_report,
        readonly=True,
    ),
    ToolSpec(
        name="lookup_glossary",
        description="查询公司内部术语表：支持缩写、别名与标准名（例如 DT / 路测 / Drive Test），返回标准说法与定义。",
        input_schema={
            "type": "object",
            "properties": {
                "term": {"type": "string", "description": "术语、缩写或别名"},
            },
            "required": ["term"],
        },
        handler=lookup_glossary,
        readonly=True,
    ),
    ToolSpec(
        name="create_action_item",
        description="把一条待办同步到 Jira / 飞书任务（幂等：同一会议同一人同一件事重复调用不会重复建单）。属于写操作。",
        input_schema={
            "type": "object",
            "properties": {
                "meeting_id": {"type": "string", "description": "来源会议 ID"},
                "task_assignee": {"type": "string", "description": "负责人显示名"},
                "task": {"type": "string", "description": "待办内容"},
                "deadline": {"type": "string", "description": "截止日期 YYYY-MM-DD，可空"},
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "urgent"],
                    "description": "优先级，默认 medium",
                },
            },
            "required": ["task_assignee", "task"],
        },
        handler=create_action_item,
        readonly=False,
    ),
]


def build_default_registry(policy=None, audit=None) -> ToolRegistry:
    """构造带全部业务工具的注册表。"""
    registry = ToolRegistry(policy=policy, audit=audit)
    for spec in TOOL_SPECS:
        registry.register(spec)
    return registry
