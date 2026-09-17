"""MCP 工具测试：四个工具的正常路径、错误路径、权限与审计。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.mcp.audit import AuditLog
from src.mcp.protocol import INVALID_PARAMS, JsonRpcError
from src.mcp.registry import ToolRegistry, ToolSpec, validate_arguments
from src.mcp.tools import build_default_registry
from tests.fakes import FakeFeishuClient, FakeJiraClient


@pytest.fixture
def registry_with_rag(offline_env, rag_index, monkeypatch) -> ToolRegistry:
    """把 RAG 检索器替换成离线索引（不加载真实向量模型）。"""
    import src.rag.service as service

    monkeypatch.setattr(service, "get_retriever", lambda *a, **kw: rag_index.retriever)
    return build_default_registry()


def _enable_writes(monkeypatch: pytest.MonkeyPatch, jira=None, feishu=None) -> tuple[Any, Any]:
    monkeypatch.setenv("MCP_ALLOW_WRITE", "1")
    jira = jira or FakeJiraClient(enabled=True)
    feishu = feishu or FakeFeishuClient(enabled=True)
    monkeypatch.setattr(
        "src.integrations.jira_client.JiraClient", lambda *a, **kw: jira
    )
    monkeypatch.setattr(
        "src.integrations.feishu_client.FeishuClient", lambda *a, **kw: feishu
    )
    return jira, feishu


# ----------------------------------------------------------------------
# 单一事实来源
# ----------------------------------------------------------------------

def test_tools_list_and_llm_catalog_match(offline_env, monkeypatch) -> None:
    """MCP 暴露的工具与喂给 LLM 的工具目录必须是同一份定义。"""
    monkeypatch.setenv("MCP_ALLOW_WRITE", "1")
    registry = build_default_registry()

    mcp_names = {tool["name"] for tool in registry.tools_list_payload()["tools"]}
    llm_names = {tool["name"] for tool in registry.llm_catalog()}

    assert mcp_names == llm_names == {
        "search_meetings",
        "get_meeting_report",
        "lookup_glossary",
        "create_action_item",
    }


def test_registry_rejects_duplicate_names() -> None:
    registry = ToolRegistry()

    async def handler(args):  # pragma: no cover
        return {}

    spec = ToolSpec(name="x", description="d", input_schema={"type": "object"}, handler=handler)
    registry.register(spec)
    with pytest.raises(ValueError):
        registry.register(spec)


def test_validate_arguments_reports_path() -> None:
    schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}, "top_k": {"type": "integer", "minimum": 1}},
        "required": ["query"],
    }

    with pytest.raises(JsonRpcError) as missing:
        validate_arguments(schema, {})
    assert missing.value.code == INVALID_PARAMS

    with pytest.raises(JsonRpcError) as wrong_type:
        validate_arguments(schema, {"query": 42})
    assert wrong_type.value.code == INVALID_PARAMS


# ----------------------------------------------------------------------
# search_meetings
# ----------------------------------------------------------------------

async def test_search_meetings_hits_meeting_source(registry_with_rag) -> None:
    result = await registry_with_rag.call(
        "search_meetings", {"query": "路测 接入频率 调整", "top_k": 3}
    )

    assert result.ok
    assert result.data["count"] >= 1
    assert all(item["meeting_id"] for item in result.data["results"])
    assert all(item["citation"] for item in result.data["results"])


async def test_search_meetings_excludes_knowledge_docs(registry_with_rag) -> None:
    """会议检索工具只返回会议纪要来源，不把内部文档混进来。"""
    result = await registry_with_rag.call("search_meetings", {"query": "版本冻结 灰度 规范"})

    assert result.ok
    for item in result.data["results"]:
        assert "会议纪要" in item["citation"]


async def test_search_meetings_validates_arguments(registry_with_rag) -> None:
    with pytest.raises(JsonRpcError):
        await registry_with_rag.call("search_meetings", {"top_k": 3})  # 缺 query

    with pytest.raises(JsonRpcError):
        await registry_with_rag.call("search_meetings", {"query": "x", "top_k": 999})


# ----------------------------------------------------------------------
# get_meeting_report
# ----------------------------------------------------------------------

async def test_get_meeting_report_reads_file(offline_env: Path) -> None:
    reports = offline_env / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "meeting-report-m-1.md").write_text("# 会议报告\n\n内容", encoding="utf-8")

    result = await build_default_registry().call(
        "get_meeting_report", {"meeting_id": "m-1"}
    )

    assert result.ok
    assert result.data["meeting_id"] == "m-1"
    assert result.data["truncated"] is False
    assert "内容" in result.data["markdown"]


async def test_get_meeting_report_missing_lists_available(offline_env: Path) -> None:
    reports = offline_env / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "meeting-report-known.md").write_text("x", encoding="utf-8")

    result = await build_default_registry().call(
        "get_meeting_report", {"meeting_id": "nope"}
    )

    assert result.ok is False
    assert "not found" in result.error
    assert "known" in result.error
    assert result.audit_status == "error"


async def test_get_meeting_report_sanitizes_meeting_id(offline_env: Path) -> None:
    """meeting_id 来自外部输入，不能用来穿越目录。"""
    result = await build_default_registry().call(
        "get_meeting_report", {"meeting_id": "../../../etc/passwd"}
    )

    assert result.ok is False
    assert "/etc/" not in result.error


async def test_get_meeting_report_truncates(offline_env: Path) -> None:
    reports = offline_env / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "meeting-report-big.md").write_text("字" * 500, encoding="utf-8")

    result = await build_default_registry().call(
        "get_meeting_report", {"meeting_id": "big", "max_chars": 200}
    )

    assert result.data["truncated"] is True
    assert len(result.data["markdown"]) == 200


# ----------------------------------------------------------------------
# lookup_glossary
# ----------------------------------------------------------------------

async def test_lookup_glossary_by_abbreviation_and_alias() -> None:
    registry = build_default_registry()

    by_term = await registry.call("lookup_glossary", {"term": "DT"})
    by_alias = await registry.call("lookup_glossary", {"term": "Drive Test"})
    by_canonical = await registry.call("lookup_glossary", {"term": "路测"})

    for result in (by_term, by_alias, by_canonical):
        assert result.ok
        assert result.data["matches"][0]["canonical"] == "路测"
        assert result.data["matches"][0]["definition"]


async def test_lookup_glossary_unknown_term() -> None:
    result = await build_default_registry().call("lookup_glossary", {"term": "量子纠缠"})

    assert result.ok is False
    assert "no glossary entry" in result.error


# ----------------------------------------------------------------------
# create_action_item（写工具）
# ----------------------------------------------------------------------

async def test_write_tool_denied_by_default(offline_env: Path) -> None:
    result = await build_default_registry().call(
        "create_action_item", {"task_assignee": "李明", "task": "写方案"}
    )

    assert result.ok is False
    assert result.denied is True
    assert "MCP_ALLOW_WRITE" in result.error

    records = AuditLog().tail(5)
    assert records[-1]["status"] == "denied"
    assert records[-1]["tool"] == "create_action_item"


async def test_write_tool_requires_allowlist_entry(offline_env, monkeypatch) -> None:
    monkeypatch.setenv("MCP_ALLOW_WRITE", "1")
    monkeypatch.setenv("MCP_TOOL_ALLOWLIST", "search_meetings")

    result = await build_default_registry().call(
        "create_action_item", {"task_assignee": "李明", "task": "写方案"}
    )

    assert result.denied is True
    assert "allowlist" in result.error


async def test_write_tool_creates_then_dedupes(offline_env, monkeypatch) -> None:
    jira, feishu = _enable_writes(monkeypatch)
    registry = build_default_registry()
    args = {
        "meeting_id": "m-1",
        "task_assignee": "李明",
        "task": "整理Q3预算方案",
        "deadline": "2026-09-30",
        "priority": "high",
    }

    first = await registry.call("create_action_item", args, actor="unit-test")
    assert first.ok
    assert first.data["duplicate"] is False
    assert first.data["jira_issue_key"] == "MEET-101"
    assert first.data["feishu_task_id"] == "t1000"
    assert first.data["targets"] == {"jira": "created", "feishu": "created"}

    second = await registry.call("create_action_item", args, actor="unit-test")
    assert second.ok
    assert second.data["duplicate"] is True
    # 幂等：外部系统里没有多出记录
    assert len(jira.created) == 1
    assert len(feishu.created) == 1

    records = AuditLog().tail(5)
    assert [r["status"] for r in records[-2:]] == ["ok", "ok"]
    assert all(r["actor"] == "unit-test" for r in records)


async def test_write_tool_reports_disabled_targets(offline_env, monkeypatch) -> None:
    _enable_writes(
        monkeypatch, jira=FakeJiraClient(enabled=False), feishu=FakeFeishuClient(enabled=False)
    )

    result = await build_default_registry().call(
        "create_action_item", {"task_assignee": "李明", "task": "写方案"}
    )

    assert result.ok
    assert result.data["targets"] == {"jira": "disabled", "feishu": "disabled"}
    assert result.data["duplicate"] is False


async def test_write_tool_rejects_bad_deadline_and_priority(offline_env, monkeypatch) -> None:
    _enable_writes(monkeypatch)
    registry = build_default_registry()

    bad_deadline = await registry.call(
        "create_action_item",
        {"task_assignee": "李明", "task": "x", "deadline": "下周五"},
    )
    assert bad_deadline.ok is False
    assert "YYYY-MM-DD" in bad_deadline.error

    with pytest.raises(JsonRpcError):
        await registry.call(
            "create_action_item",
            {"task_assignee": "李明", "task": "x", "priority": "sometime"},
        )


async def test_write_tool_empty_task_rejected(offline_env, monkeypatch) -> None:
    _enable_writes(monkeypatch)

    result = await build_default_registry().call(
        "create_action_item", {"task_assignee": "李明", "task": "   "}
    )

    assert result.ok is False
    assert "must not be empty" in result.error


# ----------------------------------------------------------------------
# 资源与提示
# ----------------------------------------------------------------------

def test_resources_list_and_read(offline_env: Path) -> None:
    from src.mcp.catalog import list_resources, read_resource

    reports = offline_env / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "meeting-report-m-9.md").write_text("# 报告 m-9", encoding="utf-8")

    resources = list_resources()["resources"]
    assert resources[0]["uri"] == "meeting://report/m-9"

    content = read_resource("meeting://report/m-9")["contents"][0]
    assert content["mimeType"] == "text/markdown"
    assert "报告 m-9" in content["text"]

    with pytest.raises(JsonRpcError):
        read_resource("file:///etc/passwd")

    with pytest.raises(JsonRpcError):
        read_resource("meeting://report/missing")
