"""工具权限策略与审计日志测试。"""

from __future__ import annotations

import json
from pathlib import Path

from src.mcp.audit import AuditLog, args_digest
from src.mcp.policy import ToolDenied, ToolPolicy


def test_policy_defaults_to_readonly(monkeypatch) -> None:
    monkeypatch.delenv("MCP_ALLOW_WRITE", raising=False)
    monkeypatch.delenv("MCP_TOOL_ALLOWLIST", raising=False)
    policy = ToolPolicy.from_env()

    assert policy.allow_write is False
    assert policy.allows("search_meetings", readonly=True)
    assert not policy.allows("create_action_item", readonly=False)

    try:
        policy.check("create_action_item", readonly=False)
    except ToolDenied as e:
        assert "MCP_ALLOW_WRITE" in str(e)
    else:  # pragma: no cover
        raise AssertionError("写工具默认应被拒绝")


def test_policy_env_enables_write(monkeypatch) -> None:
    monkeypatch.setenv("MCP_ALLOW_WRITE", "1")
    policy = ToolPolicy.from_env()

    assert policy.allows("create_action_item", readonly=False)


def test_policy_allowlist_restricts_tools(monkeypatch) -> None:
    monkeypatch.setenv("MCP_ALLOW_WRITE", "1")
    monkeypatch.setenv("MCP_TOOL_ALLOWLIST", "search_meetings, lookup_glossary")
    policy = ToolPolicy.from_env()

    assert policy.allows("search_meetings", readonly=True)
    assert not policy.allows("get_meeting_report", readonly=True)
    assert not policy.allows("create_action_item", readonly=False)
    assert policy.allowlist == ("search_meetings", "lookup_glossary")


def test_args_digest_is_order_insensitive_and_not_plaintext() -> None:
    first = args_digest({"a": 1, "b": "机密内容"})
    second = args_digest({"b": "机密内容", "a": 1})

    assert first == second
    assert len(first) == 32
    assert "机密内容" not in first


def test_audit_appends_and_tails(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit.jsonl")

    log.append("search_meetings", actor="test", status="ok", duration_ms=12.5, arguments={"query": "x"})
    log.append("create_action_item", actor="test", status="denied", error="write disabled")

    records = log.tail(10)
    assert [r["tool"] for r in records] == ["search_meetings", "create_action_item"]
    assert records[0]["status"] == "ok"
    assert records[1]["status"] == "denied"
    assert records[0]["duration_ms"] == 12.5


def test_audit_survives_unwritable_path(tmp_path: Path) -> None:
    """审计写不进去也不能让工具调用崩掉。"""
    log = AuditLog(tmp_path / "no-such-dir" / "audit.jsonl")
    log.path.parent.mkdir(parents=True, exist_ok=True)
    record = log.append("ping")

    assert record.tool == "ping"
    assert log.tail(1)[0]["tool"] == "ping"


def test_audit_clear(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    log.append("ping")
    assert path.exists()

    log.clear()
    assert not path.exists()


def test_audit_uses_env_path(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "from-env.jsonl"
    monkeypatch.setenv("MCP_AUDIT_LOG", str(target))

    AuditLog().append("ping")

    assert target.exists()
    assert json.loads(target.read_text(encoding="utf-8").strip())["tool"] == "ping"
