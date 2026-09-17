"""待办 Agent 测试：抽取归一化、幂等同步、失败上报。"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agents.action_agent import ActionAgent
from src.integrations.idempotency import SyncLedger
from tests.fakes import FakeFeishuClient, FakeJiraClient, FakeLLM


def _agent(**kwargs) -> ActionAgent:
    return ActionAgent(
        llm_client=kwargs.pop("llm", FakeLLM()),
        jira_client=kwargs.pop("jira", FakeJiraClient(enabled=False)),
        feishu_client=kwargs.pop("feishu", FakeFeishuClient(enabled=False)),
        **kwargs,
    )


def test_normalize_deadline() -> None:
    assert ActionAgent._normalize_deadline("2026-03-05") == "2026-03-05"
    assert ActionAgent._normalize_deadline("2026-03-05T10:00:00") == "2026-03-05"
    # 模糊时间表达一律丢弃，避免写进 Jira 后报格式错误
    assert ActionAgent._normalize_deadline("下周五") == ""
    assert ActionAgent._normalize_deadline("本周三前") == ""
    assert ActionAgent._normalize_deadline(None) == ""
    assert ActionAgent._normalize_deadline("") == ""


async def test_extract_actions_maps_fields(demo_transcript: str) -> None:
    items = await _agent()._extract_actions(demo_transcript)

    assert [i.assignee for i in items] == ["李明", "王芳", "赵伟"]
    assert [i.priority.value for i in items] == ["high", "medium", "medium"]
    # FakeLLM 给的是「下周五 / 本周三 / 空」，前两个被归一化丢弃
    assert [i.deadline for i in items] == ["", "", ""]


async def test_process_without_transcript_returns_empty() -> None:
    agent = _agent()
    result = await agent.process({"meeting_id": "m", "transcript_text": ""})

    assert result["actions"].action_items == []
    assert "errors" not in result


async def test_sync_creates_once_then_skips(
    tmp_path: Path, demo_transcript: str
) -> None:
    """同一会议、同一批待办重复同步：第二次全部命中台账。"""
    ledger = SyncLedger(tmp_path / "ledger.db")
    jira = FakeJiraClient(enabled=True)
    feishu = FakeFeishuClient(enabled=True)
    agent = _agent(jira=jira, feishu=feishu, ledger=ledger)
    state = {"meeting_id": "m-1", "transcript_text": demo_transcript}

    first = await agent.process(dict(state))
    assert len(jira.created) == 3
    assert len(feishu.created) == 3
    assert first["actions"].duplicates_skipped == 0

    second = await agent.process(dict(state))
    assert len(jira.created) == 3
    assert len(feishu.created) == 3
    assert second["actions"].duplicates_skipped == 6
    assert "skipped=6" in second["actions"].sync_status["jira"]
    ledger.close()


async def test_sync_without_ledger_creates_duplicates(demo_transcript: str) -> None:
    jira = FakeJiraClient(enabled=True)
    agent = _agent(jira=jira, ledger=None)
    state = {"meeting_id": "m-1", "transcript_text": demo_transcript}

    await agent.process(dict(state))
    await agent.process(dict(state))

    assert len(jira.created) == 6


async def test_sync_failures_are_reported(
    tmp_path: Path, demo_transcript: str
) -> None:
    """外部系统不可用时：不抛异常，但失败计数与原因要能查到。"""
    ledger = SyncLedger(tmp_path / "ledger.db")
    agent = _agent(
        jira=FakeJiraClient(enabled=True, fail=True),
        feishu=FakeFeishuClient(enabled=True, fail=True),
        ledger=ledger,
    )

    result = await agent.process(
        {"meeting_id": "m-1", "transcript_text": demo_transcript}
    )

    assert len(result["errors"]) == 6  # 3 条待办 × 2 个目标
    assert all("FakeJira" in e or "FakeFeishu" in e for e in result["errors"])
    assert "failed=6" in result["actions"].sync_status["jira"]
    # 失败没有写台账，因此重试时仍会尝试创建（不会把失败当成已同步）
    assert ledger.stats()["total"] == 0
    ledger.close()


async def test_failed_items_are_retried_next_round(
    tmp_path: Path, demo_transcript: str
) -> None:
    """第一次失败、第二次成功：说明失败没有污染幂等台账。"""
    ledger = SyncLedger(tmp_path / "ledger.db")
    jira = FakeJiraClient(enabled=True, fail=True)
    state = {"meeting_id": "m-1", "transcript_text": demo_transcript}

    await _agent(jira=jira, ledger=ledger).process(dict(state))
    assert jira.created == []
    assert ledger.stats()["total"] == 0

    jira.fail = False
    result = await _agent(jira=jira, ledger=ledger).process(dict(state))
    assert len(jira.created) == 3
    assert result.get("errors", []) == []
    ledger.close()


@pytest.mark.parametrize(
    "priority,expected",
    [("low", "Low"), ("medium", "Medium"), ("high", "High"), ("urgent", "Highest")],
)
def test_priority_mapping(priority: str, expected: str) -> None:
    from src.integrations.jira_client import JiraClient

    assert JiraClient.map_priority(priority) == expected
