"""LangGraph 编排层测试：图结构、端到端执行、幂等、降级。"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.graph.meeting_graph import (
    build_meeting_graph,
    compile_meeting_graph,
    run_meeting_pipeline,
)
from src.integrations.idempotency import SyncLedger
from src.models.schemas import MeetingStatus
from tests.fakes import FakeFeishuClient, FakeJiraClient, FakeLLM

EXPECTED_NODES = {"transcription", "summary", "action", "insight", "followup"}


def _build(**kwargs):
    return build_meeting_graph(
        llm_client=kwargs.pop("llm_client", FakeLLM()),
        jira_client=kwargs.pop("jira_client", FakeJiraClient(enabled=False)),
        feishu_client=kwargs.pop("feishu_client", FakeFeishuClient(enabled=False)),
        ledger=kwargs.pop("ledger", SyncLedger(":memory:")),
        **kwargs,
    )


def test_graph_registers_five_agents() -> None:
    graph = _build()
    assert EXPECTED_NODES.issubset(set(graph.nodes))


def test_graph_fan_out_and_fan_in_edges() -> None:
    """转写节点有三条出边（Fan-out），三个分析节点都指向跟进节点（Fan-in）。"""
    graph = _build()
    edges = {(source, target) for source, target in graph.edges}

    assert ("transcription", "summary") in edges
    assert ("transcription", "action") in edges
    assert ("transcription", "insight") in edges
    assert ("summary", "followup") in edges
    assert ("action", "followup") in edges
    assert ("insight", "followup") in edges


def test_graph_compiles() -> None:
    compiled = compile_meeting_graph(
        llm_client=FakeLLM(),
        jira_client=FakeJiraClient(enabled=False),
        feishu_client=FakeFeishuClient(enabled=False),
        ledger=SyncLedger(":memory:"),
    )
    assert compiled is not None


async def test_pipeline_end_to_end_offline(offline_env: Path) -> None:
    """无音频、无外部服务时，整条 Pipeline 仍然跑完并落盘报告。"""
    result = await run_meeting_pipeline(
        "unit-e2e",
        audio_data=b"",
        llm_client=FakeLLM(),
        jira_client=FakeJiraClient(enabled=False),
        feishu_client=FakeFeishuClient(enabled=False),
    )

    assert result["status"] == MeetingStatus.COMPLETED
    assert result["errors"] == []

    # 转写走内置演示数据
    assert len(result["transcript"].segments) == 8

    # 纪要 / 待办 / 洞察 三个并行 Agent 都产出了结果
    assert result["summary"].topics
    assert len(result["actions"].action_items) == 3
    assert result["insights"].speaker_stats

    report = Path(result["followup"].report_url)
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    for section in ("## 会议纪要", "## 待办事项", "## 会议洞察"):
        assert section in text


async def test_pipeline_sync_is_idempotent(offline_env: Path) -> None:
    """同一个会议跑两遍，外部系统只会各建一次单。"""
    ledger = SyncLedger(str(offline_env / "ledger.db"))
    jira = FakeJiraClient(enabled=True)
    feishu = FakeFeishuClient(enabled=True)

    common = dict(llm_client=FakeLLM(), jira_client=jira, feishu_client=feishu)

    first = await run_meeting_pipeline("idem-1", ledger=ledger, **common)
    assert len(jira.created) == 3
    assert len(feishu.created) == 3
    assert first["actions"].duplicates_skipped == 0

    second = await run_meeting_pipeline("idem-1", ledger=ledger, **common)

    # 第二次全部命中台账，没有新增任何外部记录
    assert len(jira.created) == 3
    assert len(feishu.created) == 3
    assert second["actions"].duplicates_skipped == 6
    assert second["actions"].action_items[0].jira_issue_key == jira.created[0]
    ledger.close()


async def test_pipeline_degrades_when_llm_is_down(offline_env: Path) -> None:
    """LLM 全挂时：Pipeline 不失败，错误被记录，报告仍落盘。"""
    result = await run_meeting_pipeline(
        "degraded-1",
        llm_client=FakeLLM(fail_times=None),
        jira_client=FakeJiraClient(enabled=False),
        feishu_client=FakeFeishuClient(enabled=False),
    )

    assert result["status"] == MeetingStatus.COMPLETED
    assert len(result["errors"]) == 3  # 三个并行 Agent 各记一条
    assert any("SummaryAgent" in e for e in result["errors"])

    report = Path(result["followup"].report_url).read_text(encoding="utf-8")
    assert "降级" in report


async def test_pipeline_without_ledger_has_no_dedup(offline_env: Path) -> None:
    """不传台账时退化为「每次都同步」，用于对比幂等开启前后的行为。"""
    jira = FakeJiraClient(enabled=True)
    feishu = FakeFeishuClient(enabled=True)
    common = dict(llm_client=FakeLLM(), jira_client=jira, feishu_client=feishu)

    for _ in range(2):
        await run_meeting_pipeline(
            "no-ledger", ledger=None, **common
        )

    assert len(jira.created) == 6


@pytest.mark.parametrize("meeting_id", ["a/../b", "x:y", "正常id"])
async def test_report_path_is_sanitized(
    offline_env: Path, meeting_id: str
) -> None:
    """来自 URL 的 meeting_id 不能影响最终文件路径。"""
    result = await run_meeting_pipeline(
        meeting_id,
        llm_client=FakeLLM(),
        jira_client=FakeJiraClient(enabled=False),
        feishu_client=FakeFeishuClient(enabled=False),
    )
    report = Path(result["followup"].report_url)
    assert report.parent == offline_env / "reports"
