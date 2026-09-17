"""HTTP 接口测试（FastAPI TestClient）。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.models.schemas import (
    ActionResult,
    FollowUpResult,
    MeetingInsight,
    MeetingSummary,
    TranscriptResult,
)
from src.websocket import server as server_module


@pytest.fixture
def client() -> TestClient:
    with TestClient(server_module.app) as c:
        yield c


def test_root_reports_name_and_version(client: TestClient) -> None:
    body = client.get("/").json()

    assert body["name"].startswith("MeetGraph")
    assert body["version"] == "2.0.0"
    assert body["health"] == "/healthz"


def test_healthz_reports_integration_readiness(client: TestClient) -> None:
    body = client.get("/healthz").json()

    assert body["status"] == "ok"
    assert set(body["integrations"]) == {"llm", "jira", "feishu", "whisper"}
    assert isinstance(body["active_meetings"], int)
    assert "rag" in body


def test_start_meeting_returns_websocket_url(client: TestClient) -> None:
    body = client.post("/api/v1/meeting/start").json()

    assert len(body["meeting_id"]) == 12
    assert body["meeting_id"] in body["websocket_url"]


def test_unknown_meeting_returns_error(client: TestClient) -> None:
    assert "error" in client.get("/api/v1/meeting/nope/summary").json()


def test_ask_endpoint_returns_answer_with_citations(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """知识库问答接口：把 Answer 序列化成 JSON（这里替换掉真实 QA，避免调模型）。"""
    from src.rag.qa import Answer
    from src.rag import service

    class FakeQA:
        async def ask(self, question: str, top_k: int = 5) -> Answer:
            return Answer(
                question=question,
                text="版本冻结后只允许缺陷修复与配置调整 [1]",
                citations=[
                    {
                        "index": 1,
                        "chunk_id": "c1",
                        "title": "研发流程-版本冻结与灰度发布",
                        "section": "1. 版本冻结",
                        "source_type": "knowledge",
                        "citation": "研发流程-版本冻结与灰度发布 / 1. 版本冻结",
                    }
                ],
                retrieved=[],
                used_terms=["版本冻结"],
            )

    monkeypatch.setattr(service, "get_qa", lambda *a, **kw: FakeQA())

    body = client.post("/api/v1/ask", json={"question": "版本冻结之后能改什么？"}).json()

    assert body["answered"] is True
    assert body["citations"][0]["title"] == "研发流程-版本冻结与灰度发布"
    assert body["used_terms"] == ["版本冻结"]


def test_ask_endpoint_validates_payload(client: TestClient) -> None:
    assert client.post("/api/v1/ask", json={"question": ""}).status_code == 422
    assert (
        client.post("/api/v1/ask", json={"question": "x", "top_k": 99}).status_code
        == 422
    )


def test_reindex_endpoint_returns_stats(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.rag import service

    monkeypatch.setattr(
        service,
        "reindex",
        lambda: {"chunks": 12, "docs": 3, "embedder": "hash-512"},
    )

    body = client.post("/api/v1/knowledge/reindex").json()

    assert body["status"] == "ok"
    assert body["chunks"] == 12


def test_demo_endpoint_serializes_pipeline_result(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """demo 接口把 Pipeline 结果序列化成 JSON（这里用假结果替换真实 Pipeline）。"""
    async def fake_pipeline(meeting_id: str, audio_data: bytes = b"", **kwargs):
        return {
            "meeting_id": meeting_id,
            "status": "completed",
            "transcript": TranscriptResult(meeting_id=meeting_id),
            "summary": MeetingSummary(
                title="Q3 预算评审会议", participants=["张总"], topics=[]
            ),
            "actions": ActionResult(meeting_id=meeting_id),
            "insights": MeetingInsight(meeting_id=meeting_id),
            "followup": FollowUpResult(meeting_id=meeting_id),
            "errors": [],
        }

    monkeypatch.setattr(server_module, "run_meeting_pipeline", fake_pipeline)

    body = client.post("/api/v1/meeting/demo-1/demo").json()

    assert body["status"] == "completed"
    assert body["summary"]["title"] == "Q3 预算评审会议"
    assert body["errors"] == []

    # 结果被缓存，随后可以通过 REST 查询
    assert client.get("/api/v1/meeting/demo-1/summary").json()["title"] == (
        "Q3 预算评审会议"
    )
    assert "transcript" in client.get("/api/v1/meeting/demo-1/report").json()
