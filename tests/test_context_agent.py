"""上下文检索节点（RAG 接入主链路）测试。"""

from __future__ import annotations

import pytest

from src.agents.context_agent import ContextAgent


async def test_no_retriever_returns_empty_context() -> None:
    agent = ContextAgent(retriever=None)

    result = await agent.process({"meeting_id": "m", "transcript_text": "张总: 开会"})

    assert result["context"].history == []
    assert result["context"].terms == []
    assert "errors" not in result


async def test_no_transcript_returns_empty_context(rag_index) -> None:
    agent = ContextAgent(retriever=rag_index.retriever)

    result = await agent.process({"meeting_id": "m", "transcript_text": ""})

    assert result["context"].history == []


async def test_retrieves_history_and_terms(rag_index) -> None:
    agent = ContextAgent(retriever=rag_index.retriever)
    transcript = (
        "张总: 今天我们确认 DT 数据的接入频率调整方案，"
        "上次会上说的是每周两次，这次需要再确认一下。"
    )

    result = await agent.process({"meeting_id": "m", "transcript_text": transcript})

    context = result["context"]
    assert context.history, "应当检索到历史会议内容"
    assert all("citation" in item and item["citation"] for item in context.history)
    assert "DT" in context.terms
    assert any(t["canonical"] == "路测" for t in context.term_definitions)


async def test_only_meeting_source_types_are_used(rag_index) -> None:
    agent = ContextAgent(retriever=rag_index.retriever)
    transcript = "张总: 版本冻结之后的灰发放量节奏怎么安排？"

    result = await agent.process({"meeting_id": "m", "transcript_text": transcript})

    for item in result["context"].history:
        retrieved_ids = {c.chunk_id for c in rag_index.retriever.chunks.values()}
        assert item["citation"]


def test_build_query_uses_head_and_terms(rag_index) -> None:
    agent = ContextAgent(retriever=rag_index.retriever, query_chars=20)

    query = agent.build_query("开场先说明背景，" + "后续内容" * 50 + " 涉及 DT 数据")

    assert query.startswith("开场先说明背景")
    assert len(query) < 200
    assert "DT" in query


async def test_retrieval_failure_is_recorded_not_raised() -> None:
    class BrokenRetriever:
        terminology = type("T", (), {"is_empty": True, "match": staticmethod(lambda _: [])})()

        def search(self, *args, **kwargs):
            raise RuntimeError("index corrupted")

    agent = ContextAgent(retriever=BrokenRetriever())

    result = await agent.process(
        {"meeting_id": "m", "transcript_text": "张总: 正常会议内容"}
    )

    assert result["context"].history == []
    assert any("ContextAgent" in e for e in result["errors"])


async def test_top_k_limits_history(rag_index) -> None:
    agent = ContextAgent(retriever=rag_index.retriever, top_k=1)

    result = await agent.process({"meeting_id": "m", "transcript_text": "路测数据接入频率"})

    assert len(result["context"].history) <= 1


@pytest.mark.parametrize("transcript", ["", "   "])
async def test_blank_transcript(rag_index, transcript: str) -> None:
    agent = ContextAgent(retriever=rag_index.retriever)
    result = await agent.process({"meeting_id": "m", "transcript_text": transcript})
    assert result["context"].history == []
