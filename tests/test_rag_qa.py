"""知识问答（RAG 生成侧）测试：prompt 约束、引用解析、无命中不调用 LLM。"""

from __future__ import annotations

import pytest

from src.rag.qa import NO_ANSWER_TEXT, KnowledgeQA, build_context_block
from tests.fakes import FakeAnswerLLM


def _qa(rag_index, answer_text: str = "记录如下 [1]") -> tuple[KnowledgeQA, FakeAnswerLLM]:
    llm = FakeAnswerLLM(answer_text)
    return KnowledgeQA(retriever=rag_index.retriever, llm=llm), llm


async def test_ask_returns_answer_with_citation(rag_index) -> None:
    qa, llm = _qa(rag_index, "版本冻结后只允许缺陷修复与配置调整 [1]")

    answer = await qa.ask("版本冻结之后还允许做哪些变更？")

    assert answer.answered
    assert answer.citations
    assert answer.citations[0]["index"] == 1
    assert answer.citations[0]["chunk_id"] == answer.retrieved[0].chunk.chunk_id
    assert llm.call_count == 1


async def test_prompt_contains_context_and_terminology_block(rag_index) -> None:
    qa, llm = _qa(rag_index)

    await qa.ask("DT 数据多久接入一次？")

    prompt = llm.calls[0]["messages"][-1]["content"]
    assert "## 参考资料" in prompt
    assert "[1]" in prompt
    assert "## 公司内部术语" in prompt  # 命中 DT，注入术语约束
    assert "路测" in prompt


async def test_terminology_can_be_disabled(rag_index) -> None:
    qa, llm = _qa(rag_index)

    answer = await qa.ask("DT 数据多久接入一次？", use_terminology=False)

    prompt = llm.calls[0]["messages"][-1]["content"]
    assert "## 公司内部术语" not in prompt
    assert answer.used_terms == []


async def test_out_of_range_citation_is_ignored(rag_index) -> None:
    qa, _ = _qa(rag_index, "结论在这里 [1]，还有编造的 [9]")

    answer = await qa.ask("版本冻结的规则是什么")

    assert [c["index"] for c in answer.citations] == [1]


async def test_duplicate_citation_numbers_are_deduplicated(rag_index) -> None:
    qa, _ = _qa(rag_index, "规则 A [1]，规则 B [1]")

    answer = await qa.ask("版本冻结的规则是什么")

    assert len(answer.citations) == 1


async def test_no_hit_short_circuits_without_llm_call(rag_index) -> None:
    """检索不到就直说不知道，不调用 LLM —— 这是最关键的一条防幻觉约束。"""
    qa, llm = _qa(rag_index)

    answer = await qa.ask("如何用 Rust 写一个编译器？", top_k=1)

    if not answer.answered:
        assert answer.text == NO_ANSWER_TEXT
        assert llm.call_count == 0
        assert answer.citations == []


async def test_answer_to_dict_is_serializable(rag_index) -> None:
    qa, _ = _qa(rag_index, "见 [1]")

    payload = (await qa.ask("版本冻结的规则是什么")).to_dict()

    assert {"question", "answer", "citations", "retrieved", "used_terms"} <= set(payload)
    assert isinstance(payload["retrieved"][0]["citation"], str)


def test_build_context_block_numbers_sources(rag_index) -> None:
    results = rag_index.retriever.search("版本冻结", top_k=2)
    block = build_context_block(results)

    assert block.startswith("[1]")
    assert "[2]" in block


async def test_ask_falls_back_to_default_top_k(rag_index) -> None:
    """top_k 传 0/None 时回落到默认值，而不是把检索条数变成 0。"""
    qa, _ = _qa(rag_index)

    answer = await qa.ask("版本冻结的规则", top_k=0)

    assert answer.retrieved
