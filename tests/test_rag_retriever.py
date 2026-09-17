"""混合检索与精排测试（全部离线，向量用确定性哈希向量）。"""

from __future__ import annotations

from src.rag.chunking import Chunk
from src.rag.rerank import Reranker, RerankConfig, query_phrases
from src.rag.retriever import HybridRetriever


def _search(index, query, **kwargs):
    return index.retriever.search(query, **kwargs)


def test_search_returns_citable_results(rag_index) -> None:
    results = _search(rag_index, "灰度流量比例是多少", top_k=3)

    assert results
    top = results[0]
    assert top.chunk.chunk_id in rag_index.retriever.chunks
    assert top.citation.startswith(top.chunk.doc_title)


def test_search_prefers_exact_document(rag_index) -> None:
    results = _search(rag_index, "灰度流量比例不超过多少", top_k=3)

    assert results[0].chunk.doc_id == "内部规范-版本冻结"


def test_terminology_toggle_changes_query_and_channels(rag_index) -> None:
    _, debug_on = _search(rag_index, "DT 数据多久接入一次", top_k=3, return_debug=True)
    _, debug_off = _search(
        rag_index, "DT 数据多久接入一次", top_k=3, use_terminology=False, return_debug=True
    )

    assert debug_on.query_terms == ["DT"]
    assert debug_on.expanded_query != debug_on.query
    assert "bm25:expanded" in debug_on.channels
    assert debug_off.query_terms == []
    assert debug_off.expanded_query == debug_off.query
    assert "bm25:expanded" not in debug_off.channels


def test_terminology_expansion_bridges_abbreviation(rag_index) -> None:
    """查询里只有缩写、文档里只有标准术语时，扩展是唯一的桥。"""
    with_terms = _search(rag_index, "Drive Test 接入频率", top_k=3)
    doc_ids = [r.chunk.doc_id for r in with_terms]

    assert "会议纪要-数据接入评审" in doc_ids


def test_vector_channel_can_be_disabled(rag_index) -> None:
    retriever = HybridRetriever(
        chunks=rag_index.chunks,
        embedder=rag_index.embedder,
        bm25_index=rag_index.bm25,
        vector_index=rag_index.vectors,
        terminology=rag_index.retriever.terminology,
        vector_weight=0.0,
    )

    results, debug = retriever.search("版本冻结", top_k=3, return_debug=True)

    assert results
    assert "vector:original" not in debug.channels
    assert "bm25:original" in debug.channels


def test_glossary_chunk_not_top_for_factual_question(rag_index) -> None:
    """问事实时，术语表不该靠术语密度抢占首位。"""
    results = _search(rag_index, "灰度流量比例是多少", top_k=3)

    assert results[0].chunk.source_type != "glossary"


def test_glossary_is_retrievable_for_term_question(rag_index) -> None:
    results = _search(rag_index, "DT 是什么术语", top_k=5)

    assert any(r.chunk.source_type == "glossary" for r in results)


def test_citations_are_deduplicated(rag_index) -> None:
    results = _search(rag_index, "版本冻结", top_k=5)
    citations = rag_index.retriever.citations(results)

    assert len(citations) == len({c["chunk_id"] for c in citations})


def test_rrf_fuses_two_channels(rag_index) -> None:
    results = _search(rag_index, "路测 脱敏", top_k=5)

    assert any(r.bm25_rank > 0 for r in results)
    assert all(r.score > 0 for r in results)


# ----------------------------------------------------------------------
# 精排特征单测
# ----------------------------------------------------------------------

def _chunk(
    text: str,
    section: str = "",
    source_type: str = "knowledge",
    chunk_id: str = "",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id or f"c-{abs(hash(text)) % 10**8}",
        doc_id="d",
        doc_title="测试文档",
        section=section,
        text=text,
        source_type=source_type,
    )


def test_query_phrases_are_sliding_windows() -> None:
    phrases = query_phrases("版本冻结规则", min_len=4)

    assert "版本冻结" in phrases
    assert "冻结规则" in phrases


def test_rerank_rewards_query_coverage() -> None:
    reranker = Reranker()
    query = "版本冻结 规则"

    exact = reranker.features(query, _chunk("版本冻结的规则如下"))
    unrelated = reranker.features(query, _chunk("灰度流量比例不超过 5%"))

    assert exact["coverage"] > unrelated["coverage"]
    assert exact["phrase"] == 1.0


def test_rerank_section_match_bonus() -> None:
    reranker = Reranker()
    query = "灰度发布"

    with_section = reranker.features(query, _chunk("内容", section="灰度发布"))
    without = reranker.features(query, _chunk("内容", section="其他"))

    assert with_section["section"] == 1.0
    assert without["section"] == 0.0


def test_glossary_penalty_only_without_query_term() -> None:
    reranker = Reranker()
    glossary_chunk = _chunk("术语：DT 标准说法：路测", source_type="glossary")

    without_term, _ = reranker.score("接入频率是多少", glossary_chunk, 1.0)
    with_term, _ = reranker.score("DT 是什么", glossary_chunk, 1.0)

    assert without_term < with_term


def test_term_bonus_is_disabled_when_terminology_off() -> None:
    from src.rag.terminology import Terminology

    reranker = Reranker(terminology=Terminology.from_file())
    chunk = _chunk("路测数据接入频率调整")

    on, feats_on = reranker.score("路测", chunk, 1.0, use_terminology=True)
    off, feats_off = reranker.score("路测", chunk, 1.0, use_terminology=False)

    assert feats_on["term_hits"] > 0
    assert feats_off["term_hits"] == 0
    assert on >= off


def test_reranker_without_terminology_has_no_term_bonus() -> None:
    """不传术语表时精排仍可工作，只是没有术语加成。"""
    reranker = Reranker()
    _, feats = reranker.score("路测", _chunk("路测数据接入频率调整"), 1.0)

    assert feats["term_hits"] == 0.0


def test_rerank_can_be_disabled(rag_index) -> None:
    retriever = HybridRetriever(
        chunks=rag_index.chunks,
        embedder=rag_index.embedder,
        bm25_index=rag_index.bm25,
        vector_index=rag_index.vectors,
        terminology=rag_index.retriever.terminology,
        rerank=False,
    )

    results = retriever.search("版本冻结", top_k=3)

    assert results
    assert results[0].features == {}
    assert results[0].score == results[0].base_score


def test_custom_rerank_config_is_respected() -> None:
    config = RerankConfig(coverage=0.0, phrase=0.0, section=0.0, term_bonus=0.0)
    reranker = Reranker(config=config)

    score, feats = reranker.score("版本冻结", _chunk("版本冻结规则"), 1.0)

    assert not config.enabled
    assert score == 1.0
    assert feats == {}
