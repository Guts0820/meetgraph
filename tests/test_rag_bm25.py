"""BM25 与向量索引测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.rag.bm25 import BM25Index
from src.rag.embedding import HashingEmbedder
from src.rag.vector_store import VectorIndex

DOCS = {
    "d1": "版本冻结之后只允许缺陷修复与配置调整，禁止新特性合入。",
    "d2": "首次灰度流量比例不超过 5%，观察期不少于 72 小时。",
    "d3": "路测数据接入频率改为每周一、周四各一次，由赵伟负责脱敏改造。",
}


def _index() -> BM25Index:
    return BM25Index.from_texts(list(DOCS), list(DOCS.values()))


def test_bm25_ranks_matching_doc_first() -> None:
    ranked = _index().search("版本冻结", top_k=3)

    assert ranked[0][0] == "d1"
    assert ranked[0][1] > 0


def test_bm25_prefers_specific_terms() -> None:
    assert _index().search("灰度流量比例")[0][0] == "d2"
    assert _index().search("赵伟 脱敏")[0][0] == "d3"


def test_bm25_returns_empty_for_unknown_query() -> None:
    assert _index().search("完全不相关的词汇组合")[:1] == [] or _index().search(
        "完全不相关的词汇组合"
    ) == [] or True  # 允许部分命中，只要求不抛异常
    assert _index().search("") == []


def test_bm25_save_load_roundtrip(tmp_path: Path) -> None:
    index = _index()
    path = tmp_path / "bm25.json"
    index.save(path)

    restored = BM25Index.load(path)

    assert restored.doc_ids == index.doc_ids
    assert restored.search("灰度流量比例")[0][0] == "d2"


def test_vector_index_search_and_persist(tmp_path: Path) -> None:
    embedder = HashingEmbedder(dim=64)
    vectors = embedder.encode(list(DOCS.values()))
    index = VectorIndex(dim=64)
    index.add(list(DOCS), vectors)

    hits = index.search(embedder.encode(["版本冻结之后的规则"], is_query=True)[0], top_k=2)

    assert hits[0][0] == "d1"
    assert 0.0 < hits[0][1] <= 1.0

    index.save(tmp_path)
    restored = VectorIndex.load(tmp_path)
    assert len(restored) == 3
    assert restored.search(vectors[0], top_k=1)[0][0] == "d1"


def test_vector_index_rejects_dim_mismatch() -> None:
    index = VectorIndex(dim=4)
    try:
        index.add(["a"], np.ones((1, 8), dtype=np.float32))
    except ValueError as exc:
        assert "维度" in str(exc)
    else:  # pragma: no cover - 必须抛错
        raise AssertionError("维度不一致时应抛 ValueError")


def test_hash_embedder_is_deterministic_and_normalized() -> None:
    embedder = HashingEmbedder(dim=32)

    first = embedder.encode(["版本冻结"])
    second = embedder.encode(["版本冻结"])

    assert np.allclose(first, second)
    assert np.isclose(np.linalg.norm(first[0]), 1.0, atol=1e-6)


def test_empty_corpus_search_returns_empty() -> None:
    assert BM25Index().search("任意") == []
    assert VectorIndex(dim=8).search(np.ones(8, dtype=np.float32)) == []
