"""混合检索：多路召回 → RRF 融合 → 精排 → 引用。

四路召回（各自只贡献排名，融合用 RRF）：

| 通道 | 查询 | 权重 | 作用 |
|------|------|------|------|
| 向量 + 原始查询 | 用户原话 | 1.0 | 语义近似，**不被扩展词稀释** |
| BM25 + 原始查询 | 用户原话 | 1.0 | 精确命中原话里的词与数字 |
| BM25 + 扩展查询 | 原话 + 术语标准名/别名 | 0.6 | 缩写与标准术语不一致时的召回 |
| 向量 + 扩展查询 | 原话 + 术语标准名/别名 | 0.4 | 语义层的术语桥接 |

早期版本把「原话 + 别名 + 全称」拼成一个查询去检索，实测**排序反而变差**（见
``docs/evaluation.md`` 的记录）：扩展词稀释了原查询的语义中心，且术语表 chunk
因为术语密度最高而抢占首位。因此现在：扩展只走独立通道，术语表 chunk 在精排
阶段被降权（它只是「定义证据」，不是「事实证据」）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from loguru import logger

from .chunking import Chunk
from .rerank import Reranker, RerankConfig
from .terminology import Terminology, Term


@dataclass
class RetrievalResult:
    """一条检索结果。"""

    chunk: Chunk
    score: float
    vector_score: float = 0.0
    bm25_score: float = 0.0
    vector_rank: int = 0
    bm25_rank: int = 0
    term_hits: tuple[str, ...] = ()
    base_score: float = 0.0
    features: dict[str, float] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        return self.chunk.citation

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.chunk.doc_id,
            "doc_title": self.chunk.doc_title,
            "section": self.chunk.section,
            "source_type": self.chunk.source_type,
            "citation": self.citation,
            "text": self.chunk.text,
            "score": self.score,
            "base_score": self.base_score,
            "features": self.features,
            "vector_score": self.vector_score,
            "bm25_score": self.bm25_score,
            "vector_rank": self.vector_rank,
            "bm25_rank": self.bm25_rank,
            "term_hits": list(self.term_hits),
            "metadata": self.chunk.metadata,
        }


@dataclass
class SearchDebug:
    """检索过程的可观测信息（评测与排障用）。"""

    query: str
    expanded_query: str
    query_terms: list[str] = field(default_factory=list)
    candidate_pool: int = 0
    channels: list[str] = field(default_factory=list)
    rerank_enabled: bool = False


class HybridRetriever:
    """向量 + BM25 多路召回，RRF 融合后精排。"""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        embedder,                 # 满足 Embedder 协议即可
        bm25_index,               # BM25Index
        vector_index,             # VectorIndex
        terminology: Terminology | None = None,
        rrf_k: int = 60,
        vector_weight: float = 1.0,
        bm25_weight: float = 1.0,
        expansion_weight: float = 0.6,
        expansion_vector_weight: float = 0.4,
        term_boost: float = 0.15,
        use_terminology: bool = True,
        rerank: bool = True,
        rerank_config: RerankConfig | None = None,
    ) -> None:
        self.chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self.embedder = embedder
        self.bm25 = bm25_index
        self.vectors = vector_index
        self.terminology = terminology or Terminology()
        self.rrf_k = rrf_k
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.expansion_weight = expansion_weight
        self.expansion_vector_weight = expansion_vector_weight
        self.term_boost = term_boost
        self.use_terminology = use_terminology
        self.rerank_enabled = rerank
        self.reranker = Reranker(self.terminology, rerank_config)

    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 5,
        use_terminology: bool | None = None,
        return_debug: bool = False,
    ):
        """检索：返回 ``list[RetrievalResult]``（可选附带 ``SearchDebug``）。"""
        use_terms = self.use_terminology if use_terminology is None else use_terminology
        expanded, query_terms = (
            self.terminology.expand(query) if use_terms else (query, [])
        )

        pool = max(top_k * 4, 20)
        debug = SearchDebug(
            query=query,
            expanded_query=expanded,
            query_terms=[t.term for t in query_terms],
            rerank_enabled=self.rerank_enabled,
        )

        accum: dict[str, dict] = {}

        def add_channel(hits: list[tuple[str, float]], weight: float, kind: str) -> None:
            for rank, (chunk_id, value) in enumerate(hits, 1):
                entry = accum.setdefault(
                    chunk_id,
                    {"rrf": 0.0, "vector_score": 0.0, "bm25_score": 0.0,
                     "vector_rank": 0, "bm25_rank": 0},
                )
                entry["rrf"] += weight / (self.rrf_k + rank)
                if kind == "vector":
                    entry["vector_score"] = max(entry["vector_score"], value)
                    entry["vector_rank"] = entry["vector_rank"] or rank
                else:
                    entry["bm25_score"] = max(entry["bm25_score"], value)
                    entry["bm25_rank"] = entry["bm25_rank"] or rank

        # ---- 通道 1：向量 + 原始查询（主力语义通道） ----
        if len(self.vectors) > 0 and self.vector_weight > 0:
            try:
                query_vector = self.embedder.encode([query], is_query=True)[0]
                add_channel(
                    self.vectors.search(query_vector, top_k=pool),
                    self.vector_weight,
                    "vector",
                )
                debug.channels.append("vector:original")
            except Exception as e:  # 向量通道故障不能拖垮检索
                logger.warning(f"Vector channel failed, degrading to BM25 only: {e}")

        # ---- 通道 2：BM25 + 原始查询 ----
        if self.bm25.doc_ids and self.bm25_weight > 0:
            add_channel(self.bm25.search(query, top_k=pool), self.bm25_weight, "bm25")
            debug.channels.append("bm25:original")

        # ---- 通道 3/4：术语扩展查询（独立通道，不稀释原查询） ----
        if use_terms and expanded != query:
            if self.bm25.doc_ids and self.expansion_weight > 0:
                add_channel(
                    self.bm25.search(expanded, top_k=pool),
                    self.expansion_weight,
                    "bm25",
                )
                debug.channels.append("bm25:expanded")
            if len(self.vectors) > 0 and self.expansion_vector_weight > 0:
                try:
                    expanded_vector = self.embedder.encode([expanded], is_query=True)[0]
                    add_channel(
                        self.vectors.search(expanded_vector, top_k=pool),
                        self.expansion_vector_weight,
                        "vector",
                    )
                    debug.channels.append("vector:expanded")
                except Exception as e:
                    logger.warning(f"Expanded vector channel failed: {e}")

        debug.candidate_pool = len(accum)
        if not accum:
            return ([], debug) if return_debug else []

        results: list[RetrievalResult] = []
        for chunk_id, entry in accum.items():
            chunk = self.chunks.get(chunk_id)
            if chunk is None:
                continue

            base = entry["rrf"]
            hits: tuple[str, ...] = ()
            if not self.terminology.is_empty and chunk.source_type != "glossary":
                matched: list[Term] = self.terminology.match(chunk.text)
                hits = tuple(t.term for t in matched)

            score, features = (
                self.reranker.score(
                    query,
                    chunk,
                    base,
                    use_terminology=use_terms,
                    query_has_term=bool(query_terms),
                )
                if self.rerank_enabled
                else (base, {})
            )

            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=round(score, 6),
                    base_score=round(base, 6),
                    features=features,
                    vector_score=entry["vector_score"],
                    bm25_score=entry["bm25_score"],
                    vector_rank=entry["vector_rank"],
                    bm25_rank=entry["bm25_rank"],
                    term_hits=hits,
                )
            )

        results.sort(key=lambda r: (-r.score, r.chunk.chunk_id))
        top = results[:top_k]
        return (top, debug) if return_debug else top

    # ------------------------------------------------------------------
    @staticmethod
    def citations(results: Sequence[RetrievalResult]) -> list[dict]:
        """把检索结果转成可核验的引用列表（按首次出现顺序去重）。"""
        seen: set[str] = set()
        citations: list[dict] = []
        for result in results:
            if result.chunk.chunk_id in seen:
                continue
            seen.add(result.chunk.chunk_id)
            citations.append(
                {
                    "chunk_id": result.chunk.chunk_id,
                    "doc_id": result.chunk.doc_id,
                    "title": result.chunk.doc_title,
                    "section": result.chunk.section,
                    "source_type": result.chunk.source_type,
                    "citation": result.citation,
                }
            )
        return citations
