"""精排（rerank）：在 RRF 粗排结果上做与查询的直接相关性打分。

为什么需要精排：RRF 只融合「两路的排名」，不判断 chunk 是否真的在回答这个问题。
典型失败是「同主题文档互相抢位」——问「用户标识怎么处理」，讲数据源的那一节
和讲脱敏的那一节都命中关键词，粗排分不出高下。

这里先实现**确定性特征精排**（不依赖模型、可离线复现、可单测）：

- ``coverage``：查询 token 落在 chunk 里的比例（查询词覆盖度）；
- ``phrase``：查询中的连续片段（≥4 字）是否原样出现在 chunk 里（奖励精确引文）；
- ``section``：查询词是否命中小节标题（标题命中通常意味着整节都在讲这件事）；
- ``term_bonus``：非术语表 chunk 的术语命中数（术语密度是内部文档相关性的强信号，
  但术语表本身天然密度最高，因此单独处理）。

特征权重可通过 :class:`RerankConfig` 调整，改动后用
``scripts/evaluate_rag.py --rebuild`` 回归，避免凭感觉调参。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .chunking import Chunk
from .terminology import Terminology
from .tokenize import tokenize, tokenize_query


@dataclass
class RerankConfig:
    """精排特征权重。"""

    coverage: float = 0.8
    phrase: float = 0.5
    section: float = 0.25
    term_bonus: float = 0.15
    # 术语表 chunk 在「查询不含任何术语」时降权：它天然术语密度最高，
    # 否则会把「问事实」的问题都吸到术语定义上（实测过，见 docs/evaluation.md）。
    # 查询本身就含术语时（如「KQI 和 KPI 怎么区分」）不降权——此时术语表正是答案来源。
    glossary_penalty: float = 0.6

    @property
    def enabled(self) -> bool:
        return any(
            v > 0
            for v in (self.coverage, self.phrase, self.section, self.term_bonus)
        )


def query_phrases(query: str, min_len: int = 4) -> list[str]:
    """查询里的连续片段（滑窗），用于「精确引文」特征。"""
    cleaned = "".join(ch for ch in query if ch.strip())
    phrases: list[str] = []
    for size in (min_len, min_len + 2):
        phrases.extend(
            cleaned[i : i + size] for i in range(max(len(cleaned) - size + 1, 0))
        )
    return phrases


class Reranker:
    """确定性特征精排。"""

    def __init__(
        self,
        terminology: Terminology | None = None,
        config: RerankConfig | None = None,
    ) -> None:
        self.terminology = terminology or Terminology()
        self.config = config or RerankConfig()
        # 缓存键用 (chunk_id, 文本哈希)：chunk_id 在测试里可能被复用，只按 id 缓存会串味
        self._chunk_tokens: dict[tuple[str, int], set[str]] = {}
        self._chunk_lower: dict[tuple[str, int], str] = {}

    # ------------------------------------------------------------------
    def _tokens(self, chunk: Chunk) -> set[str]:
        key = (chunk.chunk_id, hash(chunk.text))
        cached = self._chunk_tokens.get(key)
        if cached is None:
            cached = set(tokenize(chunk.text))
            self._chunk_tokens[key] = cached
        return cached

    def _lower(self, chunk: Chunk) -> str:
        key = (chunk.chunk_id, hash(chunk.text))
        cached = self._chunk_lower.get(key)
        if cached is None:
            cached = chunk.text.lower()
            self._chunk_lower[key] = cached
        return cached

    # ------------------------------------------------------------------
    def features(self, query: str, chunk: Chunk) -> dict[str, float]:
        """计算单个 chunk 的精排特征。"""
        query_tokens = set(tokenize_query(query))
        chunk_tokens = self._tokens(chunk)

        coverage = (
            len(query_tokens & chunk_tokens) / len(query_tokens) if query_tokens else 0.0
        )
        text = self._lower(chunk)
        phrases = query_phrases(query)
        phrase = 1.0 if any(p.lower() in text for p in phrases) else 0.0

        heading = f"{chunk.doc_title} {chunk.section}".lower()
        section_tokens = set(tokenize(heading))
        section = 1.0 if (query_tokens & section_tokens) else 0.0

        term_hits = 0
        if chunk.source_type != "glossary" and not self.terminology.is_empty:
            term_hits = self.terminology.boost(chunk.text)

        return {
            "coverage": round(coverage, 4),
            "phrase": phrase,
            "section": section,
            "term_hits": float(term_hits),
        }

    def score(
        self,
        query: str,
        chunk: Chunk,
        base_score: float,
        use_terminology: bool = True,
        query_has_term: bool = False,
    ) -> tuple[float, dict[str, float]]:
        """在粗排分基础上给出精排分。"""
        if not self.config.enabled:
            return base_score, {}

        feats = self.features(query, chunk)
        if not use_terminology:
            feats["term_hits"] = 0.0
        multiplier = (
            1
            + self.config.coverage * feats["coverage"]
            + self.config.phrase * feats["phrase"]
            + self.config.section * feats["section"]
            + self.config.term_bonus * feats["term_hits"]
        )
        if chunk.source_type == "glossary" and not query_has_term:
            multiplier *= self.config.glossary_penalty
        return round(base_score * multiplier, 6), feats
