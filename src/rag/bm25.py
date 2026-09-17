"""BM25 倒排检索（纯 Python，无额外依赖）。

混合检索里的「关键词通道」：向量通道擅长语义近似，BM25 擅长精确命中
（编号、人名、专有名词、数字门限）。两路用 RRF 融合，见 ``retriever.py``。
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

from .tokenize import tokenize


class BM25Index:
    """标准 BM25（Okapi）实现。

    Args:
        k1: 词频饱和参数，越大越强调词频。
        b: 文档长度归一化强度，0 表示不归一化。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self._term_freqs: list[Counter[str]] = []
        self._doc_len: list[int] = []
        self._df: Counter[str] = Counter()
        self._avg_len: float = 0.0

    # ------------------------------------------------------------------
    # 构建
    # ------------------------------------------------------------------

    @classmethod
    def from_texts(
        cls,
        doc_ids: Sequence[str],
        texts: Sequence[str],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> "BM25Index":
        index = cls(k1=k1, b=b)
        index.add_many(doc_ids, texts)
        return index

    def add_many(self, doc_ids: Sequence[str], texts: Sequence[str]) -> None:
        for doc_id, text in zip(doc_ids, texts):
            self.add(doc_id, text)
        self._finalize()

    def add(self, doc_id: str, text: str) -> None:
        tokens = tokenize(text)
        freq = Counter(tokens)
        self.doc_ids.append(doc_id)
        self._term_freqs.append(freq)
        self._doc_len.append(len(tokens))
        for term in freq:
            self._df[term] += 1

    def _finalize(self) -> None:
        total = len(self._doc_len)
        self._avg_len = sum(self._doc_len) / total if total else 0.0

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    def _idf(self, term: str) -> float:
        n = len(self.doc_ids)
        df = self._df.get(term, 0)
        if df == 0:
            return 0.0
        # BM25 的 idf，加 0.5 平滑；若为负则钳到 0（语料很小时会出现）
        value = math.log(1 + (n - df + 0.5) / (df + 0.5))
        return max(value, 0.0)

    def search(
        self, query: str, top_k: int = 10
    ) -> list[tuple[str, float]]:
        """返回 [(doc_id, 分数)]，按分数降序，零分不入榜。"""
        tokens = tokenize(query)
        if not tokens or not self.doc_ids:
            return []

        scores: dict[str, float] = {}
        avg_len = self._avg_len or 1.0

        for term in set(tokens):
            idf = self._idf(term)
            if idf == 0:
                continue
            for idx, freq in enumerate(self._term_freqs):
                tf = freq.get(term)
                if not tf:
                    continue
                doc_len = self._doc_len[idx] or 1
                denom = tf + self.k1 * (1 - self.b + self.b * doc_len / avg_len)
                scores[self.doc_ids[idx]] = (
                    scores.get(self.doc_ids[idx], 0.0) + idf * tf * (self.k1 + 1) / denom
                )

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [(doc_id, round(score, 4)) for doc_id, score in ranked[:top_k]]

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "k1": self.k1,
            "b": self.b,
            "doc_ids": self.doc_ids,
            "term_freqs": [dict(freq) for freq in self._term_freqs],
            "doc_len": self._doc_len,
            "df": dict(self._df),
            "avg_len": self._avg_len,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BM25Index":
        index = cls(k1=data.get("k1", 1.5), b=data.get("b", 0.75))
        index.doc_ids = list(data.get("doc_ids", []))
        index._term_freqs = [Counter(freq) for freq in data.get("term_freqs", [])]
        index._doc_len = list(data.get("doc_len", []))
        index._df = Counter(data.get("df", {}))
        index._avg_len = data.get("avg_len", 0.0)
        return index

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
