"""向量索引（numpy 实现）。

语料规模在万级 chunk 以内，暴力余弦检索足够快，因此不引入 faiss / chromadb，
少一个依赖、少一处版本坑。索引与元数据落盘到 ``data/index/``。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np


class VectorIndex:
    """归一化向量的暴力最近邻检索。"""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.ids: list[str] = []
        self._matrix: np.ndarray = np.zeros((0, dim), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.ids)

    # ------------------------------------------------------------------
    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        if len(ids) != len(vectors):
            raise ValueError("ids 与 vectors 数量不一致")
        if vectors.size and vectors.shape[1] != self.dim:
            raise ValueError(
                f"向量维度不一致：索引 {self.dim}，输入 {vectors.shape[1]}"
            )
        self.ids.extend(ids)
        self._matrix = (
            vectors.astype(np.float32)
            if self._matrix.size == 0
            else np.vstack([self._matrix, vectors.astype(np.float32)])
        )

    def search(self, vector: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        """返回 [(id, 余弦相似度)]，按相似度降序。"""
        if not self.ids or vector.size == 0:
            return []

        query = vector.reshape(-1).astype(np.float32)
        scores = self._matrix @ query
        order = np.argsort(-scores)[:top_k]
        # 相似度 <= 0（无公共词、零向量）不入榜：避免空查询返回一堆无关片段
        return [
            (self.ids[i], round(float(scores[i]), 6))
            for i in order
            if float(scores[i]) > 0
        ]

    # ------------------------------------------------------------------
    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self._matrix)
        (path / "vector_ids.json").write_text(
            json.dumps(self.ids, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: str | Path) -> "VectorIndex":
        path = Path(directory)
        matrix = np.load(path / "vectors.npy")
        ids = json.loads((path / "vector_ids.json").read_text(encoding="utf-8"))
        index = cls(dim=int(matrix.shape[1]) if matrix.size else 0)
        index.ids = ids
        index._matrix = matrix.astype(np.float32)
        return index
