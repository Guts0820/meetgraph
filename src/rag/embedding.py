"""向量化后端（可插拔）。

三种 embedder：

- ``HashingEmbedder``：把 token 哈希到固定维度，**确定性、零依赖、离线**。
  不追求语义质量，用于单元测试与「模型不可用」时的降级，保证 CI 永不联网。
- ``TransformersBgEmbedder``：本地加载 BGE 中文小模型（默认
  ``BAAI/bge-small-zh-v1.5``），用 venv 里已有的 ``transformers`` + ``torch``，
  不需要 sentence-transformers。
- ``OpenAICompatEmbedder``：走任意 OpenAI 兼容的 ``/embeddings`` 接口（可选）。

统一约定：输出 L2 归一化后的 ``float32`` 矩阵，余弦相似度即点积。
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Protocol, Sequence

import numpy as np
from loguru import logger

from .tokenize import tokenize

DEFAULT_BG_MODEL = "BAAI/bge-small-zh-v1.5"
# BGE 中文模型对短查询建议加指令前缀，检索效果更稳
BG_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class Embedder(Protocol):
    """向量化接口。"""

    name: str
    dim: int

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        ...


class HashingEmbedder:
    """哈希向量：确定性、离线、无依赖（测试与降级用）。"""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.name = f"hash-{dim}"

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                digest = hashlib.md5(token.encode("utf-8")).hexdigest()
                matrix[row, int(digest[:8], 16) % self.dim] += 1.0
        return _l2_normalize(matrix)


class TransformersBgEmbedder:
    """本地 BGE 中文向量模型（transformers + torch，懒加载）。"""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        max_length: int = 512,
        batch_size: int = 16,
    ) -> None:
        self.model_name = model_name or os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_BG_MODEL)
        self.name = self.model_name
        self.device = device or os.getenv("RAG_DEVICE")
        self.max_length = max_length
        self.batch_size = batch_size
        self.dim = 512
        self._model: Any = None
        self._tokenizer: Any = None

    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        # HF 直连在国内常不可达；未显式指定 endpoint 时默认走镜像
        if not os.getenv("HF_ENDPOINT") and os.getenv("RAG_HF_MIRROR", "1") != "0":
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            logger.info("HF_ENDPOINT not set, using https://hf-mirror.com")

        from transformers import AutoModel, AutoTokenizer  # 延迟导入，避免拖慢启动

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)

        import torch

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self.device)
        self._model.eval()
        self.dim = int(self._model.config.hidden_size)
        logger.info(f"Loaded embedding model {self.model_name} on {self.device} (dim={self.dim})")

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        self._ensure_loaded()
        import torch

        prefix = BG_QUERY_INSTRUCTION if is_query else ""
        vectors: list[np.ndarray] = []

        for start in range(0, len(texts), self.batch_size):
            batch = [prefix + text for text in texts[start : start + self.batch_size]]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self._model(**encoded)

            # 均值池化（按 attention mask 加权）
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            summed = (outputs.last_hidden_state * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-6)
            vectors.append((summed / counts).cpu().numpy().astype(np.float32))

        if not vectors:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _l2_normalize(np.vstack(vectors))


class OpenAICompatEmbedder:
    """任意 OpenAI 兼容 /embeddings 接口。"""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = (base_url or os.getenv("RAG_EMBEDDING_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.getenv("RAG_EMBEDDING_API_KEY", "")
        self.model = model or os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small")
        self.timeout = timeout
        self.name = f"openai-compat:{self.model}"
        self.dim = 0  # 首次调用后确定

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        import httpx

        response = httpx.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": list(texts)},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()["data"]
        matrix = np.array([item["embedding"] for item in data], dtype=np.float32)
        self.dim = int(matrix.shape[1])
        return _l2_normalize(matrix)


def resolve_embedder(prefer: str | None = None, probe: bool = True) -> Embedder:
    """按配置返回可用的 embedder，失败时降级为离线哈希向量。

    Args:
        prefer: ``hash`` / ``bge`` / ``openai`` / ``auto``；默认读环境变量
            ``RAG_EMBEDDER``（缺省 ``auto``：先试本地 BGE，不行退哈希）。
        probe: 是否真正加载模型验证可用性（建索引时建议 True，查询路径可复用缓存）。
    """
    choice = (prefer or os.getenv("RAG_EMBEDDER", "auto")).lower()

    if choice == "hash":
        return HashingEmbedder()

    if choice in ("openai", "api"):
        return OpenAICompatEmbedder()

    if choice in ("bge", "auto", "local"):
        embedder = TransformersBgEmbedder()
        if not probe:
            return embedder
        try:
            embedder.encode(["连通性探测"], is_query=True)
            return embedder
        except Exception as e:  # 模型缺失/下载失败/显存不足
            if choice == "bge":
                logger.error(f"BGE embedder unavailable: {e}")
            else:
                logger.warning(
                    f"Local embedding model unavailable ({e}); "
                    f"falling back to offline hashing embedder — "
                    f"retrieval will rely mostly on BM25"
                )
            return HashingEmbedder()

    logger.warning(f"Unknown RAG_EMBEDDER={choice!r}, using hashing embedder")
    return HashingEmbedder()
