"""进程内索引单例：给 API 与主流水线复用，避免每次请求重建索引。

加载策略：优先读 ``data/index/`` 的落盘索引；不存在时按语料现场构建（首次调用
会慢一点，之后复用）。``reindex()`` 强制重建并落盘。
"""

from __future__ import annotations

import threading
from typing import Any

from loguru import logger

from .ingest import RagIndex, build_and_save
from .qa import KnowledgeQA
from .retriever import HybridRetriever

_lock = threading.Lock()
_index: RagIndex | None = None


def get_index(rebuild: bool = False) -> RagIndex:
    """获取索引单例。"""
    global _index
    if _index is not None and not rebuild:
        return _index

    with _lock:
        if _index is not None and not rebuild:
            return _index
        try:
            _index = RagIndex.load()
        except FileNotFoundError:
            logger.info("No persisted index found, building from corpus")
            _index = build_and_save()
    return _index


def get_retriever(rebuild: bool = False) -> HybridRetriever:
    return get_index(rebuild=rebuild).retriever


def get_qa(llm: Any | None = None, rebuild: bool = False) -> KnowledgeQA:
    index = get_index(rebuild=rebuild)
    return KnowledgeQA(retriever=index.retriever, terminology=index.retriever.terminology, llm=llm)


def reindex() -> dict[str, Any]:
    """强制重建索引并落盘，返回索引概览。"""
    with _lock:
        global _index
        _index = build_and_save()
        return _index.stats()


def reset_index() -> None:
    """清空单例（测试用）。"""
    global _index
    with _lock:
        _index = None
