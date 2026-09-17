"""RAG 检索增强模块。

对外主要入口：

- :class:`~src.rag.ingest.RagIndex` / ``build_and_save`` / ``RagIndex.load``：索引构建与加载
- :class:`~src.rag.retriever.HybridRetriever`：向量 + BM25 混合检索（RRF 融合 + 术语加权）
- :class:`~src.rag.qa.KnowledgeQA`：带引用的知识问答
- :class:`~src.rag.terminology.Terminology`：公司内部术语表（查询扩展 / 加权 / 生成约束）
- :func:`~src.rag.service.get_index` / ``reindex``：进程内单例索引（API 集成用）
"""

from .bm25 import BM25Index
from .chunking import Chunk, chunk_markdown, load_markdown_dir
from .embedding import (
    HashingEmbedder,
    OpenAICompatEmbedder,
    TransformersBgEmbedder,
    resolve_embedder,
)
from .ingest import RagIndex, build_and_save, build_chunks
from .qa import Answer, KnowledgeQA
from .retriever import HybridRetriever, RetrievalResult
from .service import get_index, get_qa, get_retriever, reindex, reset_index
from .terminology import Terminology, Term
from .vector_store import VectorIndex

__all__ = [
    "Answer",
    "BM25Index",
    "Chunk",
    "HashingEmbedder",
    "HybridRetriever",
    "KnowledgeQA",
    "OpenAICompatEmbedder",
    "RagIndex",
    "RetrievalResult",
    "Term",
    "Terminology",
    "TransformersBgEmbedder",
    "VectorIndex",
    "build_and_save",
    "build_chunks",
    "chunk_markdown",
    "get_index",
    "get_qa",
    "get_retriever",
    "load_markdown_dir",
    "reindex",
    "reset_index",
    "resolve_embedder",
]
