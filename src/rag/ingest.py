"""索引构建与加载：把语料（内部文档 + 会议纪要 + 术语表）变成可检索索引。

产物落在 ``data/index/``：

```
data/index/
├── chunks.jsonl      # 全部 chunk 与元数据
├── bm25.json         # 倒排（词频、文档频、长度）
├── vectors.npy       # 归一化向量矩阵
├── vector_ids.json   # 向量行号 → chunk_id
└── meta.json         # 构建信息（embedder、语料规模、时间），便于排查
```

语料来源可替换：默认读 ``data/knowledge``（内部文档）与 ``data/meetings``
（会议纪要），换成本公司的目录即可，不需要改代码。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from loguru import logger

from .bm25 import BM25Index
from .chunking import Chunk, load_markdown_dir
from .embedding import Embedder, resolve_embedder
from .retriever import HybridRetriever
from .terminology import Terminology
from .vector_store import VectorIndex

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX_DIR = REPO_ROOT / "data" / "index"


def corpus_dirs(root: Path | None = None) -> list[tuple[Path, str]]:
    """返回 [(目录, 来源类型)]；可用 ``RAG_CORPUS_DIRS`` 覆盖（逗号分隔）。"""
    base = root or REPO_ROOT
    override = os.getenv("RAG_CORPUS_DIRS", "").strip()
    if override:
        pairs = []
        for item in override.split(","):
            item = item.strip()
            if not item:
                continue
            path = Path(item)
            pairs.append((path if path.is_absolute() else base / path, "knowledge"))
        return pairs
    return [
        (base / "data" / "knowledge", "knowledge"),
        (base / "data" / "meetings", "meeting"),
    ]


def build_chunks(
    terminology: Terminology | None = None,
    root: Path | None = None,
) -> list[Chunk]:
    """加载全部语料并切成 chunk（含术语表自身）。"""
    terms = terminology if terminology is not None else Terminology.from_file()
    chunks: list[Chunk] = []

    for directory, source_type in corpus_dirs(root):
        loaded = load_markdown_dir(directory, source_type=source_type)
        if not loaded and not directory.exists():
            logger.warning(f"Corpus directory missing: {directory}")
        chunks.extend(loaded)

    if not terms.is_empty:
        chunks.extend(terms.to_chunks())

    logger.info(f"Corpus built: {len(chunks)} chunks")
    return chunks


@dataclass
class RagIndex:
    """检索索引（chunks + 两个通道 + 检索器）。"""

    chunks: list[Chunk]
    embedder: Embedder
    bm25: BM25Index
    vectors: VectorIndex
    retriever: HybridRetriever
    built_at: str = ""

    # ------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        chunks: Sequence[Chunk],
        embedder: Embedder | None = None,
        terminology: Terminology | None = None,
        probe_embedder: bool = True,
    ) -> "RagIndex":
        if not chunks:
            raise ValueError("语料为空：请检查 data/knowledge 与 data/meetings")

        terms = terminology if terminology is not None else Terminology.from_file()
        embedder = embedder or resolve_embedder(probe=probe_embedder)

        chunk_list = list(chunks)
        ids = [c.chunk_id for c in chunk_list]
        texts = [c.text for c in chunk_list]

        bm25 = BM25Index.from_texts(ids, texts)

        vectors = VectorIndex(dim=int(embedder.dim))
        embeddings = embedder.encode(texts, is_query=False)
        vectors.add(ids, embeddings)

        retriever = HybridRetriever(
            chunks=chunk_list,
            embedder=embedder,
            bm25_index=bm25,
            vector_index=vectors,
            terminology=terms,
        )
        logger.info(
            f"Index built: {len(chunk_list)} chunks, embedder={embedder.name}, dim={embedder.dim}"
        )
        return cls(
            chunks=chunk_list,
            embedder=embedder,
            bm25=bm25,
            vectors=vectors,
            retriever=retriever,
            built_at=datetime.now().isoformat(timespec="seconds"),
        )

    # ------------------------------------------------------------------
    def save(self, directory: str | Path | None = None) -> Path:
        path = Path(directory or os.getenv("RAG_INDEX_DIR") or DEFAULT_INDEX_DIR)
        path.mkdir(parents=True, exist_ok=True)

        with (path / "chunks.jsonl").open("w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

        self.bm25.save(path / "bm25.json")
        self.vectors.save(path)
        (path / "meta.json").write_text(
            json.dumps(
                {
                    "built_at": self.built_at,
                    "embedder": self.embedder.name,
                    "dim": int(self.embedder.dim),
                    "chunks": len(self.chunks),
                    "docs": len({c.doc_id for c in self.chunks}),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info(f"Index saved to {path}")
        return path

    @classmethod
    def load(
        cls,
        directory: str | Path | None = None,
        embedder: Embedder | None = None,
        terminology: Terminology | None = None,
    ) -> "RagIndex":
        path = Path(directory or os.getenv("RAG_INDEX_DIR") or DEFAULT_INDEX_DIR)
        chunks = [
            Chunk.from_dict(json.loads(line))
            for line in (path / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        bm25 = BM25Index.load(path / "bm25.json")
        vectors = VectorIndex.load(path)
        terms = terminology if terminology is not None else Terminology.from_file()
        # 加载时不探测模型：查询时才需要真正加载（懒加载），避免服务启动被拖慢
        embedder = embedder or resolve_embedder(probe=False)

        meta_path = path / "meta.json"
        built_at = ""
        if meta_path.exists():
            meta: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
            built_at = meta.get("built_at", "")

        retriever = HybridRetriever(
            chunks=chunks,
            embedder=embedder,
            bm25_index=bm25,
            vector_index=vectors,
            terminology=terms,
        )
        logger.info(f"Index loaded from {path}: {len(chunks)} chunks (built_at={built_at})")
        return cls(
            chunks=chunks,
            embedder=embedder,
            bm25=bm25,
            vectors=vectors,
            retriever=retriever,
            built_at=built_at,
        )

    # ------------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        by_source: dict[str, int] = {}
        for chunk in self.chunks:
            by_source[chunk.source_type] = by_source.get(chunk.source_type, 0) + 1
        return {
            "chunks": len(self.chunks),
            "docs": len({c.doc_id for c in self.chunks}),
            "by_source": by_source,
            "embedder": self.embedder.name,
            "dim": int(self.embedder.dim),
            "built_at": self.built_at,
        }


def build_and_save(
    directory: str | Path | None = None,
    embedder: Embedder | None = None,
) -> RagIndex:
    """从语料重建索引并落盘（CLI / reindex 接口用）。"""
    terminology = Terminology.from_file()
    chunks = build_chunks(terminology)
    index = RagIndex.build(chunks, embedder=embedder, terminology=terminology)
    index.save(directory)
    return index
