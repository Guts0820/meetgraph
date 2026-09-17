"""Markdown 感知的文本分块。

分块策略：
- **按标题切**：``#`` / ``##`` / ``###`` 作为天然语义边界，chunk 保留标题路径，
  检索命中后能告诉用户「来自哪篇文档的哪一节」；
- **超长再滑窗**：单节超过 ``max_chars`` 时按句子边界切窗，窗口之间保留
  ``overlap_chars`` 重叠，避免结论被切在中间；
- **元数据随行**：每个 chunk 带 doc_id / 标题 / 来源类型（内部文档、会议纪要、
  术语表）/ 日期，检索结果直接可做引用。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
SENTENCE_END = "。！？!?；;\n"


@dataclass
class Chunk:
    """一个可检索的最小单元。"""

    chunk_id: str
    doc_id: str
    doc_title: str
    section: str
    text: str
    source_type: str = "knowledge"
    start: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "section": self.section,
            "text": self.text,
            "source_type": self.source_type,
            "start": self.start,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        return cls(**data)

    @property
    def citation(self) -> str:
        """人类可读的引用串，例如「数据接入规范-话单与路测 / 2. 字段与脱敏要求」。"""
        return f"{self.doc_title} / {self.section}" if self.section else self.doc_title


def _split_long(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """把超长文本切成带重叠的窗口，尽量在句子边界断开。"""
    if len(text) <= max_chars:
        return [text]

    windows: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            # 在窗口尾部附近找最后一个句末标点
            cut = max(text.rfind(ch, start + max_chars // 2, end) for ch in SENTENCE_END)
            if cut > start:
                end = cut + 1
        windows.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap_chars, start + 1)
    return [w for w in windows if w]


def chunk_markdown(
    text: str,
    doc_id: str,
    doc_title: str | None = None,
    max_chars: int = 420,
    overlap_chars: int = 80,
    source_type: str = "knowledge",
    metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """把一篇 Markdown 切成 Chunk 列表。"""
    title = doc_title or doc_id
    meta = metadata or {}
    lines = text.splitlines()

    sections: list[tuple[list[tuple[int, str]], list[str], int]] = []
    heading_stack: list[tuple[int, str]] = []
    buffer: list[str] = []
    buffer_start = 0
    offset = 0

    def flush() -> None:
        if any(line.strip() for line in buffer):
            sections.append((list(heading_stack), list(buffer), buffer_start))

    for line in lines:
        match = HEADING_RE.match(line)
        if match:
            flush()
            buffer = []
            level = len(match.group(1))
            heading = match.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading))
            buffer_start = offset + len(line) + 1
        else:
            buffer.append(line)
        offset += len(line) + 1
    flush()

    chunks: list[Chunk] = []
    for section_index, (stack, block, block_start) in enumerate(sections):
        section_path = " > ".join(h for _, h in stack)
        body = "\n".join(block).strip()
        for piece_index, piece in enumerate(
            _split_long(body, max_chars, overlap_chars)
        ):
            digest = hashlib.sha1(
                f"{doc_id}|{section_path}|{piece_index}|{piece[:64]}".encode("utf-8")
            ).hexdigest()[:12]
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}#{section_index}-{piece_index}-{digest}",
                    doc_id=doc_id,
                    doc_title=title,
                    section=section_path,
                    text=piece,
                    source_type=source_type,
                    start=block_start,
                    metadata=dict(meta),
                )
            )
    return chunks


def load_markdown_dir(
    directory: str | Path,
    source_type: str = "knowledge",
    pattern: str = "*.md",
    extra_metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """把一个目录下的 Markdown 全部切成 chunk（doc_id 取文件名主干）。"""
    root = Path(directory)
    if not root.exists():
        return []

    chunks: list[Chunk] = []
    for path in sorted(root.glob(pattern)):
        text = path.read_text(encoding="utf-8")
        chunks.extend(
            chunk_markdown(
                text,
                doc_id=path.stem,
                doc_title=path.stem,
                source_type=source_type,
                metadata={"path": str(path), **(extra_metadata or {})},
            )
        )
    return chunks
