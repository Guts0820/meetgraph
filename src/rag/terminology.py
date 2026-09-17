"""术语层：把公司内部术语表变成检索与生成的增益。

四个作用点（对应 ``docs/plan-rag.md`` 的设计）：

1. ``expand`` —— 查询扩展：用户说缩写（QoE / MRD / CDR），文档只写标准术语
   （用户体验质量 / 市场需求文档 / 话单）时，把标准术语与别名补进检索词；
2. ``match`` / ``boost`` —— 检索加权：chunk 里命中的术语越多，相关性越强；
3. ``prompt_block`` —— 生成约束：把命中的术语定义注入 prompt，要求使用公司
   标准术语、不要自行改写成近义词；
4. 评测口径 —— 术语类问题单独统计命中率，并做开关 A/B。

术语表是**唯一权威来源**：正文只写标准术语，缩写与别名统一登记在
``config/glossary.json``。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from loguru import logger

DEFAULT_GLOSSARY_PATH = Path(__file__).resolve().parents[2] / "config" / "glossary.json"


@dataclass(frozen=True)
class Term:
    """一条术语。"""

    term: str
    canonical: str
    aliases: tuple[str, ...] = ()
    definition: str = ""
    owner: str = ""

    @property
    def surface_forms(self) -> tuple[str, ...]:
        """所有可见写法（缩写 + 标准名 + 别名），去重保序。"""
        forms: list[str] = []
        for form in (self.term, self.canonical, *self.aliases):
            if form and form not in forms:
                forms.append(form)
        return tuple(forms)

    @property
    def expansion(self) -> list[str]:
        """查询扩展时追加的写法：标准名优先，其次别名。"""
        return [f for f in self.surface_forms if f != self.term]


class Terminology:
    """术语表：加载、匹配、扩展、加权、prompt 片段。"""

    def __init__(self, terms: Iterable[Term] = ()) -> None:
        self.terms: list[Term] = list(terms)
        # 长写法优先匹配，避免「话单数据」被「话单」截断后重复计数
        self._surface_index: list[tuple[str, Term]] = sorted(
            (
                (form.lower(), term)
                for term in self.terms
                for form in term.surface_forms
                if len(form) >= 2
            ),
            key=lambda item: len(item[0]),
            reverse=True,
        )

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------

    @staticmethod
    def default_path() -> Path:
        return Path(os.getenv("RAG_GLOSSARY") or DEFAULT_GLOSSARY_PATH)

    @classmethod
    def from_file(cls, path: str | Path | None = None) -> "Terminology":
        raw_path = Path(path) if path else cls.default_path()
        if not raw_path.exists():
            logger.warning(f"Glossary not found: {raw_path}; terminology layer disabled")
            return cls()

        try:
            data: dict[str, Any] = json.loads(raw_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load glossary {raw_path}: {e}")
            return cls()

        terms = [
            Term(
                term=item.get("term", ""),
                canonical=item.get("canonical") or item.get("term", ""),
                aliases=tuple(item.get("aliases", []) or []),
                definition=item.get("definition", ""),
                owner=item.get("owner", ""),
            )
            for item in data.get("terms", [])
            if item.get("term")
        ]
        logger.info(f"Loaded {len(terms)} glossary terms from {raw_path}")
        return cls(terms)

    # ------------------------------------------------------------------
    # 匹配与扩展
    # ------------------------------------------------------------------

    def match(self, text: str) -> list[Term]:
        """返回文本中命中的术语（每个术语只算一次）。"""
        if not text:
            return []

        lowered = text.lower()
        hits: list[Term] = []
        seen: set[str] = set()
        for form, term in self._surface_index:
            if term.term in seen:
                continue
            if form in lowered:
                hits.append(term)
                seen.add(term.term)
        return hits

    def expand(self, query: str) -> tuple[str, list[Term]]:
        """查询扩展：把命中术语的标准名与别名补进查询串。"""
        hits = self.match(query)
        if not hits:
            return query, []

        lowered = query.lower()
        extra: list[str] = []
        for term in hits:
            for form in term.expansion:
                if form.lower() not in lowered and form not in extra:
                    extra.append(form)

        if not extra:
            return query, hits
        expanded = f"{query} {' '.join(extra)}"
        logger.debug(f"Query expanded: {query!r} -> {expanded!r}")
        return expanded, hits

    def boost(self, text: str, max_hits: int = 3) -> int:
        """术语加权用：命中的术语数量（上限 ``max_hits``，避免长文本通吃）。"""
        return min(len(self.match(text)), max_hits)

    def prompt_block(self, terms: Iterable[Term]) -> str:
        """生成用术语约束片段；没有命中术语时返回空串。"""
        items = [t for t in terms if t.definition]
        if not items:
            return ""

        lines = [
            "## 公司内部术语（回答时必须使用下列标准术语，不要改写成近义词或自造说法）",
        ]
        for term in items:
            alias = f"（别名：{'、'.join(term.aliases)}）" if term.aliases else ""
            lines.append(f"- {term.canonical}{alias}：{term.definition}")
        return "\n".join(lines)

    @property
    def is_empty(self) -> bool:
        return not self.terms

    def to_chunks(self, max_chars: int = 420) -> list[Any]:
        """把术语表转成可检索的 chunk（doc_id = 内部术语表）。

        术语表本身也是知识库的一部分：问「KQI 和 KPI 怎么区分」时，答案就该
        来自这里。
        """
        from .chunking import Chunk

        chunks: list[Chunk] = []
        for idx, term in enumerate(self.terms):
            alias = f"别名：{'、'.join(term.aliases)}。" if term.aliases else ""
            text = (
                f"术语：{term.term}\n标准说法：{term.canonical}\n"
                f"{alias}定义：{term.definition}\n归属：{term.owner}"
            ).strip()
            chunks.append(
                Chunk(
                    chunk_id=f"内部术语表#{idx}",
                    doc_id="内部术语表",
                    doc_title="内部术语表",
                    section=term.canonical,
                    text=text,
                    source_type="glossary",
                    start=0,
                    metadata={"term": term.term, "owner": term.owner},
                )
            )
        return chunks
