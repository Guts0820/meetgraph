"""上下文检索 Agent（RAG 节点）。

位置：``transcription → context → [summary | action | insight]``。

做两件事：

1. 用本次会议的转写文本，从**历史会议纪要**里捞出相关决议片段——会上问「上次
   谁定的这个 deadline」时，纪要里能带上出处；
2. 命中的**公司术语**带上标准定义，交给下游 Agent 写 prompt 时使用，避免同一
   概念在纪要里出现多种叫法。

降级约定：没有索引、检索失败、没命中，都只写空上下文并记一条 error，
**绝不阻塞主流程**（RAG 是增强，不是依赖）。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

from loguru import logger

from ..models.schemas import RetrievedContext


class ContextAgent:
    """历史会议与术语检索节点。"""

    def __init__(
        self,
        retriever: Any | None = None,
        top_k: int = 3,
        query_chars: int = 240,
        source_types: Sequence[str] = ("meeting",),
    ) -> None:
        self.retriever = retriever
        self.top_k = top_k
        self.query_chars = query_chars
        # 默认只用「会议纪要」做历史上下文；内部文档留给知识问答场景
        self.source_types = tuple(source_types)

    # ------------------------------------------------------------------
    def build_query(self, transcript_text: str) -> str:
        """用转写开场（通常点明议题）+ 全文命中的术语构造检索查询。

        整段转写直接当查询会稀释语义，因此只取开场片段，并用术语补上关键实体。
        """
        head = transcript_text[: self.query_chars].replace("\n", " ")
        terms: list[str] = []
        if self.retriever is not None and not self.retriever.terminology.is_empty:
            terms = [t.term for t in self.retriever.terminology.match(transcript_text)][:6]
        return " ".join([head, *terms]).strip()

    # ------------------------------------------------------------------
    async def process(self, state: dict) -> dict:
        """LangGraph 节点函数 —— 检索历史决议与术语。"""
        meeting_id = state.get("meeting_id", "unknown")
        logger.info(f"[ContextAgent] Processing meeting: {meeting_id}")

        transcript_text = state.get("transcript_text", "")
        errors: list[str] = []

        # 空白转写不触发检索（否则会拿空查询去打索引，返回一堆无关片段）
        if not transcript_text.strip():
            state["context"] = RetrievedContext()
            return {"context": state["context"]}

        context = RetrievedContext(retrieved_at=datetime.now().isoformat(timespec="seconds"))

        if self.retriever is None:
            logger.debug("[ContextAgent] No retriever configured, skipping retrieval")
            state["context"] = context
            return {"context": context}

        try:
            query = self.build_query(transcript_text)
            results = self.retriever.search(query, top_k=self.top_k * 3)
            picked = [
                r for r in results if r.chunk.source_type in self.source_types
            ][: self.top_k]

            context.query = query
            context.history = [
                {
                    "citation": r.citation,
                    "doc_id": r.chunk.doc_id,
                    "section": r.chunk.section,
                    "score": r.score,
                    "text": r.chunk.text,
                }
                for r in picked
            ]

            terms = self.retriever.terminology.match(transcript_text)[:8]
            context.terms = [t.term for t in terms]
            context.term_definitions = [
                {"term": t.term, "canonical": t.canonical, "definition": t.definition}
                for t in terms
                if t.definition
            ]

            logger.info(
                f"[ContextAgent] Retrieved {len(context.history)} history chunks, "
                f"{len(context.terms)} terms"
            )
        except Exception as e:  # RAG 失败不影响主流程
            logger.warning(f"[ContextAgent] Retrieval failed, continuing without context: {e}")
            errors.append(f"ContextAgent: {e}")

        state["context"] = context
        updates: dict[str, Any] = {"context": context}
        if errors:
            updates["errors"] = errors
        return updates
