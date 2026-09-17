"""基于检索的问答：把检索结果组装成受约束的 prompt，产出带引用的答案。

防幻觉的三条硬约束（都在 prompt 与代码里落实）：

1. **只用资料回答**：prompt 明确要求「资料中没有的不要臆测」，代码层在检索
   为空时直接返回「资料中未提及」，**不调用 LLM**；
2. **必须标注来源**：要求每条结论后用 ``[编号]`` 标注，答案返回时解析出引用，
   调用方可用 :meth:`Answer.citations` 核验每条引用能否在索引里定位；
3. **使用公司标准术语**：把命中的术语定义注入 prompt，避免同一概念多种叫法。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from loguru import logger

from .retriever import HybridRetriever, RetrievalResult
from .terminology import Terminology

CITATION_RE = re.compile(r"\[(\d+)\]")

QA_SYSTEM_PROMPT = """你是公司内部知识助手。规则：
1. 只依据「参考资料」回答问题，资料里没有的信息必须明说「资料中未提及」，禁止编造；
2. 每条结论后用 [编号] 标注来源，编号与参考资料一致；
3. 使用公司内部标准术语，不要改写成近义词或自造说法；
4. 回答简洁，先给结论再给依据，不要复述整段资料。"""

NO_ANSWER_TEXT = "资料中未提及相关内容。"


@dataclass
class Answer:
    """一次问答的结果。"""

    question: str
    text: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    retrieved: list[RetrievalResult] = field(default_factory=list)
    used_terms: list[str] = field(default_factory=list)
    answered: bool = True
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.text,
            "answered": self.answered,
            "citations": self.citations,
            "used_terms": self.used_terms,
            "retrieved": [r.to_dict() for r in self.retrieved],
            "debug": self.debug,
        }


def build_context_block(results: Sequence[RetrievalResult]) -> str:
    """把检索结果编号拼成参考资料块。"""
    lines: list[str] = []
    for idx, result in enumerate(results, 1):
        lines.append(f"[{idx}] {result.citation}")
        lines.append(result.chunk.text.strip())
        lines.append("")
    return "\n".join(lines).strip()


class KnowledgeQA:
    """检索 + 生成。``llm`` 只需实现 ``async chat(messages, **kwargs) -> str``。"""

    def __init__(
        self,
        retriever: HybridRetriever,
        llm: Any | None = None,
        terminology: Terminology | None = None,
        top_k: int = 5,
        min_results: int = 1,
    ) -> None:
        self.retriever = retriever
        self.terminology = terminology or retriever.terminology
        self.top_k = top_k
        self.min_results = min_results
        self._llm = llm

    @property
    def llm(self) -> Any:
        if self._llm is None:
            from ..integrations.minimax_client import MiniMaxClient

            self._llm = MiniMaxClient()
        return self._llm

    # ------------------------------------------------------------------
    def retrieve(self, question: str, top_k: int | None = None, use_terminology: bool | None = None):
        return self.retriever.search(
            question,
            top_k=top_k or self.top_k,
            use_terminology=use_terminology,
            return_debug=True,
        )

    async def ask(
        self,
        question: str,
        top_k: int | None = None,
        use_terminology: bool | None = None,
        inject_terms: bool | None = None,
    ) -> Answer:
        """回答问题。

        Args:
            use_terminology: 是否启用术语层的检索侧能力（查询扩展 / 术语加权）；
                显式传 ``False`` 用于做「关掉术语层」的对照实验。
            inject_terms: 是否把术语定义注入 prompt（生成侧约束）；默认跟随
                ``use_terminology``。
        """
        use_terms = True if use_terminology is None else use_terminology
        inject = use_terms if inject_terms is None else inject_terms

        results, debug = self.retrieve(question, top_k, use_terminology=use_terms)
        debug_dict = {
            "query": debug.query,
            "expanded_query": debug.expanded_query,
            "query_terms": debug.query_terms,
            "candidate_pool": debug.candidate_pool,
            "channels": debug.channels,
            "rerank_enabled": debug.rerank_enabled,
        }

        if len(results) < self.min_results:
            logger.info(f"No knowledge hit for question: {question!r}")
            return Answer(
                question=question,
                text=NO_ANSWER_TEXT,
                citations=[],
                retrieved=[],
                used_terms=[],
                answered=False,
                debug=debug_dict,
            )

        # 术语约束：查询命中的术语 + Top-1 片段命中的术语（关掉术语层时为空）。
        # 只取 Top-1 是为了避免把无关片段里出现的术语定义也塞进 prompt——
        # 注入无关术语定义会稀释「必须使用公司标准术语」这条约束。
        term_pool: dict[str, Any] = {}
        if use_terms:
            for term in debug.query_terms:
                for item in self.terminology.terms:
                    if item.term == term:
                        term_pool[item.term] = item
            for name in getattr(results[0], "term_hits", ()):
                for item in self.terminology.terms:
                    if item.term == name and name not in term_pool:
                        term_pool[name] = item
        used_terms = list(term_pool)
        prompt_terms = list(term_pool.values()) if inject else []

        messages = [
            {"role": "system", "content": QA_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": self._build_user_prompt(question, results, prompt_terms),
            },
        ]
        text = await self.llm.chat(messages=messages, temperature=0.2, max_tokens=1024)

        return Answer(
            question=question,
            text=text.strip(),
            citations=self._parse_citations(text, results),
            retrieved=results,
            used_terms=used_terms,
            answered=True,
            debug=debug_dict,
        )

    # ------------------------------------------------------------------
    def _build_user_prompt(
        self,
        question: str,
        results: Sequence[RetrievalResult],
        terms: Sequence[Any],
    ) -> str:
        parts = ["## 参考资料", build_context_block(results), ""]
        block = self.terminology.prompt_block(terms) if terms else ""
        if block:
            parts += [block, ""]
        parts += [
            "## 问题",
            question,
            "",
            "## 要求",
            "- 只依据参考资料回答，资料中没有的内容写「资料中未提及」；",
            "- 每条结论后用 [编号] 标注来源；",
            "- 使用公司标准术语。",
        ]
        return "\n".join(parts)

    @staticmethod
    def _parse_citations(
        text: str, results: Sequence[RetrievalResult]
    ) -> list[dict[str, Any]]:
        """把答案里的 [编号] 解析成引用列表；编号越界则忽略（防模型编引用）。"""
        cited: list[dict[str, Any]] = []
        seen: set[int] = set()
        for raw in CITATION_RE.findall(text or ""):
            index = int(raw)
            if index in seen or not (1 <= index <= len(results)):
                continue
            seen.add(index)
            result = results[index - 1]
            cited.append(
                {
                    "index": index,
                    "chunk_id": result.chunk.chunk_id,
                    "title": result.chunk.doc_title,
                    "section": result.chunk.section,
                    "source_type": result.chunk.source_type,
                    "citation": result.citation,
                }
            )
        return cited
