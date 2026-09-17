"""
Summary Agent（摘要Agent）
- 接收转写文本，生成结构化会议纪要
- 使用 LLM 进行内容提取和组织
- 输出: 议题/讨论要点/结论/决策 四层结构
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from ..integrations.minimax_client import MiniMaxClient
from ..models.schemas import (
    MeetingStatus,
    MeetingSummary,
    TopicSummary,
)


SUMMARY_SYSTEM_PROMPT = """你是一位专业的会议纪要助手。你的任务是根据会议转写文本，生成清晰、结构化的会议纪要。

要求：
1. 准确提取会议中的每个议题
2. 每个议题包含讨论要点、参与人、结论
3. 明确列出会议做出的决策
4. 列出下一步行动计划
5. 使用中文输出

你必须严格按照以下JSON格式输出，不要添加任何其他文字："""

SUMMARY_USER_PROMPT = """请根据以下会议转写文本生成结构化会议纪要。

## 会议转写文本
{transcript}

## 输出格式（严格JSON）
{{
  "title": "会议主题（从内容推断）",
  "date": "会议日期（如无法确定则写今天）",
  "participants": ["参会人1", "参会人2"],
  "topics": [
    {{
      "title": "议题名称",
      "discussion_points": ["要点1", "要点2"],
      "participants": ["发言人1"],
      "conclusion": "该议题的结论"
    }}
  ],
  "decisions": ["决策1", "决策2"],
  "next_steps": ["下一步1", "下一步2"]
}}"""


class SummaryAgent:
    """
    摘要Agent - 并行阶段的节点之一

    架构说明:
    1. 从 state 读取 transcript_text
    2. 构造 Few-shot Prompt 调用 LLM
    3. 约束 JSON Schema 输出格式
    4. 解析并验证结果
    5. 写入 state["summary"]

    长度策略：单次调用覆盖 1 小时会议；更长的会议按议题分块后再合并
    （Map-Reduce 式），当前实现保留单次调用路径，分块作为 roadmap 项。
    """

    def __init__(self, llm_client: MiniMaxClient | None = None):
        self.llm = llm_client or MiniMaxClient()

    async def process(self, state: dict) -> dict:
        """
        LangGraph 节点函数 —— 生成会议摘要

        与 Action Agent、Insight Agent 并行执行。
        """
        meeting_id = state.get("meeting_id", "unknown")
        logger.info(f"[SummaryAgent] Processing meeting: {meeting_id}")

        transcript_text = state.get("transcript_text", "")
        errors_delta: list[str] = []
        if not transcript_text:
            logger.warning("[SummaryAgent] No transcript text available")
            state["summary"] = MeetingSummary(
                title="未知会议", date="", participants=[], topics=[],
                decisions=[], next_steps=[],
            )
            return {"summary": state["summary"]}

        try:
            summary = await self._generate_summary(transcript_text)
            state["summary"] = summary
            logger.info(
                f"[SummaryAgent] Summary generated: {summary.title}, "
                f"{len(summary.topics)} topics"
            )
        except Exception as e:
            logger.error(f"[SummaryAgent] Error: {e}")
            errors_delta.append(f"SummaryAgent: {str(e)}")
            state["summary"] = self._generate_fallback_summary(transcript_text)

        updates = {"summary": state["summary"]}
        if errors_delta:
            updates["errors"] = errors_delta
        return updates

    async def _generate_summary(self, transcript: str) -> MeetingSummary:
        """调用 LLM 生成结构化摘要"""
        messages = [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": SUMMARY_USER_PROMPT.format(transcript=transcript),
            },
        ]

        result = await self.llm.chat_json(
            messages=messages,
            temperature=0.3,
            max_tokens=4096,
        )

        topics = [
            TopicSummary(**topic) for topic in result.get("topics", [])
        ]

        return MeetingSummary(
            title=result.get("title", "会议纪要"),
            date=result.get("date", ""),
            participants=result.get("participants", []),
            topics=topics,
            decisions=result.get("decisions", []),
            next_steps=result.get("next_steps", []),
        )

    @staticmethod
    def _generate_fallback_summary(transcript: str) -> MeetingSummary:
        """LLM 调用失败时的降级方案 —— 基于规则的简单摘要"""
        lines = transcript.strip().split("\n")
        speakers = set()
        for line in lines:
            if ":" in line:
                parts = line.split(":", 1)
                speaker_part = parts[0].strip()
                # 从 "[0.0s-8.5s] 张总" 格式中提取说话人
                if "]" in speaker_part:
                    speaker_part = speaker_part.split("]", 1)[-1].strip()
                speakers.add(speaker_part)

        return MeetingSummary(
            title="会议纪要（自动摘要降级模式）",
            date="",
            participants=list(speakers),
            topics=[
                TopicSummary(
                    title="会议内容",
                    discussion_points=["（LLM调用失败，请查看原始转写文本）"],
                    participants=list(speakers),
                    conclusion="",
                )
            ],
            decisions=[],
            next_steps=[],
        )
