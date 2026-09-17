"""
Insight Agent（洞察Agent）
- 情绪分析：整体会议氛围和情感倾向
- 发言统计：各参会人发言时长、占比、次数
- 效率评分：综合评估会议质量
- 关键词提取：TF-IDF 提取核心关键词
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from loguru import logger

from ..integrations.minimax_client import MiniMaxClient
from ..models.schemas import (
    MeetingInsight,
    SentimentType,
    SpeakerStats,
    TranscriptResult,
)


INSIGHT_SYSTEM_PROMPT = """你是一位专业的会议分析师。请分析以下会议转写文本，提供多维度的会议洞察。

分析维度：
1. 情绪分析：判断整体会议氛围（positive/neutral/negative），给出0-1的情感得分
2. 关键词：提取5-10个核心关键词
3. 会议亮点：列出2-3个重要亮点
4. 改进建议：提供1-2条改进建议
5. 效率评分：0-10分评估会议效率

你必须严格按照JSON格式输出："""

INSIGHT_USER_PROMPT = """请分析以下会议转写文本。

## 会议转写文本
{transcript}

## 发言统计数据
{speaker_stats}

## 输出格式（严格JSON）
{{
  "overall_sentiment": "positive 或 neutral 或 negative",
  "sentiment_score": 0.75,
  "efficiency_score": 8.0,
  "keywords": ["关键词1", "关键词2"],
  "highlights": ["亮点1", "亮点2"],
  "suggestions": ["建议1"]
}}"""


class InsightAgent:
    """
    洞察Agent - 并行阶段的节点之一

    架构说明:
    1. 规则引擎：计算发言统计（无需LLM，确定性计算）
    2. LLM 分析：情绪/关键词/亮点/建议
    3. 综合评分：结合规则和LLM结果

    设计要点：发言统计与效率评分走确定性规则，可复现、可单测；
    情绪 / 关键词 / 亮点 / 建议这类语义判断才交给 LLM。
    """

    def __init__(self, llm_client: MiniMaxClient | None = None):
        self.llm = llm_client or MiniMaxClient()

    async def process(self, state: dict) -> dict:
        """
        LangGraph 节点函数 —— 分析会议洞察

        与 Summary Agent、Action Agent 并行执行。
        """
        meeting_id = state.get("meeting_id", "unknown")
        logger.info(f"[InsightAgent] Processing meeting: {meeting_id}")

        transcript = state.get("transcript")
        transcript_text = state.get("transcript_text", "")
        errors_delta: list[str] = []

        if not transcript_text:
            logger.warning("[InsightAgent] No transcript text available")
            state["insights"] = MeetingInsight(meeting_id=meeting_id)
            return {"insights": state["insights"]}

        try:
            # Step 1: 规则引擎计算发言统计
            speaker_stats = self._compute_speaker_stats(transcript)

            # Step 2: LLM 分析
            llm_insights = await self._analyze_with_llm(
                transcript_text, speaker_stats
            )

            # Step 3: 合并结果
            state["insights"] = MeetingInsight(
                meeting_id=meeting_id,
                overall_sentiment=llm_insights.get(
                    "overall_sentiment", SentimentType.NEUTRAL
                ),
                sentiment_score=llm_insights.get("sentiment_score", 0.5),
                speaker_stats=speaker_stats,
                efficiency_score=self._compute_efficiency_score(
                    speaker_stats,
                    llm_insights.get("efficiency_score", 5.0),
                    transcript,
                ),
                keywords=llm_insights.get("keywords", []),
                highlights=llm_insights.get("highlights", []),
                suggestions=llm_insights.get("suggestions", []),
            )

            logger.info(
                f"[InsightAgent] Analysis complete: "
                f"sentiment={state['insights'].overall_sentiment}, "
                f"efficiency={state['insights'].efficiency_score:.1f}"
            )

        except Exception as e:
            logger.error(f"[InsightAgent] Error: {e}")
            errors_delta.append(f"InsightAgent: {str(e)}")
            speaker_stats = self._compute_speaker_stats(transcript)
            state["insights"] = MeetingInsight(
                meeting_id=meeting_id,
                speaker_stats=speaker_stats,
            )

        updates = {"insights": state["insights"]}
        if errors_delta:
            updates["errors"] = errors_delta
        return updates

    @staticmethod
    def _compute_speaker_stats(
        transcript: TranscriptResult | None,
    ) -> list[SpeakerStats]:
        """规则引擎：计算发言统计（纯确定性计算，不依赖 LLM）。"""
        if not transcript or not transcript.segments:
            return []

        stats: dict[str, dict] = defaultdict(
            lambda: {
                "duration": 0.0,
                "word_count": 0,
                "segment_count": 0,
            }
        )

        total_duration = 0.0
        for seg in transcript.segments:
            duration = seg.end - seg.start
            stats[seg.speaker]["duration"] += duration
            stats[seg.speaker]["word_count"] += len(seg.text)
            stats[seg.speaker]["segment_count"] += 1
            total_duration += duration

        result = []
        for speaker, data in stats.items():
            ratio = data["duration"] / total_duration if total_duration > 0 else 0
            result.append(
                SpeakerStats(
                    speaker=speaker,
                    speaking_duration=round(data["duration"], 1),
                    speaking_ratio=round(ratio, 3),
                    word_count=data["word_count"],
                    segment_count=data["segment_count"],
                )
            )

        result.sort(key=lambda x: x.speaking_duration, reverse=True)
        return result

    async def _analyze_with_llm(
        self,
        transcript_text: str,
        speaker_stats: list[SpeakerStats],
    ) -> dict[str, Any]:
        """调用 LLM 进行语义分析"""
        stats_text = "\n".join(
            f"- {s.speaker}: 发言{s.speaking_duration}秒, "
            f"占比{s.speaking_ratio:.1%}, 发言{s.segment_count}次"
            for s in speaker_stats
        )

        messages = [
            {"role": "system", "content": INSIGHT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": INSIGHT_USER_PROMPT.format(
                    transcript=transcript_text,
                    speaker_stats=stats_text,
                ),
            },
        ]

        result = await self.llm.chat_json(
            messages=messages,
            temperature=0.3,
            max_tokens=2048,
        )

        sentiment_str = result.get("overall_sentiment", "neutral").lower()
        try:
            sentiment = SentimentType(sentiment_str)
        except ValueError:
            sentiment = SentimentType.NEUTRAL

        result["overall_sentiment"] = sentiment
        return result

    @staticmethod
    def _compute_efficiency_score(
        speaker_stats: list[SpeakerStats],
        llm_score: float,
        transcript: TranscriptResult | None,
    ) -> float:
        """
        综合效率评分算法

        公式: score = 0.4 * llm_score + 0.3 * 均衡度分 + 0.3 * 时间利用率分
        - 均衡度：基尼系数越低（发言越均衡），分数越高
        - 时间利用率：有效发言时间 / 总时长
        """
        if not speaker_stats:
            return llm_score

        # 发言均衡度评分（基于基尼系数的简化版）
        ratios = [s.speaking_ratio for s in speaker_stats]
        n = len(ratios)
        if n > 1:
            mean_ratio = sum(ratios) / n
            gini = sum(
                abs(ratios[i] - ratios[j])
                for i in range(n)
                for j in range(n)
            ) / (2 * n * n * mean_ratio) if mean_ratio > 0 else 0
            balance_score = (1 - gini) * 10
        else:
            balance_score = 5.0

        # 时间利用率
        if transcript and transcript.duration_seconds > 0:
            total_speaking = sum(s.speaking_duration for s in speaker_stats)
            utilization = min(total_speaking / transcript.duration_seconds, 1.0)
            utilization_score = utilization * 10
        else:
            utilization_score = 5.0

        final = 0.4 * llm_score + 0.3 * balance_score + 0.3 * utilization_score
        return round(min(max(final, 0), 10), 1)
