"""洞察 Agent 测试：规则统计与效率评分的确定性计算。"""

from __future__ import annotations

from src.agents.insight_agent import InsightAgent
from src.models.schemas import SentimentType, TranscriptResult, TranscriptSegment
from tests.fakes import FakeLLM


def _transcript(segments: list[tuple[str, float, float]], duration: float) -> TranscriptResult:
    return TranscriptResult(
        meeting_id="m",
        segments=[
            TranscriptSegment(speaker=s, text="内容" * 3, start=a, end=b)
            for s, a, b in segments
        ],
        duration_seconds=duration,
    )


def test_speaker_stats_are_ratio_sorted() -> None:
    transcript = _transcript(
        [("张总", 0.0, 10.0), ("李明", 10.0, 14.0), ("张总", 14.0, 20.0)], 20.0
    )

    stats = InsightAgent._compute_speaker_stats(transcript)

    assert [s.speaker for s in stats] == ["张总", "李明"]
    assert stats[0].speaking_duration == 16.0
    assert stats[0].speaking_ratio == 0.8
    assert stats[0].segment_count == 2
    assert round(sum(s.speaking_ratio for s in stats), 6) == 1.0


def test_speaker_stats_empty_transcript() -> None:
    assert InsightAgent._compute_speaker_stats(None) == []
    assert InsightAgent._compute_speaker_stats(TranscriptResult(meeting_id="m")) == []


def test_efficiency_score_single_speaker() -> None:
    """单人发言：均衡度固定 5.0，时间利用率 100%。"""
    transcript = _transcript([("张总", 0.0, 10.0)], 10.0)
    stats = InsightAgent._compute_speaker_stats(transcript)

    score = InsightAgent._compute_efficiency_score(stats, 8.0, transcript)

    # 0.4*8 + 0.3*5 + 0.3*10 = 7.7
    assert score == 7.7


def test_efficiency_score_is_bounded() -> None:
    transcript = _transcript([("A", 0.0, 5.0), ("B", 5.0, 10.0)], 10.0)
    stats = InsightAgent._compute_speaker_stats(transcript)

    assert InsightAgent._compute_efficiency_score(stats, 10.0, transcript) <= 10.0
    assert InsightAgent._compute_efficiency_score(stats, 0.0, transcript) >= 0.0


async def test_process_merges_rule_stats_and_llm_output() -> None:
    agent = InsightAgent(llm_client=FakeLLM())
    transcript = _transcript([("张总", 0.0, 10.0), ("李明", 10.0, 20.0)], 20.0)

    result = await agent.process(
        {
            "meeting_id": "m",
            "transcript": transcript,
            "transcript_text": "张总: 内容\n李明: 内容",
        }
    )

    insights = result["insights"]
    assert insights.overall_sentiment == SentimentType.POSITIVE
    assert insights.keywords == ["预算", "招聘", "采购"]
    assert len(insights.speaker_stats) == 2
    assert insights.efficiency_score > 0


async def test_process_degrades_to_rule_stats_only() -> None:
    """LLM 挂掉时仍给出统计结果，并把原因写进 errors。"""
    agent = InsightAgent(llm_client=FakeLLM(fail_times=None))
    transcript = _transcript([("张总", 0.0, 10.0)], 10.0)

    result = await agent.process(
        {
            "meeting_id": "m",
            "transcript": transcript,
            "transcript_text": "张总: 内容",
        }
    )

    assert result["insights"].speaker_stats
    assert result["insights"].keywords == []
    assert any("InsightAgent" in e for e in result["errors"])


async def test_process_without_transcript() -> None:
    result = await InsightAgent(llm_client=FakeLLM()).process(
        {"meeting_id": "m", "transcript": None, "transcript_text": ""}
    )

    assert result["insights"].speaker_stats == []
