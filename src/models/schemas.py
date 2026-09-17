"""
数据模型定义 —— 整个 Pipeline 共享的 Schema。

用 pydantic v2 定义，是 5 个 Agent 之间唯一的契约：
- 转写: TranscriptResult / TranscriptSegment
- 纪要: MeetingSummary / TopicSummary
- 待办: ActionItem / ActionResult / Priority
- 洞察: MeetingInsight / SpeakerStats / SentimentType
- 跟进: FollowUpResult

WebSocket 推送与 REST 返回统一用 model_dump() 序列化；MeetingState 是
LangGraph 节点读写用的 TypedDict 视图。
"""

from __future__ import annotations

from enum import Enum
from typing import TypedDict

from pydantic import BaseModel, ConfigDict, Field


class MeetingStatus(str, Enum):
    """会议处理状态"""

    CREATED = "created"
    RECORDING = "recording"
    PROCESSING = "processing"
    TRANSCRIBING = "transcribing"
    COMPLETED = "completed"
    FAILED = "failed"


class Priority(str, Enum):
    """行动项优先级"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class SentimentType(str, Enum):
    """会议整体情绪"""

    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class TranscriptSegment(BaseModel):
    """单条转写片段 - 包含说话人和时间戳"""

    speaker: str = "Unknown"
    text: str = ""
    start: float = 0.0
    end: float = 0.0
    confidence: float = 0.0


class TranscriptResult(BaseModel):
    """一次会议的完整转写结果"""

    meeting_id: str
    segments: list[TranscriptSegment] = Field(default_factory=list)
    language: str = "zh"
    duration_seconds: float = 0.0
    full_text: str = ""


class TopicSummary(BaseModel):
    """单个议题摘要"""

    title: str = ""
    discussion_points: list[str] = Field(default_factory=list)
    participants: list[str] = Field(default_factory=list)
    conclusion: str = ""


class MeetingSummary(BaseModel):
    """结构化会议纪要"""

    title: str = ""
    date: str = ""
    participants: list[str] = Field(default_factory=list)
    topics: list[TopicSummary] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)


class ActionItem(BaseModel):
    """单条行动项/待办事项"""

    assignee: str = "未指定"
    task: str = ""
    deadline: str = ""
    priority: Priority = Priority.MEDIUM
    context: str = ""
    jira_issue_key: str | None = None
    feishu_task_id: str | None = None


class ActionResult(BaseModel):
    """行动项提取 + 外部同步结果"""

    # 字段拼错时直接报错，而不是被 pydantic 静默丢弃
    model_config = ConfigDict(extra="forbid")

    meeting_id: str
    action_items: list[ActionItem] = Field(default_factory=list)
    sync_status: dict[str, str] = Field(default_factory=dict)
    # 命中幂等台账、跳过外部调用的次数
    duplicates_skipped: int = 0


class SpeakerStats(BaseModel):
    """单个说话人的发言统计"""

    speaker: str = ""
    speaking_duration: float = 0.0
    speaking_ratio: float = 0.0
    word_count: int = 0
    segment_count: int = 0


class MeetingInsight(BaseModel):
    """会议洞察分析结果"""

    meeting_id: str
    overall_sentiment: SentimentType = SentimentType.NEUTRAL
    sentiment_score: float = 0.5
    speaker_stats: list[SpeakerStats] = Field(default_factory=list)
    efficiency_score: float = 0.0
    keywords: list[str] = Field(default_factory=list)
    highlights: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class FollowUpResult(BaseModel):
    """跟进执行结果"""

    model_config = ConfigDict(extra="forbid")

    meeting_id: str
    summary_sent: bool = False
    recipients: list[str] = Field(default_factory=list)
    jira_issues_created: list[str] = Field(default_factory=list)
    feishu_tasks_created: list[str] = Field(default_factory=list)
    reminders_scheduled: int = 0
    report_url: str = ""
    # 通过幂等台账跳过（未重复调用外部 API）的同步次数
    duplicates_skipped: int = 0


class MeetingState(TypedDict, total=False):
    """整个 Pipeline 共享的会议状态（与 meeting_graph.GraphState 保持一致）"""

    meeting_id: str
    status: str
    audio_data: bytes
    transcript: TranscriptResult
    transcript_text: str
    summary: MeetingSummary
    actions: ActionResult
    insights: MeetingInsight
    followup: FollowUpResult
    errors: list[str]


def create_initial_state(
    meeting_id: str, audio_data: bytes = b""
) -> dict:
    """创建 Pipeline 的初始状态字典"""
    return {
        "meeting_id": meeting_id,
        "status": MeetingStatus.CREATED.value,
        "audio_data": audio_data,
        "errors": [],
    }
