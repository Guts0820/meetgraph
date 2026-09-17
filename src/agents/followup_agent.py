"""
Follow-up Agent（跟进Agent）
- 汇聚 Summary + Action + Insight 三个Agent的结果
- 生成并发送会议纪要到飞书群
- 确认所有待办已同步
- 设置跟踪提醒
"""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import re
from typing import Any

from loguru import logger

from ..integrations.feishu_client import FeishuClient
from ..models.schemas import (
    ActionResult,
    FollowUpResult,
    MeetingInsight,
    MeetingStatus,
    MeetingSummary,
)


class FollowUpAgent:
    """
    跟进Agent - Pipeline的最后一个节点（Fan-in汇聚）

    架构说明:
    1. 等待 Summary/Action/Insight 三个并行Agent全部完成
    2. 汇聚结果，生成完整的会议报告
    3. 推送到飞书群
    4. 检查待办同步状态
    5. 设置定时提醒

    面试考点:
    - Fan-in 汇聚是如何实现的？（LangGraph 多条边汇聚到同一节点）
    - 如果某个并行Agent失败了怎么办？（部分降级，跳过失败部分继续）
    - 提醒机制怎么实现？（APScheduler定时任务 / 消息队列延迟消息）
    """

    def __init__(self, feishu_client: FeishuClient | None = None):
        self.feishu = feishu_client or FeishuClient()

    async def process(self, state: dict) -> dict:
        """
        LangGraph 节点函数 —— 会后跟进

        这是 Pipeline 的最后一个节点，汇聚所有并行结果。
        """
        meeting_id = state.get("meeting_id", "unknown")
        logger.info(f"[FollowUpAgent] Processing meeting: {meeting_id}")

        summary: MeetingSummary | None = state.get("summary")
        actions: ActionResult | None = state.get("actions")
        insights: MeetingInsight | None = state.get("insights")

        result = FollowUpResult(meeting_id=meeting_id)
        errors_delta: list[str] = []

        try:
            # Step 1: 生成会议纪要 Markdown
            summary_md = self._format_summary_markdown(summary)
            actions_md = self._format_actions_markdown(actions)
            insights_md = self._format_insights_markdown(insights)

            # Step 2: 发送到飞书
            if self.feishu.is_enabled:
                title = summary.title if summary else f"会议 {meeting_id}"
                sent = await self.feishu.send_meeting_summary(
                    title=title,
                    summary_md=summary_md,
                    action_items_md=actions_md,
                    insights_md=insights_md,
                )
                result.summary_sent = sent
                if summary:
                    result.recipients = summary.participants

            # Step 3: 统计同步结果
            if actions:
                result.jira_issues_created = [
                    item.jira_issue_key
                    for item in actions.action_items
                    if item.jira_issue_key
                ]
                result.feishu_tasks_created = [
                    item.feishu_task_id
                    for item in actions.action_items
                    if item.feishu_task_id
                ]
                result.duplicates_skipped = actions.duplicates_skipped

            # Step 4: 设置提醒（简化实现）
            if actions:
                reminders = sum(
                    1 for item in actions.action_items if item.deadline
                )
                result.reminders_scheduled = reminders

            # Step 5: 生成报告
            result.report_url = self._generate_report(
                meeting_id, summary_md, actions_md, insights_md
            )

            state["followup"] = result
            state["status"] = MeetingStatus.COMPLETED

            logger.info(
                f"[FollowUpAgent] Follow-up complete: "
                f"sent={result.summary_sent}, "
                f"jira_issues={len(result.jira_issues_created)}, "
                f"feishu_tasks={len(result.feishu_tasks_created)}, "
                f"reminders={result.reminders_scheduled}"
            )

        except Exception as e:
            logger.error(f"[FollowUpAgent] Error: {e}")
            errors_delta.append(f"FollowUpAgent: {str(e)}")
            state["followup"] = result
            state["status"] = MeetingStatus.COMPLETED

        updates = {
            "followup": state["followup"],
            "status": state["status"],
        }
        if errors_delta:
            updates["errors"] = errors_delta
        return updates

    @staticmethod
    def _format_summary_markdown(summary: MeetingSummary | None) -> str:
        """将会议摘要格式化为 Markdown"""
        if not summary:
            return "*（摘要生成失败）*"

        lines = [f"## {summary.title}\n"]

        if summary.participants:
            lines.append(f"**参会人**: {', '.join(summary.participants)}\n")

        for i, topic in enumerate(summary.topics, 1):
            lines.append(f"### 议题{i}: {topic.title}")
            for point in topic.discussion_points:
                lines.append(f"- {point}")
            if topic.conclusion:
                lines.append(f"- **结论**: {topic.conclusion}")
            lines.append("")

        if summary.decisions:
            lines.append("### 会议决策")
            for d in summary.decisions:
                lines.append(f"- {d}")
            lines.append("")

        if summary.next_steps:
            lines.append("### 下一步计划")
            for s in summary.next_steps:
                lines.append(f"- {s}")

        return "\n".join(lines)

    @staticmethod
    def _format_actions_markdown(actions: ActionResult | None) -> str:
        """将待办事项格式化为 Markdown"""
        if not actions or not actions.action_items:
            return "*（无待办事项）*"

        lines = []
        for i, item in enumerate(actions.action_items, 1):
            status_parts = []
            if item.jira_issue_key:
                status_parts.append(f"Jira: {item.jira_issue_key}")
            if item.feishu_task_id:
                status_parts.append(f"飞书: {item.feishu_task_id}")
            status = f" ({', '.join(status_parts)})" if status_parts else ""

            deadline_str = f" | 截止: {item.deadline}" if item.deadline else ""
            lines.append(
                f"{i}. **{item.assignee}**: {item.task}"
                f"{deadline_str} [{item.priority.value}]{status}"
            )

        return "\n".join(lines)

    @staticmethod
    def _format_insights_markdown(insights: MeetingInsight | None) -> str:
        """将会议洞察格式化为 Markdown"""
        if not insights:
            return "*（洞察分析失败）*"

        lines = [
            f"**整体氛围**: {insights.overall_sentiment.value} "
            f"(得分: {insights.sentiment_score:.2f})",
            f"**效率评分**: {insights.efficiency_score}/10",
            "",
        ]

        if insights.speaker_stats:
            lines.append("**发言统计**:")
            for s in insights.speaker_stats:
                bar = "█" * int(s.speaking_ratio * 20)
                lines.append(
                    f"- {s.speaker}: {s.speaking_ratio:.1%} {bar} "
                    f"({s.speaking_duration}s, {s.segment_count}次)"
                )
            lines.append("")

        if insights.keywords:
            lines.append(f"**关键词**: {', '.join(insights.keywords)}")

        if insights.highlights:
            lines.append("\n**亮点**:")
            for h in insights.highlights:
                lines.append(f"- {h}")

        if insights.suggestions:
            lines.append("\n**改进建议**:")
            for s in insights.suggestions:
                lines.append(f"- {s}")

        return "\n".join(lines)

    @staticmethod
    def _generate_report(
        meeting_id: str,
        summary_md: str,
        actions_md: str,
        insights_md: str,
    ) -> str:
        """生成完整会议报告并写入 Markdown 文件。"""
        report = (
            f"# 会议报告 - {meeting_id}\n\n"
            f"生成时间: {datetime.now().isoformat()}\n\n"
            f"---\n\n"
            f"## 会议纪要\n\n{summary_md}\n\n"
            f"---\n\n"
            f"## 待办事项\n\n{actions_md}\n\n"
            f"---\n\n"
            f"## 会议洞察\n\n{insights_md}\n"
        )
        default_dir = Path(__file__).resolve().parents[2] / "reports"
        reports_dir = Path(os.getenv("REPORTS_DIR") or default_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)

        # meeting_id 可来自 WebSocket 路径，避免其影响最终的文件路径。
        safe_meeting_id = re.sub(r"[^A-Za-z0-9_.-]", "_", meeting_id)
        report_path = reports_dir / f"meeting-report-{safe_meeting_id}.md"
        report_path.write_text(report, encoding="utf-8")

        logger.info(f"Report written for meeting {meeting_id}: {report_path}")
        return str(report_path)
