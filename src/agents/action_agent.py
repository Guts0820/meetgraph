"""Action Agent（待办 Agent）。

职责：
1. 从转写文本中用 LLM 抽取行动项（谁 / 做什么 / 截止时间 / 优先级）
2. 把行动项同步到 Jira Cloud 与飞书任务
3. 通过 SyncLedger 保证同步幂等：同一个会议重复触发不会重复建单

失败处理：单个目标的同步失败不中断 Pipeline，失败原因聚合进
``state["errors"]``，同时记入 ``sync_status`` 便于排查。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from loguru import logger

from ..integrations.feishu_client import FeishuClient
from ..integrations.idempotency import SyncLedger
from ..integrations.jira_client import JiraClient
from ..integrations.minimax_client import MiniMaxClient
from ..models.schemas import ActionItem, ActionResult, Priority


ACTION_SYSTEM_PROMPT = """你是一位专业的任务提取助手。你的任务是从会议转写文本中提取所有行动项/待办事项。

提取规则：
1. 识别明确分配给某人的任务
2. 提取任务的截止时间（如果提到的话）
3. 判断任务优先级（根据语气和上下文）
4. 记录任务的上下文（为什么要做这件事）

注意：
- 只提取明确的行动项，不要凭空创造
- 截止时间格式为 YYYY-MM-DD
- 如果没有明确截止时间，留空

你必须严格按照JSON格式输出："""

ACTION_USER_PROMPT = """请从以下会议转写文本中提取所有行动项/待办事项。

今天的日期是: {today}

## 会议转写文本
{transcript}

## 输出格式（严格JSON）
{{
  "action_items": [
    {{
      "assignee": "负责人姓名",
      "task": "具体任务描述",
      "deadline": "YYYY-MM-DD 或空字符串",
      "priority": "low/medium/high/urgent",
      "context": "这个任务的背景说明"
    }}
  ]
}}"""


class ActionAgent:
    """待办 Agent —— 并行阶段的节点之一。

    幂等设计：同步前用 SyncLedger 以 (meeting_id, assignee, task) 的业务键
    查表，命中则复用已有的 Jira issue key / 飞书 task id，不再调用外部 API。
    """

    def __init__(
        self,
        llm_client: MiniMaxClient | None = None,
        jira_client: JiraClient | None = None,
        feishu_client: FeishuClient | None = None,
        ledger: SyncLedger | None = None,
    ):
        self.llm = llm_client or MiniMaxClient()
        self.jira = jira_client or JiraClient()
        self.feishu = feishu_client or FeishuClient()
        # 不显式传入时不建台账：让「只抽取、不同步」的场景（单元测试、离线
        # 评测）不去碰磁盘。
        self.ledger = ledger

    async def process(self, state: dict) -> dict:
        """LangGraph 节点函数 —— 提取待办并同步。"""
        meeting_id = state.get("meeting_id", "unknown")
        logger.info(f"[ActionAgent] Processing meeting: {meeting_id}")

        transcript_text = state.get("transcript_text", "")
        errors_delta: list[str] = []

        if not transcript_text:
            logger.warning("[ActionAgent] No transcript text available")
            state["actions"] = ActionResult(meeting_id=meeting_id, action_items=[])
            return {"actions": state["actions"]}

        try:
            action_items = await self._extract_actions(transcript_text)
            synced_items, sync_stats = await self._sync_to_external(
                action_items, meeting_id
            )

            state["actions"] = ActionResult(
                meeting_id=meeting_id,
                action_items=synced_items,
                sync_status=sync_stats["status"],
                duplicates_skipped=sync_stats["skipped"],
            )
            errors_delta.extend(sync_stats["errors"])

            logger.info(
                f"[ActionAgent] Extracted {len(synced_items)} action items, "
                f"created={sync_stats['created']}, "
                f"skipped={sync_stats['skipped']}, "
                f"failed={sync_stats['failed']}"
            )
        except Exception as e:
            logger.error(f"[ActionAgent] Error: {e}")
            errors_delta.append(f"ActionAgent: {e}")
            state["actions"] = ActionResult(meeting_id=meeting_id, action_items=[])

        updates: dict[str, Any] = {"actions": state["actions"]}
        if errors_delta:
            updates["errors"] = errors_delta
        return updates

    async def _extract_actions(self, transcript: str) -> list[ActionItem]:
        """调用 LLM 提取行动项。"""
        today = datetime.now().strftime("%Y-%m-%d")
        messages = [
            {"role": "system", "content": ACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": ACTION_USER_PROMPT.format(
                    today=today, transcript=transcript
                ),
            },
        ]

        result = await self.llm.chat_json(
            messages=messages,
            temperature=0.2,
            max_tokens=2048,
        )

        items = []
        for raw in result.get("action_items", []):
            priority_str = str(raw.get("priority", "medium")).lower()
            try:
                priority = Priority(priority_str)
            except ValueError:
                priority = Priority.MEDIUM

            items.append(
                ActionItem(
                    assignee=raw.get("assignee", "未指定"),
                    task=raw.get("task", ""),
                    deadline=self._normalize_deadline(raw.get("deadline", "")),
                    priority=priority,
                    context=raw.get("context", ""),
                )
            )

        return items

    @staticmethod
    def _normalize_deadline(value: Any) -> str:
        """只保留 YYYY-MM-DD 形式的截止时间，其余一律留空。

        Jira 对 duedate 格式敏感，宁可丢掉「下周五」这类模糊表达，也不要让
        一次格式错误把整条同步链打断。
        """
        text = str(value or "").strip()
        if not text:
            return ""
        try:
            return datetime.strptime(text[:10], "%Y-%m-%d").strftime("%Y-%m-%d")
        except ValueError:
            logger.warning(f"Drop unparsable deadline: {text!r}")
            return ""

    async def _sync_to_external(
        self, items: list[ActionItem], meeting_id: str
    ) -> tuple[list[ActionItem], dict[str, Any]]:
        """把行动项同步到 Jira 和飞书，返回（结果列表, 统计信息）。"""
        counters = {"created": 0, "skipped": 0, "failed": 0}
        errors: list[str] = []
        synced: list[ActionItem] = []

        for item in items:
            key = (
                SyncLedger.item_key(meeting_id, item.assignee, item.task)
                if self.ledger
                else ""
            )

            # ---------------- Jira ----------------
            if self.jira.is_enabled:
                existing = self.ledger.get(key, "jira") if self.ledger else None
                if existing:
                    item.jira_issue_key = existing
                    counters["skipped"] += 1
                    logger.info(f"Jira sync skipped (already synced): {existing}")
                else:
                    try:
                        jira_result = self.jira.create_issue(
                            summary=f"[会议待办] {item.task}",
                            description=(
                                f"来源：会议 {meeting_id}\n"
                                f"负责人：{item.assignee}\n"
                                f"上下文：{item.context}"
                            ),
                            assignee=self.jira.resolve_user(item.assignee),
                            due_date=item.deadline or None,
                            priority=JiraClient.map_priority(item.priority.value),
                            labels=["meeting-auto", f"meeting-{meeting_id}"],
                        )
                        item.jira_issue_key = jira_result["key"]
                        counters["created"] += 1
                        if self.ledger:
                            self.ledger.record(
                                key, "jira", jira_result["key"], meeting_id
                            )
                    except Exception as e:
                        counters["failed"] += 1
                        errors.append(f"ActionAgent(jira): {item.task} - {e}")
                        logger.error(f"Failed to sync to Jira: {item.task} - {e}")

            # ---------------- 飞书 ----------------
            if self.feishu.is_enabled:
                existing = self.ledger.get(key, "feishu") if self.ledger else None
                if existing:
                    item.feishu_task_id = existing
                    counters["skipped"] += 1
                    logger.info(f"Feishu sync skipped (already synced): {existing}")
                else:
                    try:
                        due_ts = None
                        if item.deadline:
                            due_dt = datetime.strptime(item.deadline, "%Y-%m-%d")
                            due_ts = int(due_dt.timestamp())

                        feishu_result = await self.feishu.create_task(
                            summary=f"[会议待办] {item.task}",
                            description=(
                                f"负责人：{item.assignee}\n"
                                f"来源会议：{meeting_id}\n"
                                f"上下文：{item.context}"
                            ),
                            due_timestamp=due_ts,
                        )
                        item.feishu_task_id = feishu_result.get("task_id")
                        counters["created"] += 1
                        if self.ledger and item.feishu_task_id:
                            self.ledger.record(
                                key, "feishu", item.feishu_task_id, meeting_id
                            )
                    except Exception as e:
                        counters["failed"] += 1
                        errors.append(f"ActionAgent(feishu): {item.task} - {e}")
                        logger.error(f"Failed to sync to Feishu: {item.task} - {e}")

            synced.append(item)

        detail = (
            f"created={counters['created']},"
            f"skipped={counters['skipped']},"
            f"failed={counters['failed']}"
        )
        stats: dict[str, Any] = {
            **counters,
            "errors": errors,
            "status": {
                "jira": ("enabled" if self.jira.is_enabled else "disabled")
                + ","
                + detail,
                "feishu": ("enabled" if self.feishu.is_enabled else "disabled")
                + ","
                + detail,
            },
        }
        return synced, stats
