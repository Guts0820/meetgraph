"""Jira Cloud 集成客户端 —— 自动创建和跟踪待办事项。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

# 默认的人员映射文件：会议里的显示名 → Jira 账号
DEFAULT_USER_MAP_FILE = Path(__file__).resolve().parents[2] / "config" / "jira_users.json"


def _normalize(name: Any) -> str:
    """归一化显示名：压缩空白 + 统一小写，避免「张总 / 张总 」算两个人。"""
    return " ".join(str(name or "").split()).lower()


def load_user_mapping(path: str | Path | None = None) -> dict[str, str]:
    """加载「参会人显示名 → Jira 账号」映射。

    优先级：环境变量 ``JIRA_USER_MAP``（内联 JSON）> ``JIRA_USER_MAP_FILE``
    > 仓库内 ``config/jira_users.json``。

    配置形式（值可以是账号字符串，也可以是带别名的对象）::

        {"张总": "zhang.zong", "李明": {"account": "li.ming", "aliases": ["李工"]}}

    文件缺失或格式错误只告警、不抛异常：此时所有待办都不指派负责人，宁可
    建出无负责人的单，也不要因为一个配置把 Pipeline 打断。
    """
    raw: dict[str, Any] = {}

    inline = os.getenv("JIRA_USER_MAP", "").strip()
    if inline:
        try:
            raw = json.loads(inline)
        except json.JSONDecodeError as e:
            logger.warning(f"JIRA_USER_MAP is not valid JSON, ignored: {e}")

    if not raw:
        map_file = Path(
            path or os.getenv("JIRA_USER_MAP_FILE") or DEFAULT_USER_MAP_FILE
        )
        if map_file.exists():
            try:
                raw = json.loads(map_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load user mapping {map_file}: {e}")

    mapping: dict[str, str] = {}
    for name, value in raw.items():
        if isinstance(value, dict):
            account = str(value.get("account", "")).strip()
            aliases = value.get("aliases") or []
        else:
            account = str(value).strip()
            aliases = []
        if not account:
            continue
        for alias in [name, *aliases]:
            mapping[_normalize(alias)] = account

    if mapping:
        logger.info(f"Loaded {len(mapping)} Jira user mapping(s)")
    else:
        logger.warning(
            "No Jira user mapping configured; action items will be created unassigned"
        )
    return mapping


class JiraClient:
    """
    Jira Cloud REST API 客户端

    职责:
    - 创建 Issue（从会议待办自动同步）
    - 查询 Issue 状态（用于跟踪待办完成情况）
    - 更新 Issue（添加评论等）

    API 文档: https://developer.atlassian.com/cloud/jira/platform/rest/v3/
    """

    def __init__(
        self,
        server: str | None = None,
        email: str | None = None,
        api_token: str | None = None,
        project_key: str = "MEET",
        user_mapping: dict[str, str] | None = None,
    ):
        self.server = server or os.getenv("JIRA_SERVER", "")
        self.email = email or os.getenv("JIRA_EMAIL", "")
        self.api_token = api_token or os.getenv("JIRA_API_TOKEN", "")
        self.project_key = project_key or os.getenv("JIRA_PROJECT_KEY", "MEET")
        self._jira = None
        self._enabled = bool(self.server and self.email and self.api_token)
        # 显示名 → Jira 账号；key 已做归一化处理
        self._users = (
            {_normalize(k): v for k, v in user_mapping.items()}
            if user_mapping is not None
            else load_user_mapping()
        )

    def _get_client(self):
        """懒加载 Jira 客户端"""
        if self._jira is None and self._enabled:
            from jira import JIRA
            self._jira = JIRA(
                server=self.server,
                basic_auth=(self.email, self.api_token),
            )
        return self._jira

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def create_issue(
        self,
        summary: str,
        description: str = "",
        assignee: str | None = None,
        due_date: str | None = None,
        priority: str = "Medium",
        issue_type: str = "Task",
        labels: list[str] | None = None,
    ) -> dict[str, str]:
        """
        创建 Jira Issue

        Args:
            summary: 任务标题
            description: 任务描述
            assignee: 负责人（Jira 用户名或邮箱）
            due_date: 截止日期 YYYY-MM-DD
            priority: 优先级 Low/Medium/High/Urgent
            issue_type: Issue 类型 Task/Bug/Story
            labels: 标签列表

        Returns:
            {"key": "MEET-42", "id": "10042", "url": "https://..."}
        """
        if not self._enabled:
            logger.warning("Jira integration not configured, skipping")
            return {"key": "DISABLED", "id": "", "url": ""}

        client = self._get_client()

        fields: dict[str, Any] = {
            "project": {"key": self.project_key},
            "summary": summary,
            "description": description or f"自动创建自会议助手系统\n\n{summary}",
            "issuetype": {"name": issue_type},
            "priority": {"name": priority},
        }

        if assignee:
            fields["assignee"] = {"name": assignee}
        if due_date:
            fields["duedate"] = due_date
        if labels:
            fields["labels"] = labels + ["meeting-auto"]
        else:
            fields["labels"] = ["meeting-auto"]

        issue = client.create_issue(fields=fields)
        result = {
            "key": issue.key,
            "id": str(issue.id),
            "url": f"{self.server}/browse/{issue.key}",
        }
        logger.info(f"Created Jira issue: {result['key']} - {summary}")
        return result

    def get_issue_status(self, issue_key: str) -> str:
        """查询 Issue 当前状态"""
        if not self._enabled:
            return "DISABLED"
        client = self._get_client()
        issue = client.issue(issue_key)
        return str(issue.fields.status)

    def add_comment(self, issue_key: str, comment: str) -> None:
        """为 Issue 添加评论"""
        if not self._enabled:
            return
        client = self._get_client()
        client.add_comment(issue_key, comment)
        logger.info(f"Added comment to {issue_key}")

    def resolve_user(self, display_name: str) -> str | None:
        """把会议里出现的显示名解析为 Jira 账号。

        支持精确命中与去空白/大小写不敏感命中（映射文件里可以给同一个人配
        多个别名）。解析不到时返回 None：建单时不指派负责人，而不是猜一个
        可能错的人。
        """
        account = self._users.get(_normalize(display_name))
        if account is None:
            logger.warning(
                f"No Jira account mapped for {display_name!r}; "
                f"issue will be created unassigned"
            )
        return account

    @staticmethod
    def map_priority(priority: str) -> str:
        """将系统优先级映射为 Jira 优先级"""
        mapping = {
            "low": "Low",
            "medium": "Medium",
            "high": "High",
            "urgent": "Highest",
        }
        return mapping.get(priority.lower(), "Medium")
