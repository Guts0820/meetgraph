"""飞书 Open API 集成客户端 - 消息推送和任务管理"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


class FeishuClient:
    """
    飞书开放平台 API 客户端

    职责:
    - 发送群消息（推送会议纪要）
    - 创建任务（同步待办事项）
    - Webhook 机器人消息

    API 文档: https://open.feishu.cn/document/home/index
    """

    BASE_URL = "https://open.feishu.cn/open-apis"

    def __init__(
        self,
        app_id: str | None = None,
        app_secret: str | None = None,
        webhook_url: str | None = None,
    ):
        self.app_id = app_id or os.getenv("FEISHU_APP_ID", "")
        self.app_secret = app_secret or os.getenv("FEISHU_APP_SECRET", "")
        self.webhook_url = webhook_url or os.getenv("FEISHU_WEBHOOK_URL", "")
        self._client = httpx.AsyncClient(timeout=30.0)
        self._tenant_token: str = ""
        self._token_expires_at: float = 0
        self._enabled = bool(
            (self.app_id and self.app_secret) or self.webhook_url
        )

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    async def _get_tenant_token(self) -> str:
        """获取 tenant_access_token（自动缓存和刷新）"""
        if self._tenant_token and time.time() < self._token_expires_at:
            return self._tenant_token

        if not (self.app_id and self.app_secret):
            return ""

        resp = await self._client.post(
            f"{self.BASE_URL}/auth/v3/tenant_access_token/internal",
            json={"app_id": self.app_id, "app_secret": self.app_secret},
        )
        data = resp.json()
        self._tenant_token = data.get("tenant_access_token", "")
        self._token_expires_at = time.time() + data.get("expire", 7200) - 300
        return self._tenant_token

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def send_webhook_message(
        self,
        title: str,
        content: str,
        msg_type: str = "interactive",
    ) -> bool:
        """
        通过 Webhook 机器人发送消息到飞书群

        Args:
            title: 消息标题
            content: 消息内容（Markdown格式）
            msg_type: 消息类型

        Returns:
            是否发送成功
        """
        if not self.webhook_url:
            logger.warning("Feishu webhook not configured, skipping")
            return False

        card = {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {"tag": "plain_text", "content": title},
                    "template": "blue",
                },
                "elements": [
                    {
                        "tag": "markdown",
                        "content": content,
                    }
                ],
            },
        }

        resp = await self._client.post(self.webhook_url, json=card)
        data = resp.json()
        success = data.get("code", -1) == 0
        if success:
            logger.info(f"Feishu webhook message sent: {title}")
        else:
            logger.error(f"Feishu webhook failed: {data}")
        return success

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def send_message(
        self,
        receive_id: str,
        content: str,
        receive_id_type: str = "chat_id",
        msg_type: str = "text",
    ) -> dict[str, Any]:
        """
        通过 API 发送消息

        Args:
            receive_id: 接收者ID（群ID或用户ID）
            content: 消息内容
            receive_id_type: ID类型 chat_id/open_id/user_id/email
            msg_type: 消息类型 text/interactive/post
        """
        token = await self._get_tenant_token()
        if not token:
            return {"success": False, "error": "No token"}

        resp = await self._client.post(
            f"{self.BASE_URL}/im/v1/messages",
            params={"receive_id_type": receive_id_type},
            headers={"Authorization": f"Bearer {token}"},
            json={
                "receive_id": receive_id,
                "msg_type": msg_type,
                "content": content if isinstance(content, str) else str(content),
            },
        )
        return resp.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def create_task(
        self,
        summary: str,
        description: str = "",
        due_timestamp: int | None = None,
        assignee_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        创建飞书任务

        Args:
            summary: 任务标题
            description: 任务描述
            due_timestamp: 截止时间戳（秒）
            assignee_ids: 负责人open_id列表
        """
        token = await self._get_tenant_token()
        if not token:
            return {"success": False, "error": "No token"}

        task_body: dict[str, Any] = {
            "summary": summary,
            "description": description or f"来源：会议助手自动创建",
        }
        if due_timestamp:
            task_body["due"] = {"timestamp": str(due_timestamp), "is_all_day": True}

        resp = await self._client.post(
            f"{self.BASE_URL}/task/v2/tasks",
            headers={"Authorization": f"Bearer {token}"},
            json=task_body,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("code", -1) != 0:
            raise RuntimeError(f"Feishu task creation failed: {data}")

        task = data.get("data", {}).get("task", {})
        # 飞书 Task V2 返回 task_id（部分版本返回 guid），而非通用的 id。
        task_id = task.get("task_id") or task.get("guid") or task.get("id", "")
        if not task_id:
            raise RuntimeError(
                f"Feishu returned success without a task identifier: {data}"
            )

        logger.info(f"Feishu task created: {task_id} - {summary}")
        return {"task_id": task_id, "data": data}


    async def send_meeting_summary(
        self,
        title: str,
        summary_md: str,
        action_items_md: str,
        insights_md: str,
    ) -> bool:
        """发送完整的会议纪要卡片消息"""
        content = (
            f"**会议主题**: {title}\n\n"
            f"---\n\n"
            f"**📋 会议纪要**\n{summary_md}\n\n"
            f"---\n\n"
            f"**✅ 待办事项**\n{action_items_md}\n\n"
            f"---\n\n"
            f"**📊 会议洞察**\n{insights_md}"
        )
        return await self.send_webhook_message(
            title=f"📝 会议纪要 | {title}",
            content=content,
        )

    async def close(self):
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
