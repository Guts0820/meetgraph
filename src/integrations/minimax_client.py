"""MiniMax LLM 客户端 - 支持 M2.7 模型调用"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


class MiniMaxClient:
    """
    MiniMax API 客户端

    支持 MiniMax M2.7 模型，兼容 OpenAI 接口格式。
    文档: https://platform.minimaxi.com/document/guides/chat-model/chat/api
    """

    BASE_URL = "https://api.minimax.chat/v1"

    def __init__(
        self,
        api_key: str | None = None,
        group_id: str | None = None,
        model: str = "abab6.5s-chat",
    ):
        self.api_key = api_key or os.getenv("MINIMAX_API_KEY", "")
        self.group_id = group_id or os.getenv("MINIMAX_GROUP_ID", "")
        self.model = model
        self._client = httpx.AsyncClient(
            timeout=60.0,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        reraise=True,
    )
    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: dict | None = None,
    ) -> str:
        """
        调用 MiniMax 聊天接口

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            temperature: 温度参数
            max_tokens: 最大生成token数
            response_format: 输出格式约束 (如 {"type": "json_object"})

        Returns:
            模型生成的文本
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # 注意: MiniMax chatcompletion_v2 不识别 OpenAI 风格的
        # response_format={"type": "json_object"} (返回 2013 错误)，
        # 因此不随请求发送；JSON 解析由 chat_json 的容错提取逻辑兜底。
        _ = response_format

        url = f"{self.BASE_URL}/text/chatcompletion_v2"
        if self.group_id:
            url = f"{url}?GroupId={self.group_id}"

        response = await self._client.post(url, json=payload)
        response.raise_for_status()

        data = response.json()

        base_resp = data.get("base_resp") or {}
        status_code = base_resp.get("status_code")
        if status_code not in (None, 0):
            logger.error(f"MiniMax API error: {base_resp}")
            raise ValueError(
                f"MiniMax API error ({status_code}): "
                f"{base_resp.get('status_msg', 'unknown')}"
            )

        choices = data.get("choices") or []
        if choices:
            return data["choices"][0]["message"]["content"]

        logger.error(f"MiniMax API unexpected response: {data}")
        raise ValueError(f"Unexpected API response: {data}")

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> dict:
        """调用聊天接口并解析 JSON 输出"""
        text = await self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
            raise

    async def close(self):
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
