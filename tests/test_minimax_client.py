"""LLM 客户端测试：JSON 容错解析与错误映射。"""

from __future__ import annotations

import json

import httpx
import pytest

from src.integrations.minimax_client import MiniMaxClient

# tenacity 装饰后的 chat 会重试 3 次（带退避）；单次行为通过 __wrapped__ 直接验证
CHAT_ONCE = getattr(MiniMaxClient.chat, "__wrapped__", MiniMaxClient.chat)


def _client(handler) -> MiniMaxClient:
    client = MiniMaxClient(api_key="test-key", group_id="test-group")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


async def _messages():
    return [{"role": "user", "content": "hi"}]


async def test_chat_json_extracts_json_from_prose(monkeypatch) -> None:
    client = MiniMaxClient(api_key="k")

    async def fake_chat(messages, **kwargs):
        return "好的，结果如下：\n{\"title\": \"会议\"}\n以上。"

    monkeypatch.setattr(client, "chat", fake_chat)

    assert await client.chat_json([{"role": "user", "content": "hi"}]) == {
        "title": "会议"
    }


async def test_chat_json_raises_on_non_json(monkeypatch) -> None:
    client = MiniMaxClient(api_key="k")

    async def fake_chat(messages, **kwargs):
        return "抱歉，我无法回答。"

    monkeypatch.setattr(client, "chat", fake_chat)

    with pytest.raises(json.JSONDecodeError):
        await client.chat_json([{"role": "user", "content": "hi"}])


async def test_chat_returns_content_on_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "base_resp": {"status_code": 0},
                "choices": [{"message": {"content": "你好"}}],
            },
        )

    client = _client(handler)
    assert await CHAT_ONCE(client, await _messages()) == "你好"
    await client.close()


async def test_chat_maps_api_error_status_to_exception() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"base_resp": {"status_code": 2013, "status_msg": "invalid params"}},
        )

    client = _client(handler)
    with pytest.raises(ValueError, match="2013"):
        await CHAT_ONCE(client, await _messages())
    await client.close()


async def test_chat_raises_on_unexpected_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"whatever": True})

    client = _client(handler)
    with pytest.raises(ValueError, match="Unexpected API response"):
        await CHAT_ONCE(client, await _messages())
    await client.close()


async def test_chat_sends_group_id_in_query() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(
            200, json={"base_resp": {"status_code": 0}, "choices": [{"message": {"content": "ok"}}]}
        )

    client = _client(handler)
    await CHAT_ONCE(client, await _messages())

    assert "GroupId=test-group" in seen["url"]
    assert seen["url"].startswith(MiniMaxClient.BASE_URL)
    await client.close()
