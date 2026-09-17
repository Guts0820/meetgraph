"""MCP HTTP/SSE 传输与 stdio 传输测试。

HTTP 部分用 TestClient 覆盖 POST /mcp 与 SSE 会话装配；stdio 部分真的把
MCP Server 作为子进程拉起来跑一遍握手与工具调用——「Claude Desktop 能接上」
这件事必须由真实进程验证，不能只测内部函数。
"""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from src.mcp.client import McpStdioClient, text_content
from src.websocket import server as server_module


@pytest.fixture
def client(offline_env) -> TestClient:
    with TestClient(server_module.app) as c:
        yield c


INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "pytest-client", "version": "1.0"},
    },
}


# ----------------------------------------------------------------------
# HTTP（POST /mcp）
# ----------------------------------------------------------------------

def test_http_initialize_and_tools_list(client: TestClient) -> None:
    response = client.post("/mcp", json=INITIALIZE)
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == 1
    assert body["result"]["serverInfo"]["name"] == "meetgraph"

    tools = client.post("/mcp", json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"}).json()
    names = {tool["name"] for tool in tools["result"]["tools"]}
    assert "search_meetings" in names
    assert "create_action_item" not in names  # 写工具默认不暴露


def test_http_notification_returns_202(client: TestClient) -> None:
    response = client.post(
        "/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"}
    )
    assert response.status_code == 202


def test_http_tool_call_unknown_tool(client: TestClient) -> None:
    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "nope", "arguments": {}},
        },
    )
    assert response.json()["error"]["code"] == -32602


def test_http_tool_call_denied_write(client: TestClient) -> None:
    """写工具在 HTTP 传输上同样受策略约束。"""
    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "create_action_item",
                "arguments": {"task_assignee": "李明", "task": "写方案"},
            },
        },
    )
    result = response.json()["result"]
    assert result["isError"] is True
    assert "MCP_ALLOW_WRITE" in result["content"][0]["text"]


def test_http_glossary_tool_round_trip(client: TestClient) -> None:
    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {"name": "lookup_glossary", "arguments": {"term": "KQI"}},
        },
    )
    payload = response.json()["result"]
    assert payload["isError"] is False
    assert "关键质量指标" in text_content(payload)


def test_mcp_info_reports_policy(client: TestClient) -> None:
    body = client.get("/mcp/info").json()

    assert body["allow_write"] is False
    assert body["server"]["name"] == "meetgraph"
    assert "search_meetings" in [t["name"] for t in body["tools"]]
    assert body["available_tools"] and "create_action_item" not in body["available_tools"]


# ----------------------------------------------------------------------
# SSE 会话
# ----------------------------------------------------------------------
# 说明：SSE 的**集成**测试依赖客户端实现（httpx ASGITransport 在流未关闭时
# 不支持并发请求，TestClient 关闭流又不会取消服务端生成器），因此这里直接
# 单测事件流生成器：帧格式、心跳、断开清理都是纯逻辑，确定且秒级。

async def test_sse_event_stream_frames_heartbeat_and_cleanup(offline_env) -> None:
    from src.websocket.server import _mcp_sessions, sse_event_stream

    queue: asyncio.Queue = asyncio.Queue()
    queue.put_nowait({"jsonrpc": "2.0", "id": 7, "result": {}})
    _mcp_sessions["s-1"] = queue

    checks = {"n": 0}

    async def is_disconnected() -> bool:
        checks["n"] += 1
        return checks["n"] > 3  # 第 4 次心跳检查时客户端断开

    frames: list[str] = []
    async for chunk in sse_event_stream("s-1", queue, is_disconnected, keepalive=0.01):
        frames.append(chunk)

    assert frames[0] == "event: endpoint\ndata: /mcp/messages?session_id=s-1\n\n"
    assert any(
        frame.startswith("event: message") and '"id": 7' in frame for frame in frames
    )
    assert any(": keep-alive" in frame for frame in frames)
    assert "s-1" not in _mcp_sessions  # 断开后必须清理


def test_sse_frame_without_event_name() -> None:
    from src.websocket.server import sse_frame

    assert sse_frame(None, "x") == "data: x\n\n"
    assert sse_frame("message", "y") == "event: message\ndata: y\n\n"


def test_sse_messages_unknown_session_404(client: TestClient) -> None:
    response = client.post(
        "/mcp/messages",
        params={"session_id": "does-not-exist"},
        json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
    )
    assert response.status_code == 404


def test_sse_messages_pushes_response_into_session(client: TestClient) -> None:
    """请求经 /mcp/messages 处理后，响应进入对应会话队列。"""
    queue: asyncio.Queue = asyncio.Queue()
    server_module._mcp_sessions["test-session"] = queue
    try:
        response = client.post(
            "/mcp/messages",
            params={"session_id": "test-session"},
            json={"jsonrpc": "2.0", "id": 9, "method": "tools/list"},
        )
        assert response.status_code == 202

        pushed = queue.get_nowait()
        assert pushed["id"] == 9
        assert {t["name"] for t in pushed["result"]["tools"]} >= {"search_meetings"}
    finally:
        server_module._mcp_sessions.pop("test-session", None)


# ----------------------------------------------------------------------
# stdio（真实子进程）
# ----------------------------------------------------------------------

async def test_stdio_end_to_end(offline_env) -> None:
    """真起子进程：握手 → 工具发现 → 工具调用 → 资源与提示 → 错误路径 → ping。"""
    from src.mcp.client import McpProtocolError

    async with McpStdioClient() as client:
        info = await client.initialize()
        assert info["serverInfo"]["name"] == "meetgraph"
        assert set(info["capabilities"]) == {"tools", "resources", "prompts"}

        tools = await client.list_tools()
        assert {t["name"] for t in tools} >= {"search_meetings", "lookup_glossary"}

        called = await client.call_tool("lookup_glossary", {"term": "MRD"})
        assert called["isError"] is False
        assert "市场需求文档" in text_content(called)

        denied = await client.call_tool(
            "create_action_item", {"task_assignee": "李明", "task": "写方案"}
        )
        assert denied["isError"] is True

        # 参数校验失败走 JSON-RPC 错误层（客户端调用姿势不对，不是工具执行失败）
        with pytest.raises(McpProtocolError) as invalid_args:
            await client.call_tool("lookup_glossary", {})
        assert invalid_args.value.code == -32602

        resources = await client.list_resources()
        assert isinstance(resources, list)

        prompts = await client.list_prompts()
        assert prompts[0]["name"] == "summarize_meeting"

        assert await client.ping() == {}


async def test_stdio_unknown_method_returns_error(offline_env) -> None:
    async with McpStdioClient() as client:
        await client.initialize()
        with pytest.raises(Exception) as excinfo:
            await client._call("tools/teleport")  # noqa: SLF001 - 故意打错误方法
        assert "method not supported" in str(excinfo.value)


def test_stdio_client_framing_helpers() -> None:
    """编码后的报文必须是单行 JSON，否则 stdio 传输会错帧。"""
    from src.mcp.protocol import encode_message

    encoded = encode_message({"jsonrpc": "2.0", "id": 1, "result": {"a": "b\nc"}})
    assert encoded.endswith("\n") and encoded.count("\n") == 1
    assert json.loads(encoded)["result"]["a"] == "b\nc"
