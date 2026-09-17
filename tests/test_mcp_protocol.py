"""MCP 协议层测试：报文解析、错误码、方法分发、版本协商。"""

from __future__ import annotations

import json

import pytest

from src.mcp.protocol import (
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    PROTOCOL_VERSION,
    JsonRpcError,
    encode_message,
    make_error,
    make_notification,
    make_result,
    parse_message,
)
from src.mcp.server import McpServer
from tests.fakes import FakeToolLLM  # noqa: F401  (保持 fakes 被导入以便共用)


# ----------------------------------------------------------------------
# 报文
# ----------------------------------------------------------------------

def test_parse_request_and_notification() -> None:
    request = parse_message('{"jsonrpc":"2.0","id":1,"method":"tools/list"}')
    assert request.method == "tools/list"
    assert request.id == 1
    assert request.is_notification is False

    notification = parse_message('{"jsonrpc":"2.0","method":"notifications/initialized"}')
    assert notification.is_notification is True


def test_parse_rejects_broken_payloads() -> None:
    with pytest.raises(JsonRpcError) as bad_json:
        parse_message("{not json")
    assert bad_json.value.code == PARSE_ERROR

    with pytest.raises(JsonRpcError) as no_version:
        parse_message('{"id":1,"method":"ping"}')
    assert no_version.value.code == INVALID_REQUEST

    with pytest.raises(JsonRpcError) as no_method:
        parse_message('{"jsonrpc":"2.0","id":1}')
    assert no_method.value.code == INVALID_REQUEST

    with pytest.raises(JsonRpcError) as bad_params:
        parse_message('{"jsonrpc":"2.0","id":1,"method":"ping","params":[1,2]}')
    assert bad_params.value.code == INVALID_PARAMS


def test_encode_is_single_line() -> None:
    payload = make_result(1, {"text": "第一行\n第二行"})
    encoded = encode_message(payload)

    assert encoded.count("\n") == 1 and encoded.endswith("\n")
    assert json.loads(encoded)["result"]["text"] == "第一行\n第二行"


def test_builders_shape() -> None:
    assert make_result(7, {"ok": True}) == {"jsonrpc": "2.0", "id": 7, "result": {"ok": True}}
    assert make_error(7, -32601, "nope")["error"] == {"code": -32601, "message": "nope"}
    assert make_notification("x", {"a": 1})["method"] == "x"


def test_json_rpc_error_payload_with_data() -> None:
    error = JsonRpcError(INVALID_PARAMS, "bad", {"field": "query"})
    assert error.to_payload() == {
        "code": INVALID_PARAMS,
        "message": "bad",
        "data": {"field": "query"},
    }


# ----------------------------------------------------------------------
# 服务端分发
# ----------------------------------------------------------------------

async def _handle(payload):
    return await McpServer().handle(payload)


async def test_initialize_negotiates_protocol_version() -> None:
    response = await _handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "unit-test", "version": "0"},
            },
        }
    )

    result = response["result"]
    assert result["protocolVersion"] == PROTOCOL_VERSION
    assert set(result["capabilities"]) == {"tools", "resources", "prompts"}
    assert result["serverInfo"]["name"] == "meetgraph"
    assert "instructions" in result


async def test_initialize_falls_back_for_unknown_client_version() -> None:
    response = await _handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "1999-01-01"},
        }
    )

    assert response["result"]["protocolVersion"] == PROTOCOL_VERSION


async def test_notification_gets_no_response() -> None:
    assert await _handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


async def test_ping_and_unknown_method() -> None:
    assert (await _handle({"jsonrpc": "2.0", "id": 2, "method": "ping"}))["result"] == {}

    unknown = await _handle({"jsonrpc": "2.0", "id": 3, "method": "tools/teleport"})
    assert unknown["error"]["code"] == METHOD_NOT_FOUND
    assert "tools/call" in unknown["error"]["data"]["supported"]


async def test_broken_message_returns_error_not_exception() -> None:
    response = await _handle("这不是 JSON")
    assert response["error"]["code"] == PARSE_ERROR
    assert response["id"] is None


async def test_tools_list_shape(offline_env) -> None:
    response = await _handle({"jsonrpc": "2.0", "id": 4, "method": "tools/list"})
    tools = response["result"]["tools"]

    names = {tool["name"] for tool in tools}
    # 默认只暴露只读工具（写工具需要 MCP_ALLOW_WRITE=1）
    assert names == {"search_meetings", "get_meeting_report", "lookup_glossary"}
    for tool in tools:
        assert tool["inputSchema"]["type"] == "object"
        assert tool["description"]


async def test_resources_and_prompts_dispatch(offline_env) -> None:
    listed = await _handle({"jsonrpc": "2.0", "id": 5, "method": "resources/list"})
    assert listed["result"] == {"resources": []}

    prompts = await _handle({"jsonrpc": "2.0", "id": 6, "method": "prompts/list"})
    assert prompts["result"]["prompts"][0]["name"] == "summarize_meeting"

    rendered = await _handle(
        {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "prompts/get",
            "params": {"name": "summarize_meeting", "arguments": {"meeting_id": "m-1"}},
        }
    )
    assert "meeting://report/m-1" in rendered["result"]["messages"][0]["content"]["text"]

    missing_arg = await _handle(
        {"jsonrpc": "2.0", "id": 8, "method": "prompts/get", "params": {"name": "summarize_meeting"}}
    )
    assert missing_arg["error"]["code"] == INVALID_PARAMS


async def test_unknown_tool_is_invalid_params_not_crash(offline_env) -> None:
    response = await _handle(
        {
            "jsonrpc": "2.0",
            "id": 9,
            "method": "tools/call",
            "params": {"name": "make_coffee", "arguments": {}},
        }
    )
    assert response["error"]["code"] == INVALID_PARAMS
    assert "search_meetings" in response["error"]["data"]["available"]


async def test_tools_call_requires_name_and_object_arguments(offline_env) -> None:
    no_name = await _handle(
        {"jsonrpc": "2.0", "id": 10, "method": "tools/call", "params": {"arguments": {}}}
    )
    assert no_name["error"]["code"] == INVALID_REQUEST

    bad_args = await _handle(
        {
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "params": {"name": "lookup_glossary", "arguments": "DT"},
        }
    )
    assert bad_args["error"]["code"] == INVALID_REQUEST


async def test_notification_bearing_request_id_is_answered() -> None:
    """带 id 的报文必须回响应，不带 id 的才当通知。"""
    response = await _handle({"jsonrpc": "2.0", "id": 0, "method": "ping"})
    assert response["id"] == 0
