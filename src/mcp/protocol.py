"""MCP 协议内核：JSON-RPC 2.0 报文、错误码、协议版本与能力声明。

MCP 的报文就是 JSON-RPC 2.0，stdio 传输用**换行分隔**（每条消息一行，内部
不允许出现裸换行）。这里把「解析 / 构造 / 错误码」集中在一处，服务端与客户端
共用，避免两边对协议的理解漂移。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# 协议版本：客户端在 initialize 里给出自己支持的版本，服务端回自己支持的版本
PROTOCOL_VERSION = "2024-11-05"
SUPPORTED_PROTOCOL_VERSIONS = ("2024-11-05", "2025-03-26")

SERVER_NAME = "meetgraph"
SERVER_VERSION = "2.1.0"

# JSON-RPC 2.0 标准错误码 + MCP 常用
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class JsonRpcError(Exception):
    """带 JSON-RPC 错误码的异常，服务端会把它翻译成 error 报文。"""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            payload["data"] = self.data
        return payload


@dataclass
class McpServerInfo:
    """服务端身份与能力声明。"""

    name: str = SERVER_NAME
    version: str = SERVER_VERSION
    protocol_version: str = PROTOCOL_VERSION
    tools: bool = True
    resources: bool = True
    prompts: bool = True
    extra_capabilities: dict[str, Any] = field(default_factory=dict)

    def capabilities(self) -> dict[str, Any]:
        caps: dict[str, Any] = {}
        if self.tools:
            caps["tools"] = {"listChanged": False}
        if self.resources:
            caps["resources"] = {"subscribe": False, "listChanged": False}
        if self.prompts:
            caps["prompts"] = {"listChanged": False}
        caps.update(self.extra_capabilities)
        return caps

    def to_payload(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version}


@dataclass
class Request:
    """一条入站报文（请求或通知）。"""

    method: str
    params: dict[str, Any]
    id: Any = None
    is_notification: bool = False


def make_result(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def make_error(
    request_id: Any, code: int, message: str, data: Any = None
) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}


def make_notification(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if params:
        payload["params"] = params
    return payload


def parse_message(raw: str | bytes | dict[str, Any]) -> Request:
    """解析入站报文；格式错误抛 :class:`JsonRpcError`。"""
    if isinstance(raw, dict):
        payload = raw
    else:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        text = text.strip()
        if not text:
            raise JsonRpcError(INVALID_REQUEST, "empty message")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as e:
            raise JsonRpcError(PARSE_ERROR, f"invalid JSON: {e}") from e

    if not isinstance(payload, dict) or payload.get("jsonrpc") != "2.0":
        raise JsonRpcError(INVALID_REQUEST, "not a JSON-RPC 2.0 message")

    method = payload.get("method")
    if not isinstance(method, str) or not method:
        raise JsonRpcError(INVALID_REQUEST, "missing method")

    params = payload.get("params") or {}
    if not isinstance(params, dict):
        raise JsonRpcError(INVALID_PARAMS, "params must be an object")

    has_id = "id" in payload
    return Request(
        method=method,
        params=params,
        id=payload.get("id"),
        is_notification=not has_id,
    )


def encode_message(payload: dict[str, Any]) -> str:
    """编码为一行 JSON（stdio 传输要求单行）。"""
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return text.replace("\n", " ") + "\n"
