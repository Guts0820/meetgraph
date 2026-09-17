"""MCP 服务端：协议分发 + stdio 传输。

可直接作为子进程启动（Claude Desktop / Cursor 就是这么接的）::

    python -m src.mcp.server

或 ``python scripts/mcp_server.py --transport stdio``。

分发的方法（MCP 里客户端会按需调用）：

| 方法 | 是否需要响应 | 说明 |
|------|--------------|------|
| ``initialize`` | ✅ | 版本协商 + 能力声明 |
| ``notifications/initialized`` | ❌ | 客户端就绪通知 |
| ``ping`` | ✅ | 连通性 |
| ``tools/list`` / ``tools/call`` | ✅ | 工具发现与调用 |
| ``resources/list`` / ``resources/read`` | ✅ | 会议报告资源 |
| ``prompts/list`` / ``prompts/get`` | ✅ | 提示模板 |

约定：**stdout 只输出协议报文**，日志一律走 stderr（loguru 默认），否则客户端会
因收到非协议内容而断连。
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any, IO

from loguru import logger

from .catalog import get_prompt, list_prompts, list_resources, read_resource
from .protocol import (
    INTERNAL_ERROR,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PROTOCOL_VERSION,
    SUPPORTED_PROTOCOL_VERSIONS,
    JsonRpcError,
    McpServerInfo,
    encode_message,
    make_error,
    make_result,
    parse_message,
)
from .registry import ToolRegistry
from .tools import build_default_registry


class McpServer:
    """协议服务端（与传输方式解耦：stdio 与 HTTP 都调 :meth:`handle`）。"""

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        info: McpServerInfo | None = None,
    ) -> None:
        self.registry = registry or build_default_registry()
        self.info = info or McpServerInfo()
        self.client_info: dict[str, Any] = {}
        self.negotiated_protocol: str = PROTOCOL_VERSION

    # ------------------------------------------------------------------
    async def handle(self, message: str | bytes | dict[str, Any]) -> dict[str, Any] | None:
        """处理一条报文，返回要发回的响应（通知返回 None）。"""
        request_id: Any = None
        try:
            request = parse_message(message)
            request_id = request.id

            if request.is_notification:
                await self._handle_notification(request.method, request.params)
                return None

            result = await self._dispatch(request.method, request.params)
            return make_result(request.id, result)
        except JsonRpcError as e:
            return make_error(request_id, e.code, e.message, e.data)
        except Exception as e:  # 兜底：协议层不能因为业务异常而崩
            logger.exception(f"[MCP] internal error handling message: {e}")
            return make_error(request_id, INTERNAL_ERROR, f"{type(e).__name__}: {e}")

    async def _handle_notification(self, method: str, params: dict[str, Any]) -> None:
        if method == "notifications/initialized":
            logger.debug("[MCP] client initialized notification received")
        elif method == "notifications/cancelled":
            logger.debug("[MCP] client cancelled a request")
        else:
            logger.debug(f"[MCP] ignoring unknown notification: {method}")

    async def _dispatch(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "initialize":
            return self._initialize(params)
        if method == "ping":
            return {}
        if method == "tools/list":
            return self.registry.tools_list_payload()
        if method == "tools/call":
            return await self._tools_call(params)
        if method == "resources/list":
            return list_resources()
        if method == "resources/read":
            return read_resource(params.get("uri", ""))
        if method == "prompts/list":
            return list_prompts()
        if method == "prompts/get":
            return get_prompt(params.get("name", ""), params.get("arguments"))

        raise JsonRpcError(
            METHOD_NOT_FOUND,
            f"method not supported: {method}",
            {
                "supported": [
                    "initialize",
                    "ping",
                    "tools/list",
                    "tools/call",
                    "resources/list",
                    "resources/read",
                    "prompts/list",
                    "prompts/get",
                ]
            },
        )

    # ------------------------------------------------------------------
    def _initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        """版本协商：客户端给版本，服务端在支持列表里选，否则回自己的默认版本。"""
        client_version = str(params.get("protocolVersion", "")).strip()
        self.client_info = dict(params.get("clientInfo") or {})
        self.negotiated_protocol = (
            client_version
            if client_version in SUPPORTED_PROTOCOL_VERSIONS
            else PROTOCOL_VERSION
        )
        logger.info(
            f"[MCP] initialize from {self.client_info.get('name', 'unknown')} "
            f"client_version={client_version or 'n/a'} "
            f"negotiated={self.negotiated_protocol}"
        )
        return {
            "protocolVersion": self.negotiated_protocol,
            "capabilities": self.info.capabilities(),
            "serverInfo": self.info.to_payload(),
            "instructions": (
                "MeetGraph 会议助手：可检索历史会议、读取会议报告、查询公司术语；"
                "写操作（建单）默认关闭。所有工具参数以 JSON Schema 为准。"
            ),
        }

    async def _tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        if not isinstance(name, str) or not name:
            raise JsonRpcError(INVALID_REQUEST, "tools/call requires a tool name")

        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise JsonRpcError(INVALID_REQUEST, "arguments must be an object")

        result = await self.registry.call(name, arguments, actor="mcp-client")
        return result.mcp_payload()

    # ------------------------------------------------------------------
    async def serve_stdio(
        self, stdin: IO[str] | None = None, stdout: IO[str] | None = None
    ) -> None:
        """stdio 传输：一行一条 JSON-RPC 报文，读 EOF 退出。"""
        stdin = stdin or sys.stdin
        stdout = stdout or sys.stdout
        logger.info(f"[MCP] stdio server started ({self.info.name} {self.info.version})")

        while True:
            line = await asyncio.to_thread(stdin.readline)
            if not line:  # EOF：客户端断开
                break
            if not line.strip():
                continue

            response = await self.handle(line)
            if response is not None:
                stdout.write(encode_message(response))
                stdout.flush()

        logger.info("[MCP] stdio server stopped (EOF)")


async def main() -> int:
    server = McpServer()
    try:
        await server.serve_stdio()
    except (KeyboardInterrupt, asyncio.CancelledError):  # pragma: no cover
        logger.info("[MCP] interrupted")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
