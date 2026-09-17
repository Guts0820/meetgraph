"""最小 MCP 客户端：两种形态。

- :class:`McpInProcessClient`：直接把报文喂给 ``McpServer.handle``，用于协议层
  单测（不起进程、不联网，秒级）；
- :class:`McpStdioClient`：真的把服务端作为子进程拉起来，按行收发 JSON-RPC，
  用于端到端冒烟——「Claude Desktop 能接上」这件事必须靠真实进程验证。

客户端只实现工具/资源/提示所需的握手与调用，够用即止（不想造一个 SDK）。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from .protocol import (
    PROTOCOL_VERSION,
    JsonRpcError,
    encode_message,
    parse_message,
)
from .server import McpServer

REPO_ROOT = Path(__file__).resolve().parents[2]


class McpProtocolError(RuntimeError):
    """服务端返回了 JSON-RPC 错误或不可用的响应。"""

    def __init__(self, payload: dict[str, Any]) -> None:
        error = payload.get("error", {})
        super().__init__(f"{error.get('code')}: {error.get('message')}")
        self.payload = payload
        self.code = error.get("code")


class _RequestMixin:
    """请求构造与响应解析（两种传输共用）。"""

    def __init__(self) -> None:
        self._next_id = 0
        self.server_info: dict[str, Any] = {}
        self.capabilities: dict[str, Any] = {}
        self.protocol_version: str = ""
        self.client_name = "meetgraph-test-client"

    def _request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._next_id += 1
        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": self._next_id,
            "method": method,
        }
        if params is not None:
            payload["params"] = params
        return payload

    @staticmethod
    def _unwrap(response: dict[str, Any] | None) -> dict[str, Any]:
        if response is None:
            raise McpProtocolError({"error": {"code": -1, "message": "no response"}})
        if "error" in response:
            raise McpProtocolError(response)
        return response.get("result") or {}

    def _store_initialize(self, result: dict[str, Any]) -> dict[str, Any]:
        self.protocol_version = result.get("protocolVersion", "")
        self.capabilities = result.get("capabilities", {})
        self.server_info = result.get("serverInfo", {})
        return result


class McpInProcessClient(_RequestMixin):
    """进程内客户端（协议单测用）。"""

    def __init__(self, server: McpServer | None = None) -> None:
        super().__init__()
        self.server = server or McpServer()

    async def initialize(self) -> dict[str, Any]:
        result = self._unwrap(
            await self.server.handle(
                self._request(
                    "initialize",
                    {
                        "protocolVersion": PROTOCOL_VERSION,
                        "capabilities": {},
                        "clientInfo": {"name": self.client_name, "version": "0.1.0"},
                    },
                )
            )
        )
        await self.server.handle(
            {"jsonrpc": "2.0", "method": "notifications/initialized"}
        )
        return self._store_initialize(result)

    async def list_tools(self) -> list[dict[str, Any]]:
        return self._unwrap(await self.server.handle(self._request("tools/list"))).get(
            "tools", []
        )

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._unwrap(
            await self.server.handle(
                self._request("tools/call", {"name": name, "arguments": arguments or {}})
            )
        )

    async def list_resources(self) -> list[dict[str, Any]]:
        return self._unwrap(
            await self.server.handle(self._request("resources/list"))
        ).get("resources", [])

    async def read_resource(self, uri: str) -> dict[str, Any]:
        return self._unwrap(
            await self.server.handle(self._request("resources/read", {"uri": uri}))
        )

    async def list_prompts(self) -> list[dict[str, Any]]:
        return self._unwrap(await self.server.handle(self._request("prompts/list"))).get(
            "prompts", []
        )

    async def get_prompt(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._unwrap(
            await self.server.handle(
                self._request("prompts/get", {"name": name, "arguments": arguments or {}})
            )
        )

    async def close(self) -> None:
        return None


class McpStdioClient(_RequestMixin):
    """把服务端作为子进程启动的真客户端。"""

    def __init__(
        self,
        command: list[str] | None = None,
        cwd: Path | str | None = None,
        timeout: float = 60.0,
    ) -> None:
        super().__init__()
        self.command = command or [sys.executable, "-m", "src.mcp.server"]
        self.cwd = str(cwd or REPO_ROOT)
        self.timeout = timeout
        self._process: asyncio.subprocess.Process | None = None

    async def __aenter__(self) -> "McpStdioClient":
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def start(self) -> None:
        env = dict(os.environ)
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONPATH", self.cwd)
        self._process = await asyncio.create_subprocess_exec(
            *self.command,
            cwd=self.cwd,
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,  # 日志走 stderr，客户端不读
        )

    # ------------------------------------------------------------------
    async def _send(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        assert self._process and self._process.stdin and self._process.stdout
        self._process.stdin.write(encode_message(payload).encode("utf-8"))
        await self._process.stdin.drain()

        line = await asyncio.wait_for(
            self._process.stdout.readline(), timeout=self.timeout
        )
        if not line:
            raise McpProtocolError(
                {"error": {"code": -1, "message": "server closed the connection"}}
            )
        return json.loads(line.decode("utf-8"))

    async def _call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        response = await self._send(self._request(method, params))
        return self._unwrap(response)

    async def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        assert self._process and self._process.stdin
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        self._process.stdin.write(encode_message(payload).encode("utf-8"))
        await self._process.stdin.drain()

    # ------------------------------------------------------------------
    async def initialize(self) -> dict[str, Any]:
        result = await self._call(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": self.client_name, "version": "0.1.0"},
            },
        )
        await self._notify("notifications/initialized")
        return self._store_initialize(result)

    async def list_tools(self) -> list[dict[str, Any]]:
        return (await self._call("tools/list")).get("tools", [])

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        return await self._call("tools/call", {"name": name, "arguments": arguments or {}})

    async def list_resources(self) -> list[dict[str, Any]]:
        return (await self._call("resources/list")).get("resources", [])

    async def read_resource(self, uri: str) -> dict[str, Any]:
        return await self._call("resources/read", {"uri": uri})

    async def list_prompts(self) -> list[dict[str, Any]]:
        return (await self._call("prompts/list")).get("prompts", [])

    async def get_prompt(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        return await self._call("prompts/get", {"name": name, "arguments": arguments or {}})

    async def ping(self) -> dict[str, Any]:
        return await self._call("ping")

    async def close(self) -> None:
        if self._process and self._process.returncode is None:
            try:
                if self._process.stdin:
                    self._process.stdin.close()
                await asyncio.wait_for(self._process.wait(), timeout=10)
            except (asyncio.TimeoutError, ProcessLookupError):  # pragma: no cover
                self._process.kill()
        self._process = None


def text_content(payload: dict[str, Any]) -> str:
    """从 tools/call 响应里取出文本内容（空则返回空串）。"""
    blocks = payload.get("content") or []
    texts = [b.get("text", "") for b in blocks if isinstance(b, dict)]
    return "\n".join(t for t in texts if t)


__all__ = [
    "McpInProcessClient",
    "McpProtocolError",
    "McpStdioClient",
    "JsonRpcError",
    "text_content",
]
