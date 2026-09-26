#!/usr/bin/env python
"""MCP 互通性验证：用 Anthropic 官方 `mcp` SDK 作为客户端连本项目的 MCP Server。

为什么单独写一个脚本，而不是只靠 `tests/` 里的自研客户端：

- 自研客户端与自研服务端「互相迁就」，测不出协议实现偏差。真实案例：服务端在
  Windows 管道下用 locale 编码（GBK）写 stdout，自研客户端恰好设了
  ``PYTHONIOENCODING=utf-8`` 把缺陷掩盖了，换成官方 SDK 立刻 ``UnicodeDecodeError``。
- 官方 SDK 还会做严格校验（协议版本、字段名），是「符合标准」这句话的现实检验。

用法::

    pip install mcp                     # 仅开发/验证用，不在 requirements.txt 里
    python scripts/mcp_interop_check.py

退出码 0 表示全部通过；非 0 会打印完整异常树。
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError:  # pragma: no cover - 只是缺开发依赖
    print("缺少开发依赖：pip install mcp")
    raise SystemExit(2)

SERVER = StdioServerParameters(
    command=sys.executable,
    args=["-m", "src.mcp.server"],
    cwd=str(REPO_ROOT),
)


def _field(obj: object, *names: str):
    """兼容 SDK 的 camelCase / snake_case 字段命名。"""
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def dump(e: BaseException, indent: int = 0) -> None:
    print(" " * indent + f"{type(e).__name__}: {e!r}"[:400], flush=True)
    for sub in getattr(e, "exceptions", None) or []:
        dump(sub, indent + 2)


async def run() -> None:
    async with stdio_client(SERVER) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            print("initialize  OK  protocolVersion =", _field(init, "protocol_version", "protocolVersion"))
            info = _field(init, "server_info", "serverInfo")
            print("serverInfo  OK ", info.name, info.version)
            print("capabilities   ", sorted(init.capabilities.model_dump(exclude_none=True)))

            tools = await session.list_tools()
            print("tools/list  OK ", sorted(t.name for t in tools.tools))

            result = await session.call_tool("lookup_glossary", {"term": "MRD"})
            is_error = _field(result, "is_error", "isError")
            text = result.content[0].text if result.content else ""
            print("tools/call  OK  isError =", is_error)
            print("           中文回包 =", text[:60].replace("\n", " "))
            assert is_error is False, "tools/call 不应报错"
            assert "市场需求文档" in text, "UTF-8 中文回包内容异常"

            prompts = await session.list_prompts()
            print("prompts/list OK", [p.name for p in prompts.prompts])


if __name__ == "__main__":
    try:
        asyncio.run(asyncio.wait_for(run(), timeout=60))
    except BaseException as e:  # noqa: BLE001 - 验证脚本要把所有异常类型都摊开
        print("互通性验证失败：", flush=True)
        dump(e)
        sys.stdout.flush()
        os._exit(1)
    print("官方 SDK 互通性验证通过")
    sys.stdout.flush()
    os._exit(0)
