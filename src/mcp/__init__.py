"""MCP（Model Context Protocol）实现。

对外入口：

- ``src.mcp.server``：协议服务端（stdio 传输，``python -m src.mcp.server`` 启动）
- ``src.mcp.tools``：业务工具（会议检索 / 报告 / 建单 / 术语查询）
- ``src.mcp.registry``：工具注册表（一份 Schema，MCP 与 LLM 两处消费）
- ``src.mcp.policy`` / ``src.mcp.audit``：权限与审计

不依赖官方 SDK：协议只有 7 个方法，自己实现便于讲清版本协商、传输与错误码，
且测试可以完全离线。
"""

from .audit import AuditLog
from .policy import ToolPolicy
from .protocol import PROTOCOL_VERSION, JsonRpcError, McpServerInfo
from .registry import ToolRegistry, ToolSpec
from .tools import build_default_registry

__all__ = [
    "AuditLog",
    "JsonRpcError",
    "McpServerInfo",
    "PROTOCOL_VERSION",
    "ToolPolicy",
    "ToolRegistry",
    "ToolSpec",
    "build_default_registry",
]
