"""工具权限策略。

默认姿态是**最小权限**：

- 只读工具（检索、查报告、查术语）默认可用；
- 写工具（建单/建任务）默认**关闭**，必须显式设置 ``MCP_ALLOW_WRITE=1``；
- 可用 ``MCP_TOOL_ALLOWLIST`` 进一步把可用工具收窄成白名单（逗号分隔）。

这样即使 MCP Server 被接到别的客户端上，也不会有人无意中拿它去批量建单。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


class ToolDenied(Exception):
    """工具被策略拒绝（不是参数问题，也不是执行失败）。"""


@dataclass
class ToolPolicy:
    allow_write: bool = False
    allowlist: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_env(cls) -> "ToolPolicy":
        allowlist = tuple(
            item.strip()
            for item in os.getenv("MCP_TOOL_ALLOWLIST", "").split(",")
            if item.strip()
        )
        return cls(
            allow_write=os.getenv("MCP_ALLOW_WRITE", "0").strip() in ("1", "true", "True"),
            allowlist=allowlist,
        )

    def allows(self, name: str, readonly: bool) -> bool:
        if self.allowlist and name not in self.allowlist:
            return False
        return bool(readonly or self.allow_write)

    def check(self, name: str, readonly: bool) -> None:
        if self.allows(name, readonly):
            return
        reason = (
            "tool not in allowlist"
            if self.allowlist and name not in self.allowlist
            else "write tools are disabled (set MCP_ALLOW_WRITE=1 to enable)"
        )
        raise ToolDenied(f"{name}: {reason}")
