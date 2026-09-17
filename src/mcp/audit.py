"""审计日志：每次工具调用落一行 JSONL。

刻意记录**参数摘要**（canonical JSON 的 SHA-256）而不是明文参数：审计要能回答
「谁在什么时候调了什么、成功没有」，但不应该把会议内容、人名、邮箱再抄一份到
日志里。需要复核参数时用摘要去比对调用方自己的记录。
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

DEFAULT_AUDIT_PATH = Path(__file__).resolve().parents[2] / "data" / "mcp-audit.jsonl"


def args_digest(arguments: dict[str, Any] | None) -> str:
    """参数摘要：键排序后的 canonical JSON → SHA-256。"""
    canonical = json.dumps(arguments or {}, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


@dataclass
class AuditRecord:
    ts: float
    tool: str
    actor: str
    status: str  # ok / error / denied
    duration_ms: float
    args_digest: str
    error: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AuditLog:
    """JSONL 审计日志（进程内串行写入）。"""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or os.getenv("MCP_AUDIT_LOG") or DEFAULT_AUDIT_PATH)

    def append(
        self,
        tool: str,
        actor: str = "mcp",
        status: str = "ok",
        duration_ms: float = 0.0,
        arguments: dict[str, Any] | None = None,
        error: str = "",
        **extra: Any,
    ) -> AuditRecord:
        record = AuditRecord(
            ts=round(time.time(), 3),
            tool=tool,
            actor=actor,
            status=status,
            duration_ms=round(duration_ms, 2),
            args_digest=args_digest(arguments),
            error=error[:500],
            extra=extra,
        )
        self._write(record)
        return record

    def _write(self, record: AuditRecord) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(record.to_dict(), ensure_ascii=False) + "\n"
                )
        except OSError as e:  # 审计失败不能影响工具调用本身
            logger.warning(f"Failed to write MCP audit log: {e}")

    def tail(self, limit: int = 20) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines[-limit:] if line.strip()]

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()
