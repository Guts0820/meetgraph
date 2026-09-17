"""外部系统写入的幂等台账（SQLite 实现）。

会议待办会同步到 Jira / 飞书。同步动作必须可重放：同一个会议里同一个人、
同一件事，无论 Pipeline 被触发多少次、重试多少次，外部系统里都只应该存在
一条记录。

做法：以 ``(meeting_id, assignee, task)`` 的 SHA-256 摘要作为业务唯一键，
把「业务键 → 外部系统 ID」写进本地 SQLite 台账；写入前先查台账，命中就复用
已有 ID，不再调用外部 API。这样即使进程重启、消息重投，也不会重复建单。
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from loguru import logger

# 默认落在仓库根目录的 data/ 下（已被 .gitignore 忽略）
DEFAULT_LEDGER_PATH = Path(__file__).resolve().parents[2] / "data" / "sync-ledger.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sync_ledger (
    item_key    TEXT NOT NULL,
    target      TEXT NOT NULL,
    external_id TEXT NOT NULL,
    meeting_id  TEXT NOT NULL,
    created_at  REAL NOT NULL,
    PRIMARY KEY (item_key, target)
);
CREATE INDEX IF NOT EXISTS idx_sync_ledger_meeting
    ON sync_ledger (meeting_id);
"""


class SyncLedger:
    """幂等台账。

    Args:
        db_path: SQLite 文件路径；默认取环境变量 ``SYNC_LEDGER_DB``，
            未配置时落在 ``<repo>/data/sync-ledger.db``。
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        raw = db_path or os.getenv("SYNC_LEDGER_DB") or DEFAULT_LEDGER_PATH
        self.db_path = Path(raw)
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # 并行阶段多个 Agent 可能同时写入，sqlite 连接本身不能跨线程共享，
        # 因此关闭线程检查并用锁串行化写操作。
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        logger.debug(f"SyncLedger ready: {self.db_path}")

    # ------------------------------------------------------------------
    # 业务键
    # ------------------------------------------------------------------

    @staticmethod
    def item_key(meeting_id: str, assignee: str, task: str) -> str:
        """生成行动项的业务唯一键。

        对内容做归一化（去首尾空白、统一小写、压缩内部空白），避免同一件事
        因为空格或大小写差异被算成两条。
        """
        def norm(value: str) -> str:
            return " ".join(str(value or "").split()).lower()

        raw = f"{norm(meeting_id)}|{norm(assignee)}|{norm(task)}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    # ------------------------------------------------------------------
    # 读写
    # ------------------------------------------------------------------

    def get(self, item_key: str, target: str) -> str | None:
        """查询某个行动项在目标系统里已存在的外部 ID，没有则返回 None。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT external_id FROM sync_ledger WHERE item_key = ? AND target = ?",
                (item_key, target),
            ).fetchone()
        return row[0] if row else None

    def record(
        self,
        item_key: str,
        target: str,
        external_id: str,
        meeting_id: str = "",
    ) -> None:
        """登记「业务键 → 外部 ID」映射；重复登记时覆盖为新 ID。"""
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO sync_ledger (item_key, target, external_id, meeting_id, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (item_key, target) DO UPDATE SET
                    external_id = excluded.external_id,
                    created_at  = excluded.created_at
                """,
                (item_key, target, external_id, meeting_id, time.time()),
            )
            self._conn.commit()
        logger.debug(f"Ledger recorded: {target}={external_id}")

    def stats(self) -> dict[str, Any]:
        """台账概览，用于评测和运维自检。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT target, COUNT(*) FROM sync_ledger GROUP BY target"
            ).fetchall()
        return {
            "db_path": str(self.db_path),
            "total": sum(count for _, count in rows),
            "by_target": {target: count for target, count in rows},
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "SyncLedger":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
