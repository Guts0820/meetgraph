"""幂等台账测试。"""

from __future__ import annotations

from pathlib import Path

from src.integrations.idempotency import SyncLedger


def test_item_key_is_stable_and_normalized() -> None:
    key = SyncLedger.item_key("meeting-1", "李明", "整理Q3预算方案")
    # 空白与大小写差异不应改变业务键
    assert key == SyncLedger.item_key("meeting-1", " 李明 ", "整理Q3预算方案")
    assert key == SyncLedger.item_key("Meeting-1", "李明", "整理q3预算方案")
    assert len(key) == 32


def test_item_key_differs_per_field() -> None:
    base = SyncLedger.item_key("m", "李明", "任务A")
    assert base != SyncLedger.item_key("m2", "李明", "任务A")
    assert base != SyncLedger.item_key("m", "王芳", "任务A")
    assert base != SyncLedger.item_key("m", "李明", "任务B")


def test_record_and_get(tmp_path: Path) -> None:
    ledger = SyncLedger(tmp_path / "ledger.db")
    key = SyncLedger.item_key("m", "李明", "任务A")

    assert ledger.get(key, "jira") is None

    ledger.record(key, "jira", "MEET-1", "m")
    ledger.record(key, "feishu", "t1001", "m")

    assert ledger.get(key, "jira") == "MEET-1"
    assert ledger.get(key, "feishu") == "t1001"
    # 不同目标互不干扰
    assert ledger.get(key, "confluence") is None
    ledger.close()


def test_ledger_survives_process_restart(tmp_path: Path) -> None:
    """台账落盘：进程重启后再查仍然命中，这是幂等在真实环境生效的前提。"""
    db = tmp_path / "ledger.db"
    key = SyncLedger.item_key("m", "李明", "任务A")

    first = SyncLedger(db)
    first.record(key, "jira", "MEET-7", "m")
    first.close()

    second = SyncLedger(db)
    assert second.get(key, "jira") == "MEET-7"
    second.close()


def test_record_overwrites_previous_mapping(tmp_path: Path) -> None:
    ledger = SyncLedger(tmp_path / "ledger.db")
    key = SyncLedger.item_key("m", "李明", "任务A")

    ledger.record(key, "jira", "MEET-1", "m")
    ledger.record(key, "jira", "MEET-2", "m")

    assert ledger.get(key, "jira") == "MEET-2"
    assert ledger.stats()["total"] == 1
    ledger.close()


def test_stats_groups_by_target(tmp_path: Path) -> None:
    ledger = SyncLedger(tmp_path / "ledger.db")
    for i in range(2):
        key = SyncLedger.item_key("m", f"人{i}", "任务")
        ledger.record(key, "jira", f"MEET-{i}", "m")
    ledger.record(SyncLedger.item_key("m", "人0", "任务"), "feishu", "t1", "m")

    stats = ledger.stats()
    assert stats["total"] == 3
    assert stats["by_target"] == {"jira": 2, "feishu": 1}
    ledger.close()


def test_in_memory_ledger_works() -> None:
    ledger = SyncLedger(":memory:")
    key = SyncLedger.item_key("m", "李明", "任务A")
    ledger.record(key, "jira", "MEET-9", "m")
    assert ledger.get(key, "jira") == "MEET-9"
    ledger.close()
