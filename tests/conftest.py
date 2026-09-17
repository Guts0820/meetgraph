"""pytest 共享 fixture。

约定：单元测试一律不联网、不写真实 Jira/飞书、不落盘到仓库目录，
外部依赖全部由 tests/fakes.py 里的假实现或临时目录替代。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterator

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.fakes import (  # noqa: E402
    FakeFeishuClient,
    FakeJiraClient,
    FakeLLM,
)

FIXTURES = Path(__file__).parent / "fixtures"

# 所有可能把测试连到外部世界的环境变量
EXTERNAL_ENV_VARS = (
    "MINIMAX_API_KEY",
    "MINIMAX_GROUP_ID",
    "OPENAI_API_KEY",
    "JIRA_SERVER",
    "JIRA_EMAIL",
    "JIRA_API_TOKEN",
    "JIRA_USER_MAP",
    "JIRA_USER_MAP_FILE",
    "FEISHU_APP_ID",
    "FEISHU_APP_SECRET",
    "FEISHU_WEBHOOK_URL",
)


@pytest.fixture
def demo_transcript() -> str:
    """标注集里的会议转写文本。"""
    return (FIXTURES / "demo_transcript.txt").read_text(encoding="utf-8")


@pytest.fixture
def golden_actions() -> dict[str, Any]:
    """人工标注的待办清单。"""
    return json.loads(
        (FIXTURES / "golden_actions.json").read_text(encoding="utf-8")
    )


@pytest.fixture
def offline_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """关闭全部外部集成，并把报告、台账指向临时目录。"""
    for var in EXTERNAL_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("SYNC_LEDGER_DB", str(tmp_path / "sync-ledger.db"))
    yield tmp_path


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def fake_jira() -> FakeJiraClient:
    return FakeJiraClient(enabled=True)


@pytest.fixture
def fake_feishu() -> FakeFeishuClient:
    return FakeFeishuClient(enabled=True)
