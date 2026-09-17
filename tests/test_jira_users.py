"""Jira 人员映射与优先级映射测试。"""

from __future__ import annotations

import json
from pathlib import Path

from src.integrations.jira_client import JiraClient, load_user_mapping


def test_load_from_inline_env(monkeypatch) -> None:
    monkeypatch.setenv(
        "JIRA_USER_MAP", json.dumps({"张总": "zhang.zong", "李明": "li.ming"})
    )
    mapping = load_user_mapping()

    assert mapping["张总"] == "zhang.zong"
    assert mapping["李明"] == "li.ming"


def test_inline_env_wins_over_file(monkeypatch, tmp_path: Path) -> None:
    map_file = tmp_path / "users.json"
    map_file.write_text(json.dumps({"李明": "from-file"}), encoding="utf-8")
    monkeypatch.setenv("JIRA_USER_MAP_FILE", str(map_file))
    monkeypatch.setenv("JIRA_USER_MAP", json.dumps({"李明": "from-env"}))

    assert load_user_mapping()["李明"] == "from-env"


def test_load_from_file_with_aliases(monkeypatch, tmp_path: Path) -> None:
    map_file = tmp_path / "users.json"
    map_file.write_text(
        json.dumps(
            {
                "张总": {"account": "zhang.zong", "aliases": ["Zhang", "张总（主持人）"]},
                "王芳": "wang.fang",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("JIRA_USER_MAP", raising=False)
    monkeypatch.setenv("JIRA_USER_MAP_FILE", str(map_file))

    mapping = load_user_mapping()
    assert mapping["张总"] == "zhang.zong"
    assert mapping["zhang"] == "zhang.zong"  # 别名已归一化为小写
    assert mapping["张总（主持人）"] == "zhang.zong"
    assert mapping["王芳"] == "wang.fang"


def test_broken_config_does_not_raise(monkeypatch, tmp_path: Path) -> None:
    bad = tmp_path / "users.json"
    bad.write_text("{not json", encoding="utf-8")
    monkeypatch.delenv("JIRA_USER_MAP", raising=False)
    monkeypatch.setenv("JIRA_USER_MAP_FILE", str(bad))

    assert load_user_mapping() == {}


def test_repo_default_mapping_file_is_valid() -> None:
    """仓库自带的示例映射文件必须能被解析。"""
    mapping = load_user_mapping()
    assert mapping["张总"] == "zhang.zong"
    assert mapping["李工"] == "li.ming"


def test_resolve_user_matches_normalized_names() -> None:
    client = JiraClient(
        server="https://example.atlassian.net",
        email="a@b.com",
        api_token="token",
        user_mapping={"李明": "li.ming"},
    )

    assert client.resolve_user("李明") == "li.ming"
    assert client.resolve_user(" 李明 ") == "li.ming"
    assert client.resolve_user("李明\n") == "li.ming"


def test_resolve_user_returns_none_for_unknown() -> None:
    client = JiraClient(
        server="https://example.atlassian.net",
        email="a@b.com",
        api_token="token",
        user_mapping={"李明": "li.ming"},
    )

    # 解析不到时宁可不指派，也不猜一个人
    assert client.resolve_user("张三") is None


def test_enabled_requires_full_credentials(monkeypatch) -> None:
    monkeypatch.delenv("JIRA_SERVER", raising=False)
    monkeypatch.delenv("JIRA_EMAIL", raising=False)
    monkeypatch.delenv("JIRA_API_TOKEN", raising=False)

    assert JiraClient().is_enabled is False
    assert JiraClient(server="https://x", email="e", api_token="t").is_enabled is True
