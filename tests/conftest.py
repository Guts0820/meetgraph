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
    # 指向一个不存在的索引目录：默认行为下 RAG 节点自动跳过，测试不依赖向量模型
    monkeypatch.setenv("RAG_INDEX_DIR", str(tmp_path / "no-index"))
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


# ----------------------------------------------------------------------
# RAG 相关 fixture：全部离线，向量用确定性哈希向量，绝不加载真实模型
# ----------------------------------------------------------------------

KNOWLEDGE_DOC = """# 内部规范：版本冻结

## 冻结规则

版本冻结后只允许缺陷修复与配置调整，禁止新特性合入。

## 灰度发布

首次灰度流量比例不超过 5%，观察期不少于 72 小时。
"""

MEETING_DOC = """# 会议纪要：数据接入评审

## 结论

路测数据接入频率改为每周一、周四各一次，由赵伟负责脱敏改造。
"""

GLOSSARY = {
    "version": "test",
    "terms": [
        {
            "term": "DT",
            "canonical": "路测",
            "aliases": ["Drive Test"],
            "definition": "测试终端沿路线移动采集的网络质量数据。",
            "owner": "网优组",
        },
        {
            "term": "版本冻结",
            "canonical": "版本冻结",
            "aliases": ["Code Freeze"],
            "definition": "到达约定时间点后停止合入新特性。",
            "owner": "研发部",
        },
    ],
}


@pytest.fixture
def rag_corpus(tmp_path: Path) -> Path:
    """构造一个最小语料目录（内部文档 + 会议纪要 + 术语表）。"""
    root = tmp_path / "corpus"
    (root / "knowledge").mkdir(parents=True)
    (root / "meetings").mkdir(parents=True)
    (root / "knowledge" / "内部规范-版本冻结.md").write_text(
        KNOWLEDGE_DOC, encoding="utf-8"
    )
    (root / "meetings" / "会议纪要-数据接入评审.md").write_text(
        MEETING_DOC, encoding="utf-8"
    )
    (root / "glossary.json").write_text(
        json.dumps(GLOSSARY, ensure_ascii=False), encoding="utf-8"
    )
    return root


@pytest.fixture
def rag_index(rag_corpus: Path):
    """用离线哈希向量构建索引（不加载模型、不联网）。"""
    from src.rag.chunking import load_markdown_dir
    from src.rag.embedding import HashingEmbedder
    from src.rag.ingest import RagIndex
    from src.rag.terminology import Terminology

    terminology = Terminology.from_file(rag_corpus / "glossary.json")
    chunks = load_markdown_dir(rag_corpus / "knowledge", source_type="knowledge")
    chunks += load_markdown_dir(rag_corpus / "meetings", source_type="meeting")
    chunks += terminology.to_chunks()

    return RagIndex.build(
        chunks,
        embedder=HashingEmbedder(),
        terminology=terminology,
        probe_embedder=False,
    )
