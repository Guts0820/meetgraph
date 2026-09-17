"""分块与分词测试。"""

from __future__ import annotations

from pathlib import Path

from src.rag.chunking import chunk_markdown, load_markdown_dir
from src.rag.tokenize import tokenize, tokenize_query


def test_chunk_markdown_splits_by_heading() -> None:
    text = """# 标题

导语部分。

## 第一节

第一节内容。

## 第二节

第二节内容。
"""
    chunks = chunk_markdown(text, doc_id="doc", doc_title="测试文档")

    # 标题本身也是一级 heading，因此它进入 section 路径
    assert [c.section for c in chunks] == ["标题", "标题 > 第一节", "标题 > 第二节"]
    assert chunks[0].text == "导语部分。"
    assert chunks[1].text == "第一节内容。"
    assert chunks[1].citation == "测试文档 / 标题 > 第一节"


def test_chunk_markdown_keeps_nested_heading_path() -> None:
    text = "## 一级\n\n### 二级\n\n内容\n"
    chunks = chunk_markdown(text, doc_id="doc")

    assert chunks[0].section == "一级 > 二级"


def test_chunk_ids_are_unique() -> None:
    text = "## 节\n\n" + "句子。" * 200
    chunks = chunk_markdown(text, doc_id="doc", max_chars=120, overlap_chars=20)

    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert len(chunks) > 1


def test_long_section_is_split_with_overlap() -> None:
    body = "".join(f"第{i}句内容。" for i in range(60))
    text = f"## 长节\n\n{body}"
    chunks = chunk_markdown(text, doc_id="doc", max_chars=100, overlap_chars=30)

    assert len(chunks) > 1
    assert all(len(c.text) <= 110 for c in chunks)
    # 相邻窗口有重叠：前一块的尾巴应出现在后一块里
    tail = chunks[0].text[-10:]
    assert tail in chunks[1].text


def test_empty_sections_are_dropped() -> None:
    chunks = chunk_markdown("## 空节\n\n\n## 有内容\n\n内容\n", doc_id="doc")
    assert [c.section for c in chunks] == ["有内容"]


def test_load_markdown_dir(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\n\n内容A\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("# B\n\n内容B\n", encoding="utf-8")

    chunks = load_markdown_dir(tmp_path, source_type="knowledge")

    assert {c.doc_id for c in chunks} == {"a", "b"}
    assert all(c.source_type == "knowledge" for c in chunks)
    assert load_markdown_dir(tmp_path / "missing") == []


def test_tokenize_chinese_bigram_and_words() -> None:
    tokens = tokenize("版本冻结 Code Freeze Q3")

    assert "版本" in tokens and "本冻" in tokens
    assert "code" in tokens and "freeze" in tokens
    assert "q3" in tokens


def test_tokenize_query_adds_single_chars() -> None:
    tokens = tokenize_query("验收标准")

    assert "验收" in tokens
    assert "验" in tokens  # 查询额外补单字，提高召回


def test_tokenize_handles_empty_and_short() -> None:
    assert tokenize("") == []
    assert tokenize("中") == ["中"]
