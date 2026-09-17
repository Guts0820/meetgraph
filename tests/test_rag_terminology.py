"""术语表测试：加载、匹配、查询扩展、加权、prompt 片段。"""

from __future__ import annotations

import json
from pathlib import Path

from src.rag.terminology import Terminology


def _glossary(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "glossary.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


TERMS = {
    "terms": [
        {
            "term": "DT",
            "canonical": "路测",
            "aliases": ["Drive Test", "路测数据"],
            "definition": "测试终端移动采集的数据。",
            "owner": "网优组",
        },
        {
            "term": "KQI",
            "canonical": "关键质量指标",
            "aliases": [],
            "definition": "刻画业务体验的指标集合。",
            "owner": "无线组",
        },
    ]
}


def test_load_terms(tmp_path: Path) -> None:
    terminology = Terminology.from_file(_glossary(tmp_path, TERMS))

    assert len(terminology.terms) == 2
    assert terminology.terms[0].canonical == "路测"
    assert "Drive Test" in terminology.terms[0].surface_forms


def test_missing_or_broken_glossary_is_tolerated(tmp_path: Path) -> None:
    assert Terminology.from_file(tmp_path / "nope.json").is_empty

    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert Terminology.from_file(broken).is_empty


def test_match_finds_abbreviation_and_alias(tmp_path: Path) -> None:
    terminology = Terminology.from_file(_glossary(tmp_path, TERMS))

    assert [t.term for t in terminology.match("DT 数据多久接入一次？")] == ["DT"]
    assert [t.term for t in terminology.match("Drive Test 报告")] == ["DT"]
    assert [t.term for t in terminology.match("路测数据接入")] == ["DT"]
    assert terminology.match("完全无关的句子") == []


def test_match_counts_each_term_once(tmp_path: Path) -> None:
    terminology = Terminology.from_file(_glossary(tmp_path, TERMS))

    hits = terminology.match("路测数据里既有路测也有 Drive Test，还有 DT")

    assert len(hits) == 1


def test_expand_appends_canonical_and_aliases(tmp_path: Path) -> None:
    terminology = Terminology.from_file(_glossary(tmp_path, TERMS))

    expanded, hits = terminology.expand("DT 接入频率")

    assert hits and hits[0].term == "DT"
    assert "路测" in expanded and "Drive Test" in expanded
    assert expanded.startswith("DT 接入频率")


def test_expand_is_noop_without_hits(tmp_path: Path) -> None:
    terminology = Terminology.from_file(_glossary(tmp_path, TERMS))

    expanded, hits = terminology.expand("会议纪要里写了什么")

    assert expanded == "会议纪要里写了什么"
    assert hits == []


def test_expand_is_idempotent(tmp_path: Path) -> None:
    """已经扩展过的查询再扩展一次不应继续变长（每个写法只补一次）。"""
    terminology = Terminology.from_file(_glossary(tmp_path, TERMS))

    once, _ = terminology.expand("路测 DT 的规则")
    twice, _ = terminology.expand(once)

    assert twice == once
    assert once.count("Drive Test") == 1


def test_boost_is_capped(tmp_path: Path) -> None:
    terminology = Terminology.from_file(_glossary(tmp_path, TERMS))

    assert terminology.boost("路测") == 1
    assert terminology.boost("路测 与 关键质量指标") == 2
    assert terminology.boost("路测 与 关键质量指标", max_hits=1) == 1


def test_prompt_block_contains_definitions(tmp_path: Path) -> None:
    terminology = Terminology.from_file(_glossary(tmp_path, TERMS))
    block = terminology.prompt_block(terminology.terms)

    assert "关键质量指标" in block
    assert "刻画业务体验的指标集合。" in block
    assert "不要改写成近义词" in block
    assert terminology.prompt_block([]) == ""


def test_to_chunks_makes_one_chunk_per_term(tmp_path: Path) -> None:
    terminology = Terminology.from_file(_glossary(tmp_path, TERMS))
    chunks = terminology.to_chunks()

    assert len(chunks) == 2
    assert {c.doc_id for c in chunks} == {"内部术语表"}
    assert all(c.source_type == "glossary" for c in chunks)
    assert "标准说法：路测" in chunks[0].text


def test_repo_glossary_file_is_loadable() -> None:
    """仓库自带的术语表必须能被解析，否则检索的术语层会静默失效。"""
    terminology = Terminology.from_file()

    assert len(terminology.terms) >= 15
    terms = {t.term for t in terminology.terms}
    assert {"QoE", "MRD", "CDR", "DT"} <= terms
