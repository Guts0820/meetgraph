#!/usr/bin/env python
"""RAG 检索评测：Recall@K / MRR / 术语命中率 + 术语层与检索通道的消融。

用法::

    python scripts/evaluate_rag.py                      # 用已有索引（无则现场构建）
    python scripts/evaluate_rag.py --rebuild             # 重建索引后再评测
    python scripts/evaluate_rag.py --embedder hash       # 强制离线哈希向量（不加载模型）
    python scripts/evaluate_rag.py --live                # 额外跑真实 LLM 的答案级评测

评测三张表：

1. **主配置**（向量 + BM25 混合）在术语层开/关下的 Recall@1 / Recall@5 / MRR；
2. **消融矩阵**：混合 / 纯 BM25 / 纯向量 × 术语层开/关——术语层的价值在哪个通道上
   体现、又被哪个通道掩盖，一看就知道；
3. **分类别**（术语类 / 会议历史类 / 制度规范类）的命中情况。

指标口径见 docs/evaluation.md。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

from src.rag.ingest import RagIndex, build_and_save  # noqa: E402
from src.rag.qa import KnowledgeQA  # noqa: E402
from src.rag.retriever import HybridRetriever  # noqa: E402

QA_FILE = REPO_ROOT / "tests" / "fixtures" / "rag_qa.jsonl"
TOP_K = 5

# 消融配置：权重为 0 表示该通道关闭
ABLATION_CONFIGS = {
    "hybrid": {"vector_weight": 1.0, "bm25_weight": 1.0},
    "bm25-only": {"vector_weight": 0.0, "bm25_weight": 1.0},
    "vector-only": {"vector_weight": 1.0, "bm25_weight": 0.0},
}


def load_qa(path: Path = QA_FILE) -> list[dict[str, Any]]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def make_retriever(
    index: RagIndex,
    *,
    vector_weight: float = 1.0,
    bm25_weight: float = 1.0,
    use_terminology: bool = True,
) -> HybridRetriever:
    """复用同一份索引，按参数构造检索器（消融用）。"""
    return HybridRetriever(
        chunks=index.chunks,
        embedder=index.embedder,
        bm25_index=index.bm25,
        vector_index=index.vectors,
        terminology=index.retriever.terminology,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
        use_terminology=use_terminology,
    )


def evaluate_item(retriever: HybridRetriever, item: dict[str, Any], use_terminology: bool) -> dict[str, Any]:
    results, debug = retriever.search(
        item["question"], top_k=TOP_K, use_terminology=use_terminology, return_debug=True
    )
    doc_ids = [r.chunk.doc_id for r in results]
    expected = set(item.get("expected_sources", []))

    rank = next((i for i, doc in enumerate(doc_ids, 1) if doc in expected), 0)
    context = "\n".join(r.chunk.text for r in results)
    term_hits = [t for t in item.get("expected_terms", []) if t in context]
    citations = retriever.citations(results)
    verifiable = all(c["chunk_id"] in retriever.chunks for c in citations)

    return {
        "id": item["id"],
        "category": item["category"],
        "question": item["question"],
        "expanded_query": debug.expanded_query if use_terminology else debug.query,
        "query_terms": debug.query_terms,
        "channels": debug.channels,
        "top_docs": doc_ids,
        "hit@1": bool(rank == 1),
        "hit@5": bool(rank and rank <= TOP_K),
        "rank": rank,
        "mrr": 1.0 / rank if rank else 0.0,
        "expected_terms": item.get("expected_terms", []),
        "term_hits": term_hits,
        "term_hit": bool(term_hits) if item.get("expected_terms") else None,
        "citations_verifiable": verifiable,
    }


def summarize(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    if not rows:
        return {}

    total = len(rows)
    term_rows = [r for r in rows if r["term_hit"] is not None]
    return {
        "questions": total,
        "recall@1": round(sum(r["hit@1"] for r in rows) / total, 3),
        "recall@5": round(sum(r["hit@5"] for r in rows) / total, 3),
        "mrr": round(sum(r["mrr"] for r in rows) / total, 3),
        "term_hit_rate": round(
            sum(1 for r in term_rows if r["term_hit"]) / len(term_rows), 3
        )
        if term_rows
        else None,
        "citation_verifiable_rate": round(
            sum(1 for r in rows if r["citations_verifiable"]) / total, 3
        ),
    }


def group_by_category(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    return {category: summarize(items) for category, items in sorted(grouped.items())}


async def evaluate_answers(index: RagIndex, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """答案级评测（真实 LLM）：术语层开/关两组的引用可核验率与术语一致性。

    术语层的生成侧价值就在这里体现：同样的问题、同样的知识库，只差「是否注入
    公司标准术语定义」——看答案是否使用标准术语（而不是用户的缩写），以及引用
    是否可核验。
    """
    from src.integrations.minimax_client import MiniMaxClient

    llm = MiniMaxClient()
    qa = KnowledgeQA(
        retriever=index.retriever, terminology=index.retriever.terminology, llm=llm
    )

    arms = {"with_terminology": True, "without_terminology": False}
    stats: dict[str, dict[str, int]] = {name: {"answered": 0, "cited": 0, "terms": 0, "term_total": 0} for name in arms}
    samples: dict[str, list[dict[str, Any]]] = {name: [] for name in arms}

    for item in rows[:8]:  # 控制 API 调用次数（两组 × 8 条 = 16 次）
        expected_terms = item.get("expected_terms", [])
        for arm, flag in arms.items():
            answer = await qa.ask(item["question"], use_terminology=flag)
            if not answer.answered:
                continue
            bucket = stats[arm]
            bucket["answered"] += 1
            if answer.citations and all(
                c["chunk_id"] in index.retriever.chunks for c in answer.citations
            ):
                bucket["cited"] += 1
            if expected_terms:
                bucket["term_total"] += 1
                if any(term in answer.text for term in expected_terms):
                    bucket["terms"] += 1
            if len(samples[arm]) < 3:
                samples[arm].append(
                    {
                        "id": item["id"],
                        "question": item["question"],
                        "answer": answer.text,
                        "citations": [c["citation"] for c in answer.citations],
                    }
                )

    await llm.close()

    def ratios(bucket: dict[str, int]) -> dict[str, Any]:
        answered = bucket["answered"] or 1
        return {
            "answered": bucket["answered"],
            "citation_valid_rate": round(bucket["cited"] / answered, 3),
            "term_covered_rate": (
                round(bucket["terms"] / bucket["term_total"], 3)
                if bucket["term_total"]
                else None
            ),
        }

    return {
        "with_terminology": ratios(stats["with_terminology"]),
        "without_terminology": ratios(stats["without_terminology"]),
        "samples": samples,
    }


def run_ablation(index: RagIndex, qa_items: list[dict[str, Any]]) -> dict[str, Any]:
    ablation: dict[str, Any] = {}
    for name, params in ABLATION_CONFIGS.items():
        entry: dict[str, Any] = {}
        for flag in (True, False):
            retriever = make_retriever(index, use_terminology=flag, **params)
            rows = [evaluate_item(retriever, item, flag) for item in qa_items]
            entry["with_terminology" if flag else "without_terminology"] = {
                "summary": summarize(rows),
                "rows": rows,
            }
        on = entry["with_terminology"]["summary"]
        off = entry["without_terminology"]["summary"]
        entry["delta"] = {
            key: round(on[key] - off[key], 3)
            for key in ("recall@1", "recall@5", "mrr")
            if on.get(key) is not None and off.get(key) is not None
        }
        ablation[name] = entry
    return ablation


def run_degraded_ablation(index: RagIndex, qa_items: list[dict[str, Any]]) -> dict[str, Any]:
    """语义通道缺失时的消融：无向量模型部署 / 模型加载失败的降级路径。

    这一组用离线哈希向量（无语义能力）替换 BGE，用来回答一个具体问题：
    「术语扩展到底在什么情况下有价值」——预期是在没有语义模型时，缩写与标准
    术语之间的鸿沟只能靠术语表来补。
    """
    from src.rag.embedding import HashingEmbedder

    hash_index = RagIndex.build(
        chunks=index.chunks,
        embedder=HashingEmbedder(),
        terminology=index.retriever.terminology,
        probe_embedder=False,
    )

    ablation: dict[str, Any] = {}
    for name, params in ABLATION_CONFIGS.items():
        entry: dict[str, Any] = {}
        for flag in (True, False):
            retriever = make_retriever(hash_index, use_terminology=flag, **params)
            rows = [evaluate_item(retriever, item, flag) for item in qa_items]
            entry["with_terminology" if flag else "without_terminology"] = summarize(rows)
        on = entry["with_terminology"]
        off = entry["without_terminology"]
        entry["delta"] = {
            key: round(on[key] - off[key], 3)
            for key in ("recall@1", "recall@5", "mrr")
            if on.get(key) is not None and off.get(key) is not None
        }
        ablation[name] = entry
    return ablation


def render(report: dict[str, Any]) -> str:
    on = report["ablation"]["hybrid"]["with_terminology"]
    off = report["ablation"]["hybrid"]["without_terminology"]

    lines = [
        "## RAG 检索评测结果",
        "",
        f"- 索引：{report['index']['chunks']} chunks / {report['index']['docs']} 文档，embedder=`{report['index']['embedder']}`",
        f"- 标注问答：{on['questions']} 条（术语类 8 / 会议历史类 8 / 制度规范类 8）",
        "",
        "### 主配置：向量 + BM25 混合 + 确定性精排",
        "",
        "| 指标 | 术语层 开 | 术语层 关 | 差值 |",
        "|------|-----------|-----------|------|",
        f"| Recall@1 | {on['recall@1']:.3f} | {off['recall@1']:.3f} | {on['recall@1'] - off['recall@1']:+.3f} |",
        f"| Recall@5 | {on['recall@5']:.3f} | {off['recall@5']:.3f} | {on['recall@5'] - off['recall@5']:+.3f} |",
        f"| MRR | {on['mrr']:.3f} | {off['mrr']:.3f} | {on['mrr'] - off['mrr']:+.3f} |",
        f"| 术语命中率 | {on['term_hit_rate']} | {off['term_hit_rate']} | — |",
        f"| 引用可核验率 | {on['citation_verifiable_rate']:.3f} | {off['citation_verifiable_rate']:.3f} | — |",
        "",
        "### 消融：术语层在不同检索通道下的增益",
        "",
        "| 检索配置 | 术语层 | Recall@1 | Recall@5 | MRR |",
        "|----------|--------|----------|----------|-----|",
    ]
    for name in ABLATION_CONFIGS:
        entry = report["ablation"][name]
        for flag, label in (("with_terminology", "开"), ("without_terminology", "关")):
            stats = entry[flag]
            lines.append(
                f"| {name} | {label} | {stats['recall@1']:.3f} | "
                f"{stats['recall@5']:.3f} | {stats['mrr']:.3f} |"
            )
        delta = entry["delta"]
        lines.append(
            f"| **{name} 增益** | 开−关 | {delta.get('recall@1', 0):+.3f} | "
            f"{delta.get('recall@5', 0):+.3f} | {delta.get('mrr', 0):+.3f} |"
        )

    lines += [
        "",
        "### 分类别（主配置，术语层开启）",
        "",
        "| 类别 | 题数 | Recall@1 | Recall@5 | MRR |",
        "|------|------|----------|----------|-----|",
    ]
    for category, stats in report["by_category"].items():
        lines.append(
            f"| {category} | {stats['questions']} | {stats['recall@1']:.3f} | "
            f"{stats['recall@5']:.3f} | {stats['mrr']:.3f} |"
        )

    degraded = report.get("degraded_ablation")
    if degraded:
        lines += [
            "",
            "### 语义通道缺失时（离线哈希向量：无模型部署 / 降级路径）",
            "",
            "| 检索配置 | 术语层 | Recall@1 | Recall@5 | MRR |",
            "|----------|--------|----------|----------|-----|",
        ]
        for name in ABLATION_CONFIGS:
            entry = degraded[name]
            for flag, label in (("with_terminology", "开"), ("without_terminology", "关")):
                stats = entry[flag]
                lines.append(
                    f"| {name} | {label} | {stats['recall@1']:.3f} | "
                    f"{stats['recall@5']:.3f} | {stats['mrr']:.3f} |"
                )
            delta = entry["delta"]
            lines.append(
                f"| **{name} 增益** | 开−关 | {delta.get('recall@1', 0):+.3f} | "
                f"{delta.get('recall@5', 0):+.3f} | {delta.get('mrr', 0):+.3f} |"
            )

    flipped = report.get("flipped_by_terminology", [])
    if flipped:
        lines += ["", "### 术语层带来的改善（关→开 由未命中变命中）", ""]
        for row in flipped:
            lines.append(
                f"- `{row['id']}` {row['question']}｜扩展后查询：`{row['expanded_query']}`"
            )

    missed = [r for r in report["details"] if not r["hit@5"]]
    if missed:
        lines += ["", "### 仍未命中（Top-5 内没有正确来源）", ""]
        for row in missed:
            lines.append(
                f"- `{row['id']}` {row['question']}｜实际 Top-5：{row['top_docs']}"
            )

    if report.get("answer_level"):
        ans = report["answer_level"]
        on_a = ans["with_terminology"]
        off_a = ans["without_terminology"]
        lines += [
            "",
            "### 答案级（真实 LLM，术语层开/关对照）",
            "",
            "| 指标 | 术语层 开 | 术语层 关 |",
            "|------|-----------|-----------|",
            f"| 有效回答数 | {on_a['answered']} | {off_a['answered']} |",
            f"| 引用可核验率 | {on_a['citation_valid_rate']:.3f} | {off_a['citation_valid_rate']:.3f} |",
            f"| 术语覆盖率（答案含标准术语） | {on_a['term_covered_rate']} | {off_a['term_covered_rate']} |",
        ]
        for arm, label in (("with_terminology", "术语层开启"), ("without_terminology", "术语层关闭")):
            lines += ["", f"**{label}** 的样例："]
            for sample in ans["samples"].get(arm, [])[:2]:
                lines.append(f"- {sample['question']}")
                lines.append(f"  - 答案：{sample['answer'][:200]}")
                lines.append(f"  - 引用：{'；'.join(sample['citations']) or '（无）'}")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 检索评测")
    parser.add_argument("--rebuild", action="store_true", help="评测前重建索引")
    parser.add_argument(
        "--embedder", default=None, help="hash / bge / openai（默认读 RAG_EMBEDDER）"
    )
    parser.add_argument("--live", action="store_true", help="额外做真实 LLM 的答案级评测")
    parser.add_argument("--qa-file", default=str(QA_FILE))
    args = parser.parse_args()

    from src.rag.embedding import resolve_embedder

    if args.rebuild:
        index = build_and_save(
            embedder=resolve_embedder(args.embedder) if args.embedder else None
        )
    else:
        try:
            index = RagIndex.load(
                embedder=resolve_embedder(args.embedder, probe=False)
                if args.embedder
                else None
            )
        except FileNotFoundError:
            print("未找到索引，改为现场构建…")
            index = build_and_save(
                embedder=resolve_embedder(args.embedder) if args.embedder else None
            )

    qa_items = load_qa(Path(args.qa_file))
    ablation = run_ablation(index, qa_items)
    degraded_ablation = run_degraded_ablation(index, qa_items)

    main_rows = ablation["hybrid"]["with_terminology"]["rows"]
    off_rows = {row["id"]: row for row in ablation["hybrid"]["without_terminology"]["rows"]}
    flipped = [
        row for row in main_rows if row["hit@5"] and not off_rows[row["id"]]["hit@5"]
    ]

    report: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "index": index.stats(),
        "ablation": {
            name: {
                "with_terminology": entry["with_terminology"]["summary"],
                "without_terminology": entry["without_terminology"]["summary"],
                "delta": entry["delta"],
            }
            for name, entry in ablation.items()
        },
        "by_category": group_by_category(main_rows),
        "degraded_ablation": degraded_ablation,
        "flipped_by_terminology": flipped,
        "details": main_rows,
    }

    if args.live:
        report["answer_level"] = asyncio.run(evaluate_answers(index, qa_items))

    out_dir = REPO_ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"rag-eval-{datetime.now():%Y%m%d-%H%M%S}.json"

    # 明细一并落盘，便于复现与回归对比
    detail_report = dict(report)
    detail_report["rows"] = {
        "hybrid_with": ablation["hybrid"]["with_terminology"]["rows"],
        "hybrid_without": ablation["hybrid"]["without_terminology"]["rows"],
        "bm25_with": ablation["bm25-only"]["with_terminology"]["rows"],
        "bm25_without": ablation["bm25-only"]["without_terminology"]["rows"],
        "vector_with": ablation["vector-only"]["with_terminology"]["rows"],
        "vector_without": ablation["vector-only"]["without_terminology"]["rows"],
    }
    out_file.write_text(
        json.dumps(detail_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(render(report))
    print(f"\n原始结果已写入：{out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
