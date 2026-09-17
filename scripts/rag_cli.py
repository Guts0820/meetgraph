#!/usr/bin/env python
"""知识库（RAG）命令行工具。

用法::

    python scripts/rag_cli.py reindex                     # 重建索引
    python scripts/rag_cli.py search "版本冻结的规则"       # 只看检索结果（不调 LLM）
    python scripts/rag_cli.py ask "DT 数据多久接入一次？"   # 检索 + 生成（带引用）
    python scripts/rag_cli.py stats                        # 索引概览
    python scripts/rag_cli.py ask "..." --no-terms          # 关掉术语层做对照
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

from src.rag.ingest import RagIndex, build_and_save  # noqa: E402


def ensure_index(rebuild: bool = False) -> RagIndex:
    if rebuild:
        return build_and_save()
    try:
        return RagIndex.load()
    except FileNotFoundError:
        print("未找到索引，现场构建…")
        return build_and_save()


def cmd_reindex(args: argparse.Namespace) -> int:
    index = ensure_index(rebuild=True)
    print(f"索引已重建：{index.stats()}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    index = ensure_index()
    stats = index.stats()
    print(f"chunks: {stats['chunks']}  docs: {stats['docs']}  embedder: {stats['embedder']}")
    print(f"来源分布: {stats['by_source']}")
    print(f"构建时间: {stats['built_at']}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    index = ensure_index(args.rebuild)
    results, debug = index.retriever.search(
        args.query,
        top_k=args.top_k,
        use_terminology=not args.no_terms,
        return_debug=True,
    )
    print(f"查询: {debug.query}")
    print(f"扩展后: {debug.expanded_query}")
    print(f"命中术语: {debug.query_terms}")
    print(f"召回通道: {debug.channels}\n")
    for rank, result in enumerate(results, 1):
        print(f"[{rank}] {result.citation}  score={result.score:.4f} "
              f"(vec={result.vector_score:.3f}, bm25={result.bm25_score:.3f})")
        print(f"    {result.chunk.text[:120].replace(chr(10), ' ')}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from src.rag.qa import KnowledgeQA

    index = ensure_index(args.rebuild)
    qa = KnowledgeQA(retriever=index.retriever, terminology=index.retriever.terminology)
    answer = asyncio.run(
        qa.ask(args.query, top_k=args.top_k, use_terminology=not args.no_terms)
    )
    print(f"问：{answer.question}\n")
    print(answer.text or "（无答案）")
    if answer.citations:
        print("\n引用：")
        for item in answer.citations:
            print(f"  [{item['index']}] {item['citation']}")
    if answer.used_terms:
        print(f"\n术语约束：{'、'.join(answer.used_terms)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="知识库（RAG）命令行工具")
    parser.add_argument("--rebuild", action="store_true", help="先重建索引")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("reindex", help="重建索引").set_defaults(func=cmd_reindex)
    sub.add_parser("stats", help="索引概览").set_defaults(func=cmd_stats)

    search_parser = sub.add_parser("search", help="只做检索")
    search_parser.add_argument("query")
    search_parser.add_argument("--top-k", type=int, default=5)
    search_parser.add_argument("--no-terms", action="store_true", help="关闭术语层")
    search_parser.set_defaults(func=cmd_search)

    ask_parser = sub.add_parser("ask", help="检索 + 生成（带引用）")
    ask_parser.add_argument("query")
    ask_parser.add_argument("--top-k", type=int, default=5)
    ask_parser.add_argument("--no-terms", action="store_true", help="关闭术语层")
    ask_parser.set_defaults(func=cmd_ask)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
