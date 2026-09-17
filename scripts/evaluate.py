#!/usr/bin/env python
"""MeetGraph 评测脚本 —— 用可复现的实验代替口头指标。

指标定义（全部由脚本现场测量，结果同时打印并落盘到 reports/）：

1. 并行编排收益：同样三个分析 Agent、同样的模型延迟，Fan-out（LangGraph 并行）
   与串行 await 的端到端耗时对比，给出加速比。
2. 抽取质量：在人工标注的待办清单上算 precision / recall / F1。
   加 --live 时用真实 LLM 跑（会产生 API 费用）；不加时用假 LLM，只用于验证
   评分链路本身是否正确（会打印为「fixture 自检」，不代表模型能力）。
3. 降级可靠性：把 LLM 打挂，检查 Pipeline 是否仍能跑完、错误是否被记录、
   报告是否仍落盘。
4. 报告完整率：报告中三个章节非空的比例（占位符不算通过）。

用法::

    python scripts/evaluate.py            # 离线评测，不发外部请求
    python scripts/evaluate.py --live     # 额外跑一次真实 LLM 抽取质量评测
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

from src.agents.action_agent import ActionAgent
from src.agents.insight_agent import InsightAgent
from src.agents.summary_agent import SummaryAgent
from src.agents.transcription_agent import TranscriptionAgent
from src.graph.meeting_graph import run_meeting_pipeline
from src.integrations.idempotency import SyncLedger
from src.integrations.minimax_client import MiniMaxClient
from tests.fakes import FakeFeishuClient, FakeJiraClient, FakeLLM

FIXTURES = REPO_ROOT / "tests" / "fixtures"
FAKE_LATENCY = 0.4  # 模拟每次 LLM 调用耗时 400ms
REPEATS = 3


# ----------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------

def load_transcript() -> str:
    return (FIXTURES / "demo_transcript.txt").read_text(encoding="utf-8")


def load_golden() -> list[dict[str, Any]]:
    data = json.loads((FIXTURES / "golden_actions.json").read_text(encoding="utf-8"))
    return data["action_items"]


def score_actions(
    predicted: list[Any], golden: list[dict[str, Any]]
) -> dict[str, float | int]:
    """把预测的待办与标注清单做二部匹配。

    判定规则：负责人匹配（包含关系即算）且任务描述命中该条目的任一关键词。
    这是偏宽松的近似评分，用于观察「漏提 / 多提」的趋势，不替代人工复核。
    """
    matched_pred: set[int] = set()
    matched_golden: set[int] = set()

    for gi, g in enumerate(golden):
        for pi, p in enumerate(predicted):
            if pi in matched_pred:
                continue
            assignee = str(getattr(p, "assignee", "") or "")
            task = str(getattr(p, "task", "") or "").lower()
            assignee_ok = bool(g["assignee"]) and (
                g["assignee"] in assignee or assignee in g["assignee"]
            )
            task_ok = any(kw.lower() in task for kw in g["keywords"])
            if assignee_ok and task_ok:
                matched_pred.add(pi)
                matched_golden.add(gi)
                break

    precision = len(matched_pred) / len(predicted) if predicted else 0.0
    recall = len(matched_golden) / len(golden) if golden else 0.0
    f1 = (
        2 * precision * recall / (precision + recall) if precision + recall else 0.0
    )
    deadlines = [p for p in predicted if getattr(p, "deadline", "")]
    return {
        "predicted": len(predicted),
        "golden": len(golden),
        "matched": len(matched_golden),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "deadline_fill_rate": round(len(deadlines) / len(predicted), 3)
        if predicted
        else 0.0,
    }


# ----------------------------------------------------------------------
# 实验
# ----------------------------------------------------------------------

async def experiment_parallel_speedup(tmp_dir: Path) -> dict[str, Any]:
    """Fan-out 并行 vs 串行 await 的端到端耗时对比。"""
    parallel_times: list[float] = []
    for i in range(REPEATS):
        start = time.perf_counter()
        await run_meeting_pipeline(
            f"eval-parallel-{i}",
            llm_client=FakeLLM(delay=FAKE_LATENCY),
            jira_client=FakeJiraClient(enabled=False),
            feishu_client=FakeFeishuClient(enabled=False),
            ledger=SyncLedger(":memory:"),
        )
        parallel_times.append(time.perf_counter() - start)

    transcript = TranscriptionAgent._generate_demo_transcript("eval-seq")
    transcript_text = TranscriptionAgent._format_transcript_text(transcript)

    sequential_times: list[float] = []
    for i in range(REPEATS):
        llm = FakeLLM(delay=FAKE_LATENCY)
        summary = SummaryAgent(llm)
        action = ActionAgent(
            llm, FakeJiraClient(enabled=False), FakeFeishuClient(enabled=False)
        )
        insight = InsightAgent(llm)
        state = {
            "meeting_id": f"eval-seq-{i}",
            "transcript": transcript,
            "transcript_text": transcript_text,
        }

        start = time.perf_counter()
        await summary.process(dict(state))
        await action.process(dict(state))
        await insight.process(dict(state))
        sequential_times.append(time.perf_counter() - start)

    parallel = statistics.median(parallel_times)
    sequential = statistics.median(sequential_times)
    llm_latency_total = 3 * FAKE_LATENCY

    return {
        "repeats": REPEATS,
        "llm_calls_per_run": 3,
        "simulated_llm_latency_s": FAKE_LATENCY,
        "llm_latency_total_s": llm_latency_total,
        "parallel_median_s": round(parallel, 3),
        "sequential_median_s": round(sequential, 3),
        "speedup": round(sequential / parallel, 2) if parallel else 0.0,
        "overhead_parallel_s": round(parallel - FAKE_LATENCY, 3),
    }


async def experiment_degradation(tmp_dir: Path) -> dict[str, Any]:
    """LLM 不可用时 Pipeline 是否仍能跑完并如实记录错误。"""
    runs = 3
    completed = 0
    errors_recorded = 0
    reports_written = 0
    error_samples: list[str] = []

    for i in range(runs):
        result = await run_meeting_pipeline(
            f"eval-degraded-{i}",
            llm_client=FakeLLM(fail_times=None),
            jira_client=FakeJiraClient(enabled=False),
            feishu_client=FakeFeishuClient(enabled=False),
            ledger=SyncLedger(":memory:"),
        )
        if result.get("status") == "completed":
            completed += 1
        if result.get("errors"):
            errors_recorded += 1
            error_samples = list(result["errors"])[:3]
        report = Path(result["followup"].report_url)
        if report.exists() and report.read_text(encoding="utf-8").strip():
            reports_written += 1

    return {
        "runs": runs,
        "completed": completed,
        "completed_rate": round(completed / runs, 3),
        "errors_recorded": errors_recorded,
        "reports_written": reports_written,
        "error_samples": error_samples,
    }


async def experiment_report_completeness(tmp_dir: Path) -> dict[str, Any]:
    """报告三个章节的非空比例（占位符文本不计通过）。"""
    result = await run_meeting_pipeline(
        "eval-report",
        llm_client=FakeLLM(),
        jira_client=FakeJiraClient(enabled=False),
        feishu_client=FakeFeishuClient(enabled=False),
        ledger=SyncLedger(":memory:"),
    )
    text = Path(result["followup"].report_url).read_text(encoding="utf-8")

    sections = {
        "会议纪要": "## 会议纪要",
        "待办事项": "## 待办事项",
        "会议洞察": "## 会议洞察",
    }
    placeholders = ("*（摘要生成失败）*", "*（洞察分析失败）*", "*（无待办事项）*")

    filled = {}
    for name, marker in sections.items():
        body = text.split(marker, 1)[-1]
        for other in sections.values():
            if other != marker:
                body = body.split(other, 1)[0]
        has_placeholder = any(p in body for p in placeholders)
        filled[name] = bool(body.strip()) and not has_placeholder

    ratio = sum(filled.values()) / len(filled)
    return {
        "sections": filled,
        "completeness": round(ratio, 3),
        "report_bytes": len(text.encode("utf-8")),
    }


async def experiment_extraction(live: bool) -> dict[str, Any]:
    """待办抽取质量：live 用真实 LLM，offline 只做评分链路自检。"""
    golden = load_golden()
    transcript = load_transcript()

    if live:
        llm: Any = MiniMaxClient()
        model = getattr(llm, "model", "unknown")
        mode = f"live（真实 LLM: {model}）"
    else:
        llm = FakeLLM()
        mode = "fixture 自检（假 LLM，不代表模型能力）"

    agent = ActionAgent(
        llm_client=llm,
        jira_client=FakeJiraClient(enabled=False),
        feishu_client=FakeFeishuClient(enabled=False),
    )
    predicted = await agent._extract_actions(transcript)
    scores = score_actions(predicted, golden)
    scores["mode"] = mode
    scores["items"] = [
        {
            "assignee": p.assignee,
            "task": p.task,
            "deadline": p.deadline,
            "priority": p.priority.value,
        }
        for p in predicted
    ]

    if live and isinstance(llm, MiniMaxClient):
        await llm.close()
    return scores


# ----------------------------------------------------------------------
# 输出
# ----------------------------------------------------------------------

def render(report: dict[str, Any]) -> str:
    par = report["parallel_speedup"]
    deg = report["degradation"]
    rep = report["report_completeness"]
    ext = report["extraction"]

    lines = [
        "## MeetGraph 评测结果",
        "",
        f"- 时间：{report['generated_at']}",
        f"- Python：{report['python']}；模拟 LLM 延迟：{par['simulated_llm_latency_s']}s × {par['llm_calls_per_run']} 次调用",
        "",
        "| 指标 | 数值 | 测量方式 |",
        "|------|------|----------|",
        f"| 并行编排中位耗时 | {par['parallel_median_s']}s | {par['repeats']} 次 LangGraph Fan-out 端到端 |",
        f"| 串行基线中位耗时 | {par['sequential_median_s']}s | 同参数下三个 Agent 顺序 await |",
        f"| 编排加速比 | **{par['speedup']}x** | 串行 / 并行 |",
        f"| 并行编排固有开销 | {par['overhead_parallel_s']}s | 并行耗时 − 单次 LLM 延迟 |",
        f"| 降级完成率 | {deg['completed']}/{deg['runs']} | LLM 全挂时 Pipeline 仍跑完 |",
        f"| 错误可观测率 | {deg['errors_recorded']}/{deg['runs']} | errors 字段如实记录失败 |",
        f"| 报告完整率 | {rep['completeness']:.0%} | 三个章节非空且非占位符 |",
        f"| 待办抽取 P/R/F1 | {ext['precision']} / {ext['recall']} / {ext['f1']} | 人工标注 {ext['golden']} 条，匹配 {ext['matched']} 条 |",
        f"| 待办截止时间填充率 | {ext['deadline_fill_rate']:.0%} | 抽取结果中带合法日期的比例 |",
        "",
        f"> 抽取质量口径：{ext['mode']}",
    ]

    if deg["error_samples"]:
        lines += ["", "降级时记录的错误样例："]
        lines += [f"- `{e}`" for e in deg["error_samples"]]

    lines += ["", "抽取到的待办：", "", "| 负责人 | 任务 | 截止时间 | 优先级 |", "|---|---|---|---|"]
    for item in ext["items"]:
        lines.append(
            f"| {item['assignee']} | {item['task']} | {item['deadline'] or '—'} | {item['priority']} |"
        )
    return "\n".join(lines)


async def main() -> int:
    parser = argparse.ArgumentParser(description="MeetGraph 评测脚本")
    parser.add_argument(
        "--live", action="store_true", help="用真实 LLM 评测抽取质量（产生 API 费用）"
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="meetgraph-eval-") as tmp:
        tmp_dir = Path(tmp)
        os.environ["REPORTS_DIR"] = str(tmp_dir / "reports")
        os.environ["SYNC_LEDGER_DB"] = str(tmp_dir / "sync-ledger.db")

        report = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "python": sys.version.split()[0],
            "parallel_speedup": await experiment_parallel_speedup(tmp_dir),
            "degradation": await experiment_degradation(tmp_dir),
            "report_completeness": await experiment_report_completeness(tmp_dir),
            "extraction": await experiment_extraction(args.live),
        }

    out_dir = REPO_ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"evaluation-{datetime.now():%Y%m%d-%H%M%S}.json"
    out_file.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(render(report))
    print(f"\n原始结果已写入：{out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
