#!/usr/bin/env python
"""工具调用评测：MCP 协议一致性 + Agent 工具选择质量 + 熔断与权限行为。

用法::

    python scripts/evaluate_tools.py           # 离线（Oracle/脚本化 LLM，不联网）
    python scripts/evaluate_tools.py --live     # 用真实 LLM 评工具选择与参数正确率

指标：

| 指标 | 定义 |
|------|------|
| 协议一致性 | 自研 client 走真实子进程：握手 → tools/list → tools/call → 错误路径全部符合预期 |
| 工具发现完整性 | 只读工具默认可见、写工具默认不可见（权限最小化） |
| 工具选择准确率 | 标注任务里「模型第一个调用的工具」是否等于期望（live） |
| 工具序列准确率 | 多步任务的工具序列是否完全匹配（live） |
| 参数正确率 | 期望参数键值是否出现在实际调用参数里（live，如 meeting_id / term） |
| 熔断命中率 | 构造死循环/连续失败时循环是否被正确终止（离线） |
| 权限拒绝与审计 | 写工具默认拒绝、每次调用都留审计（离线） |

`--live` 会**强制关闭写工具**（不设 MCP_ALLOW_WRITE），因此评测过程不会在真实
Jira/飞书里建任何东西：写任务只评估「模型是否选择了写工具」以及「被拒绝后的表述」。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

from src.agents.tool_agent import ToolCallingAgent  # noqa: E402
from src.mcp.audit import AuditLog  # noqa: E402
from src.mcp.client import McpProtocolError, McpStdioClient, text_content  # noqa: E402
from src.mcp.policy import ToolPolicy  # noqa: E402
from src.mcp.registry import ToolRegistry, ToolSpec  # noqa: E402
from src.mcp.tools import build_default_registry  # noqa: E402

TASKS_FILE = REPO_ROOT / "tests" / "fixtures" / "tool_tasks.jsonl"


def load_tasks() -> list[dict[str, Any]]:
    tasks = []
    for line in TASKS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            tasks.append(json.loads(line))
    return tasks


# ----------------------------------------------------------------------
# 1. 协议一致性（真实子进程）
# ----------------------------------------------------------------------

async def protocol_checks() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def record(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    async with McpStdioClient() as client:
        started = time.perf_counter()
        info = await client.initialize()
        record(
            "initialize 握手",
            info.get("serverInfo", {}).get("name") == "meetgraph",
            f"protocol={info.get('protocolVersion')}",
        )
        record(
            "能力声明",
            set(info.get("capabilities", {})) == {"tools", "resources", "prompts"},
            str(sorted(info.get("capabilities", {}))),
        )

        tools = await client.list_tools()
        names = {t["name"] for t in tools}
        record(
            "工具发现（默认只暴露只读）",
            names == {"search_meetings", "get_meeting_report", "lookup_glossary"},
            ",".join(sorted(names)),
        )
        record(
            "工具 Schema 完整",
            all(t["inputSchema"].get("type") == "object" and t.get("description") for t in tools),
        )

        called = await client.call_tool("lookup_glossary", {"term": "KQI"})
        record(
            "tools/call 正常路径",
            called.get("isError") is False and "关键质量指标" in text_content(called),
        )

        denied = await client.call_tool(
            "create_action_item", {"task_assignee": "李明", "task": "x"}
        )
        record(
            "写工具默认拒绝",
            denied.get("isError") is True and "MCP_ALLOW_WRITE" in text_content(denied),
        )

        unknown = None
        try:
            unknown = await client.call_tool("make_coffee", {})
        except McpProtocolError as e:
            record("未知工具返回 JSON-RPC 错误", e.code == -32602, f"code={e.code}")
        else:
            record(
                "未知工具返回 JSON-RPC 错误",
                False,
                f"expected error, got {unknown}",
            )

        resources = await client.list_resources()
        record("resources/list 可用", isinstance(resources, list))

        prompts = await client.list_prompts()
        record("prompts/list 可用", prompts and prompts[0]["name"] == "summarize_meeting")

        record("ping", await client.ping() == {})
        record(
            "握手+调用总耗时 < 30s",
            (time.perf_counter() - started) < 30,
            f"{(time.perf_counter() - started):.2f}s",
        )

    passed = sum(1 for c in checks if c["ok"])
    return {"checks": checks, "passed": passed, "total": len(checks)}


# ----------------------------------------------------------------------
# 2. 熔断与权限（离线，脚本化 LLM）
# ----------------------------------------------------------------------

@dataclass
class _Scripted:
    decisions: list[dict[str, Any]]

    def __post_init__(self) -> None:
        self.calls = 0

    async def chat_json(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        index = self.calls
        self.calls += 1
        return self.decisions[min(index, len(self.decisions) - 1)]


async def _tool(args: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True, "query": args.get("query", "")}


async def _boom(args: dict[str, Any]) -> dict[str, Any]:
    raise RuntimeError("simulated backend failure")


def _loop_registry(tmp_path: Path) -> ToolRegistry:
    audit = AuditLog(tmp_path / "audit.jsonl")
    registry = ToolRegistry(policy=ToolPolicy(allow_write=True), audit=audit)
    schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    registry.register(ToolSpec("probe", "查询", schema, _tool))
    registry.register(ToolSpec("flaky", "总会失败", schema, _boom))
    return registry


async def safety_checks(tmp_path: Path) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []

    # 死循环：一直用同样参数调同一工具
    registry = _loop_registry(tmp_path)
    llm = _Scripted([{"action": "probe", "arguments": {"query": "same"}}])
    run = await ToolCallingAgent(registry, llm_client=llm, max_steps=6).run("死循环")
    cases.append(
        {
            "case": "重复相同调用",
            "stopped_reason": run.stopped_reason,
            "steps": len(run.steps),
            "expected": "repeat_detected",
            "ok": run.stopped_reason == "repeat_detected" and len(run.steps) == 1,
        }
    )

    # 步数上限
    registry = _loop_registry(tmp_path)
    llm = _Scripted(
        [
            {"action": "probe", "arguments": {"query": "a"}},
            {"action": "probe", "arguments": {"query": "b"}},
        ]
    )
    run = await ToolCallingAgent(registry, llm_client=llm, max_steps=2).run("步数上限")
    cases.append(
        {
            "case": "达到最大步数",
            "stopped_reason": run.stopped_reason,
            "steps": len(run.steps),
            "expected": "max_steps",
            "ok": run.stopped_reason == "max_steps" and len(run.steps) == 2,
        }
    )

    # 连续工具失败
    registry = _loop_registry(tmp_path)
    llm = _Scripted(
        [
            {"action": "flaky", "arguments": {"query": "a"}},
            {"action": "flaky", "arguments": {"query": "b"}},
        ]
    )
    run = await ToolCallingAgent(
        registry, llm_client=llm, max_steps=5, max_consecutive_errors=2
    ).run("连续失败")
    cases.append(
        {
            "case": "连续工具失败",
            "stopped_reason": run.stopped_reason,
            "steps": len(run.steps),
            "expected": "tool_errors",
            "ok": run.stopped_reason == "tool_errors",
        }
    )

    # 参数非法（缺必填）
    registry = _loop_registry(tmp_path)
    llm = _Scripted(
        [
            {"action": "probe", "arguments": {}},
            {"action": "probe", "arguments": {"query": "fixed"}},
            {"final_answer": "修正成功"},
        ]
    )
    run = await ToolCallingAgent(registry, llm_client=llm, max_steps=4).run("参数非法")
    cases.append(
        {
            "case": "参数非法后自修复",
            "stopped_reason": run.stopped_reason,
            "steps": len(run.steps),
            "expected": "final_answer",
            "ok": run.stopped_reason == "final_answer" and run.steps[0].error == "invalid_arguments",
        }
    )

    # 审计完整性：每条执行过的调用都要有审计记录
    registry = _loop_registry(tmp_path)
    llm = _Scripted(
        [
            {"action": "probe", "arguments": {"query": "a"}},
            {"action": "flaky", "arguments": {"query": "b"}},
            {"final_answer": "结束"},
        ]
    )
    run = await ToolCallingAgent(registry, llm_client=llm).run("审计检查")
    records = registry.audit.tail(10)
    executed = len([s for s in run.steps if s.error != "invalid_arguments" and s.error != "timeout"])
    cases.append(
        {
            "case": "审计覆盖每次调用",
            "stopped_reason": run.stopped_reason,
            "steps": len(records),
            "expected": f">= {executed}",
            "ok": len(records) >= executed and all("args_digest" in r for r in records),
        }
    )

    return {
        "cases": cases,
        "passed": sum(1 for c in cases if c["ok"]),
        "total": len(cases),
    }


# ----------------------------------------------------------------------
# 3. 工具选择质量（离线 oracle 自检 / live 真实 LLM）
# ----------------------------------------------------------------------

class OracleLLM:
    """按标注脚本行事的假 LLM：用于验证评测链路与工具本身可用（不代表模型能力）。"""

    def __init__(self, task: dict[str, Any]) -> None:
        self.task = task
        self.calls = 0

    async def chat_json(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        index = self.calls
        self.calls += 1
        tools = self.task.get("expected_tools", [])
        if index < len(tools):
            name = tools[index]
            args = dict(self.task.get("expected_args", {}))
            if name == "search_meetings":
                args = {"query": args.get("query") or self.task["request"][:20]}
            elif name == "lookup_glossary":
                args = {"term": args.get("term") or "DT"}
            elif name == "get_meeting_report":
                args = {"meeting_id": args.get("meeting_id") or "m-1"}
            elif name == "create_action_item":
                args = {
                    "task_assignee": args.get("task_assignee") or "李明",
                    "task": self.task["request"][:30],
                    "deadline": args.get("deadline") or "",
                }
            return {"action": name, "arguments": args}
        return {"final_answer": f"已完成：{self.task['request'][:30]}"}


STRICT_ARG_KEYS = {"meeting_id", "term", "task_assignee", "deadline", "priority", "top_k"}


def _selection_metrics(task: dict[str, Any], run) -> dict[str, Any]:
    """计算单条任务的工具选择与参数指标。

    口径说明（这是被实测修正过的）：

    - **首个工具**：模型第一步调的工具是否等于期望——衡量「路由」能力；
    - **工具集合召回/精确**：期望工具是否都被调用过、有没有调用无关工具。
      早期用「工具序列完全相等」评分，会惩罚合理的探索调用（例如先查报告再
      检索一次确认），那不是错误；
    - **参数**：`meeting_id / term / assignee / deadline` 这类**结构化参数要求精确相等**；
      `query` 是自由文本，只要求非空——模型用自己的措辞检索是正常的。
    """
    expected_tools = task.get("expected_tools", [])
    actual_tools = run.tool_calls
    expected_args = task.get("expected_args", {})

    first_tool_ok = (
        bool(expected_tools) and bool(actual_tools) and actual_tools[0] == expected_tools[0]
    )
    exact_sequence = actual_tools == expected_tools

    if expected_tools:
        hit = set(actual_tools) & set(expected_tools)
        tool_recall = len(hit) / len(expected_tools)
    else:
        tool_recall = 1.0 if not actual_tools else 0.0

    if actual_tools:
        relevant = [t for t in actual_tools if t in expected_tools]
        tool_precision = len(relevant) / len(actual_tools)
    else:
        tool_precision = 1.0 if not expected_tools else 0.0

    arg_ok: bool | None = None
    if expected_args:
        arg_ok = True
        for key, value in expected_args.items():
            hits = [step for step in run.steps if key in step.arguments]
            if not hits:
                arg_ok = False
                break
            if key in STRICT_ARG_KEYS:
                matched = any(
                    str(step.arguments.get(key, "")).strip() == str(value).strip()
                    for step in hits
                )
            else:  # 自由文本参数（query）：非空即可
                matched = any(str(step.arguments.get(key, "")).strip() for step in hits)
            if not matched:
                arg_ok = False
                break

    return {
        "id": task["id"],
        "category": task["category"],
        "request": task["request"],
        "expected_tools": expected_tools,
        "actual_tools": actual_tools,
        "first_tool_ok": first_tool_ok,
        "exact_sequence": exact_sequence,
        "tool_recall": round(tool_recall, 3),
        "tool_precision": round(tool_precision, 3),
        "args_ok": arg_ok,
        "stopped_reason": run.stopped_reason,
        "final_answer": run.final_answer[:200],
        "steps": run.trace(),
    }


async def selection_quality(live: bool, registry: ToolRegistry, tmp_path: Path) -> dict[str, Any]:
    tasks = load_tasks()
    rows: list[dict[str, Any]] = []

    for task in tasks:
        if task["category"] == "unsupported":
            llm: Any = OracleLLM(task)
        elif live:
            from src.integrations.minimax_client import MiniMaxClient

            llm = MiniMaxClient()
        else:
            llm = OracleLLM(task)

        run = await ToolCallingAgent(
            registry, llm_client=llm, max_steps=4, max_consecutive_errors=2
        ).run(task["request"])
        rows.append(_selection_metrics(task, run))

    if live:
        try:
            await llm.close()  # type: ignore[attr-defined]
        except Exception:
            pass

    scored = [r for r in rows if r["expected_tools"]]
    readonly_rows = [
        r
        for r in scored
        if not any(tool == "create_action_item" for tool in r["expected_tools"])
    ]
    write_rows = [
        r
        for r in scored
        if any(tool == "create_action_item" for tool in r["expected_tools"])
    ]

    def first_tool_accuracy(subset: list[dict[str, Any]]) -> float | None:
        if not subset:
            return None
        return round(sum(1 for r in subset if r["first_tool_ok"]) / len(subset), 3)

    return {
        "mode": "live（真实 LLM）" if live else "oracle 自检（脚本化 LLM，不代表模型能力）",
        "tasks": len(rows),
        "first_tool_accuracy": first_tool_accuracy(scored),
        "readonly_first_tool_accuracy": first_tool_accuracy(readonly_rows),
        "write_tasks": len(write_rows),
        "write_tasks_note": (
            "评测期间写工具被策略隐藏（不设 MCP_ALLOW_WRITE），因此写任务只统计"
            "「模型是否尝试调用写工具」，不会真的在 Jira/飞书建单"
        ),
        "tool_recall": round(sum(r["tool_recall"] for r in scored) / len(scored), 3)
        if scored
        else None,
        "tool_precision": round(
            sum(r["tool_precision"] for r in scored) / len(scored), 3
        )
        if scored
        else None,
        "exact_sequence_accuracy": round(
            sum(1 for r in scored if r["exact_sequence"]) / len(scored), 3
        )
        if scored
        else None,
        "arg_accuracy": round(
            sum(1 for r in scored if r["args_ok"] is True)
            / max(len([r for r in scored if r["args_ok"] is not None]), 1),
            3,
        ),
        "completion_rate": round(
            sum(1 for r in rows if r["stopped_reason"] == "final_answer") / len(rows), 3
        ),
        "fallback_rate": round(
            sum(1 for r in rows if r["stopped_reason"] == "max_steps") / len(rows), 3
        ),
        "rows": rows,
    }


# ----------------------------------------------------------------------
# 渲染与入口
# ----------------------------------------------------------------------

def render(report: dict[str, Any]) -> str:
    proto = report["protocol"]
    safety = report["safety"]
    quality = report["selection"]

    lines = [
        "## 工具调用评测结果",
        "",
        f"- 时间：{report['generated_at']}",
        f"- MCP 协议一致性：**{proto['passed']}/{proto['total']}** 项通过（真实子进程 stdio 传输）",
        f"- 熔断与权限：**{safety['passed']}/{safety['total']}** 项符合预期",
        f"- 工具选择：{quality['mode']}，{quality['tasks']} 条标注任务",
        "",
        "### 协议一致性明细",
        "",
        "| 检查项 | 结果 | 说明 |",
        "|--------|------|------|",
    ]
    for check in proto["checks"]:
        lines.append(
            f"| {check['check']} | {'✅' if check['ok'] else '❌'} | {check.get('detail', '')} |"
        )

    lines += [
        "",
        "### 熔断与权限行为",
        "",
        "| 场景 | 期望 | 实际 | 步数 |",
        "|------|------|------|------|",
    ]
    for case in safety["cases"]:
        lines.append(
            f"| {case['case']} | {case['expected']} | {case['stopped_reason']} "
            f"| {case['steps']} |"
        )

    lines += [
        "",
        "### 工具选择质量",
        "",
        "| 指标 | 数值 |",
        "|------|------|",
        f"| 首个工具准确率（全部任务） | {quality['first_tool_accuracy']} |",
        f"| 首个工具准确率（只读任务） | {quality['readonly_first_tool_accuracy']} |",
        f"| 工具集合召回 | {quality['tool_recall']} |",
        f"| 工具集合精确率 | {quality['tool_precision']} |",
        f"| 参数正确率 | {quality['arg_accuracy']} |",
        f"| 正常收敛率（给出 final_answer） | {quality['completion_rate']} |",
        f"| 触发步数上限（回落到总结） | {quality['fallback_rate']} |",
        f"| 工具序列完全一致（参考值） | {quality['exact_sequence_accuracy']} |",
        "",
        f"> 写任务共 {quality['write_tasks']} 条：{quality['write_tasks_note']}",
        "",
        "| 任务 | 类别 | 期望工具 | 实际工具 | 首个工具 | 工具召回 |",
        "|------|------|----------|----------|----------|----------|",
    ]
    for row in quality["rows"]:
        mark = "✅" if row["first_tool_ok"] else "—"
        lines.append(
            f"| {row['id']} | {row['category']} | {row['expected_tools']} "
            f"| {row['actual_tools']} | {mark} | {row['tool_recall']} |"
        )
    return "\n".join(lines)


async def main() -> int:
    parser = argparse.ArgumentParser(description="MCP 工具调用评测")
    parser.add_argument("--live", action="store_true", help="用真实 LLM 评工具选择质量")
    parser.add_argument("--keep-index", action="store_true", help="保留临时目录（调试用）")
    args = parser.parse_args()

    import tempfile

    tmp_root = Path(tempfile.mkdtemp(prefix="meetgraph-tools-eval-"))
    os.environ["MCP_AUDIT_LOG"] = str(tmp_root / "audit.jsonl")
    os.environ["REPORTS_DIR"] = str(tmp_root / "reports")
    os.environ["SYNC_LEDGER_DB"] = str(tmp_root / "ledger.db")
    os.environ.pop("MCP_ALLOW_WRITE", None)  # 评测期间绝不写外部系统
    (tmp_root / "reports").mkdir(parents=True, exist_ok=True)

    registry = build_default_registry()
    report: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "protocol": await protocol_checks(),
        "safety": await safety_checks(tmp_root),
        "selection": await selection_quality(args.live, registry, tmp_root),
    }
    report["selection"]["registry_tools"] = [t["name"] for t in registry.llm_catalog()]

    out_dir = REPO_ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"tools-eval-{datetime.now():%Y%m%d-%H%M%S}.json"
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(render(report))
    print(f"\n原始结果已写入：{out_file}")
    if not args.keep_index:
        import shutil

        shutil.rmtree(tmp_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
