"""自主工具调用循环测试：happy path、参数纠错、熔断、超时、权限。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from src.agents.tool_agent import ToolCallingAgent
from src.mcp.audit import AuditLog
from src.mcp.policy import ToolPolicy
from src.mcp.registry import ToolRegistry, ToolSpec
from tests.fakes import FakeToolLLM


async def _lookup(args: dict[str, Any]) -> dict[str, Any]:
    return {"value": f"looked-up:{args['query']}"}


async def _boom(args: dict[str, Any]) -> dict[str, Any]:
    raise RuntimeError("backend exploded")


async def _slow(args: dict[str, Any]) -> dict[str, Any]:
    await asyncio.sleep(1.0)
    return {"value": "too late"}


async def _write(args: dict[str, Any]) -> dict[str, Any]:
    return {"created": True, "task": args["task"]}


def build_registry(
    tmp_path: Path, allow_write: bool = True, allowlist: tuple[str, ...] = ()
) -> ToolRegistry:
    registry = ToolRegistry(
        policy=ToolPolicy(allow_write=allow_write, allowlist=allowlist),
        audit=AuditLog(tmp_path / "audit.jsonl"),
    )
    string_arg = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    registry.register(
        ToolSpec(name="lookup", description="查资料", input_schema=string_arg, handler=_lookup)
    )
    registry.register(
        ToolSpec(name="boom", description="总会失败", input_schema=string_arg, handler=_boom)
    )
    registry.register(
        ToolSpec(name="slow", description="很慢", input_schema=string_arg, handler=_slow)
    )
    registry.register(
        ToolSpec(
            name="write",
            description="写操作",
            input_schema={
                "type": "object",
                "properties": {"task": {"type": "string"}},
                "required": ["task"],
            },
            handler=_write,
            readonly=False,
        )
    )
    return registry


@pytest.fixture
def registry(tmp_path: Path) -> ToolRegistry:
    return build_registry(tmp_path)


# ----------------------------------------------------------------------

async def test_happy_path_calls_tool_then_answers(registry: ToolRegistry) -> None:
    llm = FakeToolLLM(
        [
            {"action": "lookup", "arguments": {"query": "版本冻结"}},
            {"final_answer": "版本冻结后只能修缺陷 [1]"},
        ]
    )
    run = await ToolCallingAgent(registry, llm_client=llm).run("版本冻结后能改什么？")

    assert run.stopped_reason == "final_answer"
    assert run.final_answer == "版本冻结后只能修缺陷 [1]"
    assert run.tool_calls == ["lookup"]
    assert run.steps[0].ok is True
    assert "looked-up:版本冻结" in run.steps[0].observation


async def test_prompt_carries_catalog_and_observations(registry: ToolRegistry) -> None:
    llm = FakeToolLLM(
        [
            {"action": "lookup", "arguments": {"query": "x"}},
            {"final_answer": "done"},
        ]
    )
    await ToolCallingAgent(registry, llm_client=llm).run("查一下 x")

    first_prompt = llm.calls[0]["messages"][-1]["content"]
    assert "## 可用工具" in first_prompt
    assert "lookup" in first_prompt and "write" in first_prompt

    second_prompt = llm.calls[1]["messages"][-1]["content"]
    assert "## 已执行步骤" in second_prompt
    assert "looked-up:x" in second_prompt


async def test_unknown_tool_is_fed_back_then_recovers(registry: ToolRegistry) -> None:
    llm = FakeToolLLM(
        [
            {"action": "teleport", "arguments": {}},
            {"action": "lookup", "arguments": {"query": "y"}},
            {"final_answer": "ok"},
        ]
    )
    run = await ToolCallingAgent(registry, llm_client=llm).run("用不存在的工具")

    assert run.steps[0].error == "unknown_tool"
    assert "可用工具" in run.steps[0].observation
    assert run.stopped_reason == "final_answer"
    assert run.tool_calls == ["teleport", "lookup"]


async def test_invalid_arguments_are_reported_for_self_repair(registry: ToolRegistry) -> None:
    llm = FakeToolLLM(
        [
            {"action": "lookup", "arguments": {}},          # 缺必填 query
            {"action": "lookup", "arguments": {"query": "z"}},
            {"final_answer": "修正后成功"},
        ]
    )
    run = await ToolCallingAgent(registry, llm_client=llm).run("参数写错了会怎样")

    assert run.steps[0].error == "invalid_arguments"
    assert "参数校验失败" in run.steps[0].observation
    assert run.steps[1].ok is True


async def test_repeated_identical_call_is_broken(registry: ToolRegistry) -> None:
    llm = FakeToolLLM([{"action": "lookup", "arguments": {"query": "same"}}])
    run = await ToolCallingAgent(registry, llm_client=llm, max_steps=5).run("死循环测试")

    assert run.stopped_reason == "repeat_detected"
    assert len(run.steps) == 1
    assert "repeated" in run.error


async def test_max_steps_stops_and_returns_fallback(registry: ToolRegistry) -> None:
    llm = FakeToolLLM(
        [
            {"action": "lookup", "arguments": {"query": "a"}},
            {"action": "lookup", "arguments": {"query": "b"}},
        ]
    )
    run = await ToolCallingAgent(registry, llm_client=llm, max_steps=2).run("一直查下去")

    assert run.stopped_reason == "max_steps"
    assert len(run.steps) == 2
    assert "未在步数上限内得出" in run.final_answer
    assert "looked-up:a" in run.final_answer


async def test_consecutive_tool_errors_stop_the_loop(registry: ToolRegistry) -> None:
    llm = FakeToolLLM(
        [
            {"action": "boom", "arguments": {"query": "a"}},
            {"action": "boom", "arguments": {"query": "b"}},
            {"action": "boom", "arguments": {"query": "c"}},
        ]
    )
    run = await ToolCallingAgent(
        registry, llm_client=llm, max_steps=5, max_consecutive_errors=2
    ).run("工具一直失败")

    assert run.stopped_reason == "tool_errors"
    assert len(run.steps) == 2
    assert all(step.error for step in run.steps)


async def test_tool_timeout_is_reported(registry: ToolRegistry) -> None:
    llm = FakeToolLLM(
        [
            {"action": "slow", "arguments": {"query": "a"}},
            {"final_answer": "超时后放弃"},
        ]
    )
    run = await ToolCallingAgent(registry, llm_client=llm, tool_timeout=0.05).run("很慢的工具")

    assert run.steps[0].error == "timeout"
    assert "超时" in run.steps[0].observation
    assert run.stopped_reason == "final_answer"


async def test_disabled_write_tool_is_not_advertised(tmp_path: Path) -> None:
    """写工具被策略关掉时，压根不出现在工具目录里（模型看不到就不会去调）。"""
    registry = build_registry(tmp_path, allow_write=False)

    advertised = [tool["name"] for tool in registry.llm_catalog()]
    assert "write" not in advertised
    assert advertised == ["lookup", "boom", "slow"]

    llm = FakeToolLLM(
        [
            {"action": "write", "arguments": {"task": "建个单"}},
            {"final_answer": "当前没有可用的建单工具"},
        ]
    )
    run = await ToolCallingAgent(
        registry, llm_client=llm, max_steps=3, max_consecutive_errors=2
    ).run("帮我建个单")

    assert run.steps[0].error == "unknown_tool"
    assert run.stopped_reason == "final_answer"


async def test_failing_write_tool_counts_as_failure(tmp_path: Path) -> None:
    """写工具已开启但执行失败（如缺凭据）时，按工具失败处理并可熔断。"""
    registry = build_registry(tmp_path, allow_write=True)

    async def failing_write(args: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("no credentials configured")

    registry._tools["write"].handler = failing_write  # noqa: SLF001 - 测试注错

    llm = FakeToolLLM([{"action": "write", "arguments": {"task": "建个单"}}])
    run = await ToolCallingAgent(
        registry, llm_client=llm, max_steps=3, max_consecutive_errors=1
    ).run("帮我建个单")

    assert run.steps[0].ok is False
    assert "no credentials" in run.steps[0].error
    assert run.stopped_reason == "tool_errors"


async def test_llm_outage_ends_run_with_reason(registry: ToolRegistry) -> None:
    llm = FakeToolLLM([{"final_answer": "never"}], fail_from=0)
    run = await ToolCallingAgent(registry, llm_client=llm).run("LLM 挂了")

    assert run.stopped_reason == "llm_error"
    assert "FakeToolLLM" in run.error
    assert run.steps == []


async def test_no_available_tools_short_circuits(tmp_path: Path) -> None:
    registry = build_registry(tmp_path, allowlist=("nonexistent",))
    run = await ToolCallingAgent(registry, llm_client=FakeToolLLM([])).run("没有任何工具")

    assert run.stopped_reason == "tool_errors"
    assert "no tools available" in run.error


async def test_audit_records_agent_calls(registry: ToolRegistry) -> None:
    llm = FakeToolLLM(
        [
            {"action": "lookup", "arguments": {"query": "audit"}},
            {"final_answer": "done"},
        ]
    )
    await ToolCallingAgent(registry, llm_client=llm).run("审计检查")

    records = registry.audit.tail(5)
    assert records[-1]["tool"] == "lookup"
    assert records[-1]["actor"] == "tool-agent"
    assert records[-1]["status"] == "ok"


async def test_run_serializes_to_dict(registry: ToolRegistry) -> None:
    llm = FakeToolLLM(
        [
            {"action": "lookup", "arguments": {"query": "x"}},
            {"final_answer": "yes"},
        ]
    )
    payload = (await ToolCallingAgent(registry, llm_client=llm).run("序列化")).to_dict()

    assert payload["stopped_reason"] == "final_answer"
    assert payload["steps"][0]["tool"] == "lookup"
    assert payload["final_answer"] == "yes"
