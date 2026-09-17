"""自主工具调用循环（ReAct 风格）。

和主流水线的区别：主流水线是**确定性编排**（谁先谁后写死在图里），这里让 LLM
自己决定「下一步调用哪个工具、传什么参数」，直到它给出最终答案或触发熔断。

工程上真正要处理的是失败路径，而不是 happy path：

| 风险 | 处理 |
|------|------|
| 调用不存在的工具 | 回 observation 告知可用工具，让模型自我修正 |
| 参数不合法 | 复用 registry 的 Schema 校验，把错误回传（模型常常能改对） |
| 工具执行失败 | 错误作为 observation 继续；连续失败达上限则停止 |
| 工具超时 | ``asyncio.wait_for`` 限时，超时算一次失败 |
| 重复调用同一工具同一参数 | 熔断（相同 (tool, args) 第二次出现即停止） |
| 无限循环 | ``max_steps`` 上限 |

另外：**不使用原生 tool_calls**。当前 LLM 客户端（MiniMax chatcompletion_v2）不支持
OpenAI 风格的 tool_calls，因此用严格 JSON 决策协议实现同等效果；接支持原生
tool_calls 的模型时，把 ``_decide`` 换成解析 ``message.tool_calls`` 即可，循环体不变。
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from ..mcp.protocol import JsonRpcError
from ..mcp.registry import ToolRegistry

DECISION_SYSTEM_PROMPT = """你是会议助手 Agent，通过调用工具完成任务。

每一步你只能输出一个 JSON 对象，二选一：
1. 需要调用工具：{"action": "工具名", "arguments": {参数}}
2. 信息已经足够：{"final_answer": "最终回答"}

规则：
- 只能使用「可用工具」里列出的工具名，不要编造；
- 参数必须符合工具的 JSON Schema；
- 只有在拿到足够信息后才输出 final_answer，并说明依据；
- 不要重复调用完全相同的工具与参数（重复会被熔断）。"""


@dataclass
class ToolStep:
    """一步工具调用。"""

    index: int
    tool: str
    arguments: dict[str, Any]
    ok: bool
    observation: str
    duration_ms: float = 0.0
    error: str = ""
    denied: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "tool": self.tool,
            "arguments": self.arguments,
            "ok": self.ok,
            "observation": self.observation[:500],
            "duration_ms": self.duration_ms,
            "error": self.error,
            "denied": self.denied,
        }


@dataclass
class AgentRun:
    """一次自主执行的结果。"""

    request: str
    final_answer: str = ""
    steps: list[ToolStep] = field(default_factory=list)
    stopped_reason: str = "final_answer"  # final_answer / max_steps / repeat_detected / tool_errors / llm_error
    error: str = ""

    @property
    def tool_calls(self) -> list[str]:
        return [step.tool for step in self.steps]

    def trace(self) -> list[dict[str, Any]]:
        return [step.to_dict() for step in self.steps]

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request,
            "final_answer": self.final_answer,
            "stopped_reason": self.stopped_reason,
            "error": self.error,
            "steps": self.trace(),
        }


class ToolCallingAgent:
    """让 LLM 自主选择工具并执行。"""

    def __init__(
        self,
        registry: ToolRegistry,
        llm_client: Any | None = None,
        max_steps: int = 5,
        tool_timeout: float = 20.0,
        max_consecutive_errors: int = 3,
    ) -> None:
        self.registry = registry
        self._llm = llm_client
        self.max_steps = max_steps
        self.tool_timeout = tool_timeout
        self.max_consecutive_errors = max_consecutive_errors

    @property
    def llm(self) -> Any:
        if self._llm is None:
            from ..integrations.minimax_client import MiniMaxClient

            self._llm = MiniMaxClient()
        return self._llm

    # ------------------------------------------------------------------
    async def run(self, request: str, actor: str = "tool-agent") -> AgentRun:
        run = AgentRun(request=request)
        catalog = {tool["name"]: tool for tool in self.registry.llm_catalog()}
        if not catalog:
            run.stopped_reason = "tool_errors"
            run.error = "no tools available (check MCP_ALLOW_WRITE / MCP_TOOL_ALLOWLIST)"
            return run

        seen: set[str] = set()
        consecutive_errors = 0

        for step_index in range(1, self.max_steps + 1):
            try:
                decision = await self._decide(request, run, catalog)
            except Exception as e:  # LLM 不可用：不算工具失败，直接结束并如实上报
                logger.error(f"[ToolAgent] LLM decision failed: {e}")
                run.stopped_reason = "llm_error"
                run.error = f"{type(e).__name__}: {e}"
                return run

            if "final_answer" in decision:
                run.final_answer = str(decision["final_answer"]).strip()
                run.stopped_reason = "final_answer"
                return run

            tool_name = str(decision.get("action", "")).strip()
            arguments = decision.get("arguments") or decision.get("args") or {}
            if not isinstance(arguments, dict):
                arguments = {}

            if tool_name not in catalog:
                step = ToolStep(
                    index=step_index,
                    tool=tool_name or "(missing)",
                    arguments=arguments,
                    ok=False,
                    observation=(
                        f"未知工具 {tool_name!r}；可用工具：{sorted(catalog)}"
                    ),
                    error="unknown_tool",
                )
                run.steps.append(step)
                consecutive_errors += 1
                if consecutive_errors >= self.max_consecutive_errors:
                    run.stopped_reason = "tool_errors"
                    run.error = "too many consecutive invalid decisions"
                    return run
                continue

            signature = f"{tool_name}:{json.dumps(arguments, ensure_ascii=False, sort_keys=True)}"
            if signature in seen:
                run.stopped_reason = "repeat_detected"
                run.error = f"repeated identical call: {tool_name}"
                logger.warning(f"[ToolAgent] repeat detected, stopping: {tool_name}")
                return run
            seen.add(signature)

            step = await self._execute(step_index, tool_name, arguments, actor)

            if step.ok:
                consecutive_errors = 0
            else:
                consecutive_errors += 1
                if consecutive_errors >= self.max_consecutive_errors:
                    run.steps.append(step)
                    run.stopped_reason = "tool_errors"
                    run.error = f"{consecutive_errors} consecutive tool failures"
                    return run

            run.steps.append(step)

        run.stopped_reason = "max_steps"
        run.error = f"reached max_steps={self.max_steps} without a final answer"
        run.final_answer = self._fallback_answer(run)
        logger.warning("[ToolAgent] max steps reached, returning fallback answer")
        return run

    # ------------------------------------------------------------------
    async def _execute(
        self, index: int, tool_name: str, arguments: dict[str, Any], actor: str
    ) -> ToolStep:
        try:
            result = await asyncio.wait_for(
                self.registry.call(tool_name, arguments, actor=actor),
                timeout=self.tool_timeout,
            )
        except asyncio.TimeoutError:
            return ToolStep(
                index=index,
                tool=tool_name,
                arguments=arguments,
                ok=False,
                observation=f"工具 {tool_name} 超时（>{self.tool_timeout}s）",
                error="timeout",
            )
        except JsonRpcError as e:
            # 参数不合法：把校验信息回传，模型通常能自我修正
            return ToolStep(
                index=index,
                tool=tool_name,
                arguments=arguments,
                ok=False,
                observation=f"参数校验失败：{e.message}",
                error="invalid_arguments",
            )

        observation = result.text()
        return ToolStep(
            index=index,
            tool=tool_name,
            arguments=arguments,
            ok=result.ok,
            observation=observation,
            duration_ms=result.duration_ms,
            error=result.error,
            denied=result.denied,
        )

    async def _decide(
        self,
        request: str,
        run: AgentRun,
        catalog: dict[str, Any],
    ) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": DECISION_SYSTEM_PROMPT},
            {"role": "user", "content": self._build_prompt(request, run, catalog)},
        ]
        decision = await self.llm.chat_json(messages=messages, temperature=0.1, max_tokens=1024)
        if not isinstance(decision, dict):
            raise ValueError(f"decision must be a JSON object, got {type(decision).__name__}")
        return decision

    @staticmethod
    def _build_prompt(
        request: str, run: AgentRun, catalog: dict[str, Any]
    ) -> str:
        lines = [
            "## 可用工具",
            json.dumps(
                [
                    {
                        "name": name,
                        "description": tool["description"],
                        "parameters": tool["parameters"],
                        "readonly": tool["readonly"],
                    }
                    for name, tool in catalog.items()
                ],
                ensure_ascii=False,
                indent=1,
            ),
            "",
            "## 任务",
            request,
        ]

        if run.steps:
            lines += ["", "## 已执行步骤"]
            for step in run.steps:
                status = "成功" if step.ok else f"失败({step.error})"
                lines.append(
                    f"{step.index}. {step.tool} 参数={json.dumps(step.arguments, ensure_ascii=False)} "
                    f"→ {status}；结果摘要：{step.observation[:400]}"
                )

        lines += ["", "## 现在请输出下一步决策（严格 JSON，不要输出其它内容）"]
        return "\n".join(lines)

    @staticmethod
    def _fallback_answer(run: AgentRun) -> str:
        """循环被熔断时，至少把已拿到的事实整理出来，而不是返回空。"""
        if not run.steps:
            return ""
        parts = ["（未在步数上限内得出结论，以下是已获取的信息）"]
        for step in run.steps:
            if step.ok:
                parts.append(f"- {step.tool}: {step.observation[:300]}")
        return "\n".join(parts)
