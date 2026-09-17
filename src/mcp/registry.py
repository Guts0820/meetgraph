"""工具注册表：一份 Schema，MCP 与 LLM 两处消费。

设计要点：

1. **单一事实来源**：``ToolSpec`` 既是 MCP ``tools/list`` 的返回内容，也是喂给
   LLM 的工具目录（``llm_catalog()``），避免两套定义漂移；
2. **调用前校验**：按 JSON Schema 校验参数（有 jsonschema 就用它，没有则退回内置
   的最小校验），校验失败返回 ``INVALID_PARAMS`` 而不是抛栈；
3. **权限与审计**：策略拒绝 → 记 ``denied``；执行异常 → 记 ``error``；参数摘要
   落审计日志，不落明文；
4. **错误回传给模型**：工具失败以 MCP ``isError`` 结果返回（不是 JSON-RPC 层错误），
   这样 Agent 侧可以把错误当 observation 继续推理，实现自修复。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from loguru import logger

from .audit import AuditLog
from .policy import ToolDenied, ToolPolicy
from .protocol import INVALID_PARAMS, JsonRpcError

try:  # 可选依赖：装了就用标准校验器
    import jsonschema  # type: ignore
except Exception:  # pragma: no cover - 环境没装时走内置校验
    jsonschema = None


class ToolInputError(Exception):
    """工具参数在业务上不合法（例如报告不存在）。"""


@dataclass
class ToolSpec:
    """一个可被 MCP 与 LLM 共同使用的工具定义。"""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Awaitable[Any]]
    readonly: bool = True

    def mcp_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


@dataclass
class ToolCallResult:
    """一次工具调用的结果（同时服务于 MCP 响应与 Agent 循环）。"""

    tool: str
    ok: bool
    data: Any = None
    error: str = ""
    denied: bool = False
    duration_ms: float = 0.0
    audit_status: str = "ok"

    def text(self) -> str:
        """给 LLM 看的文本形式。"""
        import json

        if not self.ok:
            return f"[tool_error] {self.error}"
        return json.dumps(self.data, ensure_ascii=False)

    def mcp_payload(self) -> dict[str, Any]:
        import json

        text = (
            json.dumps(self.data, ensure_ascii=False, default=str)
            if self.ok
            else f"ERROR: {self.error}"
        )
        return {"content": [{"type": "text", "text": text}], "isError": not self.ok}


def validate_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> None:
    """按 JSON Schema 校验参数；失败抛 ``JsonRpcError(INVALID_PARAMS)``。"""
    if jsonschema is not None:
        try:
            jsonschema.validate(instance=arguments, schema=schema)
        except jsonschema.ValidationError as e:  # type: ignore[attr-defined]
            raise JsonRpcError(
                INVALID_PARAMS, f"invalid arguments: {e.message}", {"path": list(e.path)}
            ) from e
        return

    # 内置最小校验：必填、类型、枚举、范围
    for key in schema.get("required", []):
        if key not in arguments:
            raise JsonRpcError(INVALID_PARAMS, f"missing required argument: {key}")

    properties = schema.get("properties", {})
    for key, value in arguments.items():
        rule = properties.get(key)
        if not rule:
            continue
        expected = rule.get("type")
        if expected == "string" and not isinstance(value, str):
            raise JsonRpcError(INVALID_PARAMS, f"{key} must be a string")
        if expected == "integer" and not isinstance(value, int):
            raise JsonRpcError(INVALID_PARAMS, f"{key} must be an integer")
        if expected == "array" and not isinstance(value, list):
            raise JsonRpcError(INVALID_PARAMS, f"{key} must be an array")
        if "enum" in rule and value not in rule["enum"]:
            raise JsonRpcError(
                INVALID_PARAMS, f"{key} must be one of {rule['enum']}"
            )
        if isinstance(value, int) and "minimum" in rule and value < rule["minimum"]:
            raise JsonRpcError(INVALID_PARAMS, f"{key} must be >= {rule['minimum']}")
        if isinstance(value, int) and "maximum" in rule and value > rule["maximum"]:
            raise JsonRpcError(INVALID_PARAMS, f"{key} must be <= {rule['maximum']}")


class ToolRegistry:
    """工具注册表。"""

    def __init__(
        self,
        policy: ToolPolicy | None = None,
        audit: AuditLog | None = None,
    ) -> None:
        self.policy = policy or ToolPolicy.from_env()
        self.audit = audit or AuditLog()
        self._tools: dict[str, ToolSpec] = {}

    # ------------------------------------------------------------------
    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"duplicate tool: {spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        spec = self._tools.get(name)
        if spec is None:
            raise JsonRpcError(
                INVALID_PARAMS,
                f"unknown tool: {name}",
                {"available": sorted(self._tools)},
            )
        return spec

    def list_specs(self) -> list[ToolSpec]:
        return [spec for spec in self._tools.values() if self.policy.allows(spec.name, spec.readonly)]

    def tools_list_payload(self) -> dict[str, Any]:
        return {"tools": [spec.mcp_payload() for spec in self.list_specs()]}

    def llm_catalog(self) -> list[dict[str, Any]]:
        """给 LLM 的工具目录（与 MCP 暴露的工具完全一致）。"""
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.input_schema,
                "readonly": spec.readonly,
            }
            for spec in self.list_specs()
        ]

    # ------------------------------------------------------------------
    async def call(
        self, name: str, arguments: dict[str, Any] | None = None, actor: str = "mcp"
    ) -> ToolCallResult:
        """执行工具：校验 → 鉴权 → 执行 → 审计。"""
        args = arguments or {}
        spec = self.get(name)

        try:
            validate_arguments(spec.input_schema, args)
            self.policy.check(spec.name, spec.readonly)
        except JsonRpcError:
            self.audit.append(name, actor=actor, status="error", arguments=args, error="invalid_arguments")
            raise
        except ToolDenied as e:
            self.audit.append(name, actor=actor, status="denied", arguments=args, error=str(e))
            logger.warning(f"[MCP] tool denied: {e}")
            return ToolCallResult(
                tool=name, ok=False, error=str(e), denied=True, audit_status="denied"
            )

        started = time.perf_counter()
        try:
            data = await spec.handler(args)
        except ToolInputError as e:
            duration = (time.perf_counter() - started) * 1000
            self.audit.append(
                name, actor=actor, status="error", duration_ms=duration, arguments=args, error=str(e)
            )
            return ToolCallResult(
                tool=name, ok=False, error=str(e), duration_ms=duration, audit_status="error"
            )
        except Exception as e:
            duration = (time.perf_counter() - started) * 1000
            self.audit.append(
                name, actor=actor, status="error", duration_ms=duration, arguments=args, error=repr(e)
            )
            logger.error(f"[MCP] tool {name} failed: {e}")
            return ToolCallResult(
                tool=name,
                ok=False,
                error=f"{type(e).__name__}: {e}",
                duration_ms=duration,
                audit_status="error",
            )

        duration = (time.perf_counter() - started) * 1000
        self.audit.append(
            name, actor=actor, status="ok", duration_ms=duration, arguments=args
        )
        return ToolCallResult(tool=name, ok=True, data=data, duration_ms=duration)
