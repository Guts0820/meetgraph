"""MCP 的 Resources 与 Prompts 目录。

- **Resources**：会议报告（`meeting://report/{meeting_id}`）作为只读数据源，客户端
  可以列出来、按 URI 读原文——适合「先浏览有哪些会议、再决定读哪份」的交互。
- **Prompts**：一个 `summarize_meeting` 模板，把「读报告 → 产出结构化纪要」这套提示
  词固化成服务端资产，避免每个客户端各写一份。

Resources 只暴露报告文件，不暴露 `data/` 下的其它内容（索引、台账），避免把内部
存储结构变成对外契约。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .protocol import INVALID_PARAMS, JsonRpcError
from .tools import _safe_meeting_id, reports_dir

REPORT_PREFIX = "meeting://report/"


def _report_paths() -> list[Path]:
    return sorted(reports_dir().glob("meeting-report-*.md"))


def list_resources() -> dict[str, Any]:
    """列出所有会议报告资源。"""
    resources = []
    for path in _report_paths():
        meeting_id = path.stem.replace("meeting-report-", "")
        resources.append(
            {
                "uri": f"{REPORT_PREFIX}{meeting_id}",
                "name": f"会议报告 {meeting_id}",
                "description": "会议纪要 + 待办 + 洞察 + 相关历史决议",
                "mimeType": "text/markdown",
            }
        )
    return {"resources": resources}


def read_resource(uri: str) -> dict[str, Any]:
    """按 URI 读取报告原文。"""
    if not isinstance(uri, str) or not uri.startswith(REPORT_PREFIX):
        raise JsonRpcError(
            INVALID_PARAMS,
            f"unsupported resource uri: {uri!r}",
            {"supported_prefix": REPORT_PREFIX},
        )

    meeting_id = _safe_meeting_id(uri[len(REPORT_PREFIX) :])
    path = reports_dir() / f"meeting-report-{meeting_id}.md"
    if not path.exists():
        raise JsonRpcError(
            INVALID_PARAMS,
            f"resource not found: {uri}",
            {"available": [r["uri"] for r in list_resources()["resources"][:10]]},
        )

    return {
        "contents": [
            {
                "uri": uri,
                "mimeType": "text/markdown",
                "text": path.read_text(encoding="utf-8"),
            }
        ]
    }


PROMPTS: dict[str, dict[str, Any]] = {
    "summarize_meeting": {
        "name": "summarize_meeting",
        "description": "读取某场会议的报告并产出对外可发的结构化纪要（含责任人、截止时间、风险）。",
        "arguments": [
            {
                "name": "meeting_id",
                "description": "要总结的会议 ID",
                "required": True,
            },
            {
                "name": "audience",
                "description": "读者：管理层 / 项目组 / 客户",
                "required": False,
            },
        ],
    }
}


def list_prompts() -> dict[str, Any]:
    return {"prompts": list(PROMPTS.values())}


def get_prompt(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """渲染提示模板为 MCP 的 messages 结构。"""
    template = PROMPTS.get(name)
    if template is None:
        raise JsonRpcError(
            INVALID_PARAMS, f"unknown prompt: {name}", {"available": sorted(PROMPTS)}
        )

    args = arguments or {}
    meeting_id = _safe_meeting_id(args.get("meeting_id") or "")
    if not meeting_id:
        raise JsonRpcError(INVALID_PARAMS, "meeting_id is required")
    audience = str(args.get("audience") or "项目组")

    text = (
        f"请读取资源 meeting://report/{meeting_id}，为「{audience}」产出一份结构化会议纪要：\n"
        f"1. 会议结论（不超过 5 条，每条注明依据）；\n"
        f"2. 待办表：负责人 / 任务 / 截止时间 / 优先级；\n"
        f"3. 风险与阻塞项；\n"
        f"4. 与历史决议不一致的地方（如与之前会议决定冲突，请指出）。\n"
        f"使用公司标准术语，不要改写专有名词。"
    )
    return {
        "description": template["description"],
        "messages": [{"role": "user", "content": {"type": "text", "text": text}}],
    }
