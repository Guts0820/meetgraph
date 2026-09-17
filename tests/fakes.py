"""测试与离线评测共用的假实现。

目的是让单元测试和评测脚本完全不依赖外部服务：不调用 LLM、不写 Jira/飞书，
但仍然走真实的 Agent / Graph / 报告落盘代码路径。
"""

from __future__ import annotations

import asyncio
from typing import Any

DEMO_ACTION_ITEMS: list[dict[str, str]] = [
    {
        "assignee": "李明",
        "task": "整理Q3详细预算方案",
        "deadline": "下周五",
        "priority": "high",
        "context": "张总要求提交审批",
    },
    {
        "assignee": "王芳",
        "task": "拟定招聘JD",
        "deadline": "本周三",
        "priority": "medium",
        "context": "为3名高级算法工程师岗位招聘做准备",
    },
    {
        "assignee": "赵伟",
        "task": "跟进服务器采购",
        "deadline": "",
        "priority": "medium",
        "context": "对比供应商并给出采购方案",
    },
]

DEMO_SUMMARY: dict[str, Any] = {
    "title": "Q3 预算评审会议",
    "date": "2026-01-01",
    "participants": ["张总", "李明", "王芳", "赵伟"],
    "topics": [
        {
            "title": "预算执行情况",
            "discussion_points": ["Q2 执行率 87%", "研发投入占比 42%"],
            "participants": ["李明"],
            "conclusion": "预算执行符合预期",
        }
    ],
    "decisions": ["Q3 预算上调 15%"],
    "next_steps": ["李明提交预算方案"],
}

DEMO_INSIGHT: dict[str, Any] = {
    "overall_sentiment": "positive",
    "sentiment_score": 0.8,
    "efficiency_score": 8.0,
    "keywords": ["预算", "招聘", "采购"],
    "highlights": ["议题聚焦，决策明确"],
    "suggestions": ["待办截止时间可以再具体一些"],
}


class FakeLLM:
    """按 prompt 内容返回结构化结果的假 LLM。

    Args:
        delay: 每次调用的模拟耗时（秒），用于测量并行编排收益。
        fail_times: 前 N 次调用抛异常，用于验证降级路径；``None`` 表示一直失败。
    """

    def __init__(
        self,
        delay: float = 0.0,
        fail_times: int | None = 0,
        action_items: list[dict[str, str]] | None = None,
    ) -> None:
        self.delay = delay
        self.calls: list[dict[str, Any]] = []
        self._remaining_failures = fail_times
        self._action_items = (
            action_items if action_items is not None else DEMO_ACTION_ITEMS
        )

    @property
    def call_count(self) -> int:
        return len(self.calls)

    async def chat_json(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> dict[str, Any]:
        prompt = messages[-1]["content"] if messages else ""
        self.calls.append({"prompt": prompt, "kwargs": kwargs})

        if self.delay:
            await asyncio.sleep(self.delay)

        if self._remaining_failures is None:
            raise RuntimeError("FakeLLM: simulated LLM outage")
        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            raise RuntimeError("FakeLLM: simulated LLM outage")

        if "行动项" in prompt:
            return {"action_items": [dict(item) for item in self._action_items]}
        if "会议纪要" in prompt:
            return dict(DEMO_SUMMARY)
        return dict(DEMO_INSIGHT)


class FakeAnswerLLM:
    """假 LLM（``chat`` 接口）：返回固定的答案文本，并记录收到的 prompt。

    用于 RAG 问答测试——要断言的是「prompt 里有没有术语约束 / 参考资料编号」，
    而不是模型生成质量。
    """

    def __init__(self, answer: str = "根据资料，规则如下 [1]") -> None:
        self.answer = answer
        self.calls: list[dict[str, Any]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return self.answer


class FakeToolLLM:
    """按队列返回决策的假 LLM（用于自主工具调用循环测试）。

    ``decisions`` 里每个元素会被依次返回；用完后重复返回最后一个，便于构造
    「一直调用同一工具」这类死循环场景。
    """

    def __init__(self, decisions: list[Any], fail_from: int | None = None) -> None:
        self.decisions = list(decisions)
        self.fail_from = fail_from
        self.calls: list[dict[str, Any]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    async def chat_json(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        index = len(self.calls)
        self.calls.append({"messages": messages, "kwargs": kwargs})
        if self.fail_from is not None and index >= self.fail_from:
            raise RuntimeError("FakeToolLLM: simulated LLM outage")
        if not self.decisions:
            return {"final_answer": "无决策"}
        return self.decisions[min(index, len(self.decisions) - 1)]


class FakeTargetClient:
    """计数型的 Jira / 飞书假客户端。

    ``is_enabled`` 由构造参数决定，调用次数记录在 ``created`` 里，用于断言
    幂等逻辑确实阻止了重复写入。
    """

    def __init__(self, enabled: bool = True, fail: bool = False) -> None:
        self._enabled = enabled
        self.fail = fail
        self.created: list[str] = []
        self.messages: list[str] = []

    @property
    def is_enabled(self) -> bool:
        return self._enabled


class FakeJiraClient(FakeTargetClient):
    def create_issue(self, summary: str, **kwargs: Any) -> dict[str, str]:
        if self.fail:
            raise RuntimeError("FakeJira: simulated failure")
        key = f"MEET-{len(self.created) + 101}"
        self.created.append(key)
        return {"key": key, "id": key, "url": f"https://example.atlassian.net/browse/{key}"}

    def resolve_user(self, display_name: str) -> str | None:
        return {"李明": "li.ming", "王芳": "wang.fang"}.get(display_name)


class FakeFeishuClient(FakeTargetClient):
    async def create_task(self, summary: str, **kwargs: Any) -> dict[str, str]:
        if self.fail:
            raise RuntimeError("FakeFeishu: simulated failure")
        task_id = f"t{len(self.created) + 1000}"
        self.created.append(task_id)
        return {"task_id": task_id, "data": {}}

    async def send_meeting_summary(
        self,
        title: str,
        summary_md: str,
        action_items_md: str,
        insights_md: str,
    ) -> bool:
        self.messages.append(title)
        if self.fail:
            raise RuntimeError("FakeFeishu: simulated failure")
        return True
