"""
WebSocket 服务器 - 实时音频流接入和结果推送

支持两种模式:
1. 实时模式: 客户端通过 WebSocket 发送音频流，服务端实时返回转写结果
2. 文件模式: 通过 REST API 上传音频文件，异步处理后推送结果
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field

from ..graph.meeting_graph import run_meeting_pipeline
from ..models.schemas import MeetingStatus


load_dotenv()


class AskRequest(BaseModel):
    """知识库问答请求。"""

    question: str = Field(min_length=1, description="自然语言问题")
    top_k: int = Field(default=5, ge=1, le=20, description="检索条数")


def _rag_status() -> dict[str, Any]:
    """只读索引目录的 meta.json 判断 RAG 是否就绪（探活接口保持轻量，不加载模型）。"""
    from ..rag.ingest import DEFAULT_INDEX_DIR

    meta_path = Path(os.getenv("RAG_INDEX_DIR") or DEFAULT_INDEX_DIR) / "meta.json"
    if not meta_path.exists():
        return {"index": "not_built"}
    try:
        meta: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"index": "unreadable"}
    return {
        "index": "ready",
        "chunks": meta.get("chunks"),
        "docs": meta.get("docs"),
        "embedder": meta.get("embedder"),
        "built_at": meta.get("built_at"),
    }


app = FastAPI(
    title="MeetGraph 智能会议助手",
    description=(
        "基于 LangGraph 的多智能体会议纪要系统："
        "转写 / 纪要 / 待办 / 洞察 / 跟进 五个 Agent 协作完成会议全流程自动化"
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 存储活跃的 WebSocket 连接和会议状态
active_connections: dict[str, WebSocket] = {}
meeting_results: dict[str, dict] = {}


# ============================================================
# WebSocket 端点
# ============================================================

@app.websocket("/ws/meeting/{meeting_id}")
async def websocket_meeting(websocket: WebSocket, meeting_id: str):
    """
    WebSocket 会议端点

    协议:
    - 客户端发送: 音频二进制帧 / JSON控制消息
    - 服务端返回: JSON格式的处理结果

    控制消息:
    - {"type": "start"}: 开始录制
    - {"type": "stop"}: 停止录制，触发Pipeline处理
    - {"type": "ping"}: 心跳
    """
    await websocket.accept()
    active_connections[meeting_id] = websocket
    audio_buffer = bytearray()

    logger.info(f"WebSocket connected: {meeting_id}")

    try:
        await websocket.send_json({
            "type": "connected",
            "meeting_id": meeting_id,
            "message": "会议助手已连接，发送音频数据开始录制",
        })

        while True:
            data = await websocket.receive()

            if "bytes" in data and data["bytes"]:
                audio_buffer.extend(data["bytes"])
                await websocket.send_json({
                    "type": "recording",
                    "buffer_size": len(audio_buffer),
                })

            elif "text" in data and data["text"]:
                message = json.loads(data["text"])
                msg_type = message.get("type", "")

                if msg_type == "stop":
                    await websocket.send_json({
                        "type": "processing",
                        "message": "正在处理音频，请稍候...",
                    })

                    result = await run_meeting_pipeline(
                        meeting_id=meeting_id,
                        audio_data=bytes(audio_buffer),
                    )
                    meeting_results[meeting_id] = result

                    await _send_results(websocket, result)
                    audio_buffer.clear()

                elif msg_type == "demo":
                    await websocket.send_json({
                        "type": "processing",
                        "message": "运行演示模式...",
                    })
                    result = await run_meeting_pipeline(
                        meeting_id=meeting_id,
                        audio_data=b"",
                    )
                    meeting_results[meeting_id] = result
                    await _send_results(websocket, result)

                elif msg_type == "ping":
                    await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {meeting_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {meeting_id} - {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e),
            })
        except Exception:
            pass
    finally:
        active_connections.pop(meeting_id, None)


async def _send_results(websocket: WebSocket, state: dict):
    """将 Pipeline 处理结果分步推送给客户端"""
    # 转写结果
    transcript = state.get("transcript")
    if transcript:
        await websocket.send_json({
            "type": "transcript",
            "data": transcript.model_dump() if hasattr(transcript, "model_dump") else {},
        })

    # 摘要结果
    summary = state.get("summary")
    if summary:
        await websocket.send_json({
            "type": "summary",
            "data": summary.model_dump() if hasattr(summary, "model_dump") else {},
        })

    # 待办结果
    actions = state.get("actions")
    if actions:
        await websocket.send_json({
            "type": "actions",
            "data": actions.model_dump() if hasattr(actions, "model_dump") else {},
        })

    # 洞察结果
    insights = state.get("insights")
    if insights:
        await websocket.send_json({
            "type": "insights",
            "data": insights.model_dump() if hasattr(insights, "model_dump") else {},
        })

    # 跟进结果
    followup = state.get("followup")
    if followup:
        await websocket.send_json({
            "type": "followup",
            "data": followup.model_dump() if hasattr(followup, "model_dump") else {},
        })

    # 完成通知
    errors = state.get("errors", [])
    await websocket.send_json({
        "type": "completed",
        "meeting_id": state.get("meeting_id"),
        "status": state.get("status", MeetingStatus.COMPLETED),
        "errors": errors,
    })


# ============================================================
# REST API 端点
# ============================================================

@app.get("/")
async def root():
    return {
        "name": "MeetGraph 智能会议助手",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/healthz",
        "websocket": "ws://localhost:8000/ws/meeting/{meeting_id}",
    }


@app.get("/healthz")
async def healthz():
    """健康检查：报告版本、各类外部依赖是否就绪、当前活跃会议数。

    只做配置探测，不发起外部请求——探活接口不应该依赖第三方可用性。
    """
    from ..integrations.idempotency import DEFAULT_LEDGER_PATH

    ledger_path = os.getenv("SYNC_LEDGER_DB") or str(DEFAULT_LEDGER_PATH)
    return {
        "status": "ok",
        "version": "2.0.0",
        "active_meetings": len(active_connections),
        "stored_meetings": len(meeting_results),
        "integrations": {
            "llm": bool(os.getenv("MINIMAX_API_KEY") or os.getenv("OPENAI_API_KEY")),
            "jira": bool(
                os.getenv("JIRA_SERVER")
                and os.getenv("JIRA_EMAIL")
                and os.getenv("JIRA_API_TOKEN")
            ),
            "feishu": bool(
                (os.getenv("FEISHU_APP_ID") and os.getenv("FEISHU_APP_SECRET"))
                or os.getenv("FEISHU_WEBHOOK_URL")
            ),
            "whisper": os.getenv("WHISPER_MODEL_SIZE", "large-v2"),
        },
        "ledger": ledger_path,
        "rag": _rag_status(),
    }


@app.post("/api/v1/ask")
async def ask_knowledge(request: AskRequest) -> dict:
    """基于知识库回答问题：检索内部文档 + 历史会议纪要 + 术语表，返回带引用的答案。

    没有检索到相关内容时直接返回「资料中未提及」，**不会调用 LLM**——这是防幻觉的
    第一道闸。
    """
    from ..rag.service import get_qa

    qa = get_qa()
    answer = await qa.ask(request.question, top_k=request.top_k)
    return answer.to_dict()


@app.post("/api/v1/knowledge/reindex")
async def reindex_knowledge() -> dict:
    """重建知识库索引（语料目录见 data/knowledge 与 data/meetings）。

    索引构建是 CPU 密集的同步流程，放到线程里执行，避免阻塞事件循环。
    """
    from ..rag.service import reindex

    stats = await asyncio.to_thread(reindex)
    return {"status": "ok", **stats}


# ============================================================
# MCP over HTTP
# ============================================================
# 两种形态：
# 1. POST /mcp —— 一次请求一次响应（Streamable HTTP 的简化形态，客户端最省事）；
# 2. GET /mcp/sse + POST /mcp/messages —— 传统 SSE 传输：先开事件流拿到
#    session_id 与投递端点，再 POST 请求，响应从事件流里推回来。
# stdio 传输见 `python -m src.mcp.server`。

_mcp_server = None
_mcp_sessions: dict[str, asyncio.Queue] = {}
SSE_KEEPALIVE_SECONDS = 15.0


def sse_frame(event: str | None, data: str) -> str:
    """SSE 帧：``event:`` 行可选，``data:`` 行必需，空行结尾。"""
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {data}\n\n"


async def sse_event_stream(
    session_id: str,
    queue: "asyncio.Queue[dict]",
    is_disconnected,
    keepalive: float = SSE_KEEPALIVE_SECONDS,
):
    """SSE 事件流：先下发投递端点，之后把响应帧推进流里。

    独立成生成器（而不是塞在路由闭包里）是为了能直接单测帧格式、心跳与清理
    —— 流式响应的集成测试很依赖客户端实现，而这个函数本身是纯逻辑。
    """
    try:
        yield sse_frame("endpoint", f"/mcp/messages?session_id={session_id}")
        while True:
            if await is_disconnected():
                logger.info(f"[MCP] SSE client disconnected: {session_id}")
                break
            try:
                response = await asyncio.wait_for(queue.get(), timeout=keepalive)
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"  # 心跳，防止中间层掐连接
                continue
            yield sse_frame("message", json.dumps(response, ensure_ascii=False))
    finally:
        _mcp_sessions.pop(session_id, None)
        logger.info(f"[MCP] SSE session closed: {session_id}")


def _get_mcp_server():
    """进程内单例：MCP Server 的工具注册表与审计日志只建一次。"""
    global _mcp_server
    if _mcp_server is None:
        from ..mcp.server import McpServer

        _mcp_server = McpServer()
        logger.info("[MCP] server initialised for HTTP transport")
    return _mcp_server


@app.post("/mcp")
async def mcp_http(request: Request) -> Response:
    """MCP 请求/响应直返（通知类报文返回 202）。"""
    payload = await request.json()
    response = await _get_mcp_server().handle(payload)
    if response is None:
        return Response(
            status_code=202, content='{"status":"accepted"}', media_type="application/json"
        )
    return JSONResponse(response)


@app.get("/mcp/sse")
async def mcp_sse(request: Request) -> StreamingResponse:
    """MCP SSE 传输：先下发投递端点，随后把响应推进事件流。

    每个心跳周期检查一次客户端是否断开——否则断开的会话会永远留在
    ``_mcp_sessions`` 里（长连接服务的经典泄漏点）。事件流逻辑见
    :func:`sse_event_stream`。
    """
    session_id = uuid.uuid4().hex[:12]
    queue: asyncio.Queue = asyncio.Queue()
    _mcp_sessions[session_id] = queue
    logger.info(f"[MCP] SSE session opened: {session_id}")

    return StreamingResponse(
        sse_event_stream(session_id, queue, request.is_disconnected),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/mcp/messages")
async def mcp_messages(session_id: str, request: Request) -> Response:
    """SSE 传输的请求入口：处理完把响应推进对应 session 的事件流。"""
    queue = _mcp_sessions.get(session_id)
    if queue is None:
        return JSONResponse(
            {"error": f"unknown session_id: {session_id}"}, status_code=404
        )

    payload = await request.json()
    response = await _get_mcp_server().handle(payload)
    if response is not None:
        await queue.put(response)
    return Response(
        status_code=202, content='{"status":"accepted"}', media_type="application/json"
    )


@app.get("/mcp/info")
async def mcp_info() -> dict:
    """MCP 能力自检：暴露了哪些工具、写操作是否开启、审计日志在哪。"""
    server = _get_mcp_server()
    registry = server.registry
    return {
        "server": server.info.to_payload(),
        "protocol_version": server.info.protocol_version,
        "tools": [
            {"name": spec.name, "readonly": spec.readonly}
            for spec in registry._tools.values()  # noqa: SLF001 - 自检要看到全部工具
        ],
        "available_tools": [spec.name for spec in registry.list_specs()],
        "allow_write": registry.policy.allow_write,
        "allowlist": list(registry.policy.allowlist),
        "audit_log": str(registry.audit.path),
        "sessions": len(_mcp_sessions),
    }


@app.post("/api/v1/meeting/start")
async def start_meeting():
    """创建新会议"""
    meeting_id = str(uuid.uuid4())[:12]
    return {
        "meeting_id": meeting_id,
        "websocket_url": f"ws://localhost:8000/ws/meeting/{meeting_id}",
        "status": "created",
    }


@app.post("/api/v1/meeting/{meeting_id}/upload")
async def upload_audio(meeting_id: str, file: UploadFile = File(...)):
    """上传音频文件并处理"""
    audio_data = await file.read()
    logger.info(
        f"Received audio upload: {meeting_id}, size={len(audio_data)} bytes"
    )

    result = await run_meeting_pipeline(
        meeting_id=meeting_id,
        audio_data=audio_data,
    )
    meeting_results[meeting_id] = result

    return {
        "meeting_id": meeting_id,
        "status": result.get("status", "completed"),
        "errors": result.get("errors", []),
    }


@app.post("/api/v1/meeting/{meeting_id}/demo")
async def run_demo(meeting_id: str = "demo"):
    """运行演示模式（无需音频）"""
    result = await run_meeting_pipeline(
        meeting_id=meeting_id,
        audio_data=b"",
    )
    meeting_results[meeting_id] = result

    response: dict[str, Any] = {
        "meeting_id": meeting_id,
        "status": result.get("status"),
    }

    for key in ("transcript", "summary", "actions", "insights", "followup"):
        val = result.get(key)
        if val and hasattr(val, "model_dump"):
            response[key] = val.model_dump()

    response["errors"] = result.get("errors", [])
    return response


@app.get("/api/v1/meeting/{meeting_id}/transcript")
async def get_transcript(meeting_id: str):
    """获取转写结果"""
    result = meeting_results.get(meeting_id)
    if not result:
        return {"error": "Meeting not found"}
    transcript = result.get("transcript")
    if transcript and hasattr(transcript, "model_dump"):
        return transcript.model_dump()
    return {"error": "Transcript not available"}


@app.get("/api/v1/meeting/{meeting_id}/summary")
async def get_summary(meeting_id: str):
    """获取会议纪要"""
    result = meeting_results.get(meeting_id)
    if not result:
        return {"error": "Meeting not found"}
    summary = result.get("summary")
    if summary and hasattr(summary, "model_dump"):
        return summary.model_dump()
    return {"error": "Summary not available"}


@app.get("/api/v1/meeting/{meeting_id}/actions")
async def get_actions(meeting_id: str):
    """获取待办事项"""
    result = meeting_results.get(meeting_id)
    if not result:
        return {"error": "Meeting not found"}
    actions = result.get("actions")
    if actions and hasattr(actions, "model_dump"):
        return actions.model_dump()
    return {"error": "Actions not available"}


@app.get("/api/v1/meeting/{meeting_id}/insights")
async def get_insights(meeting_id: str):
    """获取会议洞察"""
    result = meeting_results.get(meeting_id)
    if not result:
        return {"error": "Meeting not found"}
    insights = result.get("insights")
    if insights and hasattr(insights, "model_dump"):
        return insights.model_dump()
    return {"error": "Insights not available"}


@app.get("/api/v1/meeting/{meeting_id}/report")
async def get_full_report(meeting_id: str):
    """获取完整报告"""
    result = meeting_results.get(meeting_id)
    if not result:
        return {"error": "Meeting not found"}

    response = {"meeting_id": meeting_id}
    for key in ("transcript", "summary", "actions", "insights", "followup"):
        val = result.get(key)
        if val and hasattr(val, "model_dump"):
            response[key] = val.model_dump()

    response["errors"] = result.get("errors", [])
    return response
