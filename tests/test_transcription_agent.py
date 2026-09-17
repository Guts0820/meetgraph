"""转写 Agent 测试：演示转写、文本格式化、模型缺失时的降级。"""

from __future__ import annotations

import sys

from src.agents.transcription_agent import TranscriptionAgent, TranscriptionConfig


async def test_demo_transcript_when_no_audio() -> None:
    agent = TranscriptionAgent()
    result = await agent.process({"meeting_id": "m", "audio_data": b""})

    assert len(result["transcript"].segments) == 8
    assert result["transcript"].segments[0].speaker == "张总"
    assert "张总" in result["transcript_text"]


def test_transcript_text_format() -> None:
    transcript = TranscriptionAgent._generate_demo_transcript("m")
    text = TranscriptionAgent._format_transcript_text(transcript)

    lines = text.strip().split("\n")
    assert len(lines) == 8
    assert lines[0].startswith("[0.0s-8.5s] 张总:")


async def test_audio_without_whisperx_falls_back_to_demo(monkeypatch) -> None:
    """whisperx 不可用时不能崩，退回演示转写。"""
    monkeypatch.setitem(sys.modules, "whisperx", None)
    agent = TranscriptionAgent()

    result = await agent.process({"meeting_id": "m", "audio_data": b"\x00" * 32})

    assert agent._model is None
    assert len(result["transcript"].segments) == 8


def test_config_reads_environment(monkeypatch) -> None:
    monkeypatch.setenv("WHISPER_MODEL_SIZE", "small")
    monkeypatch.setenv("WHISPER_DEVICE", "cuda")
    monkeypatch.setenv("WHISPER_LANGUAGE", "en")

    config = TranscriptionConfig()

    assert config.model_size == "small"
    assert config.device == "cuda"
    assert config.language == "en"
