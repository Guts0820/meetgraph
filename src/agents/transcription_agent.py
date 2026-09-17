"""
Transcription Agent（转写Agent）
- 接收音频数据，使用 WhisperX 进行语音转文字
- 使用 pyannote-audio 进行说话人识别（Speaker Diarization）
- 输出带说话人标签和时间戳的转写文本
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

from loguru import logger

from ..models.schemas import (
    MeetingStatus,
    TranscriptResult,
    TranscriptSegment,
)


class TranscriptionConfig:
    """转写配置"""

    def __init__(
        self,
        model_size: str = "large-v2",
        device: str = "cpu",
        compute_type: str = "float32",
        language: str = "zh",
        hf_token: str = "",
        batch_size: int = 16,
    ):
        self.model_size = os.getenv("WHISPER_MODEL_SIZE", model_size)
        self.device = os.getenv("WHISPER_DEVICE", device)
        self.compute_type = compute_type
        self.language = os.getenv("WHISPER_LANGUAGE", language)
        self.hf_token = hf_token or os.getenv("HF_TOKEN", "")
        self.batch_size = batch_size


class TranscriptionAgent:
    """
    转写Agent - Pipeline的第一个节点

    架构说明:
    1. 接收音频字节数据（来自 WebSocket 或文件上传）
    2. 使用 WhisperX 进行批量转写（比原版 Whisper 快 70x）
    3. wav2vec2 强制对齐获取精确时间戳
    4. pyannote-audio 进行说话人识别
    5. 合并结果，输出 TranscriptResult

    模型懒加载：只在收到真实音频时载入 WhisperX 与对齐模型，避免服务启动
    即占用显存；无音频输入时走内置演示转写，便于离线联调和单元测试。
    """

    def __init__(self, config: TranscriptionConfig | None = None):
        self.config = config or TranscriptionConfig()
        self._model = None
        self._align_model = None
        self._diarize_pipeline = None
        self._initialized = False

    def _lazy_init(self):
        """
        懒加载模型 —— 避免在导入时就加载大模型。
        生产中模型应在服务启动时预热。
        """
        if self._initialized:
            return

        try:
            import whisperx

            logger.info(
                f"Loading WhisperX model: {self.config.model_size} "
                f"on {self.config.device}"
            )
            self._model = whisperx.load_model(
                self.config.model_size,
                self.config.device,
                compute_type=self.config.compute_type,
            )
            self._initialized = True
            logger.info("WhisperX model loaded successfully")
        except ImportError:
            logger.warning(
                "WhisperX not installed, using mock transcription. "
                "Install with: pip install whisperx"
            )
            self._initialized = True

    async def process(self, state: dict) -> dict:
        """
        LangGraph 节点函数 —— 执行语音转写

        Args:
            state: MeetingState 字典，包含 audio_data 字段

        Returns:
            更新后的 state，包含 transcript 和 transcript_text
        """
        meeting_id = state.get("meeting_id", "unknown")
        logger.info(f"[TranscriptionAgent] Processing meeting: {meeting_id}")

        state["status"] = MeetingStatus.TRANSCRIBING
        errors_delta: list[str] = []

        audio_data = state.get("audio_data", b"")
        if not audio_data:
            logger.warning("No audio data provided, using demo transcript")
            state["transcript"] = self._generate_demo_transcript(meeting_id)
            state["transcript_text"] = self._format_transcript_text(
                state["transcript"]
            )
            return {
                "status": state["status"],
                "transcript": state["transcript"],
                "transcript_text": state["transcript_text"],
            }

        try:
            self._lazy_init()
            transcript = await self._transcribe(audio_data, meeting_id)
            state["transcript"] = transcript
            state["transcript_text"] = self._format_transcript_text(transcript)
            logger.info(
                f"[TranscriptionAgent] Transcription complete: "
                f"{len(transcript.segments)} segments"
            )
        except Exception as e:
            logger.error(f"[TranscriptionAgent] Error: {e}")
            errors_delta.append(f"TranscriptionAgent: {str(e)}")
            state["transcript"] = self._generate_demo_transcript(meeting_id)
            state["transcript_text"] = self._format_transcript_text(
                state["transcript"]
            )

        updates = {
            "status": state["status"],
            "transcript": state["transcript"],
            "transcript_text": state["transcript_text"],
        }
        if errors_delta:
            updates["errors"] = errors_delta
        return updates

    async def _transcribe(
        self, audio_data: bytes, meeting_id: str
    ) -> TranscriptResult:
        """执行实际的语音转写流程"""
        if self._model is None:
            return self._generate_demo_transcript(meeting_id)

        import whisperx

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            tmp.write(audio_data)
            tmp.flush()

            # Step 1: WhisperX 转写
            result = self._model.transcribe(
                tmp.name,
                batch_size=self.config.batch_size,
                language=self.config.language,
            )

            # Step 2: 时间戳对齐
            if self._align_model is None:
                model_a, metadata = whisperx.load_align_model(
                    language_code=self.config.language,
                    device=self.config.device,
                )
                self._align_model = (model_a, metadata)

            aligned = whisperx.align(
                result["segments"],
                self._align_model[0],
                self._align_model[1],
                tmp.name,
                self.config.device,
            )

            # Step 3: 说话人识别
            if self._diarize_pipeline is None and self.config.hf_token:
                self._diarize_pipeline = whisperx.DiarizationPipeline(
                    use_auth_token=self.config.hf_token,
                    device=self.config.device,
                )

            if self._diarize_pipeline:
                diarize_result = self._diarize_pipeline(tmp.name)
                final = whisperx.assign_word_speakers(
                    diarize_result, aligned
                )
            else:
                final = aligned

        segments = []
        for seg in final.get("segments", []):
            segments.append(
                TranscriptSegment(
                    speaker=seg.get("speaker", "Unknown"),
                    text=seg.get("text", "").strip(),
                    start=seg.get("start", 0.0),
                    end=seg.get("end", 0.0),
                    confidence=seg.get("confidence", 0.0),
                )
            )

        duration = segments[-1].end if segments else 0.0
        full_text = " ".join(s.text for s in segments)

        return TranscriptResult(
            meeting_id=meeting_id,
            segments=segments,
            language=self.config.language,
            duration_seconds=duration,
            full_text=full_text,
        )

    @staticmethod
    def _generate_demo_transcript(meeting_id: str) -> TranscriptResult:
        """生成演示转写结果（无音频时使用）"""
        demo_segments = [
            TranscriptSegment(
                speaker="张总",
                text="好的，我们开始今天的Q3预算评审会议。首先请李明汇报一下目前的预算执行情况。",
                start=0.0, end=8.5, confidence=0.96,
            ),
            TranscriptSegment(
                speaker="李明",
                text="好的张总。截至目前，Q2预算执行率为87%，其中研发投入占比最大，达到42%。",
                start=9.0, end=16.2, confidence=0.95,
            ),
            TranscriptSegment(
                speaker="李明",
                text="Q3我们计划将预算上调15%，主要增加在AI基础设施和人才招聘方面。",
                start=16.5, end=23.1, confidence=0.94,
            ),
            TranscriptSegment(
                speaker="王芳",
                text="关于人才招聘，我建议我们重点招聘3名高级算法工程师，预算大概在每人年薪80万左右。",
                start=23.5, end=31.0, confidence=0.93,
            ),
            TranscriptSegment(
                speaker="张总",
                text="可以。李明你来负责整理Q3的详细预算方案，下周五之前提交给我审批。",
                start=31.5, end=38.2, confidence=0.97,
            ),
            TranscriptSegment(
                speaker="张总",
                text="王芳负责拟定招聘JD，本周三前完成。另外，赵伟跟进一下服务器采购的事情。",
                start=38.5, end=46.0, confidence=0.95,
            ),
            TranscriptSegment(
                speaker="赵伟",
                text="收到，我这边已经在对比几家供应商了，预计下周一可以给出采购方案。",
                start=46.5, end=52.8, confidence=0.94,
            ),
            TranscriptSegment(
                speaker="张总",
                text="好的，那今天的会议就到这里。各位辛苦了，请大家按时完成各自的任务。",
                start=53.0, end=59.5, confidence=0.96,
            ),
        ]

        full_text = "\n".join(
            f"[{s.speaker}] {s.text}" for s in demo_segments
        )

        return TranscriptResult(
            meeting_id=meeting_id,
            segments=demo_segments,
            language="zh",
            duration_seconds=59.5,
            full_text=full_text,
        )

    @staticmethod
    def _format_transcript_text(transcript: TranscriptResult) -> str:
        """将转写结果格式化为纯文本（供后续Agent使用）"""
        lines = []
        for seg in transcript.segments:
            ts = f"[{seg.start:.1f}s-{seg.end:.1f}s]"
            lines.append(f"{ts} {seg.speaker}: {seg.text}")
        return "\n".join(lines)
