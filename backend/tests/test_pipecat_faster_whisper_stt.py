from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipecat.frames.frames import StartFrame
from pipecat.processors.frame_processor import FrameDirection

from pipecat.frames.frames import ErrorFrame

from virtual_assistant_be.pipecat.faster_whisper_stt import FasterWhisperSTTService


@pytest.fixture
def stt():
    svc = FasterWhisperSTTService(model="base", device="cpu", compute_type="int8")
    svc._settings.language = "es"
    return svc


@pytest.mark.asyncio
class TestFasterWhisperSTTService:
    async def test_init_loads_model(self, stt):
        assert stt._model is not None

    async def test_run_stt_returns_nothing_on_empty_audio(self, stt):
        frames = []
        async for frame in stt.run_stt(b""):
            frames.append(frame)
        assert len(frames) == 0

    async def test_run_stt_error_when_model_unavailable(self):
        svc = FasterWhisperSTTService(model="base", device="cpu", compute_type="int8")
        svc._model = None
        frames = []
        async for frame in svc.run_stt(b"\x00\x00" * 16000):
            frames.append(frame)
        assert len(frames) == 1
        assert isinstance(frames[0], ErrorFrame)

    async def test_run_stt_transcribes_silence(self, stt):
        audio = b"\x00\x00" * 16000  # 1 second of silence
        frames = []
        async for frame in stt.run_stt(audio):
            frames.append(frame)
        assert len(frames) == 0

    async def test_run_stt_transcribes(self, stt):
        audio = (b"\x00\x00" * 8000) + (b"\xff\x7f" * 8000)  # silence + noise
        with patch.object(stt._model, "transcribe") as mock_t:
            class FakeSegment:
                text = "hola mundo"
                avg_logprob = -0.3
                no_speech_prob = 0.1
            mock_t.return_value = (
                [FakeSegment()],
                MagicMock(language="es", language_probability=0.95),
            )
            frames = []
            async for frame in stt.run_stt(audio):
                frames.append(frame)
            assert len(frames) == 1
            assert frames[0].text == "hola mundo"

    async def test_run_stt_discards_low_confidence(self, stt):
        audio = b"\x01\x00" * 16000
        with patch.object(stt._model, "transcribe") as mock_t:
            class FakeSegment:
                text = "a"
                avg_logprob = -2.5
                no_speech_prob = 0.9
            mock_t.return_value = (
                [FakeSegment()],
                MagicMock(language="es", language_probability=0.95),
            )
            frames = []
            async for frame in stt.run_stt(audio):
                frames.append(frame)
            assert len(frames) == 0

    async def test_can_generate_metrics(self, stt):
        assert stt.can_generate_metrics() is True

    async def test_language_to_service_language(self, stt):
        from pipecat.transcriptions.language import Language
        assert stt.language_to_service_language(Language.ES) == "es"
        assert stt.language_to_service_language(Language.EN) == "en"
