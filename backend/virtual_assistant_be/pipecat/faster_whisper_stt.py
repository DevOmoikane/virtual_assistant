from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.settings import NOT_GIVEN, STTSettings, _NotGiven, assert_given
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

log = logging.getLogger(__name__)

LANG_PROB_THRESHOLD = 0.5
MIN_WORDS = 2


@dataclass
class FasterWhisperSTTSettings(STTSettings):
    beam_size: int | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    no_speech_threshold: float | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    log_prob_threshold: float | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    compression_ratio_threshold: float | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    min_silence_duration_ms: int | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    lang_prob_threshold: float | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    min_words: int | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class FasterWhisperSTTService(SegmentedSTTService):
    """Custom STT service using faster-whisper with quality-tuning parameters.

    Wraps faster-whisper directly with configurable beam search, VAD filter,
    and confidence thresholds — matching the parameters that were tuned in the
    original ``SttService`` for better transcription quality.
    """

    Settings = FasterWhisperSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        model: str = "base",
        device: str = "auto",
        compute_type: str = "int8",
        language: str | None = None,
        beam_size: int = 5,
        no_speech_threshold: float = 0.6,
        log_prob_threshold: float = -2.0,
        compression_ratio_threshold: float = 2.4,
        min_silence_duration_ms: int = 500,
        lang_prob_threshold: float = 0.5,
        min_words: int = 2,
        **kwargs,
    ):
        default_settings = self.Settings(
            model=model,
            language=language,
            beam_size=beam_size,
            no_speech_threshold=no_speech_threshold,
            log_prob_threshold=log_prob_threshold,
            compression_ratio_threshold=compression_ratio_threshold,
            min_silence_duration_ms=min_silence_duration_ms,
            lang_prob_threshold=lang_prob_threshold,
            min_words=min_words,
        )

        super().__init__(**kwargs)

        self._settings = default_settings
        self._device = device
        self._compute_type = compute_type
        self._model: WhisperModel | None = None
        self._load()

    def can_generate_metrics(self) -> bool:
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        if language is None:
            return None
        return language.value.split("-")[0].split("_")[0]

    def _load(self):
        try:
            from faster_whisper import WhisperModel
            model_name = assert_given(self._settings.model)
            if model_name is None:
                raise ValueError("faster-whisper model must be specified")
            self._model = WhisperModel(
                model_name, device=self._device, compute_type=self._compute_type
            )
            log.info("Loaded faster-whisper model '%s' (device=%s, compute=%s)",
                      model_name, self._device, self._compute_type)
        except ModuleNotFoundError as e:
            log.error("Missing faster-whisper module: %s", e)
            self._model = None

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        if not self._model:
            yield ErrorFrame("faster-whisper model not available")
            return

        await self.start_processing_metrics()

        audio_float = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        if len(audio_float) == 0:
            await self.stop_processing_metrics()
            return

        peak = np.max(np.abs(audio_float))
        if peak > 0:
            audio_float = audio_float / peak * 0.95

        language = assert_given(self._settings.language)
        beam_size = assert_given(self._settings.beam_size)
        no_speech_threshold = assert_given(self._settings.no_speech_threshold)
        log_prob_threshold = assert_given(self._settings.log_prob_threshold)
        compression_ratio_threshold = assert_given(self._settings.compression_ratio_threshold)
        min_silence_duration_ms = assert_given(self._settings.min_silence_duration_ms)
        lang_prob_threshold = assert_given(self._settings.lang_prob_threshold)
        min_words = assert_given(self._settings.min_words)

        segments, info = await asyncio.to_thread(
            self._model.transcribe,
            audio_float,
            language=language,
            beam_size=beam_size,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=min_silence_duration_ms),
            no_speech_threshold=no_speech_threshold,
            log_prob_threshold=log_prob_threshold,
            compression_ratio_threshold=compression_ratio_threshold,
        )

        texts: list[str] = []
        total_logprob = 0.0
        n_segments = 0
        for segment in segments:
            t = segment.text.strip()
            if t:
                texts.append(t)
                total_logprob += segment.avg_logprob
                n_segments += 1

        await self.stop_processing_metrics()

        joined = " ".join(texts)
        if not joined:
            return

        detected_lang: str | None = info.language if info else None
        detected_prob: float = info.language_probability if info else 0.0

        if detected_prob < lang_prob_threshold:
            detected_lang = None

        word_count = len(joined.split())
        avg_logprob = total_logprob / n_segments if n_segments else 0.0
        if word_count < min_words and avg_logprob < -1.0:
            log.info("Discarding low-confidence transcription (words=%d, logprob=%.2f)",
                      word_count, avg_logprob)
            return

        yield TranscriptionFrame(
            joined,
            self._user_id,
            time_now_iso8601(),
            detected_lang or language,
        )
