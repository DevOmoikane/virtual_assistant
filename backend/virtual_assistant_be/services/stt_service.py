from __future__ import annotations

import logging

import numpy as np
from faster_whisper import WhisperModel

from virtual_assistant_be.core.config import settings
from virtual_assistant_be.timer import Timer

log = logging.getLogger(__name__)

LANG_PROB_THRESHOLD = 0.5
MIN_WORDS = 2


class SttService:
    def __init__(self) -> None:
        self._model: WhisperModel | None = None
        self.model_size = settings.stt_model_size
        self.sample_rate = settings.stt_sample_rate
        self._ensure_model()

    def _ensure_model(self) -> WhisperModel:
        log.debug("Checking if model is loaded ...")
        if self._model is None:
            log.info("Loading whisper model '%s' ...", self.model_size)
            self._model = WhisperModel(self.model_size, device="auto", compute_type="int8")
            log.info("Whisper model loaded")
        return self._model

    def transcribe(self, audio: np.ndarray) -> tuple[str, str]:
        with Timer("stt.transcribe"):
            model = self._ensure_model()

            if len(audio) == 0:
                return "", ""

            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio / peak * 0.95

            log.info("Transcribing audio of length %d samples ...", len(audio))

            segments, info = model.transcribe(
                audio,
                language="es",
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                no_speech_threshold=0.6,
                log_prob_threshold=-2.0,
                compression_ratio_threshold=2.4,
            )

            log.info(
                "language: %s (prob: %.2f)",
                info.language if info else "", info.language_probability if info else 0.0,
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

            joined_texts: str = " ".join(texts)
            language: str = info.language if info else ""
            lang_prob: float = info.language_probability if info else 0.0

            if joined_texts:
                log.info(
                    "STT result: %s (lang: %s, prob: %.2f)",
                    joined_texts, language, lang_prob,
                )

                word_count = len(joined_texts.split())
                avg_logprob = total_logprob / n_segments if n_segments else 0.0

                if lang_prob < LANG_PROB_THRESHOLD:
                    log.info(
                        "STT language '%.2f' below threshold %.2f — discarding language",
                        lang_prob, LANG_PROB_THRESHOLD,
                    )
                    language = ""

                if word_count < MIN_WORDS and avg_logprob < -1.0:
                    log.info(
                        "STT text too short/low-confidence (words=%d, avg_logprob=%.2f) — discarding",
                        word_count, avg_logprob,
                    )
                    return "", ""

        return joined_texts, language

    def unload(self) -> None:
        self._model = None
