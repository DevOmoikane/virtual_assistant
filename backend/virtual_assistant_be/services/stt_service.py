from __future__ import annotations

import logging

import numpy as np
from faster_whisper import WhisperModel

from virtual_assistant_be.core.config import settings

log = logging.getLogger(__name__)


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

    def transcribe(self, audio: np.ndarray) -> str:
        model = self._ensure_model()

        if len(audio) == 0:
            return ""

        segments, _info = model.transcribe(
            audio,
            language="en",
            beam_size=1,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        texts: list[str] = []
        for segment in segments:
            t = segment.text.strip()
            if t:
                texts.append(t)

        return " ".join(texts)

    def unload(self) -> None:
        self._model = None
