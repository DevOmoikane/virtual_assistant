from __future__ import annotations

import logging

import numpy as np
import sounddevice as sd
from piper import PiperVoice, SynthesisConfig

from virtual_assistant_be.core.config import settings

log = logging.getLogger(__name__)


class TtsService:
    def __init__(self) -> None:
        self._voice: PiperVoice | None = None
        self.voice_path = settings.piper_voice_path

    def _ensure_voice(self) -> PiperVoice:
        if self._voice is None:
            log.info("Loading piper voice from '%s' ...", self.voice_path)
            self._voice = PiperVoice.load(self.voice_path)
            log.info("Piper voice loaded (sample rate: %d)", self._voice.config.sample_rate)
        return self._voice

    def synthesize(self, text: str) -> bytes:
        voice = self._ensure_voice()
        config = SynthesisConfig()

        audio_parts: list[bytes] = []
        for chunk in voice.synthesize(text, config):
            audio_parts.append(chunk.audio_int16_bytes)

        return b"".join(audio_parts)

    def speak(self, text: str) -> None:
        voice = self._ensure_voice()
        config = SynthesisConfig()

        for chunk in voice.synthesize(text, config):
            audio = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
            sd.play(audio, samplerate=voice.config.sample_rate)
            sd.wait()

    def unload(self) -> None:
        self._voice = None
