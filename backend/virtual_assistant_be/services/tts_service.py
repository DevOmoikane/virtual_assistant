from __future__ import annotations

import logging
import os

import numpy as np
import sounddevice as sd
from piper import PiperVoice, SynthesisConfig

from virtual_assistant_be.core.config import settings
from virtual_assistant_be.timer import Timer

log = logging.getLogger(__name__)


class TtsService:
    def __init__(self) -> None:
        self._voice: PiperVoice | None = None
        self._available: bool = False
        self.voice_path = settings.piper_voice_path

    def _ensure_voice(self) -> PiperVoice | None:
        if self._voice is not None:
            return self._voice

        if not os.path.isfile(self.voice_path):
            log.warning("Piper voice file not found: %s", self.voice_path)
            self._available = False
            return None

        try:
            log.info("Loading piper voice from '%s' ...", self.voice_path)
            self._voice = PiperVoice.load(self.voice_path)
            self._available = True
            log.info("Piper voice loaded (sample rate: %d)", self._voice.config.sample_rate)
        except Exception:
            log.warning("Failed to load piper voice", exc_info=True)
            self._available = False

        return self._voice

    def synthesize(self, text: str) -> bytes:
        voice = self._ensure_voice()
        if voice is None:
            return b""

        config = SynthesisConfig()
        audio_parts: list[bytes] = []
        for chunk in voice.synthesize(text, config):
            audio_parts.append(chunk.audio_int16_bytes)

        return b"".join(audio_parts)

    def speak(self, text: str) -> None:
        with Timer("tts.speak"):
            voice = self._ensure_voice()
            if voice is None:
                return

            config = SynthesisConfig()
            for chunk in voice.synthesize(text, config):
                audio = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
                sd.play(audio, samplerate=voice.config.sample_rate)
                sd.wait()

    def unload(self) -> None:
        self._voice = None
        self._available = False
