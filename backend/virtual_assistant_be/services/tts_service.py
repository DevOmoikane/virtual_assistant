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
        self._voices: dict[str, PiperVoice] = {}
        self._voice_paths: dict[str, str] = dict(settings.piper_voices or {})
        self._default_language: str = settings.piper_default_language
        self._stop_requested = False
        self._playing = False

    def _ensure_voice(self, language: str) -> PiperVoice | None:
        if language in self._voices:
            return self._voices[language]

        path = self._voice_paths.get(language)
        if not path:
            language = self._default_language
            if language in self._voices:
                return self._voices[language]
            path = self._voice_paths.get(language)
            if not path:
                log.warning("No voice path configured for language '%s'", language)
                return None

        if not os.path.isfile(path):
            log.warning("Piper voice file not found: %s", path)
            return None

        try:
            log.info("Loading piper voice for '%s' from '%s' ...", language, path)
            voice = PiperVoice.load(path)
            self._voices[language] = voice
            log.info("Piper voice loaded (sample rate: %d)", voice.config.sample_rate)
        except Exception:
            log.warning("Failed to load piper voice for '%s'", language, exc_info=True)
            return None

        return voice

    def synthesize(self, text: str, language: str | None = None) -> bytes:
        lang = language or self._default_language
        voice = self._ensure_voice(lang)
        if voice is None:
            return b""

        config = SynthesisConfig()
        audio_parts: list[bytes] = []
        for chunk in voice.synthesize(text, config):
            audio_parts.append(chunk.audio_int16_bytes)

        return b"".join(audio_parts)

    @property
    def is_speaking(self) -> bool:
        return self._playing

    def stop(self) -> None:
        self._stop_requested = True
        sd.stop()

    def speak(self, text: str, language: str | None = None) -> None:
        lang = language or self._default_language
        with Timer("tts.speak"):
            voice = self._ensure_voice(lang)
            if voice is None:
                return

            self._stop_requested = False
            self._playing = True
            config = SynthesisConfig()
            try:
                for chunk in voice.synthesize(text, config):
                    if self._stop_requested:
                        break
                    audio = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
                    sd.play(audio, samplerate=voice.config.sample_rate)
                    sd.wait()
            finally:
                self._playing = False

    def unload(self) -> None:
        self._voices.clear()
