from __future__ import annotations

import abc
import logging
import os
import threading

import numpy as np
import sounddevice as sd

from virtual_assistant_be.core.config import settings
from virtual_assistant_be.timer import Timer

log = logging.getLogger(__name__)


class _TtsEngine(abc.ABC):
    @abc.abstractmethod
    def synthesize(self, text: str, language: str) -> bytes: ...

    @abc.abstractmethod
    def speak(self, text: str, language: str) -> None: ...

    @abc.abstractmethod
    def stop(self) -> None: ...

    @property
    @abc.abstractmethod
    def is_speaking(self) -> bool: ...

    @abc.abstractmethod
    def unload(self) -> None: ...


class _PiperEngine(_TtsEngine):
    def __init__(self) -> None:
        from piper import PiperVoice, SynthesisConfig

        self._PiperVoice = PiperVoice
        self._SynthesisConfig = SynthesisConfig

        self._voices: dict[str, PiperVoice] = {}
        self._voice_paths: dict[str, str] = dict(settings.piper_voices or {})
        self._default_language: str = settings.piper_default_language
        self._stop_event = threading.Event()
        self._playing = False

    def _ensure_voice(self, language: str):
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
            voice = self._PiperVoice.load(path)
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

        config = self._SynthesisConfig()
        audio_parts: list[bytes] = []
        for chunk in voice.synthesize(text, config):
            audio_parts.append(chunk.audio_int16_bytes)

        return b"".join(audio_parts)

    @property
    def is_speaking(self) -> bool:
        return self._playing

    def stop(self) -> None:
        self._stop_event.set()
        sd.stop()

    def speak(self, text: str, language: str | None = None) -> None:
        lang = language or self._default_language
        with Timer("tts.speak"):
            voice = self._ensure_voice(lang)
            if voice is None:
                return

            self._stop_event.clear()
            self._playing = True
            config = self._SynthesisConfig()
            try:
                for chunk in voice.synthesize(text, config):
                    if self._stop_event.is_set():
                        break
                    audio = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
                    sd.play(audio, samplerate=voice.config.sample_rate)
                    sd.wait()
            finally:
                self._playing = False

    def unload(self) -> None:
        self._voices.clear()


class _SupertonicEngine(_TtsEngine):
    def __init__(self) -> None:
        from supertonic import TTS

        self._tts = TTS(auto_download=True)
        self._voice = self._tts.get_voice_style(voice_name=settings.supertonic_voice)
        self._stop_event = threading.Event()
        self._playing = False

    @property
    def _lang(self) -> str:
        return settings.piper_default_language

    def synthesize(self, text: str, language: str | None = None) -> bytes:
        wav, _duration = self._tts.synthesize(
            text,
            voice_style=self._voice,
            lang=language or self._lang,
        )
        audio = wav[0]
        return (audio * 32767).astype(np.int16).tobytes()

    @property
    def is_speaking(self) -> bool:
        return self._playing

    def stop(self) -> None:
        self._stop_event.set()
        sd.stop()

    def speak(self, text: str, language: str | None = None) -> None:
        self._stop_event.clear()
        self._playing = True
        try:
            if self._stop_event.is_set():
                return
            with Timer("tts.speak"):
                wav, duration = self._tts.synthesize(
                    text,
                    voice_style=self._voice,
                    lang=language or self._lang,
                )
            if self._stop_event.is_set():
                return
            audio = wav[0]
            audio_int16 = (audio * 32767).astype(np.int16)
            sd.play(audio_int16, samplerate=self._tts.sample_rate)
            sd.wait()
        finally:
            self._playing = False

    def unload(self) -> None:
        self._tts = None


class TtsService:
    def __init__(self) -> None:
        if settings.tts_engine == "supertonic":
            log.info("Using Supertonic TTS engine (voice: %s)", settings.supertonic_voice)
            self._engine: _TtsEngine = _SupertonicEngine()
        else:
            log.info("Using Piper TTS engine")
            self._engine: _TtsEngine = _PiperEngine()

    def synthesize(self, text: str, language: str | None = None) -> bytes:
        return self._engine.synthesize(text, language)

    @property
    def is_speaking(self) -> bool:
        return self._engine.is_speaking

    def stop(self) -> None:
        self._engine.stop()

    def speak(self, text: str, language: str | None = None) -> None:
        self._engine.speak(text, language)

    def unload(self) -> None:
        self._engine.unload()
