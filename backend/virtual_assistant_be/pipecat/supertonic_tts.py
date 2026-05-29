from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import numpy as np
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService

log = logging.getLogger(__name__)

SUPERTONIC_SAMPLE_RATE = 44100


@dataclass
class SupertonicTTSSettings(TTSSettings):
    pass


class SupertonicTTSService(TTSService):
    Settings = SupertonicTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        voice_id: str = "M4",
        sample_rate: int = 24000,
        settings: Settings | None = None,
        **kwargs,
    ):
        default_settings = self.Settings(model=None, voice=voice_id, language=None)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            push_start_frame=True,
            push_stop_frames=True,
            settings=default_settings,
            **kwargs,
        )

        from supertonic import TTS

        self._tts = TTS(auto_download=True)
        self._voice = self._tts.get_voice_style(voice_name=voice_id)
        self._resampler = create_stream_resampler()

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        log.debug(f"{self}: Generating TTS [{text}]")
        try:
            await self.start_tts_usage_metrics(text)
            wav, _duration = await asyncio.to_thread(
                self._tts.synthesize,
                text,
                voice_style=self._voice,
                lang=self._settings.language or "en",
            )
            await self.stop_ttfb_metrics()
            audio_int16 = (wav[0] * 32767).astype(np.int16).tobytes()
            resampled = await self._resampler.resample(
                audio_int16, SUPERTONIC_SAMPLE_RATE, self.sample_rate
            )
            yield TTSAudioRawFrame(
                audio=resampled,
                sample_rate=self.sample_rate,
                num_channels=1,
                context_id=context_id,
            )
        except Exception as e:
            log.exception(f"Supertonic TTS error: {e}")
            yield ErrorFrame(error=f"Supertonic TTS error: {e}")
        finally:
            await self.stop_ttfb_metrics()
