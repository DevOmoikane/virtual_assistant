from __future__ import annotations

import asyncio
import logging
import threading
from typing import Callable, Awaitable

import numpy as np
import sounddevice as sd

from virtual_assistant_be.core.config import settings

log = logging.getLogger(__name__)

AudioCallback = Callable[[np.ndarray], Awaitable[None]]
DeviceCallback = Callable[[list[dict]], None]

SAMPLE_RATE = settings.stt_sample_rate
FRAME_SIZE = 480
SILENCE_THRESHOLD = 0.01
SILENCE_TIMEOUT = 1.0
MIN_SPEECH_DURATION = 0.5


class AudioService:
    def __init__(self, audio_callback: AudioCallback | None = None) -> None:
        self._callback = audio_callback
        self._running = False
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        self._buffer: np.ndarray = np.array([], dtype=np.float32)
        self._buffer_lock = threading.Lock()
        self._speech_buffer: np.ndarray = np.array([], dtype=np.float32)
        self._silence_frames = 0
        self._speech_active = False

    def set_callback(self, callback: AudioCallback) -> None:
        self._callback = callback

    def start(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        if self._running:
            return
        self._running = True
        self._loop = loop or asyncio.get_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        log.info("Audio service started (rate=%d, threshold=%.3f)", SAMPLE_RATE, SILENCE_THRESHOLD)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        log.info("Audio service stopped")

    def _emit(self, audio: np.ndarray) -> None:
        if self._callback and self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(
                asyncio.ensure_future, self._callback(audio),
            )

    def _audio_callback(self, indata: np.ndarray, frames: int, _time_info, status) -> None:
        if status:
            log.warning("Audio status: %s", status)

        with self._buffer_lock:
            self._buffer = np.append(self._buffer, indata[:, 0].copy())

    def _process_vad(self, chunk: np.ndarray) -> None:
        rms = np.sqrt(np.mean(chunk**2))
        is_speech = rms > SILENCE_THRESHOLD

        if is_speech:
            self._speech_buffer = np.append(self._speech_buffer, chunk)
            self._silence_frames = 0
            if not self._speech_active:
                self._speech_active = True
        else:
            if self._speech_active:
                self._silence_frames += len(chunk) / SAMPLE_RATE
                self._speech_buffer = np.append(self._speech_buffer, chunk)

                if self._silence_frames >= SILENCE_TIMEOUT:
                    duration = len(self._speech_buffer) / SAMPLE_RATE
                    if duration >= MIN_SPEECH_DURATION:
                        log.info("Speech segment: %.2fs", duration)
                        self._emit(self._speech_buffer.copy())

                    self._speech_buffer = np.array([], dtype=np.float32)
                    self._speech_active = False
                    self._silence_frames = 0

    def _run(self) -> None:
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                callback=self._audio_callback,
                blocksize=FRAME_SIZE,
                device=settings.stt_device_id,
            ):
                while self._running:
                    sd.sleep(100)

                    with self._buffer_lock:
                        if len(self._buffer) >= FRAME_SIZE:
                            chunk = self._buffer[:FRAME_SIZE].copy()
                            self._buffer = self._buffer[FRAME_SIZE:]
                        else:
                            chunk = None

                    if chunk is not None:
                        self._process_vad(chunk)
        except Exception:
            log.exception("Audio capture error")
