from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Callable, Awaitable

import numpy as np
import sounddevice as sd

from virtual_assistant_be.core.config import settings
from virtual_assistant_be.timer import log_duration

log = logging.getLogger(__name__)

AudioCallback = Callable[[np.ndarray], Awaitable[None]]
DeviceCallback = Callable[[list[dict]], None]
InterruptCallback = Callable[[], None]

BARGE_IN_MULTIPLIER = 3.0
BARGE_IN_MIN = 0.03

SAMPLE_RATE = settings.stt_sample_rate
FRAME_SIZE = 480
SILENCE_TIMEOUT = 0.5
MIN_SPEECH_DURATION = 0.5
MAX_SPEECH_DURATION = 15.0
NOISE_FLOOR_ALPHA = 0.02
THRESHOLD_MULTIPLIER = 2.5
MIN_THRESHOLD = 0.003


class AudioService:
    def __init__(self, audio_callback: AudioCallback | None = None,
                 device_id: int | str | None = None,
                 interrupt_callback: InterruptCallback | None = None) -> None:
        self._callback = audio_callback
        self._running = False
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._device_id = device_id
        self._interrupt_callback = interrupt_callback

        self._buffer: np.ndarray = np.array([], dtype=np.float32)
        self._buffer_lock = threading.Lock()
        self._speech_buffer: np.ndarray = np.array([], dtype=np.float32)
        self._silence_frames = 0.0
        self._speech_duration = 0.0
        self._speech_active = False
        self._speech_start_time = 0.0
        self._muted = False
        self._mute_lock = threading.Lock()

        self._noise_floor: float = 0.0
        self._noise_floor_frames = 0
        self._threshold_log_time = 0.0

    def set_callback(self, callback: AudioCallback) -> None:
        self._callback = callback

    def start(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        if self._running:
            return
        self._running = True
        self._loop = loop or asyncio.get_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        log.info("Audio service started (rate=%d)", SAMPLE_RATE)

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

    def mute(self) -> None:
        with self._mute_lock:
            self._muted = True
            self._speech_buffer = np.array([], dtype=np.float32)
            self._silence_frames = 0.0
            self._speech_duration = 0.0
            self._speech_active = False

    def unmute(self) -> None:
        with self._mute_lock:
            self._muted = False

    @property
    def _speech_threshold(self) -> float:
        return max(self._noise_floor * THRESHOLD_MULTIPLIER, MIN_THRESHOLD)

    def _update_noise_floor(self, rms: float) -> None:
        if self._noise_floor_frames < 50:
            self._noise_floor += (rms - self._noise_floor) / (self._noise_floor_frames + 1)
            self._noise_floor_frames += 1
        else:
            self._noise_floor += NOISE_FLOOR_ALPHA * (rms - self._noise_floor)

    def _process_vad(self, chunk: np.ndarray) -> None:
        rms = np.sqrt(np.mean(chunk**2))
        with self._mute_lock:
            if self._muted:
                barge_thresh = max(self._noise_floor * BARGE_IN_MULTIPLIER, BARGE_IN_MIN)
                if rms > barge_thresh and self._interrupt_callback:
                    self._muted = False
                    self._interrupt_callback()
                return
        threshold = self._speech_threshold
        is_speech = rms > threshold
        frame_duration = len(chunk) / SAMPLE_RATE

        now = time.monotonic()
        if now - self._threshold_log_time > 5.0:
            self._threshold_log_time = now
            log.info("VAD: noise_floor=%.5f threshold=%.5f rms=%.5f speech=%s",
                     self._noise_floor, threshold, rms, is_speech)

        if not is_speech and not self._speech_active:
            self._update_noise_floor(rms)

        if is_speech:
            if not self._speech_active:
                log.info("TIMING vad: speech started (waiting for silence timeout)")
                self._speech_start_time = time.monotonic()
            self._speech_buffer = np.append(self._speech_buffer, chunk)
            self._silence_frames = 0.0
            self._speech_duration += frame_duration
            if not self._speech_active:
                self._speech_active = True
            if self._speech_duration >= MAX_SPEECH_DURATION:
                log_duration("vad.speech_segment", time.monotonic() - self._speech_start_time)
                self._emit(self._speech_buffer.copy())
                self._speech_buffer = np.array([], dtype=np.float32)
                self._speech_active = False
                self._silence_frames = 0.0
                self._speech_duration = 0.0
        else:
            if self._speech_active:
                self._silence_frames += frame_duration
                self._speech_buffer = np.append(self._speech_buffer, chunk)

                if self._silence_frames >= SILENCE_TIMEOUT:
                    total_duration = time.monotonic() - self._speech_start_time
                    log_duration("vad.speech_to_emit", total_duration)
                    self._emit(self._speech_buffer.copy())

                    self._speech_buffer = np.array([], dtype=np.float32)
                    self._speech_active = False
                    self._silence_frames = 0.0
                    self._speech_duration = 0.0

    def _run(self) -> None:
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                callback=self._audio_callback,
                blocksize=FRAME_SIZE,
                device=self._device_id,
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
