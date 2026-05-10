from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from typing import Callable, Awaitable

import cv2
import numpy as np

from virtual_assistant_be.core.config import settings

try:
    import mediapipe as mp
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision import (
        FaceDetector,
        FaceDetectorOptions,
        GestureRecognizer,
        GestureRecognizerOptions,
        RunningMode,
    )

    _HAS_MEDIAPIPE = True
except ImportError:
    _HAS_MEDIAPIPE = False

log = logging.getLogger(__name__)

EventCallback = Callable[[str, dict], Awaitable[None]]

MP_GESTURE_MAP = {
    "Closed_Fist": "fist",
    "Open_Palm": "open_palm",
    "Pointing_Up": "point",
    "Thumb_Down": "thumbs_down",
    "Thumb_Up": "thumbs_up",
    "Victory": "peace",
    "ILoveYou": "love",
}

_MODEL_URLS = {
    "blaze_face_short_range.tflite": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_detector/blaze_face_short_range/float16/1/"
        "blaze_face_short_range.tflite"
    ),
    "gesture_recognizer.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "gesture_recognizer/gesture_recognizer/float16/1/"
        "gesture_recognizer.task"
    ),
}


def _download_models() -> bool:
    import requests

    models_dir = settings.mediapipe_models_dir
    os.makedirs(models_dir, exist_ok=True)
    ok = True

    for filename, url in _MODEL_URLS.items():
        dest = os.path.join(models_dir, filename)
        if os.path.isfile(dest):
            continue
        log.info("Downloading %s ...", filename)
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                f.write(resp.content)
            log.info("Downloaded %s (%.1f MB)", filename, len(resp.content) / 1e6)
        except Exception:
            log.warning("Failed to download %s", filename, exc_info=True)
            ok = False

    return ok


class CameraService:
    def __init__(self, event_callback: EventCallback | None = None,
                 face_service=None) -> None:
        self._callback = event_callback
        self._face_service = face_service
        self._running = False
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        self._face_detector = None
        self._gesture_recognizer = None
        self._cap: cv2.VideoCapture | None = None

        self._person_present = False
        self._wave_history: list[float] = []
        self._face_hit_streak = 0
        self._face_miss_streak = 0
        self._last_frame_bgr: np.ndarray | None = None

        self._camera_idx = 0
        self._audio_device_id: int | None = None
        self._probe_devices()

    @property
    def audio_device_id(self) -> int | None:
        return self._audio_device_id

    def _probe_devices(self) -> None:
        self._camera_idx, cam_name = self._find_camera()
        if cam_name:
            self._audio_device_id = self._find_audio_device(cam_name)

    @staticmethod
    def _find_audio_device(camera_name: str) -> int | None:
        import sounddevice as sd
        name_lower = camera_name.lower()
        for dev in sd.query_devices():
            if dev.get("max_input_channels", 0) > 0:
                dev_name = dev.get("name", "").lower()
                common = set(name_lower.split()) & set(dev_name.split())
                if len(common) >= 2:
                    log.info("Matched audio device %d: %s", dev["index"], dev["name"])
                    return dev["index"]
        log.info("No matching audio device found for camera '%s'", camera_name)
        return None

    def set_callback(self, callback: EventCallback) -> None:
        self._callback = callback

    def start(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        if not _HAS_MEDIAPIPE:
            log.warning("MediaPipe not installed — camera service disabled")
            return
        if self._running:
            return
        self._running = True
        self._loop = loop or asyncio.get_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        log.info("Camera service started")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        if self._cap:
            self._cap.release()
        self._cap = None
        if self._face_detector:
            self._face_detector.close()
            self._face_detector = None
        if self._gesture_recognizer:
            self._gesture_recognizer.close()
            self._gesture_recognizer = None
        log.info("Camera service stopped")

    def _emit(self, event: str, data: dict | None = None) -> None:
        if self._callback and self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(
                asyncio.ensure_future, self._callback(event, data or {})
            )

    @staticmethod
    def _find_camera() -> tuple[int, str | None]:
        cfg = settings.camera_device_id
        if cfg is not None:
            return cfg, None

        candidates: list[tuple[int, str | None]] = []
        for idx in range(4):
            cap = cv2.VideoCapture(idx)
            ok = cap.isOpened()
            cap.release()
            if ok:
                name = None
                try:
                    path = f"/sys/class/video4linux/video{idx}/name"
                    if os.path.isfile(path):
                        with open(path) as f:
                            name = f.read().strip()
                except Exception:
                    pass
                candidates.append((idx, name))

        if not candidates:
            return 0, None

        for idx, name in reversed(candidates):
            if name and ("usb" in name.lower() or "webcam" in name.lower() or "camera" in name.lower()):
                log.info("Selected USB camera at index %d: %s", idx, name)
                return idx, name

        chosen, chosen_name = candidates[0]
        log.info("Selected camera index %d%s", chosen, f" ({chosen_name})" if chosen_name else "")
        return chosen, chosen_name

    def _run(self) -> None:
        face_model = settings.face_detection_model
        gesture_model = settings.gesture_recognition_model

        if not os.path.isfile(face_model) or not os.path.isfile(gesture_model):
            log.info("Models missing, downloading ...")
            if not _download_models():
                log.warning("Failed to download models, camera disabled")
                self._running = False
                return

        try:
            face_opts = FaceDetectorOptions(
                base_options=BaseOptions(model_asset_path=face_model),
                running_mode=RunningMode.IMAGE,
            )
            self._face_detector = FaceDetector.create_from_options(face_opts)
        except Exception:
            log.warning("Failed to create face detector", exc_info=True)

        try:
            gesture_opts = GestureRecognizerOptions(
                base_options=BaseOptions(model_asset_path=gesture_model),
                running_mode=RunningMode.IMAGE,
            )
            self._gesture_recognizer = GestureRecognizer.create_from_options(gesture_opts)
        except Exception:
            log.warning("Failed to create gesture recognizer", exc_info=True)

        cap = cv2.VideoCapture(self._camera_idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.camera_height)
        self._cap = cap

        log.info("Camera %d opened: %s", self._camera_idx, cap.isOpened())

        while self._running:
            ret, frame = cap.read()
            if not ret:
                self._running = False
                break

            self._last_frame_bgr = frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            self._process_faces(mp_image)
            self._process_gestures(mp_image)

        cap.release()
        log.info("Camera loop ended")

    def _process_faces(self, mp_image: mp.Image) -> None:
        if self._face_detector is None:
            return

        try:
            result = self._face_detector.detect(mp_image)
            faces_detected = len(result.detections) > 0

            if faces_detected:
                self._face_hit_streak += 1
                self._face_miss_streak = 0
                if not self._person_present and self._face_hit_streak >= 3:
                    self._person_present = True
                    self._wave_history.clear()
                    name = self._recognize_face()
                    self._emit("person_appeared", {"name": name} if name else {})
            else:
                self._face_miss_streak += 1
                self._face_hit_streak = 0
                if self._person_present and self._face_miss_streak >= 10:
                    self._person_present = False
                    self._emit("person_disappeared")
        except Exception:
            pass

    def _recognize_face(self) -> str | None:
        if not self._face_service or not self._face_service.enabled:
            return None
        frame = self._last_frame_bgr
        if frame is None:
            return None
        emb = self._face_service.get_embedding(frame)
        if emb is None:
            return None
        name, _dist = self._face_service.recognize(emb)
        return name

    def _process_gestures(self, mp_image: mp.Image) -> None:
        if self._gesture_recognizer is None:
            return

        try:
            result = self._gesture_recognizer.recognize(mp_image)
        except Exception:
            return

        if not result.gestures:
            return

        for hand_idx, gesture_list in enumerate(result.gestures):
            if not gesture_list:
                continue

            top = gesture_list[0]
            mp_gesture = top.category_name
            gesture = MP_GESTURE_MAP.get(mp_gesture, mp_gesture.lower())

            landmarks = result.hand_landmarks[hand_idx]
            wrist = landmarks[0] if landmarks else None
            x = round(wrist.x, 3) if wrist else 0.0
            y = round(wrist.y, 3) if wrist else 0.0

            self._emit("gesture_detected", {
                "gesture": gesture,
                "x": x,
                "y": y,
            })

            if gesture == "open_palm" and wrist:
                self._detect_wave(wrist.x)

    def _detect_wave(self, x: float) -> None:
        self._wave_history.append(x)
        if len(self._wave_history) > 15:
            self._wave_history.pop(0)

        if len(self._wave_history) < 10:
            return

        vals = self._wave_history[-10:]
        oscillations = sum(
            1 for i in range(1, len(vals)) if abs(vals[i] - vals[i - 1]) > 0.02
        )
        if oscillations >= 4:
            self._emit("gesture_detected", {"gesture": "wave"})
            self._wave_history.clear()
