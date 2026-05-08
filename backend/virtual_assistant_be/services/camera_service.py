from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Callable, Awaitable

import cv2
import numpy as np

from virtual_assistant_be.core.config import settings

try:
    import mediapipe as mp

    _HAS_MEDIAPIPE = True
except ImportError:
    _HAS_MEDIAPIPE = False

log = logging.getLogger(__name__)

EventCallback = Callable[[str, dict], Awaitable[None]]

LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]


def _is_finger_extended(landmarks: list, finger_tip: int, finger_pip: int) -> bool:
    return landmarks[finger_tip].y < landmarks[finger_pip].y


def _is_thumb_extended(landmarks: list) -> bool:
    return landmarks[4].x < landmarks[3].x


def _classify_gesture(landmarks: list) -> str | None:
    thumb_ext = _is_thumb_extended(landmarks)
    index_ext = _is_finger_extended(landmarks, 8, 6)
    middle_ext = _is_finger_extended(landmarks, 12, 10)
    ring_ext = _is_finger_extended(landmarks, 16, 14)
    pinky_ext = _is_finger_extended(landmarks, 20, 18)

    fingers_extended = sum([index_ext, middle_ext, ring_ext, pinky_ext])

    if thumb_ext and not any([index_ext, middle_ext, ring_ext, pinky_ext]):
        return "thumbs_up"
    if index_ext and not any([middle_ext, ring_ext, pinky_ext]):
        return "point"
    if index_ext and middle_ext and not ring_ext and not pinky_ext:
        return "peace"
    if fingers_extended == 4 and thumb_ext:
        return "open_palm"
    if fingers_extended == 0:
        return "fist"

    return None


class CameraService:
    def __init__(self, event_callback: EventCallback | None = None) -> None:
        self._callback = event_callback
        self._running = False
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        self._face_detection = None
        self._hands = None
        self._cap: cv2.VideoCapture | None = None

        self._person_present = False
        self._wave_history: list[float] = []

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
        self._face_detection = None
        self._hands = None
        log.info("Camera service stopped")

    def _emit(self, event: str, data: dict | None = None) -> None:
        if self._callback and self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(
                asyncio.ensure_future, self._callback(event, data or {})
            )

    def _run(self) -> None:
        mp = __import__("mediapipe", fromlist=["python"])

        self._face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5,
        )
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        _drawing = mp.solutions.drawing_utils

        cap = cv2.VideoCapture(settings.camera_device_id or 0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.camera_height)
        self._cap = cap

        log.info("Camera opened: %s", cap.isOpened())

        while self._running:
            ret, frame = cap.read()
            if not ret:
                self._running = False
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            self._process_faces(rgb)
            self._process_hands(rgb)

        cap.release()
        self._face_detection.close()
        self._hands.close()
        log.info("Camera loop ended")

    def _process_faces(self, rgb: np.ndarray) -> None:
        if self._face_detection is None:
            return
        results = self._face_detection.process(rgb)
        faces_detected = results.detections is not None

        if faces_detected and not self._person_present:
            self._person_present = True
            self._emit("person_appeared")
            self._wave_history.clear()
        elif not faces_detected and self._person_present:
            self._person_present = False
            self._emit("person_disappeared")

    def _process_hands(self, rgb: np.ndarray) -> None:
        if self._hands is None:
            return
        results = self._hands.process(rgb)

        if not results.multi_hand_landmarks:
            return

        for hand_landmarks in results.multi_hand_landmarks:
            gesture = _classify_gesture(hand_landmarks.landmark)
            if gesture is None:
                continue

            wrist = hand_landmarks.landmark[0]
            self._emit("gesture_detected", {
                "gesture": gesture,
                "x": round(wrist.x, 3),
                "y": round(wrist.y, 3),
            })

            if gesture == "open_palm":
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
