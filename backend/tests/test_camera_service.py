from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

import virtual_assistant_be.services.camera_service as cs
from virtual_assistant_be.services.camera_service import (
    CameraService,
)


class TestGestureMap:
    def test_all_mediapipe_categories_mapped(self):
        expected = {
            "Closed_Fist": "fist",
            "Open_Palm": "open_palm",
            "Pointing_Up": "point",
            "Thumb_Down": "thumbs_down",
            "Thumb_Up": "thumbs_up",
            "Victory": "peace",
            "ILoveYou": "love",
        }
        assert cs.MP_GESTURE_MAP == expected

    def test_all_values_are_lowercase(self):
        for v in cs.MP_GESTURE_MAP.values():
            assert v == v.lower()


class TestDownloadModels:
    @patch("requests.get")
    def test_downloads_missing_models(self, mock_get, tmp_path):
        mock_get.return_value = MagicMock(
            status_code=200, content=b"model data"
        )
        models_dir = tmp_path / "models"
        settings_mock = MagicMock()
        settings_mock.mediapipe_models_dir = str(models_dir)
        with patch(
            "virtual_assistant_be.services.camera_service.settings", settings_mock
        ):
            result = cs._download_models()

        assert result is True
        assert mock_get.call_count == 2
        for filename in [
            "blaze_face_short_range.tflite",
            "gesture_recognizer.task",
        ]:
            dest = models_dir / filename
            assert dest.exists()
            assert dest.read_bytes() == b"model data"

    @patch("requests.get")
    def test_skips_existing_models(self, mock_get, tmp_path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "blaze_face_short_range.tflite").write_bytes(b"existing")
        (models_dir / "gesture_recognizer.task").write_bytes(b"existing")
        settings_mock = MagicMock()
        settings_mock.mediapipe_models_dir = str(models_dir)
        with patch(
            "virtual_assistant_be.services.camera_service.settings", settings_mock
        ):
            result = cs._download_models()

        assert result is True
        mock_get.assert_not_called()

    @patch("requests.get")
    def test_download_failure_returns_false(self, mock_get, tmp_path):
        mock_get.side_effect = Exception("Network error")
        settings_mock = MagicMock()
        settings_mock.mediapipe_models_dir = str(tmp_path / "models")
        with patch(
            "virtual_assistant_be.services.camera_service.settings", settings_mock
        ):
            result = cs._download_models()

        assert result is False


class TestWaveDetection:
    @pytest.fixture
    def service(self):
        svc = CameraService()
        svc._wave_history = []
        return svc

    def test_single_call_no_wave(self, service):
        service._detect_wave(0.5)
        assert len(service._wave_history) == 1

    def test_insufficient_history_no_wave(self, service):
        for x in [0.5, 0.48, 0.52, 0.47, 0.51, 0.49, 0.5, 0.48, 0.52]:
            service._detect_wave(x)
        assert len(service._wave_history) == 9

    def test_wave_detected_emits_event(self, service):
        callback = AsyncMock()
        service._callback = callback
        loop = MagicMock()
        loop.is_closed.return_value = False
        service._loop = loop

        # Alternating x values to simulate hand waving
        vals = [
            0.50,
            0.55,
            0.50,
            0.55,
            0.50,
            0.55,
            0.50,
            0.55,
            0.50,
            0.55,
        ]
        for x in vals:
            service._detect_wave(x)

        # Wave should have been emitted after 10 oscillations
        gesture_calls = [
            c
            for c in callback.call_args_list
            if c[0][0] == "gesture_detected"
        ]
        assert any(c[0][1] == {"gesture": "wave"} for c in gesture_calls)
        assert len(service._wave_history) == 0

    def test_history_limited_to_15(self, service):
        for x in range(20):
            service._detect_wave(float(x) / 100.0)
        assert len(service._wave_history) <= 15


@pytest.fixture
def mock_mp_module():
    mp_mock = MagicMock()
    mp_mock.ImageFormat.SRGB = "srgb"
    mp_mock.Image = MagicMock()
    cs.mp = mp_mock
    cs.FaceDetector = MagicMock()
    cs.GestureRecognizer = MagicMock()
    cs.FaceDetectorOptions = MagicMock()
    cs.GestureRecognizerOptions = MagicMock()
    cs.BaseOptions = MagicMock()
    cs.RunningMode = MagicMock()
    cs._HAS_MEDIAPIPE = True
    yield


class TestCameraServiceLifecycle:
    @pytest.fixture(autouse=True)
    def _setup(self, mock_mp_module):
        with patch("virtual_assistant_be.services.camera_service._download_models",
                   return_value=True):
            yield

    def test_start_missing_mediapipe(self):
        cs._HAS_MEDIAPIPE = False
        svc = CameraService()
        svc.start()
        assert not svc._running
        cs._HAS_MEDIAPIPE = True

    @patch("virtual_assistant_be.services.camera_service.cv2")
    def test_start_stop(self, mock_cv2):
        cap_mock = MagicMock()
        cap_mock.isOpened.return_value = True
        cap_mock.read.return_value = (False, None)
        mock_cv2.VideoCapture.return_value = cap_mock

        svc = CameraService()
        loop = MagicMock()
        svc.start(loop=loop)
        assert svc._running
        assert svc._thread is not None

        svc.stop()
        assert not svc._running

    @patch("virtual_assistant_be.services.camera_service.cv2")
    def test_does_not_start_twice(self, mock_cv2):
        cap_mock = MagicMock()
        cap_mock.isOpened.return_value = True
        cap_mock.read.return_value = (True, MagicMock())
        mock_cv2.VideoCapture.return_value = cap_mock

        svc = CameraService()
        loop = MagicMock()
        loop.is_closed.return_value = False
        svc.start(loop=loop)
        thread1 = svc._thread
        svc.start(loop=loop)
        assert svc._thread is thread1
        svc.stop()

    def test_person_appeared_event(self):
        detection = MagicMock()
        detection.detections = [MagicMock()]
        face_detector_instance = cs.FaceDetector.create_from_options.return_value
        face_detector_instance.detect.return_value = detection

        svc = CameraService()
        svc._face_detector = face_detector_instance

        mp_image = MagicMock()
        with patch.object(svc, '_emit') as mock_emit:
            svc._process_faces(mp_image)

        assert svc._person_present is True
        mock_emit.assert_called_once_with("person_appeared")

    def test_person_disappeared_event(self):
        detection = MagicMock()
        detection.detections = []
        face_detector_instance = cs.FaceDetector.create_from_options.return_value
        face_detector_instance.detect.return_value = detection

        svc = CameraService()
        svc._person_present = True
        svc._face_detector = face_detector_instance

        mp_image = MagicMock()
        with patch.object(svc, '_emit') as mock_emit:
            svc._process_faces(mp_image)

        assert svc._person_present is False
        mock_emit.assert_called_once_with("person_disappeared")

    def test_person_stays_present_no_event(self):
        detection = MagicMock()
        detection.detections = [MagicMock()]
        face_detector_instance = cs.FaceDetector.create_from_options.return_value
        face_detector_instance.detect.return_value = detection

        svc = CameraService()
        svc._person_present = True
        svc._face_detector = face_detector_instance

        mp_image = MagicMock()
        with patch.object(svc, '_emit') as mock_emit:
            svc._process_faces(mp_image)

        mock_emit.assert_not_called()

    def test_person_appeared_clears_wave_history(self):
        detection = MagicMock()
        detection.detections = [MagicMock()]
        face_detector_instance = cs.FaceDetector.create_from_options.return_value
        face_detector_instance.detect.return_value = detection

        svc = CameraService()
        svc._face_detector = face_detector_instance
        svc._wave_history = [0.5, 0.55, 0.5]

        mp_image = MagicMock()
        with patch.object(svc, '_emit'):
            svc._process_faces(mp_image)

        assert svc._wave_history == []

    def test_gesture_detected_event(self):
        svc = CameraService()
        face_detector_instance = cs.FaceDetector.create_from_options.return_value
        gesture_recognizer_instance = (
            cs.GestureRecognizer.create_from_options.return_value
        )
        svc._face_detector = face_detector_instance
        svc._gesture_recognizer = gesture_recognizer_instance

        result = MagicMock()
        result.gestures = [[MagicMock()]]
        result.gestures[0][0].category_name = "Thumb_Up"
        result.hand_landmarks = [[MagicMock()]]
        result.hand_landmarks[0][0].x = 0.5
        result.hand_landmarks[0][0].y = 0.3
        gesture_recognizer_instance.recognize.return_value = result

        mp_image = MagicMock()
        with patch.object(svc, '_emit') as mock_emit:
            svc._process_gestures(mp_image)

        mock_emit.assert_called_once_with(
            "gesture_detected", {"gesture": "thumbs_up", "x": 0.5, "y": 0.3}
        )
