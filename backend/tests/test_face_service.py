from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import cv2
import numpy as np
import pytest

from virtual_assistant_be.core.behavior_controller import BehaviorController
from virtual_assistant_be.services.face_service import FaceService

TEST_PHOTO = "/Users/israel/dev/omoikane/virtual_assistant/resources/test_photo.jpg"


@pytest.fixture(scope="session")
def test_image_bgr() -> np.ndarray:
    img = cv2.imread(TEST_PHOTO)
    assert img is not None, f"Test photo not found: {TEST_PHOTO}"
    return img


@pytest.fixture(scope="session")
def reference_embedding(test_image_bgr) -> np.ndarray:
    with patch.object(FaceService, "_ensure_collection"):
        svc = FaceService()
        svc._ready = True
    emb = svc.get_embedding(test_image_bgr)
    assert emb is not None, "No face detected in test photo"
    svc._app = None
    return emb


class TestFaceServicePhoto:
    """Tests using the real test photo with insightface."""

    def test_embedding_extracted_from_photo(self, test_image_bgr):
        with patch.object(FaceService, "_ensure_collection"):
            svc = FaceService()
            svc._ready = True
        try:
            emb = svc.get_embedding(test_image_bgr)
            assert emb is not None, "No face detected"
            assert isinstance(emb, np.ndarray)
            assert emb.shape == (512,)
            assert emb.dtype == np.float32
        finally:
            svc._app = None

    def test_embedding_is_unit_normalized(self, test_image_bgr):
        with patch.object(FaceService, "_ensure_collection"):
            svc = FaceService()
            svc._ready = True
        try:
            emb = svc.get_embedding(test_image_bgr)
            norm = np.linalg.norm(emb)
            assert abs(norm - 1.0) < 1e-5
        finally:
            svc._app = None

    def test_embedding_is_deterministic(self, test_image_bgr):
        with patch.object(FaceService, "_ensure_collection"):
            svc = FaceService()
            svc._ready = True
        try:
            emb1 = svc.get_embedding(test_image_bgr)
            emb2 = svc.get_embedding(test_image_bgr)
            np.testing.assert_array_almost_equal(emb1, emb2, decimal=5)
        finally:
            svc._app = None

    def test_recognize_known_face(self, reference_embedding):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "distances": [[0.35]],
            "metadatas": [[{"name": "Alice"}]],
        }
        mock_resp.raise_for_status = MagicMock()

        with (
            patch.object(FaceService, "_ensure_collection"),
            patch("virtual_assistant_be.services.face_service.requests.post", return_value=mock_resp) as mock_post,
        ):
            svc = FaceService()
            svc._ready = True
            svc._collection_id = "test-collection-id"
            try:
                name, dist = svc.recognize(reference_embedding)
                assert name == "Alice"
                assert dist == 0.35
                assert svc.last_unknown_embedding is None
                mock_post.assert_called_once()
            finally:
                svc._app = None

    def test_recognize_unknown_face(self, reference_embedding):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "distances": [[0.85]],
            "metadatas": [[{"name": "Unknown"}]],
        }
        mock_resp.raise_for_status = MagicMock()

        with (
            patch.object(FaceService, "_ensure_collection"),
            patch("virtual_assistant_be.services.face_service.requests.post", return_value=mock_resp),
        ):
            svc = FaceService()
            svc._ready = True
            svc._collection_id = "test-collection-id"
            try:
                name, dist = svc.recognize(reference_embedding)
                assert name is None
                assert dist == 0.85
                assert svc.last_unknown_embedding is not None
            finally:
                svc._app = None

    def test_register_stores_embedding(self, reference_embedding):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        with (
            patch.object(FaceService, "_ensure_collection"),
            patch("virtual_assistant_be.services.face_service.requests.post", return_value=mock_resp) as mock_post,
        ):
            svc = FaceService()
            svc._ready = True
            svc._collection_id = "test-collection-id"
            try:
                ok = svc.register("Bob", reference_embedding)
                assert ok is True
                mock_post.assert_called_once()
                _, kwargs = mock_post.call_args
                assert kwargs["json"]["metadatas"][0]["name"] == "Bob"
                assert len(kwargs["json"]["embeddings"][0]) == 512
            finally:
                svc._app = None

    def test_register_failure_returns_false(self, reference_embedding):
        with (
            patch.object(FaceService, "_ensure_collection"),
            patch("virtual_assistant_be.services.face_service.requests.post", side_effect=Exception("timeout")),
        ):
            svc = FaceService()
            svc._ready = True
            svc._collection_id = "test-collection-id"
            try:
                ok = svc.register("Bob", reference_embedding)
                assert ok is False
            finally:
                svc._app = None

    def test_disabled_service_returns_none(self, test_image_bgr):
        with patch.object(FaceService, "_ensure_collection", side_effect=RuntimeError("no chroma")):
            svc = FaceService()
        assert svc.enabled is False
        assert svc.get_embedding(test_image_bgr) is None
        name, dist = svc.recognize(np.zeros(512, dtype=np.float32))
        assert name is None
        assert svc.register("X", np.zeros(512, dtype=np.float32)) is False


@pytest.fixture
def controller():
    send_fn = AsyncMock()
    ctrl = BehaviorController(send_fn=send_fn)
    return ctrl


class TestBehaviorControllerIntegration:
    """End-to-end unknown→register→recognize flow through BehaviorController."""

    @pytest.mark.asyncio
    async def test_unknown_flow_sets_pending_name(self, controller):
        controller._current_language = "en"
        with (
            patch.object(controller, "send_animation"),
            patch.object(controller, "_send_speak"),
            patch.object(controller, "_send_listen") as mock_listen,
            patch.object(controller.tts, "speak"),
            patch.object(controller.memory, "store_person_event"),
        ):
            await controller._on_person_appeared(data={})
            assert controller._pending_name is True
            mock_listen.assert_awaited_once_with(True)

    @pytest.mark.asyncio
    async def test_register_name_with_real_embedding(self, controller, reference_embedding):
        controller._current_language = "en"
        controller._pending_name = True
        controller.face_service.last_unknown_embedding = reference_embedding

        with (
            patch.object(controller, "_send_speak") as mock_speak,
            patch.object(controller, "_send_listen") as mock_listen,
            patch.object(controller.tts, "speak"),
            patch.object(controller.face_service, "register", return_value=True) as mock_register,
        ):
            ok = await controller._register_name("Bob")
            assert ok is True
            mock_register.assert_called_once_with("Bob", reference_embedding)
            mock_speak.assert_awaited_once()
            mock_listen.assert_awaited_once_with(False)
