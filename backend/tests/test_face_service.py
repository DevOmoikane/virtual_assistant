from __future__ import annotations

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

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


class TestFaceServicePersonality:
    def test_set_and_get_personality(self):
        with patch.object(FaceService, "_ensure_collection"):
            svc = FaceService()
            svc._ready = True
        try:
            svc._personalities = {}
            svc.set_personality("Alice", "cheerful")
            assert svc.get_personality("Alice") == "cheerful"
        finally:
            svc._app = None

    def test_get_personality_unknown_returns_none(self):
        with patch.object(FaceService, "_ensure_collection"):
            svc = FaceService()
            svc._ready = True
        try:
            svc._personalities = {}
            assert svc.get_personality("Unknown") is None
        finally:
            svc._app = None

    def test_set_personality_overwrites(self):
        with patch.object(FaceService, "_ensure_collection"):
            svc = FaceService()
            svc._ready = True
        try:
            svc._personalities = {}
            svc.set_personality("Bob", "formal")
            svc.set_personality("Bob", "cheerful")
            assert svc.get_personality("Bob") == "cheerful"
        finally:
            svc._app = None

    def test_personalities_persist_to_disk(self):
        import tempfile
        import os as os_module
        import virtual_assistant_be.services.face_service as fs

        test_file = os_module.path.join(tempfile.gettempdir(), "test_personality.json")
        with (
            patch.object(FaceService, "_ensure_collection"),
            patch.object(fs, "_PERSONALITY_FILE", test_file),
        ):
            svc = FaceService()
            svc._ready = True
            try:
                svc._personalities = {}
                svc.set_personality("Alice", "cheerful")
                svc2 = FaceService()
                svc2._ready = True
                try:
                    assert svc2.get_personality("Alice") == "cheerful"
                finally:
                    svc2._app = None
            finally:
                svc._app = None
                try:
                    os_module.remove(test_file)
                except OSError:
                    pass


class TestFaceServiceLanguage:
    def test_set_and_get_language(self):
        with patch.object(FaceService, "_ensure_collection"):
            svc = FaceService()
            svc._ready = True
        try:
            svc._languages = {}
            svc.set_language("Alice", "es")
            assert svc.get_language("Alice") == "es"
        finally:
            svc._app = None

    def test_get_language_unknown_returns_none(self):
        with patch.object(FaceService, "_ensure_collection"):
            svc = FaceService()
            svc._ready = True
        try:
            svc._languages = {}
            assert svc.get_language("Unknown") is None
        finally:
            svc._app = None

    def test_set_language_overwrites(self):
        with patch.object(FaceService, "_ensure_collection"):
            svc = FaceService()
            svc._ready = True
        try:
            svc._languages = {}
            svc.set_language("Bob", "en")
            svc.set_language("Bob", "es")
            assert svc.get_language("Bob") == "es"
        finally:
            svc._app = None

    def test_languages_persist_to_disk(self):
        import tempfile
        import os as os_module
        import virtual_assistant_be.services.face_service as fs

        test_file = os_module.path.join(tempfile.gettempdir(), "test_language.json")
        with (
            patch.object(FaceService, "_ensure_collection"),
            patch.object(fs, "_LANGUAGE_FILE", test_file),
        ):
            svc = FaceService()
            svc._ready = True
            try:
                svc._languages = {}
                svc.set_language("Alice", "es")
                svc2 = FaceService()
                svc2._ready = True
                try:
                    assert svc2.get_language("Alice") == "es"
                finally:
                    svc2._app = None
            finally:
                svc._app = None
                try:
                    os_module.remove(test_file)
                except OSError:
                    pass



