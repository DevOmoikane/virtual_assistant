from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from virtual_assistant_be.services.rag_service import RagService


@pytest.fixture
def service():
    svc = RagService()
    svc._collection_id = "test-collection-id"
    return svc


@pytest.fixture
def service_no_collection():
    svc = RagService()
    svc._collection_id = ""
    svc._ensure_collection = lambda: ""
    return svc


class TestRagService:
    def test_embed_single(self, service):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
        mock_resp.raise_for_status.return_value = None

        with patch("virtual_assistant_be.services.rag_service.requests.post", return_value=mock_resp):
            result = service._embed("hello")
            assert result == [0.1, 0.2, 0.3]

    def test_embed_multiple(self, service):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embeddings": [[0.1], [0.2]]}
        mock_resp.raise_for_status.return_value = None

        with patch("virtual_assistant_be.services.rag_service.requests.post", return_value=mock_resp):
            result = service._embed(["hello", "world"])
            assert result == [[0.1], [0.2]]

    def test_retrieve_returns_documents(self, service):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"documents": [["doc1", "doc2", "doc3"]]}
        mock_resp.raise_for_status.return_value = None

        with patch("virtual_assistant_be.services.rag_service.requests.post", return_value=mock_resp) as mock_post:
            with patch.object(service, "_embed", return_value=[[0.1, 0.2]]):
                result = service.retrieve("test query", k=3)
                assert result == ["doc1", "doc2", "doc3"]

                call_kwargs = mock_post.call_args[1]
                assert call_kwargs["json"]["n_results"] == 3

    def test_retrieve_empty_on_no_collection(self, service_no_collection):
        result = service_no_collection.retrieve("test")
        assert result == []

    def test_retrieve_empty_on_error(self, service):
        import requests as req_lib
        with patch("virtual_assistant_be.services.rag_service.requests.post", side_effect=req_lib.exceptions.ConnectionError("error")):
            with patch.object(service, "_embed", return_value=[[0.1]]):
                result = service.retrieve("test")
                assert result == []

    def test_ingest_chunks_and_adds(self, service):
        with patch("virtual_assistant_be.services.rag_service.requests.post", return_value=MagicMock()) as mock_post:
            with patch.object(service, "_embed", return_value=[[0.1], [0.2]]):
                n = service.ingest("hello world how are you today", source="test.txt")
                assert n > 0
                mock_post.assert_called_once()

    def test_ingest_returns_zero_on_no_collection(self, service_no_collection):
        n = service_no_collection.ingest("test", source="test.txt")
        assert n == 0

    def test_chunk_text(self, service):
        text = "word " * 1000
        chunks = service._chunk_text(text, chunk_size=100, overlap=20)
        assert len(chunks) > 0
        assert all(len(c.split()) <= 100 for c in chunks)

    def test_ask_without_context(self, service):
        with patch.object(service, "retrieve", return_value=[]):
            result = service.ask("test question")
            assert result == ""
