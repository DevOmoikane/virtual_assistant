from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from virtual_assistant_be.services.rag_service import RagService, chunk_text, clean_text


def _make_mock_client():
    """Return a mock OpenSearch client that looks like it has an index."""
    client = MagicMock()
    client.indices.exists.return_value = True
    client.search.return_value = {"hits": {"hits": []}}
    client.delete_by_query.return_value = {"deleted": 0}
    return client


@pytest.fixture
def service():
    svc = RagService()
    svc._client = _make_mock_client()
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
        text_hits = [
            {"_id": "1", "_score": 2.0, "_source": {"text": "doc1"}},
            {"_id": "2", "_score": 1.5, "_source": {"text": "doc2"}},
        ]
        knn_hits = [
            {"_id": "3", "_score": 3.0, "_source": {"text": "doc3"}},
        ]
        service._client.search.side_effect = [
            {"hits": {"hits": text_hits}},
            {"hits": {"hits": knn_hits}},
        ]

        with patch.object(service, "_embed", return_value=[0.1, 0.2, 0.3]):
            result = service.retrieve("test query", k=3)
            assert result == ["doc3", "doc1", "doc2"]

    def test_retrieve_empty_on_no_client(self, service):
        service._client = _make_mock_client()
        service._client.indices.exists.return_value = False
        result = service.retrieve("test")
        assert result == []

    def test_retrieve_empty_on_error(self, service):
        service._client.search.side_effect = Exception("error")
        with patch.object(service, "_embed", return_value=[0.1]):
            result = service.retrieve("test")
            assert result == []

    def test_ingest_chunks_and_adds(self, service):
        with patch("virtual_assistant_be.services.rag_service.helpers.bulk", return_value=(3, [])) as mock_bulk:
            with patch.object(service, "_embed", return_value=[[0.1], [0.2], [0.3]]):
                n = service.ingest("hello world how are you today", source="test.txt")
                assert n > 0
                mock_bulk.assert_called_once()

    def test_ingest_returns_zero_on_no_client(self, service):
        with patch.object(service, "_get_client", return_value=None):
            n = service.ingest("test", source="test.txt")
            assert n == 0

    def test_chunk_text(self):
        text = "word " * 1000
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        assert len(chunks) > 0
        assert all(len(c.split()) <= 100 for c in chunks)

    def test_clean_text_removes_hyphenated_breaks(self):
        result = clean_text("exam-\nple")
        assert "exam-\nple" not in result
        assert "example" in result

    def test_ask_without_context(self, service):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": ""}
        mock_resp.raise_for_status.return_value = None
        with patch.object(service, "retrieve", return_value=[]):
            with patch("virtual_assistant_be.services.rag_service.requests.post", return_value=mock_resp):
                result = service.ask("test question")
                assert result == ""
