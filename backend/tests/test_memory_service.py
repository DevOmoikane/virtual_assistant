from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from virtual_assistant_be.services.memory_service import MemoryService


@pytest.fixture
def service():
    svc = MemoryService()
    svc._collection_id = "test-mem-collection"
    # Use a temp counter file
    svc._counter_file = os.path.join(tempfile.gettempdir(), "test_person_counter.json")
    if os.path.exists(svc._counter_file):
        os.remove(svc._counter_file)
    svc._person_count = 0
    return svc


@pytest.fixture
def mock_embed():
    with patch.object(MemoryService, "_embed", return_value=[0.1, 0.2, 0.3]) as m:
        yield m


class TestMemoryService:
    def test_person_count_starts_at_zero(self):
        svc = MemoryService()
        assert svc.get_person_count() == 0

    def test_store_person_appeared_increments_count(self, service, mock_embed):
        with patch("virtual_assistant_be.services.memory_service.requests.post") as mock_post:
            service.store_person_event("appeared")
        assert service.get_person_count() == 1

    def test_store_person_appeared_multiple_times(self, service, mock_embed):
        with patch("virtual_assistant_be.services.memory_service.requests.post"):
            for _ in range(5):
                service.store_person_event("appeared")
        assert service.get_person_count() == 5

    def test_store_person_disappeared_does_not_increment(self, service, mock_embed):
        with patch("virtual_assistant_be.services.memory_service.requests.post"):
            service.store_person_event("appeared")
            service.store_person_event("disappeared")
        assert service.get_person_count() == 1

    def test_store_interaction(self, service, mock_embed):
        with patch("virtual_assistant_be.services.memory_service.requests.post") as mock_post:
            service.store_interaction("hello", "hi there!")
        mock_post.assert_called_once()

    def test_store_person_event_calls_chromadb(self, service, mock_embed):
        with patch("virtual_assistant_be.services.memory_service.requests.post") as mock_post:
            service.store_person_event("appeared")
        assert mock_post.call_count >= 1

    @patch("virtual_assistant_be.services.memory_service.requests.post")
    def test_search_no_collection_returns_empty(self, mock_post, service):
        service._collection_id = None
        with patch.object(service, "_ensure_collection", return_value=""):
            result = service.search("test query")
        assert result == []

    def test_get_recent_events_no_collection(self, service):
        service._collection_id = None
        with patch.object(service, "_ensure_collection", return_value=""):
            result = service.get_recent_events()
        assert result == []

    def test_person_count_persists_across_instances(self):
        svc1 = MemoryService()
        # Override counter file to a temp path
        tmp = os.path.join(tempfile.gettempdir(), "test_persist_counter.json")
        svc1._counter_file = tmp
        svc1._person_count = 0
        svc1._persist_person_count()

        svc2 = MemoryService()
        svc2._counter_file = tmp
        loaded = svc2._load_person_count()
        assert loaded == 0

    def test_persist_and_load_roundtrip(self):
        tmp = os.path.join(tempfile.gettempdir(), "test_roundtrip_counter.json")
        # Clean up
        for f in [tmp]:
            if os.path.exists(f):
                os.remove(f)

        svc = MemoryService()
        svc._counter_file = tmp
        svc._person_count = 42
        svc._persist_person_count()

        loaded = svc._load_person_count()
        assert loaded == 42
        os.remove(tmp)

    def test_load_missing_file_returns_zero(self):
        tmp = os.path.join(tempfile.gettempdir(), "test_nonexistent_counter.json")
        if os.path.exists(tmp):
            os.remove(tmp)
        svc = MemoryService()
        svc._counter_file = tmp
        assert svc._load_person_count() == 0

    def test_disappeared_stores_duration(self, service, mock_embed):
        with patch("virtual_assistant_be.services.memory_service.requests.post") as mock_post:
            service.store_person_event("appeared")
            service.store_person_event("disappeared")
        # Second call (disappeared) should include duration_s in metadata
        call_args = mock_post.call_args_list[1]
        meta = call_args[1]["json"]["metadatas"][0]
        assert meta["event"] == "disappeared"
        assert "duration_s" in meta
        assert meta["duration_s"] >= 0
