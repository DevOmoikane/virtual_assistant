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
    # Reset to known state
    svc._person_count = 0
    svc._visits = []
    svc._current_visit_id = None
    svc._person_appeared_at = None
    return svc


@pytest.fixture
def mock_embed():
    with patch.object(MemoryService, "_embed", return_value=[0.1, 0.2, 0.3]) as m:
        yield m


class TestMemoryService:
    def test_person_count_starts_at_zero(self):
        tmp = os.path.join(tempfile.gettempdir(), "test_zero_counter.json")
        if os.path.exists(tmp):
            os.remove(tmp)
        svc = MemoryService()
        svc._counter_file = tmp
        svc._person_count = svc._load_person_count()
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
        if os.path.exists(tmp):
            os.remove(tmp)

        svc = MemoryService()
        svc._counter_file = tmp
        svc._person_count = 42
        svc._persist_person_count()

        loaded = svc._load_person_count()
        assert loaded == 42
        os.remove(tmp)

    def test_load_missing_file_returns_zero(self):
        svc = MemoryService()
        tmp = os.path.join(tempfile.gettempdir(), "test_nonexistent_counter_" + str(id(svc)) + ".json")
        if os.path.exists(tmp):
            os.remove(tmp)
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

    # --- Visit tracking tests ---

    def test_appeared_starts_visit(self, service, mock_embed):
        with patch("virtual_assistant_be.services.memory_service.requests.post"):
            service.store_person_event("appeared")
        assert len(service._visits) == 1
        v = service._visits[0]
        assert v["visit_id"] is not None
        assert v["end_time"] is None
        assert v["duration_s"] is None
        assert v["time_of_day"] is not None
        assert v["date"] is not None
        assert service._current_visit_id == v["visit_id"]

    def test_disappeared_ends_visit(self, service, mock_embed):
        with patch("virtual_assistant_be.services.memory_service.requests.post"):
            service.store_person_event("appeared")
            service.store_person_event("disappeared")
        assert len(service._visits) == 1
        v = service._visits[0]
        assert v["end_time"] is not None
        assert v["duration_s"] is not None
        assert v["duration_s"] >= 0
        assert service._current_visit_id is None

    def test_multiple_visits(self, service, mock_embed):
        with patch("virtual_assistant_be.services.memory_service.requests.post"):
            for _ in range(3):
                service.store_person_event("appeared")
                service.store_person_event("disappeared")
        assert len(service._visits) == 3
        for v in service._visits:
            assert v["duration_s"] is not None

    def test_get_visit_stats_empty(self, service):
        stats = service.get_visit_stats()
        assert stats["total_visits"] == 0
        assert stats["completed_visits"] == 0
        assert stats["active_visit"] is False
        assert stats["today_visits"] == 0
        assert stats["visits_by_hour"] == {}

    def test_get_visit_stats_with_visits(self, service, mock_embed):
        with patch("virtual_assistant_be.services.memory_service.requests.post"):
            service.store_person_event("appeared")
            service.store_person_event("disappeared")
        stats = service.get_visit_stats()
        assert stats["total_visits"] == 1
        assert stats["completed_visits"] == 1
        assert stats["active_visit"] is False
        assert stats["today_visits"] == 1
        assert stats["average_duration_s"] >= 0
        assert len(stats["visits_by_hour"]) == 1

    def test_get_visit_stats_active_visit(self, service, mock_embed):
        with patch("virtual_assistant_be.services.memory_service.requests.post"):
            service.store_person_event("appeared")
        stats = service.get_visit_stats()
        assert stats["total_visits"] == 1
        assert stats["completed_visits"] == 0
        assert stats["active_visit"] is True

    def test_get_recent_visits_order(self, service, mock_embed):
        with patch("virtual_assistant_be.services.memory_service.requests.post"):
            service._start_visit(100.0)
            service._start_visit(200.0)
            service._start_visit(300.0)
        recent = service.get_recent_visits(2)
        assert len(recent) == 2
        assert recent[0]["start_time"] == 300.0
        assert recent[1]["start_time"] == 200.0

    def test_visits_persist_and_load(self, service):
        tmp = os.path.join(tempfile.gettempdir(), "test_visits_log.json")
        if os.path.exists(tmp):
            os.remove(tmp)

        service._visits_file = tmp
        service._start_visit(1000.0)

        svc2 = MemoryService()
        svc2._visits_file = tmp
        loaded = svc2._load_visits()

        assert len(loaded) == 1
        assert loaded[0]["visit_id"] == service._visits[0]["visit_id"]
        os.remove(tmp)

    def test_load_visits_missing_file(self, service):
        service._visits_file = os.path.join(
            tempfile.gettempdir(), "test_nonexistent_visits.json"
        )
        if os.path.exists(service._visits_file):
            os.remove(service._visits_file)
        assert service._load_visits() == []
