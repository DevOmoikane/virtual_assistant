from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_requests():
    with patch("virtual_assistant_be.services.llm_service.requests") as mock:
        yield mock


@pytest.fixture
def mock_requests_rag():
    with patch("virtual_assistant_be.services.rag_service.requests") as mock:
        yield mock


@pytest.fixture
def mock_send_fn():
    return AsyncMock()
