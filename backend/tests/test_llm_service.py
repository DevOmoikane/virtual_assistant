from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from virtual_assistant_be.services.llm_service import LlmService


@pytest.fixture
def service():
    return LlmService()


class TestLlmService:
    def test_generate_calls_ollama(self, service):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "Hello there!"}}
        mock_resp.raise_for_status.return_value = None

        with patch("virtual_assistant_be.services.llm_service.requests.post", return_value=mock_resp) as mock_post:
            result = service.generate("Hi")

            assert result == "Hello there!"
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args[1]
            assert call_kwargs["json"]["model"] == "llama3.1"
            assert call_kwargs["json"]["messages"] == [
                {"role": "user", "content": "Hi"},
            ]
            assert call_kwargs["json"]["stream"] is False

    def test_generate_with_context(self, service):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "The answer is 42."}}
        mock_resp.raise_for_status.return_value = None

        with patch("virtual_assistant_be.services.llm_service.requests.post", return_value=mock_resp):
            result = service.generate("What is the answer?", context="The meaning of life is 42.")

            assert result == "The answer is 42."

    def test_generate_with_system(self, service):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "greeting"}}
        mock_resp.raise_for_status.return_value = None

        with patch("virtual_assistant_be.services.llm_service.requests.post", return_value=mock_resp) as mock_post:
            result = service.generate("hello", system="You classify input.")

            assert result == "greeting"
            assert mock_post.call_args[1]["json"]["messages"][0]["role"] == "system"

    def test_generate_returns_empty_on_error(self, service):
        import requests as req_lib
        with patch("virtual_assistant_be.services.llm_service.requests.post", side_effect=req_lib.exceptions.ConnectionError("connection error")):
            result = service.generate("Hi")
            assert result == ""

    def test_classify_intent_greeting(self, service):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "greeting"}}
        mock_resp.raise_for_status.return_value = None

        with patch("virtual_assistant_be.services.llm_service.requests.post", return_value=mock_resp):
            result = service.classify_intent("Hello!")
            assert result == "greeting"

    def test_classify_intent_question(self, service):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "question"}}
        mock_resp.raise_for_status.return_value = None

        with patch("virtual_assistant_be.services.llm_service.requests.post", return_value=mock_resp):
            result = service.classify_intent("What time is it?")
            assert result == "question"

    def test_decide_animation_greeting(self, service):
        assert service.decide_animation("", intent="greeting") == "greet"

    def test_decide_animation_question(self, service):
        assert service.decide_animation("", intent="question") == "think"

    def test_decide_animation_goodbye(self, service):
        assert service.decide_animation("", intent="goodbye") == "greet"

    def test_decide_animation_default(self, service):
        assert service.decide_animation("", intent="other") == "listen"

    def test_decide_animation_without_intent(self, service):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "greeting"}}
        mock_resp.raise_for_status.return_value = None

        with patch("virtual_assistant_be.services.llm_service.requests.post", return_value=mock_resp):
            anim = service.decide_animation("Hi!")
            assert anim == "greet"

    def test_generate_response_returns_tuple(self, service):
        mock_resp = MagicMock()
        mock_resp.json.side_effect = [
            {"message": {"content": "question"}},
            {"message": {"content": "I don't know."}},
        ]
        mock_resp.raise_for_status.return_value = None

        with patch("virtual_assistant_be.services.llm_service.requests.post", return_value=mock_resp):
            response, intent = service.generate_response("What is this?")
            assert intent == "question"
            assert response == "I don't know."
