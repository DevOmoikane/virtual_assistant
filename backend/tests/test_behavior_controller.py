from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from virtual_assistant_be.core.behavior_controller import BehaviorController
from virtual_assistant_be.core.config import settings
from virtual_assistant_be.core.protocol import GoCommand, GoEvent
from virtual_assistant_be.core.translations import translate as t


@pytest.fixture
def controller():
    send_fn = AsyncMock()
    ctrl = BehaviorController(send_fn=send_fn)
    return ctrl


class TestBehaviorController:
    @pytest.mark.asyncio
    async def test_handle_command_ready_sends_greeting(self, controller):
        with (
            patch.object(controller, "send_animation") as mock_anim,
            patch.object(controller, "_send_speak") as mock_speak,
            patch.object(controller.tts, "speak") as mock_tts,
            patch.object(controller.camera, "start") as mock_cam,
            patch.object(controller.audio, "start") as mock_audio,
        ):
            await controller.handle_command(GoCommand(name="ready"))

        assert mock_anim.await_count >= 1
        mock_cam.assert_called_once()
        mock_audio.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_command_shutdown_stops_services(self, controller):
        with (
            patch.object(controller.camera, "stop") as mock_cam,
            patch.object(controller.audio, "stop") as mock_audio,
        ):
            await controller.handle_command(GoCommand(name="shutdown"))

        mock_cam.assert_called_once()
        mock_audio.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_command_unknown_logged(self, controller):
        with patch("virtual_assistant_be.core.behavior_controller.log.warning") as mock_log:
            await controller.handle_command(GoCommand(name="unknown_command"))
            mock_log.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_text_calls_pipeline(self, controller):
        with (
            patch.object(controller, "_send_think") as mock_think,
            patch.object(controller, "_send_speak") as mock_speak,
            patch.object(controller, "send_animation") as mock_anim,
            patch.object(controller.llm, "classify_intent", return_value="question"),
            patch.object(controller.llm, "classify_device_command", return_value=None),
            patch.object(controller.rag, "retrieve", return_value=["doc1"]),
            patch.object(controller.llm, "generate_response", return_value=("The answer is 42.", "question")),
            patch.object(controller.tts, "speak"),
            patch.object(controller.llm, "decide_animation", return_value="think"),
        ):
            await controller.handle_text("What is the answer?")

        assert mock_think.await_count == 2
        assert mock_speak.await_count >= 1
        assert mock_anim.await_count >= 1

    @pytest.mark.asyncio
    async def test_handle_text_question_uses_rag_by_default(self, controller):
        with (
            patch.object(controller, "_send_think"),
            patch.object(controller, "_send_speak"),
            patch.object(controller, "send_animation"),
            patch.object(controller.llm, "classify_intent", return_value="question"),
            patch.object(controller.llm, "classify_device_command", return_value=None),
            patch.object(controller.rag, "retrieve", return_value=["doc1"]) as mock_retrieve,
            patch.object(controller.llm, "generate_response", return_value=("42.", "question")),
            patch.object(controller.tts, "speak"),
            patch.object(controller.llm, "decide_animation", return_value="think"),
        ):
            await controller.handle_text("What is the answer?")
        mock_retrieve.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_text_question_skips_rag_when_disabled(self, controller):
        original = settings.rag_enabled
        settings.rag_enabled = False
        try:
            with (
                patch.object(controller, "_send_think"),
                patch.object(controller, "_send_speak"),
                patch.object(controller, "send_animation"),
                patch.object(controller.llm, "classify_intent", return_value="question"),
                patch.object(controller.llm, "classify_device_command", return_value=None),
                patch.object(controller.rag, "retrieve") as mock_retrieve,
                patch.object(controller.llm, "generate_response", return_value=("42.", "question")),
                patch.object(controller.tts, "speak"),
                patch.object(controller.llm, "decide_animation", return_value="think"),
            ):
                await controller.handle_text("What is the answer?")
            mock_retrieve.assert_not_called()
        finally:
            settings.rag_enabled = original

    @pytest.mark.asyncio
    async def test_handle_text_error_does_not_crash(self, controller):
        with (
            patch.object(controller, "_send_think") as mock_think,
            patch.object(controller.llm, "classify_intent", side_effect=Exception("test error")),
        ):
            await controller.handle_text("test")
            assert mock_think.await_count == 2

    @pytest.mark.asyncio
    async def test_on_person_appeared_known(self, controller):
        controller._current_language = "en"
        with (
            patch.object(controller, "send_animation") as mock_anim,
            patch.object(controller, "_send_speak") as mock_speak,
            patch.object(controller.tts, "speak"),
            patch.object(controller.memory, "store_person_event"),
        ):
            await controller._on_person_appeared(data={"name": "Alice"})
            mock_anim.assert_called_once_with("greet")
            mock_speak.assert_awaited_once_with(t("hello_name", "en", name="Alice"))

    @pytest.mark.asyncio
    async def test_on_person_appeared_unknown(self, controller):
        controller._current_language = "en"
        with (
            patch.object(controller, "send_animation") as mock_anim,
            patch.object(controller, "_send_speak") as mock_speak,
            patch.object(controller, "_send_listen") as mock_listen,
            patch.object(controller.tts, "speak"),
            patch.object(controller.memory, "store_person_event"),
        ):
            await controller._on_person_appeared(data={})
            mock_anim.assert_called_once_with("greet")
            mock_speak.assert_awaited_once_with(t("hello_unknown", "en"))
            mock_listen.assert_awaited_once_with(True)
            assert controller._pending_name is True

    @pytest.mark.asyncio
    async def test_register_name_saves_and_greets(self, controller):
        controller._current_language = "en"
        with (
            patch.object(controller, "_send_speak") as mock_speak,
            patch.object(controller, "_send_listen") as mock_listen,
            patch.object(controller.tts, "speak"),
            patch.object(controller.face_service, "register", return_value=True),
        ):
            controller._pending_name = True
            controller.face_service.last_unknown_embedding = [0.1, 0.2, 0.3]
            ok = await controller._register_name("bob")
            assert ok is True
            mock_speak.assert_awaited_once_with(t("register_success", "en", name="Bob"))
            mock_listen.assert_awaited_once_with(False)

    @pytest.mark.asyncio
    async def test_register_name_skipped_when_not_pending(self, controller):
        controller._pending_name = None
        ok = await controller._register_name("bob")
        assert ok is False

    @pytest.mark.asyncio
    async def test_on_gesture_wave(self, controller):
        with (
            patch.object(controller, "send_animation") as mock_anim,
            patch.object(controller.tts, "speak"),
        ):
            await controller._on_gesture({"gesture": "wave"})
            mock_anim.assert_called_once_with("greet")

    @pytest.mark.asyncio
    async def test_on_gesture_thumbs_up(self, controller):
        with (
            patch.object(controller, "send_animation") as mock_anim,
            patch.object(controller.tts, "speak"),
        ):
            await controller._on_gesture({"gesture": "thumbs_up"})
            mock_anim.assert_called_once_with("nod")

    @pytest.mark.asyncio
    async def test_event_is_logged(self, controller):
        with patch("virtual_assistant_be.core.behavior_controller.log.info") as mock_log:
            await controller.handle_event(GoEvent(name="animation_finished", params={"name": "greet"}))
            mock_log.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_animation_does_nothing_without_send_fn(self):
        ctrl = BehaviorController(send_fn=None)
        await ctrl.send_animation("greet")

    @pytest.mark.asyncio
    async def test_handle_text_does_not_crash_with_empty_input(self, controller):
        with (
            patch.object(controller, "_send_think"),
            patch.object(controller, "_send_speak"),
            patch.object(controller, "send_animation"),
            patch.object(controller.llm, "classify_intent", return_value=""),
            patch.object(controller.llm, "classify_device_command", return_value=None),
            patch.object(controller.llm, "generate_response", return_value=("", "other")),
            patch.object(controller.tts, "speak"),
        ):
            await controller.handle_text("")
