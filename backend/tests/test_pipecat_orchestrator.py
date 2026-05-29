from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from virtual_assistant_be.core.protocol import GoCommand, GoEvent
from virtual_assistant_be.pipecat.orchestrator import PipecatOrchestrator


@pytest.fixture
def orchestrator():
    orch = PipecatOrchestrator(send_fn=None)
    return orch


@pytest.mark.asyncio
class TestPipecatOrchestrator:
    async def test_set_send_fn_updates(self, orchestrator):
        fn = MagicMock()
        orchestrator.set_send_fn(fn)
        assert orchestrator._send_fn is fn

    async def test_send_msg_sync_fn(self, orchestrator):
        fn = MagicMock()
        orchestrator.set_send_fn(fn)
        await orchestrator._send_msg({"type": "test"})
        fn.assert_called_once_with({"type": "test"})

    async def test_send_msg_async_fn(self, orchestrator):
        fn = AsyncMock()
        orchestrator.set_send_fn(fn)
        await orchestrator._send_msg({"type": "test"})
        fn.assert_awaited_once_with({"type": "test"})

    async def test_send_msg_no_fn(self, orchestrator):
        await orchestrator._send_msg({"type": "test"})

    async def test_say_returns_translated_text(self, orchestrator):
        orchestrator._current_language = "en"
        orchestrator.personality._enabled = False
        msg = await orchestrator._say("hello_name", name="Alice")
        assert msg == "Hello Alice!"

    async def test_say_spanish(self, orchestrator):
        orchestrator._current_language = "es"
        orchestrator.personality._enabled = False
        msg = await orchestrator._say("hello_name", name="Alice")
        assert msg == "¡Hola Alice!"

    async def test_say_personalizes_when_enabled(self, orchestrator):
        orchestrator._current_language = "en"
        orchestrator.personality._enabled = True
        with patch.object(orchestrator.personality, "personalize", return_value="Hey Alice!"):
            msg = await orchestrator._say("hello_name", name="Alice")
            assert msg == "Hey Alice!"

    async def test_handle_command_ready_starts_services(self, orchestrator):
        with (
            patch.object(orchestrator, "_send_state") as mock_state,
            patch.object(orchestrator.camera, "start") as mock_cam,
            patch.object(orchestrator.telegram, "start_polling"),
        ):
            await orchestrator.handle_command(GoCommand(name="ready"))
            mock_state.assert_awaited_once_with(connected=True)
            mock_cam.assert_called_once()

    async def test_handle_command_shutdown_stops_services(self, orchestrator):
        with (
            patch.object(orchestrator, "_cleanup") as mock_cleanup,
            patch.object(orchestrator, "_send_state") as mock_state,
        ):
            await orchestrator.handle_command(GoCommand(name="shutdown"))
            mock_cleanup.assert_awaited_once()
            mock_state.assert_awaited_once_with(connected=False)

    async def test_handle_event_text_injects(self, orchestrator):
        orchestrator._pipeline_task = MagicMock()
        orchestrator._context = MagicMock()
        with patch.object(orchestrator, "_inject_user_text") as mock_inject:
            await orchestrator.handle_event(GoEvent(name="text", params={"text": "hello"}))
            mock_inject.assert_awaited_once_with("hello", orchestrator._current_language)

    async def test_handle_event_empty_text_ignored(self, orchestrator):
        with patch.object(orchestrator, "_inject_user_text") as mock_inject:
            await orchestrator.handle_event(GoEvent(name="text", params={"text": ""}))
            mock_inject.assert_not_called()

    async def test_on_person_appeared_known_sends_greeting(self, orchestrator):
        orchestrator._current_language = "en"
        orchestrator.personality._enabled = False
        with (
            patch.object(orchestrator.memory, "store_person_event"),
            patch.object(orchestrator, "_say", return_value="Hello Alice!") as mock_say,
            patch.object(orchestrator, "_send_animation") as mock_anim,
            patch.object(orchestrator, "_speak_text") as mock_speak,
        ):
            await orchestrator._on_person_appeared(data={"name": "Alice"})
            mock_say.assert_awaited_once_with("hello_name", name="Alice")
            mock_anim.assert_awaited_once_with("greet")
            mock_speak.assert_awaited_once_with("Hello Alice!")

    async def test_on_person_appeared_unknown_sets_pending(self, orchestrator):
        orchestrator._current_language = "en"
        orchestrator.personality._enabled = False
        orchestrator._pending_name_processor = MagicMock()
        with (
            patch.object(orchestrator.memory, "store_person_event"),
            patch.object(orchestrator, "_say", return_value="Hello!"),
            patch.object(orchestrator, "_send_animation"),
            patch.object(orchestrator, "_speak_text"),
        ):
            await orchestrator._on_person_appeared(data={})
            assert orchestrator._pending_name is True
            assert orchestrator._pending_name_processor.pending is True

    async def test_on_person_appeared_rate_limited(self, orchestrator):
        orchestrator._last_person_greeted = 999999.0
        with (
            patch.object(orchestrator.memory, "store_person_event") as mock_store,
        ):
            await orchestrator._on_person_appeared(data={"name": "Alice"})
            mock_store.assert_not_called()

    async def test_on_person_disappeared_resets(self, orchestrator):
        orchestrator._current_person_name = "Alice"
        orchestrator.personality.set_style("cheerful")
        with (
            patch.object(orchestrator.memory, "store_person_event"),
        ):
            await orchestrator._on_person_disappeared()
            assert orchestrator._current_person_name is None
            assert orchestrator.personality.style != "cheerful"

    async def test_inject_user_text_adds_context_and_triggers_llm(self, orchestrator):
        orchestrator._pipeline_task = AsyncMock()
        orchestrator._context = MagicMock()
        await orchestrator._inject_user_text("hello")
        orchestrator._context.add_message.assert_called_once_with({"role": "user", "content": "hello"})
        orchestrator._pipeline_task.queue_frames.assert_awaited_once()

    async def test_register_name_saves_and_speaks(self, orchestrator):
        orchestrator._pending_name = True
        orchestrator._pending_name_processor = MagicMock()
        orchestrator.face_service._ready = True
        orchestrator.face_service.last_unknown_embedding = [0.1, 0.2]
        orchestrator._current_language = "en"
        orchestrator.personality._enabled = False
        with (
            patch.object(orchestrator.face_service, "register", return_value=True),
            patch.object(orchestrator, "_speak_text") as mock_speak,
            patch.object(orchestrator, "_send_listen") as mock_listen,
        ):
            ok = await orchestrator._register_name("bob")
            assert ok is True
            mock_speak.assert_awaited_once()
            mock_listen.assert_awaited_once_with(False)
            assert orchestrator._pending_name is False
            assert orchestrator._pending_name_processor.pending is False

    async def test_register_name_skipped_when_not_pending(self, orchestrator):
        orchestrator._pending_name = False
        ok = await orchestrator._register_name("bob")
        assert ok is False

    async def test_on_gesture_wave(self, orchestrator):
        orchestrator._pipeline_task = AsyncMock()
        with (
            patch.object(orchestrator, "_send_animation") as mock_anim,
            patch.object(orchestrator, "_speak_text") as mock_speak,
        ):
            await orchestrator._on_gesture({"gesture": "wave"})
            mock_anim.assert_awaited_once_with("greet")
            mock_speak.assert_awaited_once()

    async def test_on_gesture_ignored_when_recent(self, orchestrator):
        orchestrator._last_gesture_time = 999999.0
        orchestrator._pipeline_task = MagicMock()
        with patch.object(orchestrator._pipeline_task, "queue_frames") as mock_q:
            await orchestrator._on_gesture({"gesture": "wave"})
            mock_q.assert_not_called()

    async def test_cleanup_stops_camera_telegram(self, orchestrator):
        with (
            patch.object(orchestrator.camera, "stop") as mock_cam,
            patch.object(orchestrator.telegram, "stop_polling"),
        ):
            await orchestrator._cleanup()
            mock_cam.assert_called_once()

    async def test_speak_text_queues_tts_frame(self, orchestrator):
        orchestrator._pipeline_task = AsyncMock()
        await orchestrator._speak_text("Hello")
        orchestrator._pipeline_task.queue_frames.assert_awaited_once()
        frames = orchestrator._pipeline_task.queue_frames.call_args[0][0]
        assert len(frames) == 1
        assert frames[0].text == "Hello"

    async def test_speak_text_no_task_does_not_crash(self, orchestrator):
        orchestrator._pipeline_task = None
        await orchestrator._speak_text("Hello")
