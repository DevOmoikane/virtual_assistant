from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pipecat.frames.frames import (
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from virtual_assistant_be.pipecat.custom_frames import (
    GestureFrame,
    PersonAppearedFrame,
    PersonDisappearedFrame,
)
from virtual_assistant_be.pipecat.godot_bridge_processor import GodotBridgeProcessor


@pytest.fixture
def bridge():
    send_fn = MagicMock()
    bp = GodotBridgeProcessor(send_fn=send_fn)
    bp._started = True
    return bp


@pytest.mark.asyncio
class TestGodotBridgeProcessor:
    async def test_transcription_frame_sends_heard(self, bridge):
        frame = TranscriptionFrame(text="hello", user_id="user", timestamp="0")
        await bridge.process_frame(frame, FrameDirection.DOWNSTREAM)
        bridge._send_fn.assert_called_once()
        data = bridge._send_fn.call_args[0][0]
        assert data["type"] == "heard"
        assert data["text"] == "hello"

    async def test_llm_text_frame_sends_speak(self, bridge):
        frame = LLMTextFrame(text="Hello world")
        await bridge.process_frame(frame, FrameDirection.DOWNSTREAM)
        bridge._send_fn.assert_called_once()
        data = bridge._send_fn.call_args[0][0]
        assert data["type"] == "speak"
        assert data["text"] == "Hello world"

    async def test_llm_start_sends_think_true(self, bridge):
        frame = LLMFullResponseStartFrame()
        await bridge.process_frame(frame, FrameDirection.DOWNSTREAM)
        bridge._send_fn.assert_called_once()
        data = bridge._send_fn.call_args[0][0]
        assert data["type"] == "think"
        assert data["active"] is True

    async def test_llm_end_sends_think_false(self, bridge):
        frame = LLMFullResponseEndFrame()
        await bridge.process_frame(frame, FrameDirection.DOWNSTREAM)
        bridge._send_fn.assert_called_once()
        data = bridge._send_fn.call_args[0][0]
        assert data["type"] == "think"
        assert data["active"] is False

    async def test_tts_speak_frame_sends_speak(self, bridge):
        frame = TTSSpeakFrame(text="Hello via TTS")
        await bridge.process_frame(frame, FrameDirection.DOWNSTREAM)
        bridge._send_fn.assert_called_once()
        data = bridge._send_fn.call_args[0][0]
        assert data["type"] == "speak"
        assert data["text"] == "Hello via TTS"

    async def test_person_appeared_with_name(self, bridge):
        frame = PersonAppearedFrame(person_name="Alice")
        await bridge.process_frame(frame, FrameDirection.DOWNSTREAM)
        calls = bridge._send_fn.call_args_list
        assert len(calls) == 2
        assert calls[0][0][0]["type"] == "animation"
        assert calls[0][0][0]["name"] == "greet"
        assert calls[1][0][0]["type"] == "state"
        assert calls[1][0][0]["person"] == "Alice"

    async def test_person_appeared_without_name(self, bridge):
        frame = PersonAppearedFrame(person_name=None)
        await bridge.process_frame(frame, FrameDirection.DOWNSTREAM)
        calls = bridge._send_fn.call_args_list
        assert len(calls) == 2
        assert calls[1][0][0].get("person") is None

    async def test_person_disappeared_sends_idle(self, bridge):
        frame = PersonDisappearedFrame()
        await bridge.process_frame(frame, FrameDirection.DOWNSTREAM)
        calls = bridge._send_fn.call_args_list
        assert len(calls) == 2
        assert calls[0][0][0]["type"] == "animation"
        assert calls[0][0][0]["name"] == "idle"

    async def test_gesture_frame_sends_animation(self, bridge):
        frame = GestureFrame(gesture="wave", x=0.5, y=0.3)
        await bridge.process_frame(frame, FrameDirection.DOWNSTREAM)
        calls = bridge._send_fn.call_args_list
        assert len(calls) == 1
        assert calls[0][0][0]["type"] == "animation"
        assert calls[0][0][0]["name"] == "wave"
        assert calls[0][0][0]["params"]["x"] == 0.5

    async def test_unknown_frame_passed_through(self, bridge):
        frame = LLMFullResponseEndFrame()
        push_mock = AsyncMock()
        bridge.push_frame = push_mock
        await bridge.process_frame(frame, FrameDirection.DOWNSTREAM)
        push_mock.assert_awaited_once_with(frame, FrameDirection.DOWNSTREAM)

    async def test_set_send_fn_updates(self, bridge):
        old_fn = bridge._send_fn
        old_fn.reset_mock()
        new_fn = MagicMock()
        bridge.set_send_fn(new_fn)
        frame = TTSSpeakFrame(text="test")
        await bridge.process_frame(frame, FrameDirection.DOWNSTREAM)
        new_fn.assert_called_once()
        old_fn.assert_not_called()
