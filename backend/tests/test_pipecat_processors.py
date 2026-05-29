from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pipecat.frames.frames import (
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TextFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from virtual_assistant_be.pipecat.custom_frames import GestureFrame
from virtual_assistant_be.pipecat.processors import (
    GestureProcessor,
    MemoryProcessor,
    PersonalityProcessor,
    RAGProcessor,
)


class TestRAGProcessor:
    @pytest.mark.asyncio
    async def test_no_retrieve_fn_passes_through(self):
        proc = RAGProcessor(retrieve_fn=None)
        proc._started = True
        ctx = {"messages": [{"role": "user", "content": "hello"}]}
        frame = LLMContextFrame(context=ctx)
        push = AsyncMock()
        proc.push_frame = push
        await proc.process_frame(frame, FrameDirection.DOWNSTREAM)
        push.assert_awaited_once_with(frame, FrameDirection.DOWNSTREAM)

    @pytest.mark.asyncio
    async def test_injects_context_into_single_user_message(self):
        retrieve = MagicMock(return_value=["doc1", "doc2"])
        proc = RAGProcessor(retrieve_fn=retrieve)
        proc._started = True
        ctx = {"messages": [{"role": "user", "content": "what is X?"}]}
        frame = LLMContextFrame(context=ctx)
        push = AsyncMock()
        proc.push_frame = push
        await proc.process_frame(frame, FrameDirection.DOWNSTREAM)
        assert len(ctx["messages"]) == 2
        assert ctx["messages"][0]["role"] == "system"
        assert "doc1" in ctx["messages"][0]["content"]
        assert "doc2" in ctx["messages"][0]["content"]
        push.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_injection_without_user_message(self):
        retrieve = MagicMock(return_value=["doc"])
        proc = RAGProcessor(retrieve_fn=retrieve)
        proc._started = True
        ctx = {"messages": []}
        frame = LLMContextFrame(context=ctx)
        push = AsyncMock()
        proc.push_frame = push
        await proc.process_frame(frame, FrameDirection.DOWNSTREAM)
        assert len(ctx["messages"]) == 0

    @pytest.mark.asyncio
    async def test_ignores_non_llmcontext_frame(self):
        retrieve = MagicMock()
        proc = RAGProcessor(retrieve_fn=retrieve)
        proc._started = True
        frame = LLMTextFrame(text="test")
        push = AsyncMock()
        proc.push_frame = push
        await proc.process_frame(frame, FrameDirection.DOWNSTREAM)
        retrieve.assert_not_called()
        push.assert_awaited_once()


class TestPersonalityProcessor:
    @pytest.mark.asyncio
    async def test_disabled_passes_through(self):
        personalize = MagicMock()
        proc = PersonalityProcessor(personalize_fn=personalize, enabled=False)
        proc._started = True
        frame = LLMTextFrame(text="Hello")
        push = AsyncMock()
        proc.push_frame = push
        await proc.process_frame(frame, FrameDirection.DOWNSTREAM)
        personalize.assert_not_called()
        push.assert_awaited_once_with(frame, FrameDirection.DOWNSTREAM)

    @pytest.mark.asyncio
    async def test_enabled_personalizes_llm_text(self):
        personalize = MagicMock(return_value="Hey there!")
        proc = PersonalityProcessor(personalize_fn=personalize, enabled=True)
        proc._started = True
        frame = LLMTextFrame(text="Hello")
        push = AsyncMock()
        proc.push_frame = push
        await proc.process_frame(frame, FrameDirection.DOWNSTREAM)
        personalize.assert_called_once_with("Hello", "en")
        push.assert_awaited_once()
        pushed = push.call_args[0][0]
        assert pushed.text == "Hey there!"

    @pytest.mark.asyncio
    async def test_ignores_non_llm_text(self):
        personalize = MagicMock()
        proc = PersonalityProcessor(personalize_fn=personalize, enabled=True)
        frame = LLMFullResponseStartFrame()
        push = AsyncMock()
        proc.push_frame = push
        await proc.process_frame(frame, FrameDirection.DOWNSTREAM)
        personalize.assert_not_called()


class TestMemoryProcessor:
    @pytest.mark.asyncio
    async def test_stores_interaction_on_end_frame(self):
        store = MagicMock()
        proc = MemoryProcessor(store_fn=store)
        push = AsyncMock()
        proc.push_frame = push
        await proc.process_frame(TextFrame(text="user query"), FrameDirection.DOWNSTREAM)
        await proc.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
        store.assert_not_called()

    @pytest.mark.asyncio
    async def test_ignores_frames_without_store_fn(self):
        proc = MemoryProcessor(store_fn=None)
        push = AsyncMock()
        proc.push_frame = push
        await proc.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
        push.assert_awaited_once()


class TestGestureProcessor:
    @pytest.mark.asyncio
    async def test_ignores_non_llm_text(self):
        personalize = MagicMock()
        proc = PersonalityProcessor(personalize_fn=personalize, enabled=True)
        proc._started = True
        frame = LLMFullResponseStartFrame()
        push = AsyncMock()
        proc.push_frame = push
        await proc.process_frame(frame, FrameDirection.DOWNSTREAM)
        personalize.assert_not_called()


class TestMemoryProcessor:
    @pytest.mark.asyncio
    async def test_stores_interaction_on_end_frame(self):
        store = MagicMock()
        proc = MemoryProcessor(store_fn=store)
        proc._started = True
        push = AsyncMock()
        proc.push_frame = push

        await proc.process_frame(TextFrame(text="user query"), FrameDirection.DOWNSTREAM)
        await proc.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
        store.assert_not_called()

    @pytest.mark.asyncio
    async def test_ignores_frames_without_store_fn(self):
        proc = MemoryProcessor(store_fn=None)
        proc._started = True
        push = AsyncMock()
        proc.push_frame = push
        await proc.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
        push.assert_awaited_once()


class TestGestureProcessor:
    @pytest.mark.asyncio
    async def test_open_palm_sends_interruption(self):
        proc = GestureProcessor()
        proc._started = True
        push = AsyncMock()
        proc.push_frame = push
        frame = GestureFrame(gesture="open_palm")
        await proc.process_frame(frame, FrameDirection.DOWNSTREAM)
        assert push.await_count == 2
        assert isinstance(push.await_args_list[0].args[0], InterruptionFrame)
        assert push.await_args_list[0].args[1] == FrameDirection.UPSTREAM
        assert push.await_args_list[1].args[0] is frame
        assert push.await_args_list[1].args[1] == FrameDirection.DOWNSTREAM

    @pytest.mark.asyncio
    async def test_non_gesture_frame_passes_through(self):
        proc = GestureProcessor()
        proc._started = True
        push = AsyncMock()
        proc.push_frame = push
        frame = LLMTextFrame(text="hi")
        await proc.process_frame(frame, FrameDirection.DOWNSTREAM)
        push.assert_awaited_once_with(frame, FrameDirection.DOWNSTREAM)

    @pytest.mark.asyncio
    async def test_other_gesture_passes_through(self):
        proc = GestureProcessor()
        proc._started = True
        push = AsyncMock()
        proc.push_frame = push
        frame = LLMTextFrame(text="hi")
        await proc.process_frame(frame, FrameDirection.DOWNSTREAM)
        push.assert_awaited_once_with(frame, FrameDirection.DOWNSTREAM)

    @pytest.mark.asyncio
    async def test_other_gesture_passes_through(self):
        proc = GestureProcessor()
        push = AsyncMock()
        proc.push_frame = push
        frame = GestureFrame(gesture="wave")
        await proc.process_frame(frame, FrameDirection.DOWNSTREAM)
        push.assert_awaited_once_with(frame, FrameDirection.DOWNSTREAM)
