import asyncio
import logging
import time
from typing import Callable

from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    LLMContextFrame,
    LLMTextFrame,
    LLMFullResponseEndFrame,
    TTSStoppedFrame,
    TranscriptionFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from virtual_assistant_be.pipecat.custom_frames import GestureFrame

log = logging.getLogger(__name__)


class RAGProcessor(FrameProcessor):
    def __init__(self, retrieve_fn: Callable[[str], list[str]] | None = None, **kwargs):
        super().__init__(**kwargs)
        self._retrieve_fn = retrieve_fn

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame) and self._retrieve_fn:
            context = frame.context
            if hasattr(context, "messages"):
                msgs = context.messages
            elif isinstance(context, dict):
                msgs = context.get("messages", [])
            else:
                msgs = []
            user_msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
            if len(user_msgs) >= 1:
                query = user_msgs[-1].get("content", "")
                docs = await asyncio.to_thread(self._retrieve_fn, query)
                if docs:
                    context_text = "\n\n".join(docs)
                    if hasattr(context, "add_message"):
                        context.add_message({
                            "role": "system",
                            "content": f"Context information:\n{context_text}",
                        })
                    elif isinstance(context, dict):
                        context.setdefault("messages", []).insert(-1, {
                            "role": "system",
                            "content": f"Context information:\n{context_text}",
                        })

        await self.push_frame(frame, direction)


class PersonalityProcessor(FrameProcessor):
    def __init__(
        self,
        personalize_fn: Callable[[str, str], str] | None = None,
        enabled: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._personalize_fn = personalize_fn
        self._enabled = enabled
        self._buffer: list[str] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if not self._enabled or not self._personalize_fn:
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMTextFrame):
            self._buffer.append(frame.text)
            return

        if isinstance(frame, LLMFullResponseEndFrame):
            full_text = "".join(self._buffer)
            self._buffer.clear()
            if full_text:
                personalized = self._personalize_fn(full_text, "")
                await self.push_frame(LLMTextFrame(text=personalized), direction)
            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)


class MemoryProcessor(FrameProcessor):
    def __init__(
        self,
        store_fn: Callable[[str, str], None] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._store_fn = store_fn
        self._user_text: str | None = None
        self._assistant_text: str | None = None

    def set_user_text(self, text: str) -> None:
        self._user_text = text

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMTextFrame):
            self._assistant_text = frame.text

        if isinstance(frame, LLMFullResponseEndFrame):
            if self._store_fn and self._user_text and self._assistant_text:
                self._store_fn(self._user_text, self._assistant_text)
            self._user_text = None
            self._assistant_text = None

        await self.push_frame(frame, direction)


class GestureProcessor(FrameProcessor):
    def __init__(self, on_tts_stopped: Callable | None = None, **kwargs):
        super().__init__(**kwargs)
        self._on_tts_stopped = on_tts_stopped

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, GestureFrame) and frame.gesture == "open_palm":
            await self.push_frame(InterruptionFrame(), FrameDirection.UPSTREAM)

        if isinstance(frame, TTSStoppedFrame):
            log.debug("GestureProcessor received TTSStoppedFrame")
            if self._on_tts_stopped:
                self._on_tts_stopped()

        await self.push_frame(frame, direction)


class PendingNameProcessor(FrameProcessor):
    def __init__(
        self,
        register_callback: Callable[[str], None] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._register_callback = register_callback
        self.pending = False
        self._intercepted = False
        self._intercept_cooldown: float = 0.0
        self._cooldown_seconds: float = 8.0

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if self.pending and isinstance(frame, TranscriptionFrame):
            text = frame.text.strip()
            if text and self._register_callback:
                now = time.monotonic()
                if now < self._intercept_cooldown:
                    return
                self._intercept_cooldown = now + self._cooldown_seconds
                self._intercepted = True
                await self._register_callback(text)
            return

        if self._intercepted and isinstance(frame, UserStoppedSpeakingFrame):
            self._intercepted = False
            return

        await self.push_frame(frame, direction)
