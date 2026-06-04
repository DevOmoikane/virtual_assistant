from __future__ import annotations

import asyncio
import inspect
import logging
import time
from typing import Callable

from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from virtual_assistant_be.core.protocol import (
    AnimationCmd,
    HeardIndicator,
    ListenIndicator,
    SpeakCmd,
    StateUpdate,
    ThinkIndicator,
    serialize,
)
from virtual_assistant_be.pipecat.custom_frames import (
    GestureFrame,
    PersonAppearedFrame,
    PersonDisappearedFrame,
)

log = logging.getLogger(__name__)

SendFn = Callable[[dict], None]


class UserBridgeProcessor(FrameProcessor):
    """Sits before user_agg — handles user-side frames that user_agg consumes.

    - TranscriptionFrame → "heard" indicator
    - UserStartedSpeakingFrame → listen=True
    - UserStoppedSpeakingFrame → listen=False
    """

    def __init__(self, send_fn: SendFn | None = None, on_user_interaction: Callable[[], None] | None = None, **kwargs):
        super().__init__(**kwargs)
        self._send_fn: SendFn | None = send_fn
        self._on_user_interaction = on_user_interaction

    def set_send_fn(self, send_fn: SendFn | None) -> None:
        self._send_fn = send_fn

    async def _send(self, data: dict) -> None:
        if self._send_fn:
            try:
                if inspect.iscoroutinefunction(self._send_fn):
                    await self._send_fn(data)
                else:
                    self._send_fn(data)
            except Exception:
                log.exception("UserBridge send failed")

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            await self._send(serialize(HeardIndicator(text=frame.text)))
            if self._on_user_interaction:
                self._on_user_interaction()
        elif isinstance(frame, UserStartedSpeakingFrame):
            await self._send(serialize(ListenIndicator(active=True)))
            if self._on_user_interaction:
                self._on_user_interaction()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            await self._send(serialize(ListenIndicator(active=False)))

        await self.push_frame(frame, direction)


class GodotBridgeProcessor(FrameProcessor):
    """Sits before assistant_agg — handles LLM/assistant-side frames.

    - LLMTextFrame → "speak" indicator
    - LLMFullResponseStartFrame → think=True
    - LLMFullResponseEndFrame → think=False
    - TTSSpeakFrame → "speak" indicator
    - PersonAppearedFrame → greet animation + state update
    - PersonDisappearedFrame → idle animation + state update
    - GestureFrame → animation command
    """

    def __init__(self, send_fn: SendFn | None = None, **kwargs):
        super().__init__(**kwargs)
        self._send_fn: SendFn | None = send_fn

    def set_send_fn(self, send_fn: SendFn | None) -> None:
        self._send_fn = send_fn

    async def _send(self, data: dict) -> None:
        if self._send_fn:
            try:
                if inspect.iscoroutinefunction(self._send_fn):
                    await self._send_fn(data)
                else:
                    self._send_fn(data)
            except Exception:
                log.exception("GodotBridge send failed")

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMTextFrame):
            await self._send(serialize(SpeakCmd(text=frame.text)))
        elif isinstance(frame, LLMFullResponseStartFrame):
            await self._send(serialize(ThinkIndicator(active=True)))
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self._send(serialize(ThinkIndicator(active=False)))
        elif isinstance(frame, TTSSpeakFrame):
            await self._send(serialize(SpeakCmd(text=frame.text)))
        elif isinstance(frame, PersonAppearedFrame):
            await self._send(serialize(AnimationCmd(name="greet")))
            now_str = str(time.time())
            await self._send(serialize(StateUpdate(connected=True, person=frame.person_name, at=now_str)))
        elif isinstance(frame, PersonDisappearedFrame):
            await self._send(serialize(AnimationCmd(name="idle")))
            await self._send(serialize(StateUpdate(connected=True)))
        elif isinstance(frame, GestureFrame):
            await self._send(serialize(AnimationCmd(name=frame.gesture, params={"x": frame.x, "y": frame.y})))
            if frame.gesture == "greet":
                await self._send(serialize(ListenIndicator(active=True)))

        await self.push_frame(frame, direction)
