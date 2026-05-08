from __future__ import annotations

import asyncio
import logging
from typing import Callable, Awaitable

from virtual_assistant_be.core.protocol import GoEvent, GoCommand, AnimationCmd, StateUpdate, serialize
from virtual_assistant_be.services.llm_service import LlmService
from virtual_assistant_be.services.rag_service import RagService
from virtual_assistant_be.services.tts_service import TtsService

SendFn = Callable[[dict], Awaitable[None]]

log = logging.getLogger(__name__)


class BehaviorController:
    def __init__(self, send_fn: SendFn | None = None) -> None:
        self._send: SendFn | None = send_fn
        self._executor = asyncio.get_event_loop().run_in_executor

        self.llm = LlmService()
        self.rag = RagService()
        self.tts = TtsService()

    def set_send_fn(self, send_fn: SendFn) -> None:
        self._send = send_fn

    async def handle_event(self, msg: GoEvent) -> None:
        log.info("Event from Godot: %s %s", msg.name, msg.params)

    async def handle_command(self, msg: GoCommand) -> None:
        log.info("Command from Godot: %s %s", msg.name, msg.params)

        match msg.name:
            case "ready":
                await self._greet()
            case "shutdown":
                await self._shutdown()
            case _:
                log.warning("Unknown command: %s", msg.name)

    async def handle_text(self, text: str) -> None:
        await self._send_think(True)
        try:
            intent = await self._executor(None, self.llm.classify_intent, text)

            context: str | None = None
            if intent in ("question",):
                docs = await self._executor(None, self.rag.retrieve, text)
                if docs:
                    context = "\n\n".join(docs)

            response, resolved_intent = await self._executor(
                None, self.llm.generate_response, text, context,
            )

            if response:
                log.info("LLM response: %s", response[:100])
                await self._send_speak(response)
                await self._executor(None, self.tts.speak, response)

            anim = self.llm.decide_animation(text, resolved_intent)
            await self._send_animation(anim)
        except Exception:
            log.exception("handle_text failed")
        finally:
            await self._send_think(False)

    async def _greet(self) -> None:
        if self._send is None:
            return
        await self._send(serialize(AnimationCmd(name="greet")))
        await self._send(serialize(StateUpdate(connected=True)))

        greeting = "Hello! I'm ready to help."
        self._executor(None, self.tts.speak, greeting)
        await self._send_speak(greeting)
        await self._send_think(False)

    async def _shutdown(self) -> None:
        log.info("Shutting down services")
        if self._send:
            await self._send(serialize(StateUpdate(connected=False)))

    async def send_state(self, **kwargs) -> None:
        if self._send:
            await self._send(serialize(StateUpdate(**kwargs)))

    async def send_animation(self, name: str, **params) -> None:
        if self._send:
            await self._send(serialize(AnimationCmd(name=name, params=params or None)))

    async def _send_speak(self, text: str) -> None:
        if self._send:
            from virtual_assistant_be.core.protocol import SpeakCmd
            await self._send(serialize(SpeakCmd(text=text)))

    async def _send_think(self, active: bool) -> None:
        if self._send:
            from virtual_assistant_be.core.protocol import ThinkIndicator
            await self._send(serialize(ThinkIndicator(active=active)))
