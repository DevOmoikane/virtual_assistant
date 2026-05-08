from __future__ import annotations

import asyncio
import logging
from typing import Callable, Awaitable

import numpy as np

from virtual_assistant_be.core.protocol import (
    GoEvent,
    GoCommand,
    AnimationCmd,
    StateUpdate,
    SpeakCmd,
    DeviceCmd,
    ListenIndicator,
    ThinkIndicator,
    serialize,
)
from virtual_assistant_be.services.llm_service import LlmService
from virtual_assistant_be.services.rag_service import RagService
from virtual_assistant_be.services.stt_service import SttService
from virtual_assistant_be.services.tts_service import TtsService
from virtual_assistant_be.services.camera_service import CameraService
from virtual_assistant_be.services.audio_service import AudioService
from virtual_assistant_be.services.memory_service import MemoryService
from virtual_assistant_be.services.command_service import CommandService

SendFn = Callable[[dict], Awaitable[None]]

log = logging.getLogger(__name__)

_GREETING = "Hello! I'm ready to help."


class BehaviorController:
    def __init__(self, send_fn: SendFn | None = None) -> None:
        self._send: SendFn | None = send_fn
        self.llm = LlmService()
        self.rag = RagService()
        self.stt = SttService()
        self.tts = TtsService()
        self.camera = CameraService(event_callback=self._on_camera_event)
        self.audio = AudioService(audio_callback=self._on_audio_chunk)
        self.memory = MemoryService()
        self.commands = CommandService()

    async def _run_in_executor(self, fn, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, fn, *args)

    def set_send_fn(self, send_fn: SendFn) -> None:
        self._send = send_fn

    async def _on_camera_event(self, event: str, data: dict) -> None:
        log.debug("Camera event: %s %s", event, data)

        match event:
            case "person_appeared":
                await self._on_person_appeared()
            case "person_disappeared":
                await self._on_person_disappeared()
            case "gesture_detected":
                await self._on_gesture(data)

    async def _on_person_appeared(self) -> None:
        await self._run_in_executor(self.memory.store_person_event, "appeared")
        await self.send_animation("greet")
        await self._run_in_executor(self.tts.speak, "Hello there!")
        await self._send_speak("Hello there!")

    async def _on_person_disappeared(self) -> None:
        await self._run_in_executor(self.memory.store_person_event, "disappeared")
        await self.send_animation("idle")
        await self._send(serialize(StateUpdate(connected=True)))

    async def _on_gesture(self, data: dict) -> None:
        gesture = data.get("gesture", "")
        match gesture:
            case "wave":
                await self.send_animation("greet")
                await self._run_in_executor(self.tts.speak, "I see you waving!")
                await self._send_speak("I see you waving!")
            case "thumbs_up":
                await self.send_animation("nod")
                await self._run_in_executor(self.tts.speak, "Got it!")
                await self._send_speak("Got it!")
            case "open_palm":
                await self.send_animation("listen")
            case "point":
                await self.send_animation("think")
            case "fist":
                await self.send_animation("surprised")
            case _:
                pass

    async def _on_audio_chunk(self, audio: np.ndarray) -> None:
        await self._send_listen(True)
        try:
            text = await self._run_in_executor(self.stt.transcribe, audio)
            text = text.strip()
            if text:
                log.info("STT: %s", text)
                await self.handle_text(text)
        except Exception:
            log.exception("Audio transcription failed")
        finally:
            await self._send_listen(False)

    async def handle_event(self, msg: GoEvent) -> None:
        log.info("Event from Godot: %s %s", msg.name, msg.params)

    async def handle_command(self, msg: GoCommand) -> None:
        log.info("Command from Godot: %s %s", msg.name, msg.params)

        match msg.name:
            case "ready":
                await self._on_ready()
            case "shutdown":
                await self._on_shutdown()
            case _:
                log.warning("Unknown command: %s", msg.name)

    async def handle_text(self, text: str) -> None:
        await self._send_think(True)
        try:
            intent = await self._run_in_executor(self.llm.classify_intent, text)

            device_cmd = await self._run_in_executor(self.llm.classify_device_command, text)
            if device_cmd:
                await self._execute_device_command(device_cmd)

            context: str | None = None
            if intent in ("question",):
                docs = await self._run_in_executor(self.rag.retrieve, text)
                if docs:
                    context = "\n\n".join(docs)

            response, resolved_intent = await self._run_in_executor(
                self.llm.generate_response, text, context,
            )

            if response:
                log.info("LLM response: %s", response[:100])
                await self._send_speak(response)
                await self._run_in_executor(self.tts.speak, response)
                await self._run_in_executor(self.memory.store_interaction, text, response)

            anim = self.llm.decide_animation(text, resolved_intent)
            await self.send_animation(anim)
        except Exception:
            log.exception("handle_text failed")
        finally:
            await self._send_think(False)

    async def _execute_device_command(self, cmd: dict) -> None:
        result = {"status": "unknown_command", "device": "unknown"}
        try:
            match cmd.get("device"):
                case "lights":
                    result = self.commands.execute_lights(cmd.get("action", "toggle"))
                case "door":
                    result = self.commands.execute_door(cmd.get("action", "toggle"))
                case "send_message":
                    result = self.commands.execute_send_message(
                        cmd.get("action", ""), cmd.get("message", ""),
                    )
                case "home_assistant":
                    result = self.commands.execute_home_assistant(cmd.get("command", ""))
        except Exception:
            log.exception("Device command failed")
            result = {"status": "error", "device": cmd.get("device", "unknown")}

        await self._send(
            serialize(DeviceCmd(
                device=result.get("device", ""),
                action=result.get("action", ""),
                status=result.get("status", ""),
                message=result.get("message", ""),
            ))
        )
        log.info("Device command result: %s", result)

    async def _on_ready(self) -> None:
        await self.send_animation("greet")
        await self._send(serialize(StateUpdate(connected=True)))
        try:
            await self._run_in_executor(self.tts.speak, _GREETING)
            await self._send_speak(_GREETING)
        except Exception:
            log.warning("TTS not available, skipping greeting speech")

        loop = asyncio.get_running_loop()
        self.camera.start(loop)
        self.audio.start(loop)

    async def _on_shutdown(self) -> None:
        log.info("Shutting down")
        await self._cleanup()
        if self._send:
            await self._send(serialize(StateUpdate(connected=False)))

    async def _cleanup(self) -> None:
        self.camera.stop()
        self.audio.stop()

    async def send_state(self, **kwargs) -> None:
        if self._send:
            await self._send(serialize(StateUpdate(**kwargs)))

    async def send_animation(self, name: str, **params) -> None:
        if self._send:
            await self._send(serialize(AnimationCmd(name=name, params=params or None)))

    async def _send_speak(self, text: str) -> None:
        if self._send:
            await self._send(serialize(SpeakCmd(text=text)))

    async def _send_think(self, active: bool) -> None:
        if self._send:
            await self._send(serialize(ThinkIndicator(active=active)))

    async def _send_listen(self, active: bool) -> None:
        if self._send:
            await self._send(serialize(ListenIndicator(active=active)))
