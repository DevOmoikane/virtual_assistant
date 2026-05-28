from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, Awaitable

import numpy as np

from virtual_assistant_be.core.config import settings
from virtual_assistant_be.core.translations import translate as t
from virtual_assistant_be.timer import Timer, log_duration
from virtual_assistant_be.core.protocol import (
    GoEvent,
    GoCommand,
    AnimationCmd,
    StateUpdate,
    SpeakCmd,
    DeviceCmd,
    ListenIndicator,
    ThinkIndicator,
    HeardIndicator,
    serialize,
)
from virtual_assistant_be.services.llm_service import LlmService
from virtual_assistant_be.services.rag_service import RagService
from virtual_assistant_be.services.stt_service import SttService
from virtual_assistant_be.services.tts_service import TtsService
from virtual_assistant_be.services.mcp_tts_client import McpTtsClient
from virtual_assistant_be.services.camera_service import CameraService
from virtual_assistant_be.services.audio_service import AudioService
from virtual_assistant_be.services.memory_service import MemoryService
from virtual_assistant_be.services.command_service import CommandService
from virtual_assistant_be.services.telegram_service import TelegramService
from virtual_assistant_be.services.face_service import FaceService
from virtual_assistant_be.services.personality_service import PersonalityService

SendFn = Callable[[dict], Awaitable[None]]

log = logging.getLogger(__name__)


class BehaviorController:
    def __init__(self, send_fn: SendFn | None = None) -> None:
        self._send: SendFn | None = send_fn
        self.llm = LlmService()
        self.rag = RagService()
        self.stt = SttService()
        self.tts = TtsService()
        self.mcp_tts = McpTtsClient(server_url=settings.mcp_tts_server_url)
        self.face_service = FaceService()
        self.camera = CameraService(
            event_callback=self._on_camera_event,
            face_service=self.face_service,
        )
        self.audio = AudioService(
            audio_callback=self._on_audio_chunk,
            device_id=self.camera.audio_device_id,
        )
        self.memory = MemoryService()
        self.telegram = TelegramService()
        self.commands = CommandService(telegram_service=self.telegram)
        self.personality = PersonalityService()
        self.telegram.set_message_callback(self._on_telegram_message)

        self._last_person_greeted: float = 0.0
        self._last_gesture_time: float = 0.0
        self._pending_name: str | None = None
        self._current_language: str = settings.piper_default_language
        self._current_person_name: str | None = None
        self._processing_text = False

    async def _run_in_executor(self, fn, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, fn, *args)

    def _lang(self, language: str | None = None) -> str:
        return language or self._current_language

    async def _is_speaking(self) -> bool:
        if self.mcp_tts.is_connected:
            return await self.mcp_tts.is_speaking()
        return self.tts.is_speaking

    async def _speak(self, text: str, language: str | None = None) -> None:
        lang = self._lang(language)
        self.audio.mute()
        try:
            if self.mcp_tts.is_connected:
                await self.mcp_tts.speak(text, lang)
            else:
                await self._run_in_executor(self.tts.speak, text, lang)
        finally:
            self.audio.unmute()

    async def _say(self, key: str, **fmt_args: str) -> str:
        lang = self._lang()
        msg = t(key, lang, **fmt_args)
        if self.personality.enabled:
            msg = await self._run_in_executor(self.personality.personalize, msg, lang)
        await self._speak(msg, lang)
        await self._send_speak(msg)
        return msg

    async def _interrupt_speech(self) -> None:
        log.info("Interrupted by user")
        if self.mcp_tts.is_connected:
            await self.mcp_tts.stop()
        else:
            self.tts.stop()
        self.audio.unmute()
        self._processing_text = False
        await self._send_listen(True)

    def set_send_fn(self, send_fn: SendFn) -> None:
        self._send = send_fn

    async def _on_camera_event(self, event: str, data: dict) -> None:
        log.debug("Camera event: %s %s", event, data)

        match event:
            case "person_appeared":
                await self._on_person_appeared(data)
            case "person_disappeared":
                await self._on_person_disappeared()
            case "gesture_detected":
                await self._on_gesture(data)

    async def _on_person_appeared(self, data: dict) -> None:
        now = time.monotonic()
        if now - self._last_person_greeted < 30.0:
            return
        self._last_person_greeted = now
        await self._run_in_executor(self.memory.store_person_event, "appeared")

        name = data.get("name")
        self._current_person_name = name
        if name:
            stored_personality = await self._run_in_executor(
                self.face_service.get_personality, name
            )
            if stored_personality and stored_personality != "default" and stored_personality != settings.personality_style:
                self.personality.set_style(stored_personality)
                log.info("Applied stored personality '%s' for '%s'", stored_personality, name)
            stored_lang = await self._run_in_executor(
                self.face_service.get_language, name
            )
            if stored_lang:
                self._current_language = stored_lang
                log.info("Applied stored language '%s' for '%s'", stored_lang, name)
            await self.send_animation("greet")
            await self._say("hello_name", name=name)
        else:
            await self.send_animation("greet")
            await self._say("hello_unknown")
            self._pending_name = True
            await self._send_listen(True)

    async def _on_person_disappeared(self) -> None:
        await self._run_in_executor(self.memory.store_person_event, "disappeared")
        self._current_person_name = None
        self.personality.reset_style()
        await self.send_animation("idle")
        await self._send(serialize(StateUpdate(connected=True)))

    async def _on_gesture(self, data: dict) -> None:
        gesture = data.get("gesture", "")
        if gesture in ("none", "closed_fist", ""):
            return
        now = time.monotonic()
        if now - self._last_gesture_time < 5.0:
            return
        self._last_gesture_time = now
        match gesture:
            case "wave":
                await self.send_animation("greet")
                await self._say("gesture_wave")
            case "thumbs_up":
                await self.send_animation("nod")
                await self._say("gesture_thumbs_up")
            case "open_palm":
                if await self._is_speaking():
                    await self._interrupt_speech()
                else:
                    await self.send_animation("listen")
            case "point":
                await self.send_animation("think")
            case "fist":
                await self.send_animation("surprised")
            case _:
                pass

    async def _on_telegram_message(self, sender: str, text: str, chat_id: int) -> None:
        log.info("Telegram from %s: %s", sender, text)
        if self._send:
            await self._say("telegram_message", sender=sender, text=text)

    async def _on_audio_chunk(self, audio: np.ndarray) -> None:
        if self._processing_text or await self._is_speaking():
            log.info("Already speaking or processing, ignoring audio chunk")
            return
        await self._send_listen(True)
        try:
            t0 = time.monotonic()
            text, language = await self._run_in_executor(self.stt.transcribe, audio)
            text = text.strip()
            if text:
                self._current_language = language or self._current_language
                if self._current_person_name:
                    await self._run_in_executor(
                        self.face_service.set_language,
                        self._current_person_name, self._current_language,
                    )
                log_duration("pipeline.transcribe_to_text", time.monotonic() - t0)
                await self.handle_text(text, language)
            else:
                log_duration("pipeline.transcribe_empty", time.monotonic() - t0)
        except Exception:
            log.exception("Audio transcription failed")
        finally:
            await self._send_listen(False)

    async def handle_event(self, msg: GoEvent) -> None:
        log.info("Event from Godot: %s %s", msg.name, msg.params)
        match msg.name:
            case "text":
                text = (msg.params or {}).get("text", "")
                if text:
                    await self.handle_text(text)

    async def handle_command(self, msg: GoCommand) -> None:
        log.info("Command from Godot: %s %s", msg.name, msg.params)

        match msg.name:
            case "ready":
                await self._on_ready()
            case "shutdown":
                await self._on_shutdown()
            case _:
                log.warning("Unknown command: %s", msg.name)

    async def _register_name(self, text: str) -> bool:
        if not self._pending_name or not self.face_service.enabled:
            return False
        name = text.strip().title()
        emb = self.face_service.last_unknown_embedding
        if emb is not None:
            ok = await self._run_in_executor(self.face_service.register, name, emb)
            if ok:
                await self._say("register_success", name=name)
            else:
                await self._say("register_failure")
        self._pending_name = None
        await self._send_listen(False)
        return True

    async def _handle_change_personality(
        self, text: str, language: str | None = None
    ) -> None:
        if not self._current_person_name:
            await self._say("personality_need_face")
            return

        normalized = await self._run_in_executor(
            self.personality.normalize_personality, text,
        )
        await self._run_in_executor(
            self.face_service.set_personality,
            self._current_person_name, normalized,
        )
        self.personality.set_style(normalized)
        await self._say("personality_changed", personality=normalized)

    async def handle_text(self, text: str, language: str | None = None) -> None:
        if self._processing_text:
            log.info("Already processing text, dropping: %s", text[:60])
            return
        self._processing_text = True
        t_start = time.monotonic()

        await self._send_heard(text)

        if await self._register_name(text):
            log_duration("pipeline.register_name", time.monotonic() - t_start)
            self._processing_text = False
            return

        await self._send_think(True)
        try:
            t0 = time.monotonic()
            intent = await self._run_in_executor(self.llm.classify_intent, text)
            log_duration("pipeline.classify_intent", time.monotonic() - t0)

            if intent == "change_personality":
                await self._handle_change_personality(text, language)
                self._processing_text = False
                return

            t0 = time.monotonic()
            device_cmd = await self._run_in_executor(self.llm.classify_device_command, text)
            if device_cmd:
                log_duration("pipeline.classify_device_command", time.monotonic() - t0)
                await self._execute_device_command(device_cmd)
            else:
                log_duration("pipeline.classify_device_command", time.monotonic() - t0)

            context: str | None = None
            if intent in ("question",) and settings.rag_enabled:
                t0 = time.monotonic()
                docs = await self._run_in_executor(self.rag.retrieve, text)
                log_duration("pipeline.rag_retrieve", time.monotonic() - t0)
                if docs:
                    context = "\n\n".join(docs)

            t0 = time.monotonic()
            response, resolved_intent = await self._run_in_executor(
                self.llm.generate_response, text, context, self._lang(language),
            )
            log_duration("pipeline.generate_response", time.monotonic() - t0)

            if response:
                log.info("LLM response: %s", response[:100])
                await self._send_speak(response)
                t0 = time.monotonic()
                await self._speak(response, language)
                log_duration("pipeline.tts_speak", time.monotonic() - t0)
                await self._run_in_executor(self.memory.store_interaction, text, response)

            anim = self.llm.decide_animation(text, resolved_intent)
            await self.send_animation(anim)

            log_duration("pipeline.handle_text_total", time.monotonic() - t_start)
        except Exception:
            log.exception("handle_text failed")
        finally:
            self._processing_text = False
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
                        cmd.get("action", ""), cmd.get("message", ""), cmd.get("contact", ""),
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
            await self.mcp_tts.connect()
        except Exception:
            log.warning("MCP TTS server not available, falling back to local TTS")

        loop = asyncio.get_running_loop()
        self.camera.start(loop)
        self.audio.start(loop)
        await self._send_listen(True)
        await self._run_in_executor(self.telegram.start_polling)

    async def _on_shutdown(self) -> None:
        log.info("Shutting down")
        await self._cleanup()
        if self._send:
            await self._send(serialize(StateUpdate(connected=False)))

    async def _cleanup(self) -> None:
        self.camera.stop()
        self.audio.stop()
        await self.mcp_tts.disconnect()
        await self._run_in_executor(self.telegram.stop_polling)

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

    async def _send_heard(self, text: str) -> None:
        if self._send:
            await self._send(serialize(HeardIndicator(text=text)))

    async def _send_listen(self, active: bool) -> None:
        if self._send:
            await self._send(serialize(ListenIndicator(active=active)))
