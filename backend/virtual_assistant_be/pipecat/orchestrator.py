from __future__ import annotations

import asyncio
import inspect
import logging
import time
from pathlib import Path

import aiohttp
import requests

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    EndFrame,
    LLMRunFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.adapters.schemas.function_schema import FunctionSchema

from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
    ToolsSchema,
)
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.piper.tts import PiperTTSService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.turns.user_mute.always_user_mute_strategy import AlwaysUserMuteStrategy

from virtual_assistant_be.core.config import settings
from virtual_assistant_be.core.protocol import (
    AnimationCmd,
    GoCommand,
    GoEvent,
    HeardIndicator,
    ListenIndicator,
    StateUpdate,
    serialize,
)
from virtual_assistant_be.core.translations import lang_name, translate as t
from virtual_assistant_be.services.camera_service import CameraService
from virtual_assistant_be.services.command_service import CommandService
from virtual_assistant_be.services.face_service import FaceService
from virtual_assistant_be.services.llm_service import LlmService
from virtual_assistant_be.services.mcp_service import MCPService
from virtual_assistant_be.services.memory_service import MemoryService
from virtual_assistant_be.services.personality_service import PersonalityService
from virtual_assistant_be.services.rag_service import RagService
from virtual_assistant_be.services.telegram_service import TelegramService
from virtual_assistant_be.pipecat.custom_frames import (
    GestureFrame,
    PersonAppearedFrame,
    PersonDisappearedFrame,
    TelegramMessageFrame,
)
from virtual_assistant_be.pipecat.godot_bridge_processor import (
    GodotBridgeProcessor,
    SendFn,
    UserBridgeProcessor,
)
from virtual_assistant_be.pipecat.processors import (
    GestureProcessor,
    IntentRouter,
    MemoryProcessor,
    PendingNameProcessor,
    PersonalityProcessor,
    RAGProcessor,
)
from virtual_assistant_be.pipecat.supertonic_tts import SupertonicTTSService

log = logging.getLogger(__name__)


class PipecatOrchestrator:
    def __init__(self, send_fn: SendFn | None = None) -> None:
        self._send_fn: SendFn | None = send_fn

        self.rag = RagService()
        self.face_service = FaceService()
        self.memory = MemoryService()
        self.telegram = TelegramService()
        self.commands = CommandService(telegram_service=self.telegram)
        self.personality = PersonalityService()
        self.camera = CameraService(
            event_callback=self._on_camera_event,
            face_service=self.face_service,
        )
        self.mcp = MCPService(settings.mcp_servers)
        self.telegram.set_message_callback(self._on_telegram_message)
        self.llm_service = LlmService()

        self._pending_name: bool = False
        self._current_language: str = settings.piper_default_language
        self._current_person_name: str | None = None
        self._last_person_greeted: float = 0.0
        self._last_gesture_time: float = 0.0
        self._pipeline_task: PipelineTask | None = None
        self._runner: PipelineRunner | None = None
        self._context: LLMContext | None = None
        self._llm: OLLamaLLMService | None = None
        self._memory_processor: MemoryProcessor | None = None
        self._pending_name_processor: PendingNameProcessor | None = None
        self._transport: LocalAudioTransport | None = None
        self._aiohttp_session: aiohttp.ClientSession | None = None
        self._name_retries: int = 0
        self._max_name_retries: int = 3
        self._idle_task: asyncio.Task | None = None
        self._idle_timeout: int = settings.idle_conversation_timeout
        self._last_idle_initiate: float = 0.0
        self._mcp_refresh_task: asyncio.Task | None = None

    def set_send_fn(self, send_fn: SendFn | None) -> None:
        self._send_fn = send_fn

    async def _send_msg(self, data: dict) -> None:
        fn = self._send_fn
        if fn:
            if inspect.iscoroutinefunction(fn):
                await fn(data)
            else:
                fn(data)

    async def _speak_text(self, text: str) -> None:
        if not self._pipeline_task:
            return
        log.debug("_speak_text: pushing TTSSpeakFrame for '%s'", text[:80])
        await self._pipeline_task.queue_frames([
            TTSSpeakFrame(text=text),
        ])

    def _lang(self, language: str | None = None) -> str:
        return language or self._current_language

    async def _say(self, key: str, rephrase: bool = True, **fmt_args: str) -> str:
        lang = self._lang()
        msg = t(key, lang, **fmt_args)
        if rephrase and self.personality.enabled:
            msg = await asyncio.to_thread(self.personality.personalize, msg, lang)
        return msg

    def _build_system_prompt(self, language: str) -> str:
        base = t("system_prompt", language, name=settings.assistant_name, style=settings.personality_style)
        lang_instr = t("sys_respond_in", language, lang_name=lang_name(language))
        return f"{base}\n\n{lang_instr}"

    def _route_switch_language(self, lang: str) -> str | None:
        if lang == self._current_language:
            return None
        old_lang = self._current_language
        self._current_language = lang
        if self._current_person_name:
            self.face_service.set_language(self._current_person_name, lang)
        msgs = self._context.messages if self._context else []
        for i, msg in enumerate(msgs):
            if msg.get("role") == "system":
                msgs[i] = {"role": "system", "content": self._build_system_prompt(lang)}
                break
        log.info("Language switched from '%s' to '%s'", old_lang, lang)
        return f"Language switched to {lang_name(lang)}."

    def _route_change_personality(self, text: str) -> str | None:
        trait = self.llm_service.extract_personality(text)
        if not trait:
            return None
        self.personality.set_style(trait)
        if self._current_person_name:
            self.face_service.set_personality(self._current_person_name, trait)
        log.info("Personality changed to '%s' for '%s'", trait, self._current_person_name)
        return t("personality_changed", self._current_language, personality=trait)

    async def _create_pipeline(self) -> None:
        self._context = LLMContext()

        tts_engine = settings.tts_engine
        aiohttp_session = aiohttp.ClientSession() if tts_engine == "xtts" else None
        self._aiohttp_session = aiohttp_session

        if tts_engine == "piper":
            voice_id = (settings.piper_voices or {}).get(
                self._current_language, "en_US-lessac-medium"
            )
            voice_name = Path(voice_id).stem
            tts = PiperTTSService(
                voice_id=voice_name,
                download_dir=Path(voice_id).parent if "/" in str(voice_id) else None,
                sample_rate=24000,
            )
        elif tts_engine == "kokoro":
            tts = KokoroTTSService(
                voice_id="af_bella",
                sample_rate=24000,
            )
        elif tts_engine == "xtts":
            from pipecat.services.xtts.tts import XTTSService
            tts = XTTSService(
                voice_id="default",
                base_url="http://localhost:8000",
                aiohttp_session=aiohttp_session,
                sample_rate=24000,
            )
        elif tts_engine == "supertonic":
            tts = SupertonicTTSService(
                voice_id=settings.supertonic_voice,
                sample_rate=24000,
            )
        else:
            log.warning("Unknown tts_engine '%s', falling back to piper", tts_engine)
            tts = PiperTTSService(
                voice_id="en_US-lessac-medium",
                sample_rate=24000,
            )

        self._transport = LocalAudioTransport(
            LocalAudioTransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
            )
        )

        stt_engine = settings.stt_engine
        if stt_engine == "faster_whisper":
            from virtual_assistant_be.pipecat.faster_whisper_stt import FasterWhisperSTTService
            stt = FasterWhisperSTTService(
                model=settings.stt_model_size,
                device="auto",
                compute_type="int8",
                language="es",
            )
        elif stt_engine == "whisper":
            stt = WhisperSTTService(
                model=settings.stt_model_size,
                device="auto",
                compute_type="int8",
                language="es",
            )
        else:
            log.warning("Unknown stt_engine '%s', falling back to whisper", stt_engine)
            stt = WhisperSTTService(
                model=settings.stt_model_size,
                device="auto",
                compute_type="int8",
                language="es",
            )

        self._llm = OLLamaLLMService(
            model=settings.ollama_gen_model,
            base_url=f"{settings.ollama_url}/v1",
        )
        llm = self._llm
        self._register_llm_tools(llm)

        user_bridge = UserBridgeProcessor(
            send_fn=self._send_fn,
            on_user_interaction=self._reset_idle_timer,
        )
        godot_bridge = GodotBridgeProcessor(send_fn=self._send_fn)

        user_agg, assistant_agg = LLMContextAggregatorPair(
            self._context,
            user_params=LLMUserAggregatorParams(
                vad_analyzer=SileroVADAnalyzer(),
                user_mute_strategies=[AlwaysUserMuteStrategy()],
            ),
        )

        def _retrieve_docs(query: str) -> list[str]:
            return self.rag.retrieve(query)

        def _personalize(text: str, _: str) -> str:
            return self.personality.personalize(text, self._current_language)

        def _store_interaction(user_text: str, assistant_text: str) -> None:
            self.memory.store_interaction(user_text, assistant_text)

        self._memory_processor = MemoryProcessor(store_fn=_store_interaction)

        self._pending_name_processor = PendingNameProcessor(
            register_callback=self._register_name,
        )

        pipeline = Pipeline([
            self._transport.input(),
            stt,
            user_bridge,
            self._pending_name_processor,
            IntentRouter(
                llm_service=self.llm_service,
                on_switch_language=self._route_switch_language,
                on_change_personality=self._route_change_personality,
            ),
            user_agg,
            RAGProcessor(
                retrieve_fn=_retrieve_docs,
                language_fn=lambda: self._current_language,
            ),
            llm,
            PersonalityProcessor(
                personalize_fn=_personalize,
                enabled=self.personality.enabled,
            ),
            self._memory_processor,
            godot_bridge,
            tts,
            GestureProcessor(),
            assistant_agg,
            self._transport.output(),
        ])

        self._pipeline_task = PipelineTask(
            pipeline,
            params=PipelineParams(
                enable_metrics=False,
                enable_usage_metrics=False,
            ),
            enable_rtvi=False,
        )

        self._runner = PipelineRunner(handle_sigint=False)

        self._pipeline_task.set_reached_downstream_filter(
            (GestureFrame, PersonAppearedFrame, PersonDisappearedFrame, TelegramMessageFrame)
        )

    def _register_llm_tools(self, llm: OLLamaLLMService) -> None:
        pass

    async def _register_mcp_tools(self) -> None:
        if not self._llm or not self._context:
            return
        llm = self._llm
        tool_schemas: list[FunctionSchema] = []

        if self.mcp.enabled:
            for full_name, description, properties, required in self.mcp.list_all_tools():
                schema = FunctionSchema(
                    name=full_name,
                    description=description,
                    properties=properties,
                    required=required,
                )
                tool_schemas.append(schema)
                handler = self._make_mcp_handler(full_name)
                llm.register_function(full_name, handler)

        if tool_schemas:
            self._context.set_tools(ToolsSchema(standard_tools=tool_schemas))

    def _make_mcp_handler(self, full_name: str):
        async def handler(params):
            result = await self.mcp.call_tool(full_name, params.arguments)
            if result.isError:
                text = "Error: " + (
                    result.content[0].text if result.content else "Unknown error"
                )
            else:
                text = result.content[0].text if result.content else ""
            await params.result_callback(text)
        return handler

    def _build_mcp_context_text(self) -> str:
        lines: list[str] = []
        for conn in self.mcp._connections.values():
            if not conn.tools:
                continue
            lines.append(f"[{conn.config.name}] Available tools:")
            for tool in conn.tools:
                desc = (tool.description or "").strip()
                params = list((tool.inputSchema or {}).get("properties", {}).keys())
                param_str = f" ({', '.join(params)})" if params else ""
                lines.append(f"  - {tool.name}{param_str}: {desc}")
        return "\n".join(lines)

    async def _refresh_mcp_live_context(self) -> str | None:
        if not self.mcp._tool_map:
            return None
        try:
            result = await self.mcp.call_tool("home-assistant_GetLiveContext", {})
            if result.isError:
                return None
            text = result.content[0].text if result.content else ""
            if text:
                return "Current device states:\n" + text
        except (KeyError, ValueError, Exception) as e:
            log.debug("GetLiveContext not available: %s", e)
        return None

    async def _inject_mcp_context(self) -> None:
        if not self._context or not self.mcp._tool_map:
            return
        parts: list[str] = []
        caps = self._build_mcp_context_text()
        if caps:
            parts.append(caps)
        live = await self._refresh_mcp_live_context()
        if live:
            parts.append(live)
        if not parts:
            return
        text = "\n\n".join(parts)
        msgs = self._context.messages
        idx = next((i for i, m in enumerate(msgs) if isinstance(m, dict) and m.get("role") == "system" and m.get("content", "").startswith("[MCP]")), None)
        msg = {"role": "system", "content": f"[MCP]\n{text}"}
        if idx is not None:
            msgs[idx] = msg
        else:
            msgs.insert(1, msg)

    async def _start_mcp_refresh_task(self) -> None:
        await self._inject_mcp_context()
        try:
            while True:
                await asyncio.sleep(300)
                await self._inject_mcp_context()
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("MCP refresh failed")

    async def _inject_user_text(self, text: str, language: str | None = None) -> None:
        await self._send_msg(serialize(HeardIndicator(text=text)))
        self._context.add_message({
            "role": "user",
            "content": text,
        })
        await self._pipeline_task.queue_frames([
            LLMRunFrame(),
        ])
        self._reset_idle_timer()

    async def start(self) -> None:
        await self._create_pipeline()

        self._context.add_message({"role": "system", "content": self._build_system_prompt(self._current_language)})
        self._context.add_message({
            "role": "assistant",
            "content": t("hello_name", self._lang(), name="") or "Hello!",
        })

        async def run_task():
            await self._runner.run(self._pipeline_task)

        asyncio.create_task(run_task())
        await asyncio.sleep(0.1)

    async def stop(self) -> None:
        self._cancel_idle_timer()
        if self._mcp_refresh_task:
            self._mcp_refresh_task.cancel()
            self._mcp_refresh_task = None
        if self._pipeline_task:
            await self._pipeline_task.queue_frame(EndFrame())
            self._pipeline_task = None
        self._transport = None
        if self._aiohttp_session:
            await self._aiohttp_session.close()
            self._aiohttp_session = None

    async def handle_event(self, msg: GoEvent) -> None:
        log.info("Event from Godot: %s %s", msg.name, msg.params)
        match msg.name:
            case "text":
                text = (msg.params or {}).get("text", "")
                if text:
                    await self._inject_user_text(text, self._current_language)

    async def handle_command(self, msg: GoCommand) -> None:
        log.info("Command from Godot: %s %s", msg.name, msg.params)
        match msg.name:
            case "ready":
                await self._on_ready()
            case "shutdown":
                await self._on_shutdown()
            case "clear_data":
                await self._on_clear_data()

    async def _on_ready(self) -> None:
        await self._send_state(connected=True)
        self.camera.start(asyncio.get_running_loop())
        await asyncio.to_thread(self.telegram.start_polling)
        await self.mcp.start()
        await self._register_mcp_tools()
        self._mcp_refresh_task = asyncio.ensure_future(self._start_mcp_refresh_task())

    async def _on_shutdown(self) -> None:
        log.info("Shutting down")
        await self._cleanup()
        await self._send_state(connected=False)

    async def _on_clear_data(self) -> None:
        log.info("Clearing all data")
        await asyncio.to_thread(self.memory.clear_all)
        await asyncio.to_thread(self.face_service.clear_all)
        if self._context:
            self._context = LLMContext()
            self._context.add_message({"role": "system", "content": self._build_system_prompt(self._current_language)})
            self._context.add_message({
                "role": "assistant",
                "content": t("hello_name", self._lang(), name="") or "Hello!",
            })
            if self._pipeline_task and self.mcp.enabled:
                await self._register_mcp_tools()
        self._pending_name = False
        self._current_person_name = None
        self._current_language = settings.piper_default_language
        self._name_retries = 0
        self._cancel_idle_timer()
        await self._send_state(connected=True, data_cleared=True)
        log.info("All data cleared")

    async def _cleanup(self) -> None:
        self.camera.stop()
        await asyncio.to_thread(self.telegram.stop_polling)
        await self.mcp.stop()

    async def _on_camera_event(self, event: str, data: dict) -> None:
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
        await asyncio.to_thread(self.memory.store_person_event, "appeared")

        name = data.get("name")
        self._current_person_name = name
        if name:
            stored_personality = await asyncio.to_thread(
                self.face_service.get_personality, name
            )
            if stored_personality and stored_personality not in ("default", settings.personality_style):
                self.personality.set_style(stored_personality)
                log.info("Applied stored personality '%s' for '%s'", stored_personality, name)
            stored_lang = await asyncio.to_thread(self.face_service.get_language, name)
            if stored_lang:
                self._current_language = stored_lang
                log.info("Applied stored language '%s' for '%s'", stored_lang, name)
            greeting = await self._say("hello_name", name=name)
            await self._send_animation("greet")
            await self._speak_text(greeting)
            if self._pipeline_task:
                await self._pipeline_task.queue_frames([
                    PersonAppearedFrame(person_name=name),
                ])
            self._reset_idle_timer()
        else:
            greeting = await self._say("hello_unknown")
            await self._send_animation("greet")
            await self._speak_text(greeting)
            self._pending_name = True
            if self._pending_name_processor:
                self._pending_name_processor.pending = True
                self._pending_name_processor._intercept_cooldown = time.monotonic() + self._pending_name_processor._cooldown_seconds
            if self._pipeline_task:
                await self._pipeline_task.queue_frames([
                    PersonAppearedFrame(person_name=None),
                ])

    async def _on_person_disappeared(self) -> None:
        await asyncio.to_thread(self.memory.store_person_event, "disappeared")
        self._cancel_idle_timer()
        self._current_person_name = None
        self.personality.reset_style()
        if self._pipeline_task:
            await self._pipeline_task.queue_frames([
                PersonDisappearedFrame(),
            ])

    async def _on_gesture(self, data: dict) -> None:
        gesture = data.get("gesture", "")
        if gesture in ("none", "closed_fist", ""):
            return
        now = time.monotonic()
        if now - self._last_gesture_time < 5.0:
            return
        self._last_gesture_time = now

        log.info("Gesture detected: %s", gesture)

        gesture_actions = {
            "wave": ("greet", "gesture_wave"),
            "thumbs_up": ("nod", "gesture_thumbs_up"),
            "point": ("think", "gesture_point"),
            "fist": ("surprised", "gesture_fist"),
        }

        if gesture in ("open_palm", "point"):
            if self._pipeline_task:
                msg = await self._say("gesture_interrupt", rephrase=False)
                await self._pipeline_task.queue_frames([
                    GestureFrame(gesture="open_palm"),
                    TTSSpeakFrame(text=msg),
                ])
            return

        if gesture in gesture_actions:
            anim, phrase_key = gesture_actions[gesture]
            await self._send_animation(anim)
            msg = await self._say(phrase_key, rephrase=False)
            await self._speak_text(msg)
            self._reset_idle_timer()

    async def _on_telegram_message(self, sender: str, text: str, chat_id: int) -> None:
        log.info("Telegram from %s: %s", sender, text)
        if self._pipeline_task:
            await self._inject_user_text(f"[Telegram from {sender}]: {text}")

    async def _classify_is_name(self, text: str) -> bool:
        if not self._llm:
            return True
        prompt = (
            f'Is "{text}" a person\'s name (first name or full name)? '
            f"Answer only 'yes' or 'no'."
        )
        url = f"{settings.ollama_url}/api/chat"
        payload = {
            "model": settings.ollama_gen_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        log.debug("classify_is_name request: text=%s", text)
        try:
            resp = await asyncio.to_thread(
                lambda: requests.post(url, json=payload, timeout=10)
            )
            resp.raise_for_status()
            data = resp.json()
            content = (data.get("message", {}) or {}).get("content", "") or ""
            log.debug("classify_is_name response: %s", content)
            return content.strip().lower().startswith("y")
        except Exception as e:
            log.warning("classify_is_name error: %s", e)
            return True

    async def _register_name(self, text: str) -> bool:
        if not self._pending_name or not self.face_service.enabled:
            return False
        is_name = await self._classify_is_name(text)
        if not is_name:
            self._name_retries += 1
            if self._name_retries >= self._max_name_retries:
                self._pending_name = False
                if self._pending_name_processor:
                    self._pending_name_processor.pending = False
            await self._inject_user_text(text, self._current_language)
            return True
        self._name_retries = 0
        name = text.strip().title()
        emb = self.face_service.last_unknown_embedding
        if emb is not None:
            ok = await asyncio.to_thread(self.face_service.register, name, emb)
            if ok:
                msg = await self._say("register_success", name=name)
            else:
                msg = await self._say("register_failure")
        else:
            msg = await self._say("register_failure")
        self._pending_name = False
        if self._pending_name_processor:
            self._pending_name_processor.pending = False
        await self._speak_text(msg)
        await self._send_listen(False)
        return True

    async def _send_state(self, **kwargs) -> None:
        await self._send_msg(serialize(StateUpdate(**kwargs)))

    async def _send_animation(self, name: str, **params) -> None:
        await self._send_msg(serialize(AnimationCmd(name=name, params=params or None)))

    async def _send_listen(self, active: bool) -> None:
        await self._send_msg(serialize(ListenIndicator(active=active)))

    # ── Proactive conversation (idle timer) ──────────────────────────

    def _reset_idle_timer(self) -> None:
        self._cancel_idle_timer()
        if not self._current_person_name:
            return
        cfg = self.face_service.get_conversation_config(self._current_person_name)
        if not cfg.get("initiate", False):
            return
        timeout = cfg.get("idle_timeout") or self._idle_timeout
        if timeout <= 0:
            return

        async def _wait_and_initiate():
            try:
                await asyncio.sleep(timeout)
                # Enforce a minimum gap between proactive initiations
                if time.monotonic() - self._last_idle_initiate < timeout * 2:
                    return
                if self._pipeline_task and self._current_person_name:
                    self._last_idle_initiate = time.monotonic()
                    prompt = t("idle_initiate", self._current_language)
                    await self._inject_user_text(prompt)
            except asyncio.CancelledError:
                pass

        self._idle_task = asyncio.create_task(_wait_and_initiate())

    def _cancel_idle_timer(self) -> None:
        if self._idle_task:
            self._idle_task.cancel()
            self._idle_task = None
