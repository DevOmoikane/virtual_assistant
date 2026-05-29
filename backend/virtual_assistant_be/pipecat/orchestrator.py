from __future__ import annotations

import asyncio
import inspect
import logging
import time
from pathlib import Path

import aiohttp

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
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
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
    ListenIndicator,
    StateUpdate,
    serialize,
)
from virtual_assistant_be.core.translations import translate as t
from virtual_assistant_be.services.camera_service import CameraService
from virtual_assistant_be.services.command_service import CommandService
from virtual_assistant_be.services.face_service import FaceService
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
from virtual_assistant_be.pipecat.godot_bridge_processor import GodotBridgeProcessor, SendFn
from virtual_assistant_be.pipecat.processors import (
    GestureProcessor,
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
        self.telegram.set_message_callback(self._on_telegram_message)

        self._pending_name: bool = False
        self._current_language: str = settings.piper_default_language
        self._current_person_name: str | None = None
        self._last_person_greeted: float = 0.0
        self._last_gesture_time: float = 0.0

        self._pipeline_task: PipelineTask | None = None
        self._runner: PipelineRunner | None = None
        self._context: LLMContext | None = None
        self._memory_processor: MemoryProcessor | None = None
        self._pending_name_processor: PendingNameProcessor | None = None
        self._transport: LocalAudioTransport | None = None
        self._aiohttp_session: aiohttp.ClientSession | None = None

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
        if self._pipeline_task:
            await self._pipeline_task.queue_frames([
                TTSSpeakFrame(text=text),
            ])

    def _lang(self, language: str | None = None) -> str:
        return language or self._current_language

    async def _say(self, key: str, **fmt_args: str) -> str:
        lang = self._lang()
        msg = t(key, lang, **fmt_args)
        if self.personality.enabled:
            msg = await asyncio.to_thread(self.personality.personalize, msg, lang)
        return msg

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

        stt = WhisperSTTService(
            model=settings.stt_model_size,
            device="auto",
            compute_type="int8",
            language="es",
        )

        llm = OLLamaLLMService(
            model=settings.ollama_gen_model,
            base_url=f"{settings.ollama_url}/v1",
        )
        self._register_llm_tools(llm)

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

        def _personalize(text: str, language: str) -> str:
            return self.personality.personalize(text, language)

        def _store_interaction(user_text: str, assistant_text: str) -> None:
            self.memory.store_interaction(user_text, assistant_text)

        self._memory_processor = MemoryProcessor(store_fn=_store_interaction)

        self._pending_name_processor = PendingNameProcessor(
            register_callback=self._register_name,
        )

        pipeline = Pipeline([
            self._transport.input(),
            stt,
            self._pending_name_processor,
            user_agg,
            RAGProcessor(retrieve_fn=_retrieve_docs),
            llm,
            PersonalityProcessor(
                personalize_fn=_personalize,
                enabled=self.personality.enabled,
            ),
            assistant_agg,
            self._memory_processor,
            godot_bridge,
            tts,
            self._transport.output(),
            GestureProcessor(),
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
        async def lights_handler(params):
            result = await asyncio.to_thread(
                self.commands.execute_lights,
                params.args.get("action", "toggle"),
            )
            await params.result_callback(result)

        async def door_handler(params):
            result = await asyncio.to_thread(
                self.commands.execute_door,
                params.args.get("action", "toggle"),
            )
            await params.result_callback(result)

        async def send_message_handler(params):
            result = await asyncio.to_thread(
                self.commands.execute_send_message,
                params.args.get("action", ""),
                params.args.get("message", ""),
                params.args.get("contact", ""),
            )
            await params.result_callback(result)

        async def home_assistant_handler(params):
            result = await asyncio.to_thread(
                self.commands.execute_home_assistant,
                params.args.get("command", ""),
            )
            await params.result_callback(result)

        llm.register_function("lights", lights_handler)
        llm.register_function("door", door_handler)
        llm.register_function("send_message", send_message_handler)
        llm.register_function("home_assistant", home_assistant_handler)

    async def _inject_user_text(self, text: str, language: str | None = None) -> None:
        await self._context.add_message({
            "role": "user",
            "content": text,
        })
        await self._pipeline_task.queue_frames([
            LLMRunFrame(),
        ])

    async def start(self) -> None:
        await self._create_pipeline()

        initial_prompt = t("system_prompt", "en")

        self._context.add_message({"role": "system", "content": initial_prompt})
        self._context.add_message({
            "role": "assistant",
            "content": t("hello_name", self._lang(), name="") or "Hello!",
        })

        async def run_task():
            await self._runner.run(self._pipeline_task)

        asyncio.create_task(run_task())
        await asyncio.sleep(0.1)

    async def stop(self) -> None:
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

    async def _on_ready(self) -> None:
        await self._send_state(connected=True)
        self.camera.start(asyncio.get_running_loop())
        await asyncio.to_thread(self.telegram.start_polling)

    async def _on_shutdown(self) -> None:
        log.info("Shutting down")
        await self._cleanup()
        await self._send_state(connected=False)

    async def _cleanup(self) -> None:
        self.camera.stop()
        await asyncio.to_thread(self.telegram.stop_polling)

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
        else:
            greeting = await self._say("hello_unknown")
            await self._send_animation("greet")
            await self._speak_text(greeting)
            self._pending_name = True
            if self._pending_name_processor:
                self._pending_name_processor.pending = True
            if self._pipeline_task:
                await self._pipeline_task.queue_frames([
                    PersonAppearedFrame(person_name=None),
                ])

    async def _on_person_disappeared(self) -> None:
        await asyncio.to_thread(self.memory.store_person_event, "disappeared")
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

        if gesture == "open_palm":
            if self._pipeline_task:
                await self._pipeline_task.queue_frames([
                    GestureFrame(gesture="open_palm"),
                ])
            return

        if gesture in gesture_actions:
            anim, phrase_key = gesture_actions[gesture]
            await self._send_animation(anim)
            msg = await self._say(phrase_key)
            await self._speak_text(msg)

    async def _on_telegram_message(self, sender: str, text: str, chat_id: int) -> None:
        log.info("Telegram from %s: %s", sender, text)
        if self._pipeline_task:
            await self._inject_user_text(f"[Telegram from {sender}]: {text}")

    async def _register_name(self, text: str) -> bool:
        if not self._pending_name or not self.face_service.enabled:
            return False
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
