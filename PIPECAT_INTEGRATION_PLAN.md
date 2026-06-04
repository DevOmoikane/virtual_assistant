# Pipecat Integration Plan: STT → LLM → TTS Pipeline Replacement

## Goal

Replace the current custom `BehaviorController` pipeline (VAD → STT → intent → LLM → TTS) with [Pipecat](https://github.com/pipecat-ai/pipecat)'s frame-based pipeline, while keeping the existing peripheral services (CameraService, FaceService, AudioService, MemoryService, RAGService, TelegramService, CommandService, PersonalityService) mostly unchanged.

**Key changes:**
- Remove MCP TTS server and MCP Godot bridge (both sidecars)
- TTS runs locally on the backend via Pipecat's `PiperTTSService`
- Godot communication happens through a custom `GodotBridgeProcessor` (a Pipecat `FrameProcessor`)
- The core conversation loop uses Pipecat's `Pipeline` + `PipelineWorker` + `PipelineRunner`

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           PipecatOrchestrator                                    │
│  (replaces BehaviorController, manages lifecycle + external events)              │
│                                                                                  │
│  ┌───────────────────────── Pipecat Pipeline ───────────────────────────────┐   │
│  │                                                                          │   │
│  │  [AudioService] ──► VAD ──► SttService.transcribe()                      │   │
│  │       └── inject TranscriptionFrame + UserStoppedSpeakingFrame ──►       │   │
│  │                                                                          │   │
│  │          │ TranscriptionFrame                                            │   │
│  │          ▼                                                               │   │
│  │  [LLMUserAggregator]  (collects user turns into context)                 │   │
│  │          │ LLMContextFrame (when turn complete)                          │   │
│  │          ▼                                                               │   │
│  │  [RAGProcessor] (custom)  — retrieves docs for questions                 │   │
│  │          │ (injects context into LLM messages)                           │   │
│  │          ▼                                                               │   │
│  │  [OLLamaLLMService]  — Ollama via OpenAI-compatible API                  │   │
│  │     registers tool functions: lights, door, send_message, home_assistant │   │
│  │          │ LLMTextFrame(s) → LLMFullResponseEndFrame                     │   │
│  │          ▼                                                               │   │
│  │  [PersonalityProcessor] (custom)  — rephrase with style                  │   │
│  │          │ LLMTextFrame (modified)                                       │   │
│  │          ▼                                                               │   │
│  │  [GodotBridgeProcessor] (custom)  — intercepts frames, forwards to Godot │   │
│  │     TranscriptionFrame ──► {"type":"heard", "text": ...}                 │   │
│  │     LLMTextFrame       ──► {"type":"speak", "text": ...}                 │   │
│  │     PersonAppearedFrame──► {"type":"animation","name":"greet"} + speak   │   │
│  │     GestureFrame       ──► {"type":"animation","name": ...}              │   │
│  │     StateUpdate        ──► {"type":"state", ...}                         │   │
│  │          │ (passes all frames downstream after forwarding)                │   │
│  │          ▼                                                               │   │
│  │  [PiperTTSService]  — local TTS playback                                 │   │
│  │          │ TTSAudioRawFrame                                              │   │
│  │          ▼                                                               │   │
│  │  [LLMAssistantAggregator]  — stores assistant response in context        │   │
│  │          ▼                                                               │   │
│  │  [MemoryProcessor] (custom)  — stores interaction to ChromaDB            │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  ┌─────────────────── External Event Injection ─────────────────────────────┐   │
│  │                                                                          │   │
│  │ CameraService (thread) ──► asyncio event loop:                           │   │
│  │   person appeared known   → inject LLM context msg + queue LLMRunFrame   │   │
│  │   person appeared unknown → set pending_name + TTSSpeakFrame("Name?")    │   │
│  │   person disappeared      → push PersonDisappearedFrame                  │   │
│  │   gesture detected        → push GestureFrame to pipeline                │   │
│  │                                                                          │   │
│  │ TelegramService (thread) ──► asyncio event loop:                         │   │
│  │   message received        → inject LLM context msg + queue LLMRunFrame   │   │
│  │                                                                          │   │
│  │ Godot WS input ──► handle_event() ──► inject into pipeline or context    │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Integration Strategy: Text-Level (Phase 1)

Instead of feeding raw audio into Pipecat (which would require PyAudio and a VADProcessor), we:

1. **Keep AudioService as-is** — captures mic via sounddevice, runs adaptive VAD, produces speech segments
2. **Keep SttService as-is** — transcribes via faster-whisper, returns `(text, language)`
3. **Inject at text level** — when a speech segment is transcribed, push `TranscriptionFrame` + `UserStoppedSpeakingFrame` into the Pipecat pipeline via `worker.queue_frames()`

This gives us Pipecat's frame routing, turn-taking, interruption handling, and TTS — without replacing proven audio capture/VAD code.

**Phase 2 option:** Migrate to full Pipecat audio pipeline (LocalAudioTransport + VADProcessor + WhisperSTTService) if PyAudio dependency is acceptable.

---

## Files to Create

### `backend/virtual_assistant_be/pipecat/__init__.py`
Empty package init.

### `backend/virtual_assistant_be/pipecat/custom_frames.py`
Custom Pipecat Frame subclasses for events that aren't part of the standard Pipecat frame set:

```python
from pipecat.frames.frames import DataFrame

class PersonAppearedFrame(DataFrame):
    name: str | None

class PersonDisappearedFrame(DataFrame):
    pass

class GestureFrame(DataFrame):
    gesture: str  # "wave", "thumbs_up", "open_palm", "point", "fist"
    x: float
    y: float

class TelegramMessageFrame(DataFrame):
    sender: str
    text: str
    chat_id: int
```

### `backend/virtual_assistant_be/pipecat/godot_bridge_processor.py`
Custom `FrameProcessor` that sits in the Pipecat pipeline and forwards relevant frames to Godot via WebSocket.

**Key behavior:**
- `TranscriptionFrame` → `{"type": "heard", "text": text}`
- `LLMTextFrame` → `{"type": "speak", "text": text}`
- `TTSSpeakFrame` → `{"type": "speak", "text": text}`  (for pre-defined greetings)
- `PersonAppearedFrame(name=known)` → `{"type": "speak", ...}` greeting by name + `{"type": "animation", "name": "greet"}`
- `PersonAppearedFrame(name=None)` → `{"type": "speak", ...}` "What's your name?" + `{"type": "listen", "active": true}`
- `PersonDisappearedFrame` → `{"type": "animation", "name": "idle"}`
- `GestureFrame(gesture="wave")` → `{"type": "animation", "name": "greet"}`
- `GestureFrame(gesture="thumbs_up")` → `{"type": "animation", "name": "nod"}`
- `GestureFrame(gesture="fist")` → `{"type": "animation", "name": "surprised"}`
- `GestureFrame(gesture="point")` → `{"type": "animation", "name": "think"}`
- `StateUpdate` → `{"type": "state", ...}`

All frames are pushed downstream after processing.

**Has `set_send_fn(send_fn: Callable)`** — injected by `ws.py` with the Godot WebSocket send function.

### `backend/virtual_assistant_be/pipecat/processors.py`
Custom FrameProcessors for peripheral services:

**`RAGProcessor`:**
- Intercepts `LLMContextFrame`
- If query looks like a question, calls `RagService.retrieve(query)`
- Injects retrieved context into LLM messages via `LLMMessagesAppendFrame`
- Passes other frames through unchanged

**`PersonalityProcessor`:**
- Intercepts `LLMTextFrame`
- Calls `PersonalityService.personalize(text, language)`
- Replaces text content with personalized version
- Passes through unchanged if personality is disabled

**`MemoryProcessor`:**
- Listens for `LLMFullResponseEndFrame` to know a turn is complete
- Collects last user utterance + assistant response
- Stores via `MemoryService.store_interaction(user_text, assistant_text)`

**`GestureProcessor`:**
- Intercepts `GestureFrame`
- `open_palm` → pushes `InterruptionFrame` upstream (stops TTS)
- `wave`/`thumbs_up` → pushes through (GodotBridgeProcessor handles animation)
- `point`/`fist` → pushes through (animation only, no LLM interruption)

### `backend/virtual_assistant_be/pipecat/orchestrator.py`
The main orchestrator that replaces `BehaviorController`. Handles:

- **Pipeline setup** — creates all services, processors, pipeline, worker, runner
- **Audio event handling** — receives speech segments from AudioService, calls STT, injects frames
- **Camera event handling** — receives person/gesture events, injects custom frames
- **Telegram event handling** — receives messages, injects LLM context
- **Godot command handling** — `ready` starts everything, `shutdown` cleans up
- **Name registration** — `_pending_name` flag, next user text triggers FaceService.register()

```python
class PipecatOrchestrator:
    def __init__(self):
        # Create all services (same as current BehaviorController)
        self.llm = LlmService()
        self.stt = SttService()
        self.rag = RagService()
        self.face_service = FaceService()
        self.memory = MemoryService()
        self.telegram = TelegramService()
        self.commands = CommandService(telegram_service=self.telegram)
        self.personality = PersonalityService()
        self.camera = CameraService(...)
        self.audio = AudioService(...)

        # Pipecat-specific
        self._pipeline: Pipeline | None = None
        self._worker: PipelineWorker | None = None
        self._runner: PipelineRunner | None = None
        self._godot_bridge: GodotBridgeProcessor | None = None
        self._send_fn: SendFn | None = None

        # State
        self._pending_name = False
        self._current_language = "es"
        self._current_person_name = None
        self._last_known_embedding = None
        self._last_gesture_time = 0.0
        self._last_person_greeted = 0.0

    def set_send_fn(self, send_fn):
        self._send_fn = send_fn
        if self._godot_bridge:
            self._godot_bridge.set_send_fn(send_fn)

    async def start(self):
        # 1. Build Pipecat pipeline
        tts = PiperTTSService(...)
        llm = OLLamaLLMService(
            model=settings.ollama_gen_model,
            base_url=f"{settings.ollama_url}/v1",
        )
        self._register_llm_tools(llm)

        context = LLMContext()
        user_agg, assistant_agg = LLMContextAggregatorPair(context, ...)

        self._godot_bridge = GodotBridgeProcessor(
            send_fn=self._send_fn,
            face_service=self.face_service,
            translations=self._say,  # for greeting generation
        )

        pipeline = Pipeline([
            user_agg,
            RAGProcessor(self.rag),
            llm,
            PersonalityProcessor(self.personality),
            self._godot_bridge,
            tts,
            assistant_agg,
            MemoryProcessor(self.memory),
        ])

        self._worker = PipelineWorker(pipeline, ...)
        self._runner = PipelineRunner()
        await self._runner.add_workers(self._worker)

        # 2. Start peripheral services
        self.camera.start(self._on_camera_event)
        self.audio.start(self._on_audio_chunk)
        self.telegram.set_message_callback(self._on_telegram_message)
        self.telegram.start_polling()

        # 3. Start pipeline runner as background task
        asyncio.create_task(self._runner.run())

        # 4. Queue initial greeting
        context.add_message({"role": "system", "content": initial_prompt})
        await self._worker.queue_frames([LLMRunFrame()])

    async def stop(self):
        await self._worker.queue_frames([EndFrame()])
        self.camera.stop()
        self.audio.stop()
        self.telegram.stop_polling()

    async def _on_audio_chunk(self, audio: np.ndarray):
        # Same as current: guard processing_text, run STT, inject into pipeline
        if self._processing_text or self._tts_busy:
            return
        text, language = await asyncio.to_thread(self.stt.transcribe, audio)
        if not text:
            return

        # Handle name registration
        if self._pending_name:
            await self._register_name(text)
            return

        # Inject into Pipecat pipeline
        frame = TranscriptionFrame(text=text, user_id="user",
                                    timestamp=str(time.time()),
                                    language=language)
        await self._worker.queue_frames([frame, UserStoppedSpeakingFrame()])

    def _register_llm_tools(self, llm: OLLamaLLMService):
        llm.register_function("lights", self.commands.execute_lights)
        llm.register_function("door", self.commands.execute_door)
        llm.register_function("send_message", self.commands.execute_send_message)
        llm.register_function("home_assistant", self.commands.execute_home_assistant)
```

---

## Files to Remove

| File | Reason |
|---|---|
| `backend/tools/tts_mcp_server.py` | TTS now runs locally via PiperTTSService |
| `backend/tools/godot_mcp_bridge.py` | Replaced by GodotBridgeProcessor in-pipeline |
| `backend/virtual_assistant_be/services/mcp_tts_client.py` | No longer needed |
| `backend/virtual_assistant_be/services/mcp_godot_client.py` | Was unused, now unneeded |

---

## Files to Modify

### `backend/virtual_assistant_be/core/behavior_controller.py`
**Major rewrite** — replace with `PipecatOrchestrator` integration. The controller still holds:
- CameraService, AudioService, TelegramService lifecycle
- FaceService integration (name registration, per-person personalities)
- Name registration state management
- Gesture debouncing

All pipeline logic (STT → LLM → TTS → animations) is handled by Pipecat processors.

### `backend/virtual_assistant_be/api/routes/ws.py`
**Changes:**
- Import and create `PipecatOrchestrator` instead of `BehaviorController`
- Inject Godot `send_json` function into `PipecatOrchestrator.set_send_fn()`
- Route `event`/`command` messages to `PipecatOrchestrator.handle_event()` / `handle_command()`
- On disconnect: call `orchestrator.stop()`

### `backend/virtual_assistant_be/core/config.py` + `backend/config.yaml`
**Remove** `mcp:` section and all MCP-related config fields:
- `mcp_tts_server_url`
- `mcp_godot_server_url`

**Optionally add** if needed for PiperTTSService voice path overrides.

### `backend/pyproject.toml`
**Remove:**
- `mcp>=1.27.1` (if no longer needed anywhere)
- `supertonic>=1.3.1` (no longer needed)
- `websockets` (if only used by MCP — check first)

**Add:**
- `pipecat-ai[whisper,piper,silero]` (the Pipecat framework with local deps)

### `godot_client/scripts/websocket.gd`
**Remove the entire MCP bridge WebSocket:**
- `bridge_socket` variable
- `_bridge_connected` tracking
- `_bridge_reconnect_timer`
- `_connect_to_bridge()` method
- `_schedule_bridge_reconnect()` method
- `_bridge_was_connected` / `_bridge_connected_sent` in `_poll_socket()`
- Bridge polling call in `_process()`
- `bridge_connected` / `bridge_disconnected` signal emissions
- `bridge_connected` / `bridge_disconnected` signal definitions

### `godot_client/scripts/lobby_scene.gd`
**Remove:**
- `_on_bridge_connected()` method
- `_on_bridge_disconnected()` method
- `websocket_node.bridge_connected.connect(...)` in `_ready()`
- `websocket_node.bridge_disconnected.connect(...)` in `_ready()`
- Any overlay updates referencing bridge state

---

## Implementation Order

### Step 1: Install Pipecat
```bash
cd backend
uv add pipecat-ai[whisper,piper,silero]
```

Verify with a quick test that `from pipecat.services.whisper.stt import WhisperSTTService` imports correctly.

### Step 2: Explore Pipecat locally
Run the simplest local pipeline example to verify end-to-end:
```python
# Minimal smoke test: import and instantiate
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.worker import PipelineWorker
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.piper.tts import PiperTTSService
```

### Step 3: Create custom frames (`pipecat/custom_frames.py`)
Define `PersonAppearedFrame`, `PersonDisappearedFrame`, `GestureFrame`, `TelegramMessageFrame`.

### Step 4: Create GodotBridgeProcessor (`pipecat/godot_bridge_processor.py`)
The most critical new file — implements the frame-to-Godot-WebSocket forwarding.

### Step 5: Create pipeline processors (`pipecat/processors.py`)
`RAGProcessor`, `PersonalityProcessor`, `MemoryProcessor`, `GestureProcessor`.

### Step 6: Create PipecatOrchestrator (`pipecat/orchestrator.py`)
The main orchestrator. Integrates with existing services.

### Step 7: Update ws.py
Wire `PipecatOrchestrator` instead of `BehaviorController`.

### Step 8: Remove MCP files
Delete `tts_mcp_server.py`, `godot_mcp_bridge.py`, `mcp_tts_client.py`, `mcp_godot_client.py`.

### Step 9: Update Godot scripts
Remove bridge WebSocket from `websocket.gd` and bridge handlers from `lobby_scene.gd`.

### Step 10: Update config
Remove MCP config from `config.yaml` and `config.py`. Update `pyproject.toml`.

### Step 11: Test
- Run `uv run pytest` to verify existing tests still pass (may need test updates)
- Run end-to-end test with real hardware
- Verify: STT → LLM → TTS (local) → Godot bridge (heard, speak, animation)
- Verify: person greeting, gesture handling, interruption
- Verify: no MCP connection errors at startup

---

## Key Design Decisions

| Decision | Choice | Reasoning |
|---|---|---|
| Audio input | Keep existing AudioService (sounddevice) | Avoid PyAudio dependency, preserve adaptive VAD + device matching |
| STT | Keep existing SttService (faster-whisper) | Inject TranscriptionFrame at text level; simpler than SegmentedSTTService+VAD |
| TTS | PiperTTSService from Pipecat | Drops supertonic+Piper dependency; uses piper-tts which is already installed |
| VAD for turn-taking | AudioService's existing VAD | UserStoppedSpeakingFrame is pushed when speech segment is complete |
| LLM tool calling | OLLamaLLMService.register_function() | Native Pipecat function calling for lights/door/telegram |
| Godot WebSocket | Single connection (port 7700 only) | Remove bridge port 7802; only backend WS remains |
| Pipeline restart | `worker.queue_frames([EndFrame()])` then recreate | Cleanest lifecycle management |

---

## Open Questions

1. **Echo**: With TTS on the backend machine, will the mic pick up speaker output? The existing `mute()`/`unmute()` logic in AudioService should handle this (it clears buffers during TTS), but physical echo may still occur depending on speaker/mic placement.

2. **PyAudio vs sounddevice**: Phase 1 uses sounddevice (existing). If we want Phase 2 (full Pipecat audio pipeline), we need `portaudio` dev headers + `pyaudio`.

3. **PiperTTSService voice paths**: Pipecat's PiperTTSService downloads models by voice ID from HuggingFace. Our current setup uses local ONNX file paths. We may need to configure `download_dir` or write a thin wrapper to point to the existing model files.

4. **Tests**: The existing behavior controller tests mock the pipeline stages. These will need updates for the Pipecat-based architecture.

5. **supertonic removal**: Confirm nothing outside the MCP TTS server depends on `supertonic`.

---

## Rollback Plan

If the integration fails or causes issues:
1. The MCP files to remove (`tts_mcp_server.py`, `godot_mcp_bridge.py`, `mcp_tts_client.py`, `mcp_godot_client.py`) are self-contained — no other code imports them
2. The old `BehaviorController` can be kept as `BehaviorController.legacy.py` for reference
3. Git revert on `ws.py`, `config.yaml`, `config.py`, `pyproject.toml`, Godot scripts
4. Remove `pipecat-ai` dependency
5. Restore old files from git
