# AGENTS Guidelines for This Repository

This repository contains two sections, a backend made in python with its own virtual environment created using "uv", and a godot project, that is the client that connects to the backend through websockets.

## Backend (Python/FastAPI)

- **Entrypoint**: `backend/main.py` → runs uvicorn on `virtual_assistant_be.api.app:app` at `0.0.0.0:7700`
- **Python**: 3.12+, dependency manager is `uv` (`uv sync`, `uv run`)
- **Config**: `backend/config.yaml` + env var overrides — Ollama, ChromaDB, OpenSearch, Piper TTS, MediaPipe, camera, STT, RAG, Telegram, personality settings. Default language is `es`.
- **FastAPI app** (`backend/virtual_assistant_be/api/app.py`): 5 routers — WS (`/api/ws`), health, RAG CRUD, RAG tools (upload/ingest/delete), Telegram contacts CRUD. CORS wide-open.
- **WebSocket** (`backend/virtual_assistant_be/api/routes/ws.py`): single-client limit; parses incoming `event`/`command` JSON messages; sends `animation`, `speak`, `state`, `listen`, `think`, `device` messages to Godot.

### Protocol (`backend/virtual_assistant_be/core/protocol.py`)
- Outgoing: `AnimationCmd`, `StateUpdate`, `ListenIndicator`, `ThinkIndicator`, `SpeakCmd`, `DeviceCmd` — each has `type` field.
- Incoming: `GoEvent` (from Godot, e.g. text input), `GoCommand` (e.g. ready/shutdown).
- Serialize strips `None` values; parse dispatches by `type`.

### Pipecat Pipeline (`backend/virtual_assistant_be/pipecat/`)
The core STT→LLM→TTS conversation loop is handled by the **Pipecat** framework (`pipecat-ai`).

- **Orchestrator** (`pipecat/orchestrator.py` — `PipecatOrchestrator`): manages lifecycle (ready→start, shutdown→stop), handles external events (camera, telegram), injects frames into the Pipecat pipeline. Retains name registration state, gesture debouncing.
- **Pipeline** (`pipecat/processors.py`):
  ```
  transport.input() → stt → PendingNameProcessor → user_agg →
  RAGProcessor → LLM(Ollama) → PersonalityProcessor → assistant_agg →
  MemoryProcessor → GodotBridgeProcessor → TTS(Supertonic) →
  GestureProcessor → transport.output()
  ```
- **GodotBridgeProcessor** (`pipecat/godot_bridge_processor.py`): custom `FrameProcessor` that forwards `TranscriptionFrame`→`"heard"`, `LLMTextFrame`→`"speak"`, `GestureFrame`→`"animation"` etc. to Godot via WebSocket.
- **Custom frames** (`pipecat/custom_frames.py`): `PersonAppearedFrame`, `PersonDisappearedFrame`, `GestureFrame`, `TelegramMessageFrame` for external event injection.
- **TTS**: `SupertonicTTSService` (pipecat/supertonic_tts.py) wraps supertonic TTS. Configurable via `tts_engine` (piper/kokoro/xtts/supertonic).
- **External TTS messages**: `_speak_text()` pushes `TTSSpeakFrame` directly into the pipeline (no async queue). The TTS service internally serializes via `_processing_text`.
- **LLM tool calling**: `OLLamaLLMService.register_function()` for device commands (lights, door, telegram, home_assistant).
- **Barge-in/interruption**: Pipecat's `InterruptionFrame` mechanism + `GestureProcessor` (open_palm gesture).

### Services
| Service | File | Description |
|---|---|---|
| `CameraService` | `services/camera_service.py` | OpenCV capture + MediaPipe face detection + gesture recognition. Auto-selects USB webcam (scans indices 0-3). Startup settle (150 frames). Wave detection via hand x-oscillation. |
| `AudioService` | `services/audio_service.py` | sounddevice InputStream with adaptive VAD: dynamic noise-floor-based threshold (2.5x multiplier), 0.3s silence timeout, 0.5s min / 15s max speech. Mute/unmute for echo suppression (buffer cleared on both). |
| `FaceService` | `services/face_service.py` | insightface `buffalo_l` model for embeddings; ChromaDB collection `faces` for storage/query (L2 distance, threshold 0.8). Register/get_embedding/recognize. |
| `SttService` | `services/stt_service.py` | faster-whisper (`base` model, int8, auto device). Returns (text, language). |
| `TtsService` | `services/tts_service.py` | Local fallback TTS (Piper or Supertonic). Primary TTS is via SupertonicTTSService (Pipecat). |
| `LlmService` | `services/llm_service.py` | Ollama chat API. Classify intent (greeting/question/command/opinion/goodbye/other), classify device command (JSON extraction), generate response, decide animation. Used by Pipecat's custom processors. |
| `RagService` | `services/rag_service.py` | OpenSearch (KNN + BM25 hybrid) for document retrieval. Ingestion/chunking (300 tokens, 75 overlap). Falls back to ChromaDB via MemoryService if OpenSearch unavailable. |
| `MemoryService` | `services/memory_service.py` | ChromaDB for interaction history & person events; local JSON for person counter + visit log with stats (total visits, duration, hourly distribution). |
| `CommandService` | `services/command_service.py` | Stub implementations for lights, door, home_assistant. Telegram message sending via contact lookup. |
| `TelegramService` | `services/telegram_service.py` | Polling-based bot for receiving messages (forwards to PipecatOrchestrator). Contacts CRUD persisted as JSON. |
| `PersonalityService` | `services/personality_service.py` | LLM-based rephrasing with cached results. Controlled by `personality.enabled` flag. |

### Other backend modules
- **`timer.py`**: `Timer` context manager + `log_duration` for pipeline timing.
- **`core/translations.py`**: YAML-based i18n (`en`, `es` keys). `translate(key, language, **kwargs)`.
- **API routes** (`/api/routes/`): `health.py` (status endpoint), `rag.py` (document CRUD + upload/ingest HTML UI at `/rag`), `telegram.py` (contact CRUD), `ws.py` (WebSocket).
- **Tests**: pytest with `pytest-asyncio`. 11 test files using mocked `requests`. Run with `uv run pytest`.

## Godot Client (Godot 4.6, GL Compatibility)

- **Entrypoint**: `scenes/lobby_scene.tscn` (uid `c1dqqk0pnvax1`) — main 3D scene with a character (`Carlitos`), `Camera3D`, `WorldEnvironment`, `MeshInstance3D` with circle-sweep shader material.
- **Resolution**: 648×1152 viewport.
- **Input**: `ToggleConnected` action mapped to Ctrl+T.

### Scripts
| Script | Description |
|---|---|
| `scripts/websocket.gd` | Single `WebSocketPeer` client to `ws://localhost:7700/api/ws` (backend). Auto-reconnects (3s delay). Sends `{"type":"command","name":"ready"}` on connect. Parses `animation`, `speak`, `listen`, `think`, `state`, `heard` messages → emits signals. Polled in `_process()`. |
| `scripts/character.gd` | `Character` class (autoload-style via `extends Node`). `AnimationPlayer` on `$UAL2_Standard/AnimationPlayer`. Maps action names to animations (greet→"Yes", listen→"Idle_FoldArms", think→"Idle_FoldArms", nod→"Yes", shake→"Idle_No", surprised→"Chest_Open", speak→"Idle_TalkingPhone"). Disconnected shader override via `disconnected_shader` material. |
| `scripts/lobby_scene.gd` | Bridges `websocket` signals to `Carlitos` character methods (`execute_action`, `set_connected`, `set_disconnected`). Routes `speaking`/`listening`/`thinking` signals to corresponding character animations. |

### Structure
- **Character**: `assets/characters/carlitos.tscn` — Mixamo-rigged low-poly model with `UAL2_Standard` skeleton and `AnimationPlayer`.
- **Animations**: `assets/animations/` — imported Mixamo animations.
- **Shaders**: `shaders/materials/` (materials), `shaders/shaderlib/` (shader library).
- **Addons**: `mixamo_animation_retargeter`, `shader_library` (GodotSL).

### Character connections
`character.gd`'s `_ready()` no longer has self-connect calls. Signal wiring is done in `lobby_scene.gd` which connects `websocket` node signals to `Carlitos` node methods. `lobby_scene.gd` maps `speaking`/`listening`/`thinking` to `execute_action("speak")`, etc., so the character plays the corresponding animation automatically.

### Notable observations
- `character.gd` uses `connect()` calls in `_ready()` but these connect to its own signals (which are never emitted by the script itself) rather than the websocket node's signals.
- The websocket auto-reconnect and message handling is fully functional.
- The lobby scene has a `websocket` child node with `websocket.gd` script, and a `Carlitos` child node (instance of `carlitos.tscn`) that contains the `Character` script.

# Task Tracking

## Completed
- Design face recognition service architecture
- Create FaceService with embedding extraction and ChromaDB integration
- Integrate FaceService into CameraService for per-frame recognition
- Update BehaviorController to greet known persons and ask unknown
- Add insightface dependency and run test (all 91 pass)
- Fix missing `await` bug in `_register_name` (line 184)
- Add tests for known/unknown person greeting + name registration
- Implement audio echo suppression (mute mic during TTS playback)
- Add `Timer` utility and full-pipeline timing instrumentation (VAD, STT, LLM, RAG, TTS)
- Auto-select USB webcam over built-in camera (scans indices 0-3, prefers USB via sysfs)
- Webcam mic auto-selection (matches audio device to camera name via sounddevice)
- Fix VAD silence timeout latency: 1.0s → 0.5s
- AudioService accepts device_id parameter for mic selection
- Camera-to-audio device matching wired through BehaviorController
- Adaptive VAD noise floor: replaces hard-coded SILENCE_THRESHOLD with dynamic threshold (2.5x noise floor)
- Interruptible TTS: via barge-in (loud sound while muted) or open-palm gesture
- Barge-in detection: while muted, RMS checked against 3x noise-floor threshold to detect user interruption
- Gesture interruption: open palm detected by camera during TTS stops speech immediately
- Audio speech buffer cleared on mute to prevent residual echo processing
- Fix Bug 1 (echo ghost audio): `audio_service.py:unmute()` clears `_buffer` and `_raw_buffer`
- Fix Bug 2 (interrupt loses follow-up): `_interrupt_speech()` sets `self._processing_text = False`
- Fix Bug 3 (VAD latency): `SILENCE_TIMEOUT` lowered from 0.5s → 0.3s
- Update `behavior_controller.py` to use MCP TTS client with fallback to local TTS
- Fix Godot signal wiring: remove broken self-connects in `character.gd`, wire via `lobby_scene.gd`
- Fix `WebSocketServer` not found in Godot 4.6 — replace with second `WebSocketPeer` client to bridge WS

## Completed (Pipecat Integration)
- Install `pipecat-ai[kokoro,piper,silero,whisper]>=1.2.1` dependency
- Create custom frame types (`pipecat/custom_frames.py`): PersonAppearedFrame, PersonDisappearedFrame, GestureFrame, TelegramMessageFrame
- Create GodotBridgeProcessor (`pipecat/godot_bridge_processor.py`): async send, handles TranscriptionFrame→"heard", LLMTextFrame→"speak", GestureFrame→"animation" to Godot
- Create pipeline processors (`pipecat/processors.py`): RAGProcessor, PersonalityProcessor, MemoryProcessor, GestureProcessor
- Create PipecatOrchestrator (`pipecat/orchestrator.py`): lifecycle management, external event injection, frame routing via PipelineTask/PipelineRunner
- Create SupertonicTTSService (`pipecat/supertonic_tts.py`): custom Pipecat TTS service wrapping supertonic
- Update `ws.py` to wire PipecatOrchestrator instead of BehaviorController (sync send fn via asyncio.ensure_future)
- Remove MCP sidecars: deleted `tts_mcp_server.py`, `godot_mcp_bridge.py`, `mcp_tts_client.py`, `mcp_godot_client.py`
- Remove bridge WebSocket from Godot: simplified `websocket.gd` (single ws connection), cleaned `lobby_scene.gd`, removed `debug_overlay.gd` bridge status
- Remove MCP config from `config.yaml`/`config.py`
- Update `pyproject.toml` (removed `mcp`, `websockets`; kept `pipecat-ai`, `supertonic`)
- Sync Godot script changes to remote server (10.73.19.117)
- Add `FasterWhisperSTTService` for faster-whisper integration (pipecat/faster_whisper_stt.py)
- Fix TTS queue: remove async queue/worker/event mechanism, push `TTSSpeakFrame` directly (TTS already serializes internally via `_processing_text`)

## Deployment

- Whenever you modify files in `godot_client/`, sync them to the remote server at `israel@10.73.19.117:/home/israel/dev/omoikane/visual_assistant` (same relative path as the workspace). That machine runs the Godot app. Use `rsync -avz --delete` or `scp` to sync.

## Known Issues
- Tests are slow (~225s for behavior controller tests) because `CameraService.__init__` scans 4 camera indices, each taking ~2s to timeout when no camera is available. Consider adding a faster fail mechanism for CI/test environments.

## Current Issues (WIP)
- TTS only speaks one message — second `TTSSpeakFrame` fails silently. Suspicions:
  - `_audio_context_task_handler` background task pushes `TTSStoppedFrame` asynchronously — timing issue reaching `GestureProcessor`
  - `_processing_text` flag may prevent second TTSSpeakFrame from being processed
  - Pipeta TTS internal context management may interfere across sequential TTSSpeakFrames
  - Try: bypass TTS queue, push directly (done — testing needed)
- Next step: deploy and test if direct push fixes the issue. If not, add frame-level logging through TTS→GestureProcessor path.

