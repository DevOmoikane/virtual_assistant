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

### Behavior Controller (`backend/virtual_assistant_be/core/behavior_controller.py`)
- Orchestrator that connects all services. Lifecycle: on `ready` → starts camera, audio, telegram polling; on `shutdown` → cleans up.
- Pipeline: VAD → STT → intent classification → device command check → (RAG retrieval for questions) → LLM response → TTS playback. Full timing instrumentation via `Timer`.
- Person greeting: camera detects face → `face_service` recognizes → greets known by name / asks unknown → name registration.
- Gesture handling: wave, thumbs_up, open_palm (interrupt or listen), point, fist. 5s debounce between gestures.
- Barge-in: loud sound during TTS unmutes mic and stops speech.

### Services
| Service | File | Description |
|---|---|---|
| `CameraService` | `services/camera_service.py` | OpenCV capture + MediaPipe face detection + gesture recognition. Auto-selects USB webcam (scans indices 0-3). Startup settle (150 frames). Wave detection via hand x-oscillation. |
| `AudioService` | `services/audio_service.py` | sounddevice InputStream with adaptive VAD: dynamic noise-floor-based threshold (2.5x multiplier), 0.5s silence timeout, 0.5s min / 15s max speech. Barge-in detection (3x noise floor while muted). Mute/unmute for echo suppression. |
| `FaceService` | `services/face_service.py` | insightface `buffalo_l` model for embeddings; ChromaDB collection `faces` for storage/query (L2 distance, threshold 0.8). Register/get_embedding/recognize. |
| `SttService` | `services/stt_service.py` | faster-whisper (`base` model, int8, auto device). Returns (text, language). |
| `TtsService` | `services/tts_service.py` | Piper TTS per-language voices. Synchronous playback via sounddevice. `stop()` for interruption. |
| `LlmService` | `services/llm_service.py` | Ollama chat API. Classify intent (greeting/question/command/opinion/goodbye/other), classify device command (JSON extraction), generate response, decide animation. |
| `RagService` | `services/rag_service.py` | OpenSearch (KNN + BM25 hybrid) for document retrieval. Ingestion/chunking (300 tokens, 75 overlap). Falls back to ChromaDB via MemoryService if OpenSearch unavailable. |
| `MemoryService` | `services/memory_service.py` | ChromaDB for interaction history & person events; local JSON for person counter + visit log with stats (total visits, duration, hourly distribution). |
| `CommandService` | `services/command_service.py` | Stub implementations for lights, door, home_assistant. Telegram message sending via contact lookup. |
| `TelegramService` | `services/telegram_service.py` | Polling-based bot for receiving messages (forwards to BehaviorController). Contacts CRUD persisted as JSON. |
| `PersonalityService` | `services/personality_service.py` | LLM-based rephrasing with cached results. Controlled by `personality.enabled` flag. |

### Other backend modules
- **`timer.py`**: `Timer` context manager + `log_duration` for pipeline timing.
- **`core/translations.py`**: YAML-based i18n (`en`, `es` keys). `translate(key, language, **kwargs)`.
- **API routes** (`/api/routes/`): `health.py` (status endpoint), `rag.py` (document CRUD + upload/ingest HTML UI at `/rag`), `telegram.py` (contact CRUD), `ws.py` (WebSocket).
- **Tests**: pytest with `pytest-asyncio`. 11 test files using mocked `requests`. Run with `uv run pytest`.

### Known issues


## Godot Client (Godot 4.6, GL Compatibility)

- **Entrypoint**: `scenes/lobby_scene.tscn` (uid `c1dqqk0pnvax1`) — main 3D scene with a character (`Carlitos`), `Camera3D`, `WorldEnvironment`, `MeshInstance3D` with circle-sweep shader material.
- **Resolution**: 648×1152 viewport.
- **Input**: `ToggleConnected` action mapped to Ctrl+T.

### Scripts
| Script | Description |
|---|---|
| `scripts/websocket.gd` | `WebSocketPeer` to `ws://localhost:7700/api/ws`. Auto-reconnects (3s delay). Sends `{"type":"command","name":"ready"}` on connect. Parses `animation`, `speak`, `listen`, `think`, `state` messages → emits signals. |
| `scripts/character.gd` | `Character` class (autoload-style via `extends Node`). `AnimationPlayer` on `$UAL2_Standard/AnimationPlayer`. Maps action names to animations (greet→"Yes", listen→"Idle_FoldArms", think→"Idle_FoldArms", nod→"Yes", shake→"Idle_No", surprised→"Chest_Open", speak→"Idle_TalkingPhone"). Disconnected shader override via `disconnected_shader` material. |
| `scripts/lobby_scene.gd` | Minimal `Node3D` script — currently a stub (only handles `ToggleConnected` keybind, does nothing). |

### Structure
- **Character**: `assets/characters/carlitos.tscn` — Mixamo-rigged low-poly model with `UAL2_Standard` skeleton and `AnimationPlayer`.
- **Animations**: `assets/animations/` — imported Mixamo animations.
- **Shaders**: `shaders/materials/` (materials), `shaders/shaderlib/` (shader library).
- **Addons**: `mixamo_animation_retargeter`, `shader_library` (GodotSL).

### Character connections
`character.gd` connects to `websocket.gd` signals (`connected`, `disconnected`, `execute_action`, `speaking`, `listening`, `thinking`) but the signal connections **appear to be from Character to itself** (`connect("connected", _on_connected)`) — actual linking to the websocket node's signals is not implemented (the signals need wiring in the scene tree).

### Notable observations
- `character.gd` uses `connect()` calls in `_ready()` but these connect to its own signals (which are never emitted by the script itself) rather than the websocket node's signals.
- `lobby_scene.gd` is a stub — no logic bridges websocket events to character animations.
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

## Known Issues
- Godot `character.gd` connects to its own signals instead of the websocket node's signals — no actual animation/state wiring in lobby scene
