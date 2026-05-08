# Phase 1 Implementation Plan

## Files to Create

### 1. `backend/virtual_assistant_be/core/__init__.py`
Empty file to make `core` a Python package.

### 2. `backend/virtual_assistant_be/core/config.py`
All settings in one dataclass, loaded from env vars with sensible defaults:
- `host`, `port` (7700)
- `ollama_url`, `ollama_gen_model` (llama3.1), `ollama_embed_model` (nomic-embed-text)
- `chroma_url`, `chroma_collection`
- `stt_model_size` (base), `stt_sample_rate`, `stt_chunk_duration`, `stt_device_id`
- `piper_voice_path`
- `camera_device_id`, `camera_width`, `camera_height`

### 3. `backend/virtual_assistant_be/core/protocol.py`
Message type definitions for Godot ↔ Backend communication:

**Backend → Godot (outgoing):**
- `AnimationCmd` — `{"type": "animation", "name": "...", "params": {...}}`
- `StateUpdate` — `{"type": "state", "connected": true}`
- `ListenIndicator` — `{"type": "listen", "active": true}`
- `ThinkIndicator` — `{"type": "think", "active": true}`
- `SpeakCmd` — `{"type": "speak", "text": "..."}`

**Godot → Backend (incoming):**
- `GoEvent` — `{"type": "event", "name": "...", "params": {...}}`
- `GoCommand` — `{"type": "command", "name": "...", "params": {...}}`

Helper functions: `serialize()` and `parse()`.

### 4. `backend/virtual_assistant_be/core/behavior_controller.py`
Stub orchestrator class:
- `__init__(send_fn)` — optional send callback
- `set_send_fn(send_fn)` — set the callback to send JSON to Godot
- `handle_event(msg)` — handle incoming events from Godot
- `handle_command(msg)` — handle incoming commands from Godot
- On `"ready"` command: send greet animation + connected state

## Files to Modify

### 5. `backend/virtual_assistant_be/api/routes/ws.py`
Rewrite WebSocket endpoint:
- Accept only ONE connection at a time (reject subsequent connections with close code 403)
- Create `BehaviorController` with a send callback that pushes JSON to the WebSocket
- Parse incoming text frames via `protocol.parse()`
- Dispatch to `behavior_controller.handle_event()` or `handle_command()`
- On disconnect: send state update (connected=false), allow reconnection
- Log all messages for debugging
