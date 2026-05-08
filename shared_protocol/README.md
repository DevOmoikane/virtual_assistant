# Virtual Assistant Protocol

WebSocket text frames, JSON-encoded. Single connection from Godot to Python backend.

## Backend → Godot

### Animation Command
```json
{"type": "animation", "name": "greet", "params": {}}
```

| name | Animation |
|------|-----------|
| `idle` | Imported idle breathing |
| `greet` | Yes (nod/wave) |
| `listen` | Idle_FoldArms |
| `think` | Idle_FoldArms |
| `nod` | Yes |
| `shake` | Idle_No |
| `surprised` | Chest_Open |
| `speak` | Idle_TalkingPhone |

### State Update
```json
{"type": "state", "connected": true}
```

### Speak Command
```json
{"type": "speak", "text": "Hello world"}
```
Display text on screen. Audio played via TV speaker.

### Listen Indicator
```json
{"type": "listen", "active": true}
```

### Think Indicator
```json
{"type": "think", "active": true}
```

## Godot → Backend

### Ready Command
```json
{"type": "command", "name": "ready"}
```
Sent on successful WebSocket connection. Triggers backend to start camera, mic, and send greet.

### Shutdown Command
```json
{"type": "command", "name": "shutdown"}
```

### Event
```json
{"type": "event", "name": "animation_finished", "params": {"name": "greet"}}
```
