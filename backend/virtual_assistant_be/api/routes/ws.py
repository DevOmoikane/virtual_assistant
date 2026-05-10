from __future__ import annotations

import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from virtual_assistant_be.core.protocol import parse, serialize
from virtual_assistant_be.core.behavior_controller import BehaviorController

router = APIRouter(prefix="/api/ws", tags=["websocket"])
log = logging.getLogger(__name__)

_connected: bool = False


@router.websocket("")
async def websocket_endpoint(websocket: WebSocket):
    global _connected

    if _connected:
        log.warning("Rejecting second connection — only one Godot client allowed")
        await websocket.close(code=403)
        return

    await websocket.accept()
    _connected = True
    log.info("Godot client connected")

    controller = BehaviorController()

    async def send_to_godot(data: dict) -> None:
        try:
            await websocket.send_json(data)
        except Exception:
            log.exception("Failed to send to Godot")

    controller.set_send_fn(send_to_godot)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                log.warning("Invalid JSON from Godot: %s", raw)
                continue

            msg = parse(data)
            if msg is None:
                log.warning("Unknown message type from Godot: %s", data.get("type"))
                continue

            match msg.type:
                case "event":
                    await controller.handle_event(msg)
                case "command":
                    await controller.handle_command(msg)
    except WebSocketDisconnect:
        log.info("Godot client disconnected")
    except Exception:
        log.exception("WebSocket error")
    finally:
        await controller._cleanup()
        _connected = False
        try:
            await websocket.close()
        except Exception:
            pass
