#!/usr/bin/env python3
"""MCP Bridge for Godot — translates MCP tool calls into WebSocket messages.

Godot connects TO this bridge as a WebSocket client (port 7802).
The bridge also serves MCP SSE on port 7801 for external agents/LLMs.

Usage:
    uv run python tools/godot_mcp_bridge.py --ws-port 7802 --port 7801
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging

import anyio
from mcp.server.fastmcp import FastMCP
from websockets.asyncio.server import serve as ws_serve
from websockets.asyncio.server import ServerConnection as WSConnection

log = logging.getLogger(__name__)

_godot_clients: set[WSConnection] = set()
_ws_port: int = 7802

mcp = FastMCP("Godot Bridge")


async def _broadcast_to_godot(data: dict) -> None:
    """Send a JSON message to all connected Godot clients."""
    msg = json.dumps(data)
    dead: list[WSConnection] = []
    for client in _godot_clients:
        try:
            await client.send(msg)
        except Exception:
            dead.append(client)
    for d in dead:
        _godot_clients.discard(d)


async def _handle_godot_connection(ws: WSConnection) -> None:
    """Handle a Godot client connection."""
    _godot_clients.add(ws)
    log.info("Godot client connected (%d total)", len(_godot_clients))
    try:
        async for _ in ws:
            pass
    except Exception:
        pass
    finally:
        _godot_clients.discard(ws)
        log.info("Godot client disconnected (%d remaining)", len(_godot_clients))


@mcp.tool()
async def play_animation(name: str) -> str:
    """Play a character animation (idle, greet, listen, think, nod, shake, surprised, speak)."""
    await _broadcast_to_godot({"type": "animation", "name": name})
    return f"Playing animation: {name}"


@mcp.tool()
async def show_text(text: str) -> str:
    """Display subtitles or speech bubble text."""
    await _broadcast_to_godot({"type": "speak", "text": text})
    return "ok"


@mcp.tool()
async def set_state(
    listening: bool | None = None,
    thinking: bool | None = None,
    connected: bool | None = None,
) -> str:
    """Set indicator states (listening, thinking, connected)."""
    if listening is not None:
        await _broadcast_to_godot({"type": "listen", "active": listening})
    if thinking is not None:
        await _broadcast_to_godot({"type": "think", "active": thinking})
    if connected is not None:
        await _broadcast_to_godot({"type": "state", "connected": connected})
    return "ok"


@mcp.tool()
async def execute_command(name: str, params: str | None = None) -> str:
    """Send an arbitrary command to the backend via Godot.

    Params should be a JSON string of key-value pairs.
    """
    parsed_params = json.loads(params) if params else {}
    await _broadcast_to_godot({
        "type": "command",
        "name": name,
        "params": parsed_params,
    })
    return f"Command sent: {name}"


async def _run_ws_server(stop_event: asyncio.Event) -> None:
    """Run the WebSocket server for Godot clients."""
    async with ws_serve(_handle_godot_connection, "0.0.0.0", _ws_port, origins=None):
        log.info("Godot WS server listening on port %d", _ws_port)
        await stop_event.wait()


async def main() -> None:
    global _ws_port
    parser = argparse.ArgumentParser(description="MCP Bridge for Godot")
    parser.add_argument("--ws-port", type=int, default=7802,
                        help="WebSocket server port for Godot to connect to")
    parser.add_argument("--port", type=int, default=7801,
                        help="MCP SSE port")
    parser.add_argument("--host", default="0.0.0.0",
                        help="MCP bind address")
    args = parser.parse_args()

    _ws_port = args.ws_port
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    mcp.settings.host = args.host
    mcp.settings.port = args.port

    stop_event = asyncio.Event()
    async with asyncio.TaskGroup() as tg:
        tg.create_task(_run_ws_server(stop_event))
        log.info("Starting Godot MCP bridge on %s:%d (Godot WS on :%d)",
                 args.host, args.port, args.ws_port)
        await mcp.run_sse_async()


if __name__ == "__main__":
    anyio.run(main)
