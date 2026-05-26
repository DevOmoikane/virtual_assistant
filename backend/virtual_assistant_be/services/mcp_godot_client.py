from __future__ import annotations

import logging
from typing import Any

from mcp import ClientSession
from mcp.client.sse import sse_client

log = logging.getLogger(__name__)


class McpGodotClient:
    """MCP client for the Godot bridge.

    Connects to the Godot MCP bridge server (running alongside Godot)
    and exposes methods to control the character and scene.
    """

    def __init__(self, server_url: str = "http://localhost:7801/sse") -> None:
        self._server_url = server_url
        self._session: ClientSession | None = None
        self._client_ctx: Any = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        if self._connected:
            return
        self._client_ctx = sse_client(url=self._server_url)
        streams = await self._client_ctx.__aenter__()
        read, write = streams
        self._session = await ClientSession(read, write).__aenter__()
        await self._session.initialize()
        self._connected = True
        log.info("Connected to Godot MCP bridge at %s", self._server_url)

    async def disconnect(self) -> None:
        if not self._connected:
            return
        self._connected = False
        try:
            if self._session:
                await self._session.__aexit__(None, None, None)
            if self._client_ctx:
                await self._client_ctx.__aexit__(None, None, None)
        except Exception:
            log.exception("Error disconnecting Godot MCP client")
        self._session = None
        self._client_ctx = None
        log.info("Disconnected from Godot MCP bridge")

    async def play_animation(self, name: str) -> str:
        result = await self._session.call_tool("play_animation", {"name": name})
        if result.content:
            return result.content[0].text
        return "ok"

    async def show_text(self, text: str) -> str:
        result = await self._session.call_tool("show_text", {"text": text})
        if result.content:
            return result.content[0].text
        return "ok"

    async def set_state(
        self,
        listening: bool | None = None,
        thinking: bool | None = None,
        connected: bool | None = None,
    ) -> str:
        args: dict[str, Any] = {}
        if listening is not None:
            args["listening"] = listening
        if thinking is not None:
            args["thinking"] = thinking
        if connected is not None:
            args["connected"] = connected
        result = await self._session.call_tool("set_state", args)
        if result.content:
            return result.content[0].text
        return "ok"

    async def execute_command(self, name: str, params: dict | None = None) -> str:
        args: dict[str, Any] = {"name": name}
        if params:
            import json
            args["params"] = json.dumps(params)
        result = await self._session.call_tool("execute_command", args)
        if result.content:
            return result.content[0].text
        return "ok"
