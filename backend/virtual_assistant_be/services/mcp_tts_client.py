from __future__ import annotations

import logging
from typing import Any

from mcp import ClientSession
from mcp.client.sse import sse_client

log = logging.getLogger(__name__)


class McpTtsClient:
    """MCP client for a remote TTS server.

    Connects to a TTS MCP server (running on the machine with speakers)
    and exposes speak/stop/is_speaking methods.
    """

    def __init__(self, server_url: str = "http://localhost:7800/sse") -> None:
        self._server_url = server_url
        self._session: ClientSession | None = None
        self._client_ctx: Any = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        """Open persistent SSE connection and initialize session."""
        if self._connected:
            return
        self._client_ctx = sse_client(url=self._server_url)
        streams = await self._client_ctx.__aenter__()
        read, write = streams
        self._session = await ClientSession(read, write).__aenter__()
        await self._session.initialize()
        self._connected = True
        log.info("Connected to TTS MCP server at %s", self._server_url)

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
            log.exception("Error disconnecting TTS MCP client")
        self._session = None
        self._client_ctx = None
        log.info("Disconnected from TTS MCP server")

    async def speak(self, text: str, language: str | None = None) -> str:
        """Send text to TTS server for synthesis and playback.

        Returns the status response ('ok' or 'stopped').
        """
        lang = language or "es"
        args: dict[str, Any] = {"text": text, "language": lang}
        result = await self._session.call_tool("speak", args)
        if result.content:
            return result.content[0].text
        return "ok"

    async def stop(self) -> str:
        """Interrupt current speech playback."""
        result = await self._session.call_tool("stop", {})
        if result.content:
            return result.content[0].text
        return "stopped"

    async def is_speaking(self) -> bool:
        """Check if TTS server is currently playing audio."""
        result = await self._session.call_tool("is_speaking", {})
        if result.content:
            return result.content[0].text.lower() == "true"
        return False
