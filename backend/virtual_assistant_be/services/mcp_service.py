from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from mcp.types import CallToolResult, Tool

from virtual_assistant_be.core.config import MCPServerConfig

log = logging.getLogger(__name__)


class MCPConnection:
    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config
        self.session: ClientSession | None = None
        self.tools: list[Tool] = []
        self._stack: AsyncExitStack | None = None

    async def start(self) -> None:
        self._stack = AsyncExitStack()
        if self.config.transport == "stdio":
            params = StdioServerParameters(
                command=self.config.command,
                args=list(self.config.args),
                env=self.config.env,
            )
            streams = await self._stack.enter_async_context(stdio_client(params))
        elif self.config.transport == "sse":
            streams = await self._stack.enter_async_context(
                sse_client(self.config.url)
            )
        else:
            raise ValueError(f"Unknown MCP transport: {self.config.transport}")
        read, write = streams
        self.session = await self._stack.enter_async_context(
            ClientSession(read, write)
        )
        await self.session.initialize()
        result = await self.session.list_tools()
        self.tools = list(result.tools)
        log.info(
            "MCP '%s' connected (%s transport) with %d tools",
            self.config.name,
            self.config.transport,
            len(self.tools),
        )

    async def stop(self) -> None:
        if self._stack:
            await self._stack.aclose()
            self._stack = None
            self.session = None
            self.tools = []
            log.info("MCP '%s' disconnected", self.config.name)


class MCPService:
    def __init__(self, servers: tuple[MCPServerConfig, ...]) -> None:
        self._connections: dict[str, MCPConnection] = {}
        self._tool_map: dict[str, tuple[str, ClientSession]] = {}
        for cfg in servers:
            self._connections[cfg.name] = MCPConnection(cfg)

    async def start(self) -> None:
        results = await asyncio.gather(
            *[conn.start() for conn in self._connections.values()],
            return_exceptions=True,
        )
        for conn, result in zip(self._connections.values(), results):
            if isinstance(result, Exception):
                log.error("MCP '%s' failed to connect: %s", conn.config.name, result)
        for conn in self._connections.values():
            for tool in conn.tools:
                key = f"{conn.config.name}_{tool.name}"
                self._tool_map[key] = (conn.config.name, conn.session)
                log.debug(
                    "  MCP tool: %s (%s)", key, tool.description or ""
                )

    async def stop(self) -> None:
        tasks = [conn.stop() for conn in self._connections.values()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._connections.clear()
        self._tool_map.clear()

    @property
    def enabled(self) -> bool:
        return len(self._connections) > 0

    def list_all_tools(self) -> list[tuple[str, str, dict[str, Any], list[str]]]:
        result: list[tuple[str, str, dict[str, Any], list[str]]] = []
        for conn in self._connections.values():
            for tool in conn.tools:
                full_name = f"{conn.config.name}_{tool.name}"
                schema = tool.inputSchema or {}
                properties = schema.get("properties", {})
                required = schema.get("required", [])
                result.append(
                    (full_name, tool.description or "", properties, required)
                )
        return result

    async def call_tool(
        self, full_name: str, arguments: dict[str, Any]
    ) -> CallToolResult:
        entry = self._tool_map.get(full_name)
        if not entry:
            raise ValueError(f"Unknown MCP tool: {full_name}")
        server_name, session = entry
        tool_name = full_name[len(server_name) + 1:]
        log.info("Calling MCP tool %s/%s", server_name, tool_name)
        return await session.call_tool(tool_name, arguments)
