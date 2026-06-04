from __future__ import annotations

import asyncio
import json
import sys

import pytest

from virtual_assistant_be.core.config import MCPServerConfig
from virtual_assistant_be.services.mcp_service import MCPService

TEST_SERVER_SRC = """
import asyncio
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("test-server")

@mcp.tool(description="Echo the input back")
async def echo(message: str) -> str:
    return f"Echo: {message}"

@mcp.tool(description="Add two numbers")
async def add(a: float, b: float) -> float:
    return a + b

asyncio.run(mcp.run_stdio_async())
"""


@pytest.fixture
def test_server_script(tmp_path):
    path = tmp_path / "test_mcp_server.py"
    path.write_text(TEST_SERVER_SRC)
    return str(path)


@pytest.mark.asyncio
async def test_mcp_service_connect_and_list_tools(test_server_script):
    config = MCPServerConfig(
        name="test",
        transport="stdio",
        command=sys.executable,
        args=(test_server_script,),
    )
    service = MCPService(servers=(config,))
    assert service.enabled

    await service.start()
    try:
        tools = service.list_all_tools()
        assert len(tools) == 2

        names = [t[0] for t in tools]
        assert "test_echo" in names
        assert "test_add" in names

        echo_desc = [t[1] for t in tools if t[0] == "test_echo"][0]
        assert echo_desc == "Echo the input back"
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_mcp_service_call_tool(test_server_script):
    config = MCPServerConfig(
        name="test",
        transport="stdio",
        command=sys.executable,
        args=(test_server_script,),
    )
    service = MCPService(servers=(config,))

    await service.start()
    try:
        result = await service.call_tool("test_echo", {"message": "hello"})
        assert not result.isError
        assert result.content[0].text == "Echo: hello"

        result = await service.call_tool("test_add", {"a": 40, "b": 2})
        assert not result.isError
        assert result.content[0].text == "42.0"
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_mcp_service_unknown_tool(test_server_script):
    config = MCPServerConfig(
        name="test",
        transport="stdio",
        command=sys.executable,
        args=(test_server_script,),
    )
    service = MCPService(servers=(config,))

    await service.start()
    try:
        with pytest.raises(ValueError, match="Unknown MCP tool"):
            await service.call_tool("test_nonexistent", {})
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_mcp_service_no_servers():
    service = MCPService(servers=())
    assert not service.enabled
    assert service.list_all_tools() == []
    await service.start()
    await service.stop()
