import asyncio
import json
import logging
import signal
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("ws_test")

try:
    import websockets
except ImportError:
    log.error("websockets not installed. Run: pip install websockets")
    sys.exit(1)


async def test_connection():
    uri = "ws://localhost:7700/api/ws"
    log.info("Connecting to %s ...", uri)

    try:
        async with websockets.connect(uri, open_timeout=10) as ws:
            log.info("Connected!")

            # Start a background reader task
            async def reader():
                async for msg in ws:
                    data = json.loads(msg)
                    log.info("<<< %s", json.dumps(data))

            reader_task = asyncio.create_task(reader())
            await asyncio.sleep(0.5)

            # Send "ready" command
            log.info(">>> Sending 'ready' command")
            await ws.send(json.dumps({"type": "command", "name": "ready"}))

            # Wait for init sequence (services start + greeting)
            await asyncio.sleep(6)

            # Send a text command to test the full pipeline
            log.info(">>> Sending 'text' event")
            await ws.send(json.dumps({"type": "event", "name": "text", "params": {"text": "Hello, what can you do?"}}))

            # Wait for LLM response
            await asyncio.sleep(20)

            log.info("Test complete. Disconnecting...")
            reader_task.cancel()
            try:
                await reader_task
            except asyncio.CancelledError:
                pass

    except websockets.exceptions.WebSocketException as e:
        log.error("Connection failed: %s", e)
        log.error("Is the server running? Try: uv run python main.py")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(test_connection())
