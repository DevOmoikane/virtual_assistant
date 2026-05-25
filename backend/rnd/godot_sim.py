import asyncio
import json
import sys
import time
from datetime import datetime

try:
    import websockets
except ImportError:
    print("Run this script using `uv run` from the backend directory")
    print("  uv run python rnd/godot_sim.py")
    sys.exit(1)

WS_URL = "ws://localhost:7700/api/ws"

ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_DIM = "\033[2m"
ANSI_GREEN = "\033[92m"
ANSI_YELLOW = "\033[93m"
ANSI_BLUE = "\033[94m"
ANSI_MAGENTA = "\033[95m"
ANSI_CYAN = "\033[96m"
ANSI_RED = "\033[91m"


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:12]


def _colorize(msg_type: str, data: dict) -> str:
    ts = _ts()
    match msg_type:
        case "animation":
            name = data.get("name", "")
            return (
                f"{ANSI_DIM}{ts}{ANSI_RESET} "
                f"{ANSI_GREEN}{ANSI_BOLD}▸ animation{ANSI_RESET} "
                f"{ANSI_GREEN}{name}{ANSI_RESET}"
            )
        case "speak":
            text = data.get("text", "")
            return (
                f"{ANSI_DIM}{ts}{ANSI_RESET} "
                f"{ANSI_CYAN}{ANSI_BOLD}▸ speak{ANSI_RESET}  "
                f"{ANSI_CYAN}\"{text}\"{ANSI_RESET}"
            )
        case "listen":
            active = data.get("active", False)
            icon = f"{ANSI_GREEN}●{ANSI_RESET}" if active else f"{ANSI_DIM}○{ANSI_RESET}"
            return (
                f"{ANSI_DIM}{ts}{ANSI_RESET} "
                f"{icon} listen={active}"
            )
        case "think":
            active = data.get("active", False)
            icon = f"{ANSI_YELLOW}●{ANSI_RESET}" if active else f"{ANSI_DIM}○{ANSI_RESET}"
            return (
                f"{ANSI_DIM}{ts}{ANSI_RESET} "
                f"{icon} think={active}"
            )
        case "state":
            connected = data.get("connected", False)
            return (
                f"{ANSI_DIM}{ts}{ANSI_RESET} "
                f"{ANSI_MAGENTA}▸ state{ANSI_RESET}  "
                f"connected={connected}"
            )
        case "device":
            return (
                f"{ANSI_DIM}{ts}{ANSI_RESET} "
                f"{ANSI_RED}▸ device{ANSI_RESET} "
                f"{json.dumps(data)}"
            )
        case _:
            return (
                f"{ANSI_DIM}{ts}{ANSI_RESET} "
                f"{ANSI_BOLD}▸ {msg_type}{ANSI_RESET} "
                f"{json.dumps(data)}"
            )


async def receive_loop(ws: websockets.WebSocketClientProtocol) -> None:
    try:
        async for raw in ws:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                print(f"{ANSI_RED}invalid json{ANSI_RESET}: {raw}")
                continue

            msg_type = data.get("type", "?")
            print(_colorize(msg_type, data))
    except websockets.exceptions.ConnectionClosed:
        pass


async def send_loop(ws: websockets.WebSocketClientProtocol) -> None:
    loop = asyncio.get_running_loop()
    print(
        f"{ANSI_DIM}── commands ──────────────────────────────{ANSI_RESET}\n"
        f"  {ANSI_BOLD}/text <msg>{ANSI_RESET}   {ANSI_DIM}send text event to the assistant{ANSI_RESET}\n"
        f"  {ANSI_BOLD}/shutdown{ANSI_RESET}     {ANSI_DIM}shutdown the assistant{ANSI_RESET}\n"
        f"  {ANSI_BOLD}/quit{ANSI_RESET}          {ANSI_DIM}disconnect and exit{ANSI_RESET}\n"
        f"  {ANSI_BOLD}/help{ANSI_RESET}          {ANSI_DIM}show this help{ANSI_RESET}\n"
        f"{ANSI_DIM}──────────────────────────────────────────{ANSI_RESET}"
    )

    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        line = line.strip()
        if not line:
            continue

        if line.startswith("/"):
            parts = line[1:].split(maxsplit=1)
            cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""

            match cmd:
                case "quit" | "exit":
                    print(f"{ANSI_DIM}disconnecting...{ANSI_RESET}")
                    await ws.close()
                    break
                case "shutdown":
                    payload = json.dumps({"type": "command", "name": "shutdown"})
                    await ws.send(payload)
                    print(f"{ANSI_DIM}{_ts()} sent: shutdown{ANSI_RESET}")
                case "text":
                    if args:
                        payload = json.dumps({
                            "type": "event",
                            "name": "text",
                            "params": {"text": args},
                        })
                        await ws.send(payload)
                        print(f"{ANSI_DIM}{_ts()} sent: text \"{args}\"{ANSI_RESET}")
                    else:
                        print(f"{ANSI_RED}usage: /text <message>{ANSI_RESET}")
                case "help":
                    print(
                        f"  {ANSI_BOLD}/text <msg>{ANSI_RESET}   send text event\n"
                        f"  {ANSI_BOLD}/shutdown{ANSI_RESET}     send shutdown command\n"
                        f"  {ANSI_BOLD}/quit{ANSI_RESET}          disconnect and exit\n"
                        f"  {ANSI_BOLD}/help{ANSI_RESET}          show this help"
                    )
                case _:
                    print(f"{ANSI_RED}unknown command: /{cmd}{ANSI_RESET}")
        else:
            payload = json.dumps({
                "type": "event",
                "name": "text",
                "params": {"text": line},
            })
            await ws.send(payload)
            print(f"{ANSI_DIM}{_ts()} sent: text \"{line}\"{ANSI_RESET}")


async def main() -> None:
    print(
        f"{ANSI_BOLD}Godot Simulator{ANSI_RESET}\n"
        f"{ANSI_DIM}connecting to {WS_URL} ...{ANSI_RESET}"
    )

    try:
        async with websockets.connect(WS_URL) as ws:
            ready = json.dumps({"type": "command", "name": "ready"})
            await ws.send(ready)
            print(f"{ANSI_DIM}{_ts()} sent: ready{ANSI_RESET}")

            recv_task = asyncio.create_task(receive_loop(ws))
            send_task = asyncio.create_task(send_loop(ws))

            done, pending = await asyncio.wait(
                [recv_task, send_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in pending:
                task.cancel()

    except websockets.exceptions.ConnectionClosed:
        print(f"\n{ANSI_RED}connection closed{ANSI_RESET}")
    except OSError as e:
        print(f"\n{ANSI_RED}connection failed:{ANSI_RESET} {e}")
        print(f"  is the backend running at {WS_URL}?")


if __name__ == "__main__":
    asyncio.run(main())
