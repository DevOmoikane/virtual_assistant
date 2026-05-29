#!/usr/bin/env python3
"""
Godot Simulator — TUI dashboard that replaces the Godot client for development.

Usage:
  uv run python rnd/godot_sim.py

Commands (type at the prompt):
  /text <msg>   send text to the assistant
  /shutdown     shutdown the assistant
  /quit         disconnect and exit
  /help         show this help
"""

from __future__ import annotations

import asyncio
import json
import shutil
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

# ── ANSI ──────────────────────────────────────────────────
C = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "magenta": "\033[95m",
    "cyan": "\033[96m",
    "red": "\033[91m",
    "bg_green": "\033[42m",
    "bg_yellow": "\033[43m",
    "bg_red": "\033[41m",
    "bg_blue": "\033[44m",
    "clear_line": "\033[2K",
    "clear_screen": "\033[2J\033[H",
    "hide_cursor": "\033[?25l",
    "show_cursor": "\033[?25h",
    "save_cursor": "\033[s",
    "restore_cursor": "\033[u",
}

# ── State ─────────────────────────────────────────────────
class SimState:
    connected: bool = False
    listening: bool = False
    thinking: bool = False
    person: str | None = None
    last_heard: str = ""
    last_speak: str = ""
    last_animation: str = ""
    language: str = "es"
    log: list[str] = []
    max_log: int = 100

state = SimState()


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _term_size() -> tuple[int, int]:
    c = shutil.get_terminal_size()
    return c.columns, c.lines


def _trunc(s: str, w: int) -> str:
    return s if len(s) <= w else s[: w - 1] + "…"


# ── Rendering ─────────────────────────────────────────────
def _render(cols: int, rows: int) -> str:
    lines: list[str] = []
    sep = "─" * cols

    # ── Header ──
    conn = f"{C['bg_green']}● Connected{C['reset']}" if state.connected else f"{C['bg_red']}○ Disconnected{C['reset']}"
    listen = f"{C['green']}●{C['reset']}" if state.listening else f"{C['dim']}○{C['reset']}"
    think = f"{C['yellow']}●{C['reset']}" if state.thinking else f"{C['dim']}○{C['reset']}"
    lang = state.language.upper()
    person = state.person or "—"

    lines.append(f"{C['bold']}Godot Simulator{C['reset']}   {sep[:max(0, cols-20)]}")
    lines.append(f"  {conn}    Listen: {listen}    Think: {think}")
    lines.append(f"  Person: {person}    Lang: {lang}")
    lines.append(f"  Last anim: {C['green']}{state.last_animation or '—'}{C['reset']}")
    lines.append(sep)

    # ── Heard area ──
    lines.append(f"{C['bold']}Heard:{C['reset']}")
    if state.last_heard:
        lines.append(f"  {C['cyan']}\"{_trunc(state.last_heard, cols-4)}\"{C['reset']}")
    else:
        lines.append(f"  {C['dim']}(nothing yet){C['reset']}")
    lines.append("")

    # ── Speak area ──
    lines.append(f"{C['bold']}Speak:{C['reset']}")
    if state.last_speak:
        text = _trunc(state.last_speak, cols - 4)
        lines.append(f"  {C['yellow']}\"{text}\"{C['reset']}")
    else:
        lines.append(f"  {C['dim']}(nothing yet){C['reset']}")
    lines.append(sep)

    # ── Log area ──
    used = len(lines) + 2  # +2 for input prompt area
    available = rows - used - 1
    if available > 0 and state.log:
        lines.append(f"{C['dim']}── log ──{C['reset']}")
        for entry in state.log[-available:]:
            lines.append(f"  {entry}")
        if len(state.log) > available:
            lines.append(f"  {C['dim']}... {len(state.log) - available} more{C['reset']}")
    lines.append(sep)

    return "\n".join(lines)


def _log(msg: str) -> None:
    state.log.append(f"{C['dim']}{_ts()}{C['reset']} {msg}")
    if len(state.log) > state.max_log:
        state.log = state.log[-state.max_log:]


def _redraw() -> None:
    cols, rows = _term_size()
    out = _render(cols, rows)
    sys.stdout.write(C["save_cursor"])
    sys.stdout.write(C["clear_screen"])
    sys.stdout.write(out)
    sys.stdout.write(f"\n{C['bold']}> {C['reset']}")
    sys.stdout.write(C["restore_cursor"])
    sys.stdout.flush()


# ── WebSocket message handling ────────────────────────────
def _handle_msg(data: dict) -> None:
    msg_type = data.get("type", "?")

    match msg_type:
        case "animation":
            name = data.get("name", "")
            state.last_animation = name
            _log(f"{C['green']}▸ animation{C['reset']} {name}")
        case "speak":
            text = data.get("text", "")
            state.last_speak = text
            _log(f"{C['yellow']}▸ speak{C['reset']} \"{text[:60]}{'…' if len(text) > 60 else ''}\"")
        case "listen":
            state.listening = data.get("active", False)
            _log(f"▸ listen={state.listening}")
        case "think":
            state.thinking = data.get("active", False)
            _log(f"▸ think={state.thinking}")
        case "state":
            state.connected = data.get("connected", state.connected)
            _log(f"{C['magenta']}▸ state{C['reset']} connected={state.connected}")
        case "heard":
            text = data.get("text", "")
            state.last_heard = text
            _log(f"{C['cyan']}▸ heard{C['reset']} \"{text[:60]}{'…' if len(text) > 60 else ''}\"")
        case "device":
            _log(f"{C['red']}▸ device{C['reset']} {json.dumps(data)}")
        case _:
            _log(f"▸ {msg_type} {json.dumps(data)}")

    _redraw()


# ── Loops ─────────────────────────────────────────────────
async def receive_loop(ws: websockets.WebSocketClientProtocol) -> None:
    try:
        async for raw in ws:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                _log(f"{C['red']}invalid json{C['reset']}: {raw}")
                _redraw()
                continue
            _handle_msg(data)
    except websockets.exceptions.ConnectionClosed:
        pass


async def send_loop(ws: websockets.WebSocketClientProtocol) -> None:
    loop = asyncio.get_running_loop()
    help_text = (
        f"  {C['bold']}/text <msg>{C['reset']}   send text event to the assistant\n"
        f"  {C['bold']}/shutdown{C['reset']}     shutdown the assistant\n"
        f"  {C['bold']}/quit{C['reset']}          disconnect and exit\n"
        f"  {C['bold']}/help{C['reset']}          show this help"
    )

    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        line = line.strip()
        if not line:
            _redraw()
            continue

        if line.startswith("/"):
            parts = line[1:].split(maxsplit=1)
            cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""

            match cmd:
                case "quit" | "exit":
                    _log(f"disconnecting...")
                    _redraw()
                    await ws.close()
                    break
                case "shutdown":
                    payload = json.dumps({"type": "command", "name": "shutdown"})
                    await ws.send(payload)
                    _log(f"sent: shutdown")
                    _redraw()
                case "text":
                    if args:
                        payload = json.dumps({
                            "type": "event",
                            "name": "text",
                            "params": {"text": args},
                        })
                        await ws.send(payload)
                        _log(f"sent: text \"{args}\"")
                        _redraw()
                    else:
                        _log(f"{C['red']}usage: /text <message>{C['reset']}")
                        _redraw()
                case "help":
                    print(f"\n{help_text}\n", end="")
                    _redraw()
                case _:
                    _log(f"{C['red']}unknown: /{cmd}{C['reset']}")
                    _redraw()
        else:
            payload = json.dumps({
                "type": "event",
                "name": "text",
                "params": {"text": line},
            })
            await ws.send(payload)
            _log(f"sent: text \"{line}\"")
            _redraw()


async def main() -> None:
    sys.stdout.write(C["hide_cursor"])
    try:
        cols, _ = _term_size()
        sys.stdout.write(C["clear_screen"])
        sys.stdout.write(f"{C['bold']}Godot Simulator{C['reset']}\n")
        sys.stdout.write(f"{C['dim']}connecting to {WS_URL} ...{C['reset']}\n")
        sys.stdout.flush()

        try:
            async with websockets.connect(WS_URL) as ws:
                ready = json.dumps({"type": "command", "name": "ready"})
                await ws.send(ready)
                state.connected = True
                _log("connected, sent ready")
                _redraw()

                recv_task = asyncio.create_task(receive_loop(ws))
                send_task = asyncio.create_task(send_loop(ws))

                done, pending = await asyncio.wait(
                    [recv_task, send_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for task in pending:
                    task.cancel()

        except websockets.exceptions.ConnectionClosed:
            state.connected = False
            _log(f"{C['red']}connection closed{C['reset']}")
            _redraw()
        except OSError as e:
            sys.stdout.write(C["clear_screen"])
            print(f"{C['red']}connection failed:{C['reset']} {e}")
            print(f"  is the backend running at {WS_URL}?")
    finally:
        sys.stdout.write(C["show_cursor"])
        sys.stdout.write("\n")
        sys.stdout.flush()


if __name__ == "__main__":
    asyncio.run(main())
