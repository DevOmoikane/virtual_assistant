#!/usr/bin/env python3
"""MCP TTS Server — runs on the machine with speakers.

Exposes tools for text-to-speech synthesis and playback via Supertonic.

Usage:
    uv run python tools/tts_mcp_server.py --voice M4 --port 7800 --host 0.0.0.0
"""

from __future__ import annotations

import argparse
import logging
import threading

import numpy as np
import sounddevice as sd
from mcp.server.fastmcp import FastMCP

log = logging.getLogger(__name__)

_voice_name: str = "M4"
_tts = None
_tts_lock = threading.Lock()
_stop_event = threading.Event()
_playing = False
_play_lock = threading.Lock()


def _get_tts():
    global _tts
    if _tts is None:
        with _tts_lock:
            if _tts is None:
                from supertonic import TTS
                _tts = TTS(auto_download=True)
    return _tts


mcp = FastMCP("TTS Server")


@mcp.tool()
def speak(text: str, language: str = "es") -> str:
    """Synthesize and play text through local speakers."""
    global _playing
    if not text.strip():
        return "ok"

    _stop_event.clear()
    tts = _get_tts()
    voice = tts.get_voice_style(voice_name=_voice_name)

    with _play_lock:
        _playing = True
        try:
            wav, duration = tts.synthesize(
                text,
                voice_style=voice,
                lang=language,
            )
            if _stop_event.is_set():
                return "stopped"
            audio = wav[0]
            audio_int16 = (audio * 32767).astype(np.int16)
            sd.play(audio_int16, samplerate=tts.sample_rate)
            sd.wait()
        finally:
            _playing = False
    return "ok"


@mcp.tool()
def stop() -> str:
    """Stop current speech playback immediately."""
    global _playing
    _stop_event.set()
    sd.stop()
    with _play_lock:
        _playing = False
    return "stopped"


@mcp.tool()
def is_speaking() -> bool:
    """Check if audio is currently playing."""
    return _playing


def main() -> None:
    global _voice_name
    parser = argparse.ArgumentParser(description="MCP TTS Server (Supertonic)")
    parser.add_argument("--voice", default="M4", help="Supertonic voice name")
    parser.add_argument("--port", type=int, default=7800, help="MCP SSE port")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    args = parser.parse_args()

    _voice_name = args.voice
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    mcp.settings.host = args.host
    mcp.settings.port = args.port
    from mcp.server.transport_security import TransportSecuritySettings
    mcp.settings.transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
    )
    log.info("Starting TTS MCP server on %s:%d (voice=%s)",
             args.host, args.port, args.voice)

    import uvicorn
    app = mcp.sse_app()
    uvicorn.run(app, host=args.host, port=args.port, proxy_headers=False)


if __name__ == "__main__":
    main()
