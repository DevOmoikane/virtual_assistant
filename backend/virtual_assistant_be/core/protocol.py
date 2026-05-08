from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class AnimationCmd:
    type: str = "animation"
    name: str = ""
    params: dict[str, Any] | None = None


@dataclass
class StateUpdate:
    type: str = "state"
    connected: bool = True


@dataclass
class ListenIndicator:
    type: str = "listen"
    active: bool = False


@dataclass
class ThinkIndicator:
    type: str = "think"
    active: bool = False


@dataclass
class SpeakCmd:
    type: str = "speak"
    text: str = ""


@dataclass
class DeviceCmd:
    type: str = "device"
    device: str = ""
    action: str = ""
    status: str = ""
    message: str = ""


@dataclass
class GoEvent:
    type: str = "event"
    name: str = ""
    params: dict[str, Any] | None = None


@dataclass
class GoCommand:
    type: str = "command"
    name: str = ""
    params: dict[str, Any] | None = None


OUTGOING_TYPES = frozenset({"animation", "state", "listen", "think", "speak", "device"})
INCOMING_TYPES = frozenset({"event", "command"})


def serialize(msg: AnimationCmd | StateUpdate | ListenIndicator | ThinkIndicator | SpeakCmd | DeviceCmd) -> dict:
    return {k: v for k, v in asdict(msg).items() if v is not None}


def parse(raw: dict) -> GoEvent | GoCommand | None:
    match raw.get("type"):
        case "event":
            return GoEvent(name=raw.get("name", ""), params=raw.get("params"))
        case "command":
            return GoCommand(name=raw.get("name", ""), params=raw.get("params"))
    return None
