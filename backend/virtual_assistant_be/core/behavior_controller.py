from __future__ import annotations

import logging
from typing import Callable, Awaitable

from virtual_assistant_be.core.protocol import GoEvent, GoCommand, AnimationCmd, StateUpdate, serialize

SendFn = Callable[[dict], Awaitable[None]]

log = logging.getLogger(__name__)


class BehaviorController:
    def __init__(self, send_fn: SendFn | None = None):
        self._send: SendFn | None = send_fn

    def set_send_fn(self, send_fn: SendFn) -> None:
        self._send = send_fn

    async def handle_event(self, msg: GoEvent) -> None:
        log.info("Event from Godot: %s %s", msg.name, msg.params)

    async def handle_command(self, msg: GoCommand) -> None:
        log.info("Command from Godot: %s %s", msg.name, msg.params)

        match msg.name:
            case "ready":
                await self._greet()

    async def _greet(self) -> None:
        if self._send is None:
            return
        await self._send(serialize(AnimationCmd(name="greet")))
        await self._send(serialize(StateUpdate(connected=True)))

    async def send_state(self, **kwargs) -> None:
        if self._send is None:
            return
        await self._send(serialize(StateUpdate(**kwargs)))

    async def send_animation(self, name: str, **params) -> None:
        if self._send is None:
            return
        await self._send(serialize(AnimationCmd(name=name, params=params or None)))


