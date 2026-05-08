from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from virtual_assistant_be.services.telegram_service import TelegramService

log = logging.getLogger(__name__)


class CommandService:
    def __init__(self, telegram_service: TelegramService | None = None) -> None:
        self._telegram = telegram_service

    def execute_lights(self, action: str) -> dict:
        log.info("Lights turned %s (stub)", action)
        return {"device": "lights", "action": action, "status": "ok"}

    def execute_door(self, action: str) -> dict:
        log.info("Door %s (stub)", action)
        return {"device": "door", "action": action, "status": "ok"}

    def execute_send_message(self, platform: str, message: str, contact: str = "") -> dict:
        if platform == "telegram" and self._telegram:
            if contact:
                result = self._telegram.send_message_to_contact(contact, message)
            else:
                return {"device": "send_message", "action": platform, "status": "error", "message": "No contact specified"}
            if result.get("ok"):
                return {"device": "send_message", "action": platform, "status": "ok", "message": message}
            return {"device": "send_message", "action": platform, "status": "error", "message": result.get("error", "send failed")}

        log.info("Message sent via %s (stub): %s", platform, message)
        return {
            "device": "send_message",
            "action": platform,
            "status": "ok",
            "message": message,
        }

    def execute_home_assistant(self, command: str) -> dict:
        log.info("Home assistant command (stub): %s", command)
        return {"device": "home_assistant", "action": command, "status": "ok"}
