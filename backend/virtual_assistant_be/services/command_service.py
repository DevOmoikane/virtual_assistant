from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class CommandService:
    def execute_lights(self, action: str) -> dict:
        log.info("Lights turned %s (stub)", action)
        return {"device": "lights", "action": action, "status": "ok"}

    def execute_door(self, action: str) -> dict:
        log.info("Door %s (stub)", action)
        return {"device": "door", "action": action, "status": "ok"}

    def execute_send_message(self, platform: str, message: str) -> dict:
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
