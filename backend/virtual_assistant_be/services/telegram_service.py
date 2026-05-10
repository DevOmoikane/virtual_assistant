from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Callable

import requests

from virtual_assistant_be.core.config import settings

log = logging.getLogger(__name__)

_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
_CONTACTS_FILE = os.path.join(_DATA_DIR, "telegram_contacts.json")


class TelegramService:
    def __init__(self) -> None:
        self._token: str = settings.telegram_bot_token.strip()
        self._enabled = settings.telegram_enabled and bool(self._token)
        self._api_base = f"https://api.telegram.org/bot{self._token}"
        self._contacts: list[dict] = self._load_contacts()
        self._polling = False
        self._poll_thread: threading.Thread | None = None
        self._last_update_id: int = 0
        self._message_callback: Callable[[str, str, int], None] | None = None
        self._lock = threading.Lock()
        if not self._enabled:
            log.warning("Telegram bot disabled — no token configured")

    # --- Contacts ---

    def add_contact(self, name: str, username: str = "", chat_id: int = 0) -> dict:
        contact = {"name": name, "username": username, "chat_id": chat_id}
        with self._lock:
            self._contacts = [c for c in self._contacts if c["name"] != name]
            self._contacts.append(contact)
            self._persist_contacts()
        log.info("Telegram contact added: %s (%s)", name, username or chat_id)
        return contact

    def remove_contact(self, name: str) -> bool:
        with self._lock:
            before = len(self._contacts)
            self._contacts = [c for c in self._contacts if c["name"] != name]
            if len(self._contacts) < before:
                self._persist_contacts()
                log.info("Telegram contact removed: %s", name)
                return True
        return False

    def get_contact(self, name: str) -> dict | None:
        for c in self._contacts:
            if c["name"] == name:
                return c
        return None

    def list_contacts(self) -> list[dict]:
        return list(self._contacts)

    def _persist_contacts(self) -> None:
        try:
            os.makedirs(_DATA_DIR, exist_ok=True)
            with open(_CONTACTS_FILE, "w") as f:
                json.dump(self._contacts, f)
        except Exception:
            log.warning("Failed to persist telegram contacts", exc_info=True)

    def _load_contacts(self) -> list[dict]:
        try:
            with open(_CONTACTS_FILE) as f:
                return json.load(f)
        except FileNotFoundError:
            return []
        except Exception:
            log.warning("Failed to load telegram contacts", exc_info=True)
            return []

    # --- Send ---

    def send_message(self, chat_id: int, text: str) -> dict:
        if not self._enabled:
            return {"ok": False, "error": "Telegram bot not configured"}
        try:
            resp = requests.post(
                f"{self._api_base}/sendMessage",
                json={"chat_id": chat_id, "text": text},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("ok"):
                msg_id = data["result"]["message_id"]
                log.info("Telegram message sent to chat %s (msg_id=%s)", chat_id, msg_id)
                return {"ok": True, "message_id": msg_id}
            return {"ok": False, "error": data.get("description", "unknown")}
        except requests.RequestException as e:
            log.warning("Failed to send telegram message: %s", e)
            return {"ok": False, "error": str(e)}

    def send_message_to_contact(self, contact_name: str, text: str) -> dict:
        contact = self.get_contact(contact_name)
        if not contact:
            return {"ok": False, "error": f"Contact '{contact_name}' not found"}
        if not contact.get("chat_id"):
            return {"ok": False, "error": f"Contact '{contact_name}' has no chat_id"}
        return self.send_message(contact["chat_id"], text)

    # --- Polling (for future receive stage) ---

    def set_message_callback(self, callback: Callable[[str, str, int], None]) -> None:
        self._message_callback = callback

    def start_polling(self) -> None:
        if not self._enabled or self._polling:
            return
        self._polling = True
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()
        log.info("Telegram polling started")

    def stop_polling(self) -> None:
        self._polling = False
        if self._poll_thread:
            self._poll_thread.join(timeout=3)
            self._poll_thread = None
        log.info("Telegram polling stopped")

    def _poll_loop(self) -> None:
        while self._polling:
            try:
                resp = requests.get(
                    f"{self._api_base}/getUpdates",
                    params={
                        "offset": self._last_update_id + 1,
                        "timeout": 10,
                        "allowed_updates": ["message"],
                    },
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()
                if data.get("ok"):
                    for update in data.get("result", []):
                        self._last_update_id = update["update_id"]
                        self._handle_update(update)
            except requests.RequestException:
                pass
            except Exception:
                log.warning("Telegram poll error", exc_info=True)

    def _handle_update(self, update: dict) -> None:
        msg = update.get("message")
        if not msg:
            return
        chat_id = msg["chat"]["id"]
        text = msg.get("text", "")
        sender = msg["from"]["first_name"] if msg.get("from") else "Unknown"
        if not text:
            return
        log.info("Telegram message from %s (chat %s): %s", sender, chat_id, text)
        if self._message_callback:
            self._message_callback(sender, text, chat_id)
