from __future__ import annotations

import logging
from typing import Any

import requests

from virtual_assistant_be.core.config import settings
from virtual_assistant_be.core.translations import lang_name, translate as t
from virtual_assistant_be.timer import Timer

log = logging.getLogger(__name__)


class LlmService:
    def __init__(self) -> None:
        self.base_url = settings.ollama_url.rstrip("/")
        self.gen_model = settings.ollama_gen_model

    def generate(
        self,
        prompt: str,
        context: str | None = None,
        system: str | None = None,
    ) -> str:
        messages: list[dict[str, str]] = []

        if system:
            messages.append({"role": "system", "content": system})

        if context:
            user_msg = f"""Use the following context to answer the question.

Context:
{context}

Question: {prompt}"""
        else:
            user_msg = prompt

        messages.append({"role": "user", "content": user_msg})

        log.debug("Ollama request: model=%s messages=%s", self.gen_model, messages)
        with Timer("llm.generate"):
            try:
                resp = requests.post(
                    f"{self.base_url}/api/chat",
                    json={"model": self.gen_model, "messages": messages, "stream": False},
                    timeout=30,
                )
                resp.raise_for_status()
                content = resp.json()["message"]["content"]
                log.debug("Ollama response: %s", content)
                return content
            except requests.RequestException as e:
                log.error("Ollama generate error: %s", e)
                return ""

    def classify_intent(self, text: str) -> str:
        with Timer("llm.classify_intent"):
            system = (
                "Classify the user's intent into ONE label: "
                "greeting, question, opinion, goodbye, switch_language, change_personality, device_command, other. "
                "For switch_language also detect the target language. "
                "If switch_language: reply with the label and code separated by '|' "
                "(e.g. 'switch_language|en' or 'switch_language|es'). "
                "Otherwise reply with only the label."
            )
            result = self.generate(text.strip(), system=system).strip().lower()
        return result

    def decide_animation(self, text: str, intent: str | None = None) -> str:
        if intent is None:
            intent = self.classify_intent(text)

        mapping = {
            "greeting": "greet",
            "goodbye": "greet",
            "question": "think",
            "opinion": "listen",
            "command": "listen",
            "change_personality": "think",
        }
        return mapping.get(intent, "listen")



    def classify_device_command(self, text: str) -> dict | None:
        with Timer("llm.classify_device_command"):
            system = (
                "You extract device commands from user input. "
                "Respond with a JSON object with keys 'device', 'action', and optionally 'message'/'contact'/'command'. "
                "Devices: lights (actions: on/off/toggle), door (actions: open/close), "
                "send_message (actions: platform like telegram/discord/whatsapp, "
                "with 'contact' (recipient name) and 'message' fields), "
                "home_assistant (with 'command' field containing the raw command). "
                "If no device command is detected, respond with an empty JSON object {}."
                "Reply with ONLY the JSON, no other text."
            )
            response = self.generate(text.strip(), system=system).strip()
        if not response or response == "{}":
            return None
        try:
            import json
            cmd = json.loads(response)
            if "device" in cmd and cmd["device"]:
                return cmd
        except json.JSONDecodeError:
            log.warning("Failed to parse device command from LLM response: %s", response)
        return None

    def extract_personality(self, text: str) -> str:
        with Timer("llm.extract_personality"):
            system = (
                "Extract the personality trait the user wants the assistant to adopt. "
                "Reply with only a short adjective or phrase describing the desired personality, "
                "no explanation, no quotes."
            )
            result = self.generate(text.strip(), system=system).strip().lower().strip("\"'")
        return result

    def generate_response(self, user_input: str, context: str | None = None, language: str | None = None) -> tuple[str, str]:
        with Timer("llm.generate_response"):
            intent = self.classify_intent(user_input)
            system = t("sys_respond_in", language, lang_name=lang_name(language)) if language else None
            if context:
                response = self.generate(user_input, context=context, system=system)
            else:
                response = self.generate(user_input, system=system)
        return response, intent
