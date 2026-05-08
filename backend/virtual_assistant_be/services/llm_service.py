from __future__ import annotations

import logging
from typing import Any

import requests

from virtual_assistant_be.core.config import settings

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

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json={"model": self.gen_model, "messages": messages, "stream": False},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except requests.RequestException as e:
            log.error("Ollama generate error: %s", e)
            return ""

    def classify_intent(self, text: str) -> str:
        system = (
            "You classify user input into one of: "
            "greeting, question, command, opinion, goodbye, other. "
            "Reply with only the label, no explanation."
        )
        return self.generate(text.strip(), system=system).strip().lower()

    def decide_animation(self, text: str, intent: str | None = None) -> str:
        if intent is None:
            intent = self.classify_intent(text)

        mapping = {
            "greeting": "greet",
            "goodbye": "greet",
            "question": "think",
            "opinion": "listen",
            "command": "listen",
        }
        return mapping.get(intent, "listen")

    def generate_response(self, user_input: str, context: str | None = None) -> tuple[str, str]:
        intent = self.classify_intent(user_input)
        if context:
            response = self.generate(user_input, context=context)
        else:
            response = self.generate(user_input)
        return response, intent
