from __future__ import annotations

import hashlib
import json
import logging

from virtual_assistant_be.core.config import settings
from virtual_assistant_be.services.llm_service import LlmService
from virtual_assistant_be.timer import Timer

log = logging.getLogger(__name__)


class PersonalityService:
    def __init__(self) -> None:
        cfg = settings
        self._enabled = cfg.personality_enabled
        self._style = cfg.personality_style
        self._llm = LlmService()
        self._cache: dict[str, str] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    def personalize(self, text: str, language: str = "en") -> str:
        if not self._enabled:
            return text

        cache_key = hashlib.md5(f"{self._style}:{language}:{text}".encode()).hexdigest()
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        system = (
            f"You are a {self._style} virtual assistant. "
            f"Rephrase the following message to match your personality. "
            f"Keep the exact same meaning. Keep all named entities and numbers exactly as they are. "
            f"Respond in {language.upper()} language only. "
            f"Respond with ONLY the rephrased message, no quotes, no explanation, no extra text."
        )
        with Timer("personality.personalize"):
            result = self._llm.generate(text, system=system)

        if result:
            cleaned = result.strip().strip("\"'")
            self._cache[cache_key] = cleaned
            return cleaned

        return text

    def clear_cache(self) -> None:
        self._cache.clear()
