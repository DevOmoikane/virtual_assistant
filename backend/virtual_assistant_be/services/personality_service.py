from __future__ import annotations

import hashlib
import json
import logging

from virtual_assistant_be.core.config import settings
from virtual_assistant_be.core.translations import lang_name, translate as t
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

    def _translate_style(self, style: str, language: str) -> str:
        if language == "en":
            return style

        cache_key = hashlib.md5(f"style_translate:{style}:{language}".encode()).hexdigest()
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        system = t("sys_translate_to", language, lang_name=lang_name(language))
        with Timer("personality.translate_style"):
            result = self._llm.generate(style, system=system)

        if result:
            translated = result.strip().strip("\"'")
            self._cache[cache_key] = translated
            return translated

        return style

    def personalize(self, text: str, language: str = "en") -> str:
        if not self._enabled:
            return text

        cache_key = hashlib.md5(f"{self._style}:{language}:{text}".encode()).hexdigest()
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        translated_style = self._translate_style(self._style, language)
        system = (
            t("sys_you_are", language, style=translated_style) + " "
            + t("sys_rephrase", language, lang_name=lang_name(language))
        )
        with Timer("personality.personalize"):
            result = self._llm.generate(text, system=system)

        if result:
            cleaned = result.strip().strip("\"'")
            self._cache[cache_key] = cleaned
            return cleaned

        return text

    @property
    def style(self) -> str:
        return self._style

    def set_style(self, style: str) -> None:
        log.info("Personality style changed: '%s' -> '%s'", self._style, style)
        self._style = style

    def reset_style(self) -> None:
        self.set_style(settings.personality_style)

    def normalize_personality(self, text: str) -> str:
        cache_key = hashlib.md5(f"normalize:{text}".encode()).hexdigest()
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        system = (
            "Extract the personality trait the user wants for the assistant. "
            "Normalize it to a short concise description (1-5 words). "
            "For example: 'be more cheerful' -> 'cheerful', "
            "'act like a pirate' -> 'like a pirate', "
            "'be formal' -> 'formal', "
            "'be funny and sarcastic' -> 'funny and sarcastic', "
            "'act like a butler' -> 'like a butler', "
            "'go back to default' or 'be normal' -> 'default'. "
            "Reply with ONLY the normalized personality, no explanation, no quotes."
        )
        with Timer("personality.normalize"):
            result = self._llm.generate(text.strip(), system=system)

        if result:
            cleaned = result.strip().strip("\"'").lower()
            self._cache[cache_key] = cleaned
            return cleaned

        return "default"

    def clear_cache(self) -> None:
        self._cache.clear()
