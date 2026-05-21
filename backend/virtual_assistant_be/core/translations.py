from __future__ import annotations

import os
import logging

import yaml

log = logging.getLogger(__name__)

_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
_TRANSLATIONS_PATH = os.path.join(_BASE_DIR, "translations.yaml")

_translations: dict[str, dict[str, str]] | None = None


def _load() -> dict[str, dict[str, str]]:
    global _translations
    if _translations is not None:
        return _translations

    try:
        with open(_TRANSLATIONS_PATH) as f:
            _translations = yaml.safe_load(f) or {}
    except FileNotFoundError:
        log.warning("Translations file not found: %s", _TRANSLATIONS_PATH)
        _translations = {}
    except Exception:
        log.warning("Failed to load translations", exc_info=True)
        _translations = {}

    return _translations


def translate(key: str, language: str = "en", **kwargs) -> str:
    data = _load()
    lang_dict = data.get(language) or data.get("en", {})
    template = lang_dict.get(key)
    if template is None:
        log.warning("Missing translation key '%s' for language '%s'", key, language)
        return key
    try:
        return template.format(**kwargs)
    except KeyError as e:
        log.warning(
            "Missing format argument '%s' for translation key '%s'", e, key
        )
        return template


t = translate
