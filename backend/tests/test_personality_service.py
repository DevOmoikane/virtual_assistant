from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from virtual_assistant_be.core.config import settings
from virtual_assistant_be.services.personality_service import PersonalityService


class TestPersonalityService:
    def test_created_with_current_settings(self):
        svc = PersonalityService()
        assert svc.enabled is settings.personality_enabled

    def test_disabled_returns_original_text(self):
        original = settings.personality_enabled
        settings.personality_enabled = False
        try:
            svc = PersonalityService()
            result = svc.personalize("Hello Alice!")
            assert result == "Hello Alice!"
        finally:
            settings.personality_enabled = original

    def test_enabled_calls_llm_and_caches(self):
        original_enabled = settings.personality_enabled
        original_style = settings.personality_style
        settings.personality_enabled = True
        settings.personality_style = "cheerful"
        try:
            svc = PersonalityService()
            with patch.object(svc, "_llm") as mock_llm:
                mock_llm.generate.return_value = "Hey Alice, great to see you!"

                result = svc.personalize("Hello Alice!")
                assert result == "Hey Alice, great to see you!"
                mock_llm.generate.assert_called_once()

                result2 = svc.personalize("Hello Alice!")
                assert result2 == "Hey Alice, great to see you!"
                mock_llm.generate.assert_called_once()
        finally:
            settings.personality_enabled = original_enabled
            settings.personality_style = original_style

    def test_enabled_fallback_on_empty(self):
        original_enabled = settings.personality_enabled
        settings.personality_enabled = True
        try:
            svc = PersonalityService()
            with patch.object(svc, "_llm") as mock_llm:
                mock_llm.generate.return_value = ""

                result = svc.personalize("Hello Alice!")
                assert result == "Hello Alice!"
        finally:
            settings.personality_enabled = original_enabled

    def test_cache_differs_by_style(self):
        original_enabled = settings.personality_enabled
        settings.personality_enabled = True
        try:
            svc = PersonalityService()
            with patch.object(svc, "_llm") as mock_llm:
                mock_llm.generate.side_effect = [
                    "Hey there!",
                    "Greetings!",
                ]

                svc._style = "cheerful"
                r1 = svc.personalize("Hello!")

                svc._style = "formal"
                r2 = svc.personalize("Hello!")

                assert r1 == "Hey there!"
                assert r2 == "Greetings!"
                assert mock_llm.generate.call_count == 2
        finally:
            settings.personality_enabled = original_enabled

    def test_cache_key_includes_language(self):
        original_enabled = settings.personality_enabled
        settings.personality_enabled = True
        try:
            svc = PersonalityService()
            with patch.object(svc, "_llm") as mock_llm:
                mock_llm.generate.side_effect = [
                    "Hello!",               # personalize en → no translate call needed
                    "alegre",               # translate style to es
                    "¡Hola!",               # personalize es
                ]

                r1 = svc.personalize("Hi", language="en")
                r2 = svc.personalize("Hi", language="es")

                assert r1 == "Hello!"
                assert r2 == "¡Hola!"
                assert mock_llm.generate.call_count == 3
        finally:
            settings.personality_enabled = original_enabled

    def test_personalize_respects_language(self):
        original_enabled = settings.personality_enabled
        original_style = settings.personality_style
        settings.personality_enabled = True
        settings.personality_style = "friendly"
        try:
            svc = PersonalityService()
            with patch.object(svc, "_llm") as mock_llm:
                mock_llm.generate.side_effect = [
                    "amigable",              # translate style to es
                    "¡Hola Alice!",          # personalize in es
                ]

                result = svc.personalize("Hello Alice!", language="es")
                assert result == "¡Hola Alice!"
                system = mock_llm.generate.call_args[1]["system"]
                assert "español" in system or "Responde únicamente" in system
        finally:
            settings.personality_enabled = original_enabled
            settings.personality_style = original_style

    def test_clear_cache(self):
        original_enabled = settings.personality_enabled
        settings.personality_enabled = True
        try:
            svc = PersonalityService()
            with patch.object(svc, "_llm") as mock_llm:
                mock_llm.generate.side_effect = [
                    "Hey!",                  # personalizes "Hi" in en
                    "Hey!",                  # after cache clear, personalizes again
                ]

                svc.personalize("Hi")
                svc.clear_cache()

                svc.personalize("Hi")
                assert mock_llm.generate.call_count == 2
        finally:
            settings.personality_enabled = original_enabled


    def test_normalize_personality_calls_llm_and_caches(self):
        original_enabled = settings.personality_enabled
        settings.personality_enabled = False
        try:
            svc = PersonalityService()
            with patch.object(svc._llm, "generate", return_value="cheerful") as mock_gen:
                result = svc.normalize_personality("be more cheerful")
                assert result == "cheerful"
                mock_gen.assert_called_once()

                result2 = svc.normalize_personality("be more cheerful")
                assert result2 == "cheerful"
                mock_gen.assert_called_once()
        finally:
            settings.personality_enabled = original_enabled

    def test_normalize_personality_different_inputs_same_cache(self):
        original_enabled = settings.personality_enabled
        settings.personality_enabled = False
        try:
            svc = PersonalityService()
            with patch.object(svc._llm, "generate") as mock_gen:
                mock_gen.side_effect = ["cheerful", "formal"]
                r1 = svc.normalize_personality("be more cheerful")
                r2 = svc.normalize_personality("act formally")
                assert r1 == "cheerful"
                assert r2 == "formal"
                assert mock_gen.call_count == 2
        finally:
            settings.personality_enabled = original_enabled

    def test_normalize_personality_fallback_on_empty(self):
        original_enabled = settings.personality_enabled
        settings.personality_enabled = False
        try:
            svc = PersonalityService()
            with patch.object(svc._llm, "generate", return_value=""):
                result = svc.normalize_personality("be weird")
                assert result == "default"
        finally:
            settings.personality_enabled = original_enabled

    def test_set_style_updates_style(self):
        svc = PersonalityService()
        original = svc.style
        svc.set_style("cheerful")
        assert svc.style == "cheerful"
        svc.reset_style()
        assert svc.style == original
