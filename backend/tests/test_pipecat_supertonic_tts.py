from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestSupertonicTTSService:
    def test_can_be_instantiated(self):
        with patch("supertonic.TTS") as mock_tts:
            mock_tts.return_value.get_voice_style.return_value = MagicMock()
            from virtual_assistant_be.pipecat.supertonic_tts import SupertonicTTSService
            svc = SupertonicTTSService(voice_id="M4", sample_rate=24000)
            assert svc is not None
            mock_tts.assert_called_once_with(auto_download=True)

    def test_can_generate_metrics(self):
        with patch("supertonic.TTS") as mock_tts:
            mock_tts.return_value.get_voice_style.return_value = MagicMock()
            from virtual_assistant_be.pipecat.supertonic_tts import SupertonicTTSService
            svc = SupertonicTTSService(voice_id="M4", sample_rate=24000)
            assert svc.can_generate_metrics() is True
