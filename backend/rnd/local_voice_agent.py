"""
Local voice agent using Pipecat — all services run locally:

    STT: Whisper (faster-whisper, local models)
    LLM: Ollama (local via OpenAI-compatible API)
    TTS: Piper (local voice synthesis)
    VAD: Silero (local)
Transport: PyAudio mic input + speaker output

Usage:
    uv sync --extra dev  # ensure all extras are installed
    uv run python rnd/local_voice_agent.py

Requires extras: whisper, piper, silero, local
Install: uv add "pipecat-ai[whisper,piper,silero,local]"
On MacOS you also need: brew install portaudio
"""

import asyncio
import sys

from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.turns.user_mute.always_user_mute_strategy import AlwaysUserMuteStrategy
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.piper.tts import PiperTTSService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

SYSTEM_PROMPT = (
    "You are a helpful assistant in a voice conversation. "
    "Your responses will be spoken aloud, so avoid emojis, bullet points, "
    "or other formatting that can't be spoken. Keep your answers brief and "
    "conversational — one or two sentences is fine."
)


async def main():
    # ── Audio transport (mic in, speaker out) ──────────────────────────
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        )
    )

    # ── Speech-to-text: faster-whisper (local) ─────────────────────────
    stt = WhisperSTTService(
        model="base",
        device="auto",
        compute_type="int8",
    )

    # ── Text-to-speech: Piper (local) ──────────────────────────────────
    tts = PiperTTSService(
        voice_id="en_US-lessac-medium",
        sample_rate=24000,
    )

    # ── LLM: Ollama (local) ────────────────────────────────────────────
    llm = OLLamaLLMService(
        model="llama3.2",
        base_url="http://localhost:11434/v1",
    )

    # ── Conversation context & aggregators ────────────────────────────
    context = LLMContext()

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            user_mute_strategies=[AlwaysUserMuteStrategy()],
        ),
    )

    # ── Pipeline ──────────────────────────────────────────────────────
    #
    #   transport.input()  → captures mic audio
    #   stt                → transcribes to text (TranscriptionFrame)
    #   user_aggregator    → captures user speech into context (upstream)
    #   llm                → generates response (LLMTextFrame)
    #   tts                → synthesises speech, pushes TTSTextFrame
    #   transport.output() → plays audio through speakers
    #   assistant_aggregator → captures assistant text into context (downstream)
    #
    pipeline = Pipeline([
        transport.input(),
        stt,
        user_aggregator,
        llm,
        tts,
        transport.output(),
        assistant_aggregator,
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    context.add_message({"role": "system", "content": SYSTEM_PROMPT})
    context.add_message({
        "role": "assistant",
        "content": "Hello! I'm ready to help. What can I do for you?",
    })
    await task.queue_frames([LLMRunFrame()])

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
