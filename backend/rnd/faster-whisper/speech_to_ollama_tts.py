import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import requests
from piper import PiperVoice, SynthesisConfig
from piper.audio_playback import AudioPlayer
import threading
import argparse
import sys
import time

SAMPLE_RATE = 16000
CHUNK_DURATION = 3
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:latest"
PIPER_VOICE_PATH = "../../tools/piper/en_US-lessac-medium.onnx"

audio_buffer = np.array([], dtype=np.float32)
buffer_lock = threading.Lock()
whisper_model = None
piper_voice = None

def list_audio_devices():
    print("Available audio input devices:")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            print(f"  {i}: {dev['name']} (channels: {dev['max_input_channels']})")

def audio_callback(indata, frames, time_info, status):
    global audio_buffer
    if status:
        print(f"Audio status: {status}", file=sys.stderr)
    with buffer_lock:
        audio_buffer = np.append(audio_buffer, indata[:, 0].copy())

def process_audio():
    global audio_buffer
    
    with buffer_lock:
        if len(audio_buffer) >= SAMPLE_RATE * CHUNK_DURATION:
            chunk = audio_buffer[:SAMPLE_RATE * CHUNK_DURATION].copy()
            audio_buffer = audio_buffer[SAMPLE_RATE * CHUNK_DURATION:]
        else:
            return None
    
    return chunk

def query_ollama(text):
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": text, "stream": False},
            timeout=30
        )
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.exceptions.RequestException as e:
        print(f"Ollama error: {e}")
        return None

def speak_text(text):
    global piper_voice
    if piper_voice is None:
        return
    
    syn_config = SynthesisConfig()
    silence = bytes([0] * piper_voice.config.sample_rate)
    
    if AudioPlayer.is_available():
        with AudioPlayer(piper_voice.config.sample_rate) as player:
            for audio_chunk in piper_voice.synthesize(text, syn_config):
                player.play(audio_chunk.audio_int16_bytes)
            player.play(silence)

def transcribe_query_speak(device_id=None):
    global whisper_model, piper_voice
    
    whisper_model = WhisperModel("base", device="auto", compute_type="int8")
    piper_voice = PiperVoice.load(PIPER_VOICE_PATH)
    
    print("Real-time Speech → Ollama → TTS")
    print("=" * 50)
    list_audio_devices()
    print(f"\nModel: {OLLAMA_MODEL}")
    print(f"TTS Voice: {PIPER_VOICE_PATH.split('/')[-1]}")
    print("Listening... (Ctrl+C to stop)\n")
    
    try:
        stream_kwargs = {'samplerate': SAMPLE_RATE, 'channels': 1, 'callback': audio_callback}
        if device_id is not None:
            stream_kwargs['device'] = device_id
        
        with sd.InputStream(**stream_kwargs):
            while True:
                sd.sleep(CHUNK_DURATION * 1000)
                
                chunk = process_audio()
                if chunk is None:
                    continue
                
                segments, info = whisper_model.transcribe(
                    chunk, 
                    language="en", 
                    beam_size=1, 
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                
                for segment in segments:
                    text = segment.text.strip()
                    if text:
                        print(f"🎤 You: {text}")
                        print("🤔 Thinking...")
                        
                        response = query_ollama(text)
                        if response:
                            print(f"🦙 Ollama: {response}")
                            print("🔊 Speaking...")
                            speak_text(response)
                            print()
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time speech to Ollama with TTS")
    parser.add_argument('--device', type=int, help='Audio input device ID')
    parser.add_argument('--model', type=str, default=OLLAMA_MODEL, help='Ollama model to use')
    parser.add_argument('--voice', type=str, default=PIPER_VOICE_PATH, help='Piper voice model path')
    args = parser.parse_args()
    
    OLLAMA_MODEL = args.model
    PIPER_VOICE_PATH = args.voice
    transcribe_query_speak(args.device)
