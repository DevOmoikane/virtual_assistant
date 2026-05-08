import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import threading
import argparse
import sys

SAMPLE_RATE = 16000
CHUNK_DURATION = 2
COMMANDS = ["stop", "start", "pause", "resume", "exit", "quit", "help", "yes", "no"]

audio_buffer = np.array([], dtype=np.float32)
buffer_lock = threading.Lock()
model = None

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
    print("Processing audio chunk...")
    global audio_buffer
    
    with buffer_lock:
        if len(audio_buffer) >= SAMPLE_RATE * CHUNK_DURATION:
            chunk = audio_buffer[:SAMPLE_RATE * CHUNK_DURATION].copy()
            audio_buffer = audio_buffer[SAMPLE_RATE * CHUNK_DURATION:]
        else:
            return None
    
    return chunk

def detect_commands(device_id=None):
    global model
    model = WhisperModel("base", device="auto", compute_type="int8")
    
    print("Real-time Command Detection")
    print("=" * 50)
    list_audio_devices()
    print(f"\nListening for commands: {', '.join(COMMANDS)}")
    print("Say 'exit' or 'quit' to stop\n")
    
    try:
        stream_kwargs = {'samplerate': SAMPLE_RATE, 'channels': 1, 'callback': audio_callback}
        if device_id is not None:
            stream_kwargs['device'] = device_id
        
        with sd.InputStream(**stream_kwargs):
            while True:
                sd.sleep(CHUNK_DURATION * 1000)
                
                chunk = process_audio()
                if chunk is None:
                    print("No audio data available")
                    continue
                
                segments, info = model.transcribe(
                    chunk, 
                    language="en", 
                    beam_size=1, 
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                
                for segment in segments:
                    text = segment.text.lower().strip()
                    print(f"Recognized: '{text}' (confidence: {segment.avg_logprob:.2f})")
                    if any(cmd in text for cmd in COMMANDS):
                        print(f"✓ Command detected: '{text}'")
                        if "exit" in text or "quit" in text:
                            print("Exiting...")
                            return
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time command detection with faster-whisper")
    parser.add_argument('--device', type=int, help='Audio input device ID')
    args = parser.parse_args()
    
    detect_commands(args.device)
