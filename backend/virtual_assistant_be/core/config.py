import os
from dataclasses import dataclass

_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_MODELS_DIR = os.path.join(_BASE_DIR, "tools", "mediapipe_models")


@dataclass
class Settings:
    host: str = "0.0.0.0"
    port: int = 7700

    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_gen_model: str = os.getenv("OLLAMA_GEN_MODEL", "llama3.1")
    ollama_embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    chroma_url: str = os.getenv("CHROMA_URL", "http://localhost:8000")
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "rag_docs")

    stt_model_size: str = os.getenv("STT_MODEL_SIZE", "base")
    stt_sample_rate: int = 16000
    stt_chunk_duration: float = 3.0
    stt_device_id: int | None = None

    piper_voice_path: str = os.getenv(
        "PIPER_VOICE_PATH",
        os.path.expanduser("~/dev/omoikane/visual_assistant/backend/tools/piper/en_US-lessac-medium.onnx"),
    )

    camera_device_id: int | None = None
    camera_width: int = 640
    camera_height: int = 480

    mediapipe_models_dir: str = os.getenv("MEDIAPIPE_MODELS_DIR", _MODELS_DIR)
    face_detection_model: str = os.path.join(_MODELS_DIR, "blaze_face_short_range.tflite")
    gesture_recognition_model: str = os.path.join(_MODELS_DIR, "gesture_recognizer.task")

    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")


settings = Settings()
