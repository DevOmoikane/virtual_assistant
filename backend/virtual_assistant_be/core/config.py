import os
from dataclasses import dataclass


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


settings = Settings()
