import os
from dataclasses import dataclass

import yaml

_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_CONFIG_PATH = os.path.join(_BASE_DIR, "config.yaml")
_MODELS_DIR = os.path.join(_BASE_DIR, "tools", "mediapipe_models")


def _load_yaml() -> dict:
    try:
        with open(_CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def _expand(val: str) -> str:
    return os.path.expanduser(val) if isinstance(val, str) else val


@dataclass
class Settings:
    host: str = "0.0.0.0"
    port: int = 7700

    ollama_url: str = "http://localhost:11434"
    ollama_gen_model: str = "llama3.1"
    ollama_embed_model: str = "nomic-embed-text"

    chroma_url: str = "http://localhost:8000"
    chroma_collection: str = "rag_docs"

    opensearch_host: str = "localhost"
    opensearch_port: int = 9200
    opensearch_index: str = "documents"

    stt_engine: str = "faster_whisper"
    stt_model_size: str = "base"
    stt_sample_rate: int = 16000
    stt_chunk_duration: float = 3.0
    stt_device_id: int | None = None
    stt_vad_bypass: bool = False

    tts_engine: str = "piper"
    supertonic_voice: str = "M4"

    piper_default_language: str = "en"
    piper_voices: dict[str, str] | None = None

    camera_device_id: int | None = None
    camera_width: int = 640
    camera_height: int = 480

    mediapipe_models_dir: str = ""
    face_detection_model: str = ""
    gesture_recognition_model: str = ""

    rag_enabled: bool = True
    rag_engine: str = "opensearch"
    telegram_enabled: bool = True
    telegram_bot_token: str = ""

    personality_enabled: bool = False
    personality_style: str = "friendly and courteous"
    assistant_name: str = "Virtual Assistant"

    def _apply_yaml(self, cfg: dict) -> None:
        self.host = cfg.get("host", self.host)
        self.port = cfg.get("port", self.port)

        ollama = cfg.get("ollama", {})
        self.ollama_url = ollama.get("url", self.ollama_url)
        self.ollama_gen_model = ollama.get("gen_model", self.ollama_gen_model)
        self.ollama_embed_model = ollama.get("embed_model", self.ollama_embed_model)

        chroma = cfg.get("chroma", {})
        self.chroma_url = chroma.get("url", self.chroma_url)
        self.chroma_collection = chroma.get("collection", self.chroma_collection)

        os_cfg = cfg.get("opensearch", {})
        self.opensearch_host = os_cfg.get("host", self.opensearch_host)
        self.opensearch_port = os_cfg.get("port", self.opensearch_port)
        self.opensearch_index = os_cfg.get("index", self.opensearch_index)

        stt = cfg.get("stt", {})
        self.stt_engine = stt.get("engine", self.stt_engine)
        self.stt_model_size = stt.get("model_size", self.stt_model_size)
        self.stt_sample_rate = stt.get("sample_rate", self.stt_sample_rate)
        self.stt_chunk_duration = stt.get("chunk_duration", self.stt_chunk_duration)
        self.stt_device_id = stt.get("device_id", self.stt_device_id)
        self.stt_vad_bypass = stt.get("vad_bypass", self.stt_vad_bypass)

        tts_cfg = cfg.get("tts", {})
        self.tts_engine = tts_cfg.get("engine", self.tts_engine)

        supertonic_cfg = cfg.get("supertonic", {})
        self.supertonic_voice = supertonic_cfg.get("voice", self.supertonic_voice)

        piper_cfg = cfg.get("piper", {})
        self.piper_default_language = piper_cfg.get(
            "default_language", self.piper_default_language
        )
        voices = piper_cfg.get("voices", {})
        if voices:
            self.piper_voices = {k: _expand(v) for k, v in voices.items()}
        elif self.piper_voices is None:
            self.piper_voices = {
                "en": os.path.expanduser("./tools/piper/en_US-lessac-medium.onnx"),
            }

        camera = cfg.get("camera", {})
        self.camera_device_id = camera.get("device_id", self.camera_device_id)
        self.camera_width = camera.get("width", self.camera_width)
        self.camera_height = camera.get("height", self.camera_height)

        mp = cfg.get("mediapipe", {})
        mp_dir = mp.get("models_dir") or _MODELS_DIR
        self.mediapipe_models_dir = _expand(mp_dir)
        self.face_detection_model = os.path.join(
            self.mediapipe_models_dir, "blaze_face_short_range.tflite"
        )
        self.gesture_recognition_model = os.path.join(
            self.mediapipe_models_dir, "gesture_recognizer.task"
        )

        rag_cfg = cfg.get("rag", {})
        self.rag_enabled = rag_cfg.get("enabled", self.rag_enabled)
        self.rag_engine = rag_cfg.get("engine", self.rag_engine)

        telegram = cfg.get("telegram", {})
        self.telegram_enabled = telegram.get("enabled", self.telegram_enabled)
        self.telegram_bot_token = telegram.get("bot_token", self.telegram_bot_token)

        personality = cfg.get("personality", {})
        self.personality_enabled = personality.get("enabled", self.personality_enabled)
        self.personality_style = personality.get("style", self.personality_style)
        self.assistant_name = personality.get("name", self.assistant_name)

    def _apply_env(self) -> None:
        env_map = {
            "host": "HOST",
            "port": "PORT",
            "ollama_url": "OLLAMA_URL",
            "ollama_gen_model": "OLLAMA_GEN_MODEL",
            "ollama_embed_model": "OLLAMA_EMBED_MODEL",
            "chroma_url": "CHROMA_URL",
            "chroma_collection": "CHROMA_COLLECTION",
            "stt_model_size": "STT_MODEL_SIZE",
            "piper_default_language": "PIPER_DEFAULT_LANGUAGE",
            "mediapipe_models_dir": "MEDIAPIPE_MODELS_DIR",
            "telegram_enabled": "TELEGRAM_ENABLED",
            "telegram_bot_token": "TELEGRAM_BOT_TOKEN",
        }
        for attr, env_var in env_map.items():
            val = os.getenv(env_var)
            if val is not None:
                if attr == "port":
                    val = int(val)
                elif attr == "mediapipe_models_dir":
                    val = _expand(val)
                setattr(self, attr, val)

        # Recompute file paths that depend on models_dir
        if os.getenv("MEDIAPIPE_MODELS_DIR") or not self.face_detection_model:
            self.face_detection_model = os.path.join(
                self.mediapipe_models_dir, "blaze_face_short_range.tflite"
            )
            self.gesture_recognition_model = os.path.join(
                self.mediapipe_models_dir, "gesture_recognizer.task"
            )


settings = Settings()
settings._apply_yaml(_load_yaml())
# settings._apply_env()
