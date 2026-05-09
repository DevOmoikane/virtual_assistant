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

    stt_model_size: str = "base"
    stt_sample_rate: int = 16000
    stt_chunk_duration: float = 3.0
    stt_device_id: int | None = None

    piper_voice_path: str = os.getenv(
        "PIPER_VOICE_PATH",
        os.path.expanduser("./tools/piper/en_US-lessac-medium.onnx"),
    )

    camera_device_id: int | None = None
    camera_width: int = 640
    camera_height: int = 480

    mediapipe_models_dir: str = ""
    face_detection_model: str = ""
    gesture_recognition_model: str = ""

    telegram_bot_token: str = ""

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

        stt = cfg.get("stt", {})
        self.stt_model_size = stt.get("model_size", self.stt_model_size)
        self.stt_sample_rate = stt.get("sample_rate", self.stt_sample_rate)
        self.stt_chunk_duration = stt.get("chunk_duration", self.stt_chunk_duration)
        self.stt_device_id = stt.get("device_id", self.stt_device_id)

        piper = cfg.get("piper_voice_path", "")
        if piper:
            self.piper_voice_path = _expand(piper)

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

        telegram = cfg.get("telegram", {})
        self.telegram_bot_token = telegram.get("bot_token", self.telegram_bot_token)

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
            "piper_voice_path": "PIPER_VOICE_PATH",
            "mediapipe_models_dir": "MEDIAPIPE_MODELS_DIR",
            "telegram_bot_token": "TELEGRAM_BOT_TOKEN",
        }
        for attr, env_var in env_map.items():
            val = os.getenv(env_var)
            if val is not None:
                if attr == "port":
                    val = int(val)
                elif attr == "piper_voice_path":
                    val = _expand(val)
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
settings._apply_env()
