from __future__ import annotations

import logging
import threading
import uuid

import numpy as np
import requests

from virtual_assistant_be.core.config import settings

log = logging.getLogger(__name__)

CHROMA_BASE = (
    f"{settings.chroma_url.rstrip('/')}"
    f"/api/v2/tenants/default_tenant/databases/default_database/collections"
)
RECOGNITION_THRESHOLD = 0.8


class FaceService:
    def __init__(self) -> None:
        self._app = None
        self._lock = threading.Lock()
        self._collection_id: str | None = None
        self._collection_name = "faces"
        self._ready = False
        self._init_error: str | None = None
        self.last_unknown_embedding: np.ndarray | None = None

        try:
            self._initialize()
        except Exception as e:
            log.warning("Face recognition disabled: %s", e)
            self._init_error = str(e)

    def _initialize(self) -> None:
        import insightface
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_thresh=0.5, det_size=(640, 640))
        self._app = app
        log.info("FaceAnalysis model loaded")

        self._ensure_collection()
        self._ready = True
        log.info("FaceService ready")

    @property
    def enabled(self) -> bool:
        return self._ready

    # --- ChromaDB collection management ---

    def _ensure_collection(self) -> None:
        try:
            resp = requests.get(CHROMA_BASE, timeout=10)
            resp.raise_for_status()
            for col in resp.json():
                if col.get("name") == self._collection_name:
                    self._collection_id = col["id"]
                    log.info("Using existing ChromaDB collection: %s", self._collection_name)
                    return

            resp = requests.post(
                CHROMA_BASE,
                json={"name": self._collection_name, "metadata": {"hnsw:space": "l2"}},
                timeout=10,
            )
            resp.raise_for_status()
            self._collection_id = resp.json()["id"]
            log.info("Created ChromaDB collection: %s", self._collection_name)
        except Exception as e:
            log.warning("Failed to setup ChromaDB collection '%s': %s", self._collection_name, e)
            raise

    def _collection_url(self) -> str:
        return f"{CHROMA_BASE}/{self._collection_id}"

    # --- Embedding extraction ---

    def get_embedding(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        if not self._ready:
            return None
        try:
            faces = self._app.get(frame_bgr)
            if not faces:
                return None
            face = faces[0]
            emb = face.normed_embedding
            if emb is None:
                return None
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            return emb.astype(np.float32)
        except Exception:
            return None

    # --- Recognition ---

    def recognize(self, embedding: np.ndarray) -> tuple[str | None, float]:
        if not self._ready or self._collection_id is None:
            self.last_unknown_embedding = embedding
            return None, 1.0

        try:
            resp = requests.post(
                f"{self._collection_url()}/query",
                json={
                    "query_embeddings": [embedding.tolist()],
                    "n_results": 1,
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            distances = data.get("distances", [[]])
            metadatas = data.get("metadatas", [[]])

            if distances and distances[0] and metadatas and metadatas[0]:
                dist = distances[0][0]
                meta = metadatas[0][0]
                if dist < RECOGNITION_THRESHOLD:
                    name = meta.get("name", "Unknown")
                    log.debug("Recognized: %s (distance=%.4f)", name, dist)
                    return name, dist

            self.last_unknown_embedding = embedding
            return None, distances[0][0] if distances and distances[0] else 1.0
        except Exception:
            self.last_unknown_embedding = embedding
            return None, 1.0

    # --- Registration ---

    def register(self, name: str, embedding: np.ndarray) -> bool:
        if not self._ready or self._collection_id is None:
            return False

        face_id = str(uuid.uuid4())
        try:
            resp = requests.post(
                f"{self._collection_url()}/add",
                json={
                    "ids": [face_id],
                    "embeddings": [embedding.tolist()],
                    "metadatas": [{"name": name}],
                },
                timeout=10,
            )
            resp.raise_for_status()
            log.info("Registered face for '%s' (id=%s)", name, face_id)
            return True
        except Exception as e:
            log.warning("Failed to register face for '%s': %s", name, e)
            return False
