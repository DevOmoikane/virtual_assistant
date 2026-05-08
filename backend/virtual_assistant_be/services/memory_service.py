from __future__ import annotations

import logging
import time
import uuid

import requests

from virtual_assistant_be.core.config import settings

log = logging.getLogger(__name__)

CHROMA_BASE = (
    f"{settings.chroma_url.rstrip('/')}"
    f"/api/v2/tenants/default_tenant/databases/default_database/collections"
)


class MemoryService:
    def __init__(self) -> None:
        self.chroma_url = settings.chroma_url.rstrip("/")
        self.ollama_url = settings.ollama_url.rstrip("/")
        self.embed_model = settings.ollama_embed_model
        self.collection_name = "memories"
        self._collection_id: str | None = None
        self._person_count: int = 0

    def _embed(self, texts: str | list[str]) -> list[float] | list[list[float]]:
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        resp = requests.post(
            f"{self.ollama_url}/api/embed",
            json={"model": self.embed_model, "input": texts},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        embeds = data.get("embeddings", [])

        if not embeds:
            embeds = []
            for t in texts:
                r = requests.post(
                    f"{self.ollama_url}/api/embeddings",
                    json={"model": self.embed_model, "prompt": t},
                    timeout=30,
                )
                embeds.append(r.json()["embedding"])

        return embeds[0] if single else embeds

    def _ensure_collection(self) -> str:
        if self._collection_id:
            return self._collection_id

        try:
            resp = requests.get(CHROMA_BASE, timeout=10)
            resp.raise_for_status()
            for c in resp.json():
                if c["name"] == self.collection_name:
                    self._collection_id = c["id"]
                    return self._collection_id
        except requests.RequestException as e:
            log.warning("ChromaDB not available: %s", e)
            return ""

        try:
            resp = requests.post(
                CHROMA_BASE,
                json={
                    "name": self.collection_name,
                    "metadata": {"hnsw:space": "cosine"},
                },
                timeout=10,
            )
            resp.raise_for_status()
            self._collection_id = resp.json()["id"]
        except requests.RequestException as e:
            log.warning("Failed to create ChromaDB collection: %s", e)

        return self._collection_id or ""

    def store_interaction(self, user_text: str, assistant_text: str) -> None:
        collection_id = self._ensure_collection()
        if not collection_id:
            return

        doc = f"User asked: {user_text}\nAssistant replied: {assistant_text}"
        embeds = self._embed(doc)

        try:
            requests.post(
                f"{CHROMA_BASE}/{collection_id}/add",
                json={
                    "ids": [str(uuid.uuid4())],
                    "embeddings": [embeds] if not isinstance(embeds[0], list) else [embeds],
                    "metadatas": [{"type": "interaction", "timestamp": time.time()}],
                    "documents": [doc],
                },
                timeout=10,
            )
        except requests.RequestException:
            log.warning("Failed to store interaction", exc_info=True)

    def store_person_event(self, event_type: str) -> None:
        collection_id = self._ensure_collection()
        if not collection_id:
            return

        if event_type == "appeared":
            self._person_count += 1
        count = self._person_count

        doc = f"A person {event_type}. Total person visits: {count}."
        embeds = self._embed(doc)

        try:
            requests.post(
                f"{CHROMA_BASE}/{collection_id}/add",
                json={
                    "ids": [str(uuid.uuid4())],
                    "embeddings": [embeds] if not isinstance(embeds[0], list) else [embeds],
                    "metadatas": [
                        {
                            "type": "person_event",
                            "event": event_type,
                            "person_count": count,
                            "timestamp": time.time(),
                        }
                    ],
                    "documents": [doc],
                },
                timeout=10,
            )
        except requests.RequestException:
            log.warning("Failed to store person event", exc_info=True)

    def get_person_count(self) -> int:
        return self._person_count

    def search(self, query: str, k: int = 3) -> list[str]:
        collection_id = self._ensure_collection()
        if not collection_id:
            return []

        q_embed = self._embed(query)
        if not isinstance(q_embed[0], list):
            q_embed = [q_embed]

        try:
            resp = requests.post(
                f"{CHROMA_BASE}/{collection_id}/query",
                json={"query_embeddings": q_embed, "n_results": k},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("documents", [[]])[0]
        except requests.RequestException as e:
            log.error("ChromaDB query error: %s", e)
            return []
