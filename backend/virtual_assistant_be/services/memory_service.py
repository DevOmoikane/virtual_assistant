from __future__ import annotations

import json
import logging
import os
import time
import uuid

import requests

from virtual_assistant_be.core.config import settings

log = logging.getLogger(__name__)

_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
_COUNTER_FILE = os.path.join(_DATA_DIR, "person_counter.json")

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
        self._person_count = self._load_person_count()
        self._person_appeared_at: float | None = None

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

        now = time.time()
        duration: float | None = None

        if event_type == "appeared":
            self._person_count += 1
            self._persist_person_count()
            self._person_appeared_at = now
        elif event_type == "disappeared" and self._person_appeared_at is not None:
            duration = round(now - self._person_appeared_at, 1)
            self._person_appeared_at = None

        count = self._person_count

        if duration is not None:
            doc = f"A person left after {duration} seconds. Total person visits: {count}."
        else:
            doc = f"A person {event_type}. Total person visits: {count}."

        meta: dict = {
            "type": "person_event",
            "event": event_type,
            "person_count": count,
            "timestamp": now,
        }
        if duration is not None:
            meta["duration_s"] = duration

        embeds = self._embed(doc)

        try:
            requests.post(
                f"{CHROMA_BASE}/{collection_id}/add",
                json={
                    "ids": [str(uuid.uuid4())],
                    "embeddings": [embeds] if not isinstance(embeds[0], list) else [embeds],
                    "metadatas": [meta],
                    "documents": [doc],
                },
                timeout=10,
            )
        except requests.RequestException:
            log.warning("Failed to store person event", exc_info=True)

    def get_person_count(self) -> int:
        return self._person_count

    def _persist_person_count(self) -> None:
        try:
            os.makedirs(_DATA_DIR, exist_ok=True)
            with open(_COUNTER_FILE, "w") as f:
                json.dump({"person_count": self._person_count}, f)
        except Exception:
            log.warning("Failed to persist person count", exc_info=True)

    def _load_person_count(self) -> int:
        try:
            with open(_COUNTER_FILE) as f:
                return json.load(f).get("person_count", 0)
        except FileNotFoundError:
            return 0
        except Exception:
            log.warning("Failed to load person count", exc_info=True)
            return 0

    def get_recent_events(self, n: int = 10) -> list[dict]:
        collection_id = self._ensure_collection()
        if not collection_id:
            return []

        try:
            resp = requests.post(
                f"{CHROMA_BASE}/{collection_id}/get",
                json={"limit": n},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            items = []
            for i in range(len(data.get("ids", []))):
                meta = (data.get("metadatas") or [{}])[i] or {}
                doc = (data.get("documents") or [""])[i] or ""
                items.append({
                    "id": data["ids"][i],
                    "type": meta.get("type", "unknown"),
                    "event": meta.get("event", ""),
                    "person_count": meta.get("person_count"),
                    "duration_s": meta.get("duration_s"),
                    "timestamp": meta.get("timestamp"),
                    "document": doc[:200],
                })
            items.sort(key=lambda x: x.get("timestamp") or 0, reverse=True)
            return items[:n]
        except requests.RequestException:
            log.warning("Failed to fetch recent events", exc_info=True)
            return []

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
