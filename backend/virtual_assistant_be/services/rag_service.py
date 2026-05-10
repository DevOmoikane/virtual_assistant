from __future__ import annotations

import logging
import uuid

import requests

from virtual_assistant_be.core.config import settings
from virtual_assistant_be.timer import Timer

log = logging.getLogger(__name__)

CHROMA_BASE = (
    f"{settings.chroma_url.rstrip('/')}"
    f"/api/v2/tenants/default_tenant/databases/default_database/collections"
)


class RagService:
    def __init__(self) -> None:
        self.chroma_url = settings.chroma_url.rstrip("/")
        self.ollama_url = settings.ollama_url.rstrip("/")
        self.embed_model = settings.ollama_embed_model
        self.collection_name = settings.chroma_collection
        self._collection_id: str | None = None

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
                json={"name": self.collection_name, "metadata": {"hnsw:space": "cosine"}},
                timeout=10,
            )
            resp.raise_for_status()
            self._collection_id = resp.json()["id"]
        except requests.RequestException as e:
            log.warning("Failed to create ChromaDB collection: %s", e)

        return self._collection_id or ""

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        words = text.split()
        chunks: list[str] = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunks.append(" ".join(words[start:end]))
            start += chunk_size - overlap
        return chunks

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        with Timer("rag.retrieve"):
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

    def ingest(self, text: str, source: str) -> int:
        collection_id = self._ensure_collection()
        if not collection_id:
            return 0

        chunks = self._chunk_text(text)
        ids = [str(uuid.uuid4()) for _ in chunks]
        metas = [{"source": source, "chunk": i} for i in range(len(chunks))]
        embeds = self._embed(chunks)

        try:
            resp = requests.post(
                f"{CHROMA_BASE}/{collection_id}/add",
                json={
                    "ids": ids,
                    "embeddings": embeds,
                    "metadatas": metas,
                    "documents": chunks,
                },
                timeout=30,
            )
            resp.raise_for_status()
            log.info("Ingested %d chunks from %s", len(chunks), source)
            return len(chunks)
        except requests.RequestException as e:
            log.error("ChromaDB ingest error: %s", e)
            return 0

    def ask(self, query: str) -> str:
        docs = self.retrieve(query)
        context = "\n\n".join(docs) if docs else ""

        if not context:
            return ""

        prompt = (
            f"Only answer using the context below. If the answer is not in the context, say you don't know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}"
        )

        try:
            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model": settings.ollama_gen_model, "prompt": prompt, "stream": False},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["response"]
        except requests.RequestException as e:
            log.error("Ollama ask error: %s", e)
            return ""
