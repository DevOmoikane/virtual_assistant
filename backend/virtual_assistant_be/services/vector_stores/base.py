from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import requests

from virtual_assistant_be.core.config import settings

log = logging.getLogger(__name__)


class VectorStore(ABC):

    def __init__(self) -> None:
        self.ollama_url = settings.ollama_url.rstrip("/")
        self.embed_model = settings.ollama_embed_model

    def embed(self, texts: str | list[str]) -> list[float] | list[list[float]]:
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        resp = requests.post(
            f"{self.ollama_url}/api/embed",
            json={"model": self.embed_model, "input": texts},
            timeout=1200,
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
                    timeout=1200,
                )
                embeds.append(r.json()["embedding"])

        return embeds[0] if single else embeds

    @abstractmethod
    def ingest(self, text: str, source: str) -> int:
        ...

    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> list[str]:
        ...

    @abstractmethod
    def list_documents(self) -> list[dict]:
        ...

    @abstractmethod
    def delete(self, document_name: str) -> int:
        ...
