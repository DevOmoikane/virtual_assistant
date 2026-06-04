from __future__ import annotations

import logging
import os
import re

import requests

from virtual_assistant_be.core.config import settings

log = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 75) -> list[str]:
    text = clean_text(text)
    tokens = text.split()
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunks.append(" ".join(tokens[start:end]))
        start += chunk_size - overlap
    return chunks


class RagService:
    def __init__(self) -> None:
        self.ollama_url = settings.ollama_url.rstrip("/")
        self.embed_model = settings.ollama_embed_model

        engine = settings.rag_engine
        if engine == "turbovec":
            from virtual_assistant_be.services.vector_stores.turbovec_store import TurbovecStore
            self._store = TurbovecStore()
            log.info("RagService using TurbovecStore")
        else:
            from virtual_assistant_be.services.vector_stores.opensearch_store import OpenSearchStore
            self._store = OpenSearchStore()
            log.info("RagService using OpenSearchStore")

    def _embed(self, texts: str | list[str]) -> list[float] | list[list[float]]:
        return self._store.embed(texts)

    def ingest(self, text: str, source: str) -> int:
        return self._store.ingest(text, source)

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        return self._store.retrieve(query, k)

    def list_documents(self) -> list[dict]:
        return self._store.list_documents()

    def delete(self, document_name: str) -> int:
        return self._store.delete(document_name)

    def ask(self, query: str) -> str:
        docs = self.retrieve(query, k=5)
        context = "\n\n".join(docs) if docs else ""

        if context:
            prompt = (
                "You are a knowledgeable chatbot assistant. "
                f"Use the following context to answer the question.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}"
            )
        else:
            prompt = (
                "You are a knowledgeable chatbot assistant. "
                f"Answer the question to the best of your knowledge.\n\n"
                f"Question: {query}"
            )

        try:
            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model": settings.ollama_gen_model, "prompt": prompt, "stream": False},
                timeout=1200,
            )
            resp.raise_for_status()
            return resp.json()["response"]
        except requests.RequestException as e:
            log.error("Ollama ask error: %s", e)
            return ""
