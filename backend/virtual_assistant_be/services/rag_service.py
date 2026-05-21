from __future__ import annotations

import json
import logging
import re
import uuid

import numpy as np
import requests
from opensearchpy import OpenSearch, helpers

from virtual_assistant_be.core.config import settings
from virtual_assistant_be.timer import Timer

log = logging.getLogger(__name__)

_INDEX_CONFIG = {
    "settings": {
        "index": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "knn": True,
        }
    },
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "embedding": {
                "type": "knn_vector",
                "dimension": 768,
                "method": {
                    "engine": "faiss",
                    "space_type": "l2",
                    "name": "hnsw",
                    "parameters": {},
                },
            },
            "document_name": {"type": "keyword"},
            "chunk_index": {"type": "integer"},
        }
    },
}


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
        self._client: OpenSearch | None = None
        self._index_name = settings.opensearch_index

    # ------------------------------------------------------------------
    # OpenSearch client
    # ------------------------------------------------------------------

    def _get_client(self) -> OpenSearch | None:
        if self._client is not None:
            return self._client
        try:
            self._client = OpenSearch(
                hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}],
                http_compress=True,
                timeout=30,
                max_retries=3,
                retry_on_timeout=True,
            )
            info = self._client.info()
            log.info("Connected to OpenSearch: %s", info["version"]["number"])
        except Exception:
            log.warning("OpenSearch not available, RAG will be degraded")
            self._client = None
        return self._client

    def _ensure_index(self) -> bool:
        client = self._get_client()
        if client is None:
            return False
        if not client.indices.exists(index=self._index_name):
            try:
                client.indices.create(index=self._index_name, body=_INDEX_CONFIG)
                log.info("Created OpenSearch index '%s'", self._index_name)
            except Exception as e:
                log.warning("Failed to create index '%s': %s", self._index_name, e)
                return False
        return True

    # ------------------------------------------------------------------
    # Embeddings (Ollama nomic-embed-text)
    # ------------------------------------------------------------------

    def _embed(self, texts: str | list[str]) -> list[float] | list[list[float]]:
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

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, text: str, source: str) -> int:
        client = self._get_client()
        if client is None:
            return 0
        if not self._ensure_index():
            return 0

        chunks = chunk_text(text)
        if not chunks:
            return 0

        embeds = self._embed(chunks)
        if not embeds:
            return 0

        actions = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeds)):
            doc_id = f"{source}_{i}"
            emb_list = emb if isinstance(emb, list) else emb.tolist()
            actions.append({
                "_index": self._index_name,
                "_id": doc_id,
                "_source": {
                    "text": chunk,
                    "embedding": emb_list,
                    "document_name": source,
                    "chunk_index": i,
                },
            })

        success, errors = helpers.bulk(client, actions)
        log.info("Ingested %d chunks from '%s' (%d errors)", success, source, len(errors))
        return success

    # ------------------------------------------------------------------
    # Retrieval (hybrid: BM25 + KNN)
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        with Timer("rag.retrieve"):
            client = self._get_client()
            if client is None:
                return self._fallback_retrieve(query, k)

            if not self._ensure_index():
                return []

            q_embed = self._embed(query)
            if isinstance(q_embed, list) and len(q_embed) > 0 and not isinstance(q_embed[0], (int, float)):
                q_embed = q_embed[0]
            if isinstance(q_embed, np.ndarray):
                q_embed = q_embed.tolist()

            # BM25 text search
            text_body = {
                "query": {"match": {"text": query}},
                "size": k,
            }

            # KNN vector search
            knn_body = {
                "query": {"knn": {"embedding": {"vector": q_embed, "k": k}}},
                "size": k,
            }

            try:
                text_resp = client.search(index=self._index_name, body=text_body)
                knn_resp = client.search(index=self._index_name, body=knn_body)
            except Exception as e:
                log.error("OpenSearch search error: %s", e)
                return []

            # Merge & deduplicate with score weighting (0.3 BM25 + 0.7 KNN)
            seen: dict[str, tuple[float, str]] = {}
            max_text_score = 1.0
            max_knn_score = 1.0

            text_hits = text_resp["hits"]["hits"]
            knn_hits = knn_resp["hits"]["hits"]

            if text_hits:
                max_text_score = text_hits[0]["_score"]
            if knn_hits:
                max_knn_score = knn_hits[0]["_score"]

            for hit in text_hits:
                sid = hit["_id"]
                score = (hit["_score"] / max_text_score) * 0.3 if max_text_score else 0
                if sid not in seen or score > seen[sid][0]:
                    seen[sid] = (score, hit["_source"]["text"])

            for hit in knn_hits:
                sid = hit["_id"]
                score = (hit["_score"] / max_knn_score) * 0.7 if max_knn_score else 0
                if sid not in seen or score > seen[sid][0]:
                    seen[sid] = (score, hit["_source"]["text"])

            results = [text for _, text in sorted(seen.values(), key=lambda x: -x[0])]
            return results[:k]

    def _fallback_retrieve(self, query: str, k: int) -> list[str]:
        """Fallback when OpenSearch is not available — simple keyword match."""
        log.warning("OpenSearch unavailable, falling back to keyword retrieval")
        try:
            from virtual_assistant_be.services.memory_service import MemoryService
            mem = MemoryService()
            return mem.search(query, k=k)
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Document management
    # ------------------------------------------------------------------

    def list_documents(self) -> list[dict]:
        client = self._get_client()
        if client is None:
            return []
        if not client.indices.exists(index=self._index_name):
            return []

        resp = client.search(
            index=self._index_name,
            body={
                "size": 0,
                "aggs": {
                    "docs": {
                        "terms": {"field": "document_name", "size": 10000},
                        "aggs": {
                            "chunk_count": {"value_count": {"field": "chunk_index"}}
                        },
                    }
                },
            },
        )
        buckets = resp.get("aggregations", {}).get("docs", {}).get("buckets", [])
        return [
            {
                "name": b["key"],
                "chunks": b.get("doc_count", 0),
            }
            for b in buckets
        ]

    def delete(self, document_name: str) -> int:
        client = self._get_client()
        if client is None:
            return 0
        if not client.indices.exists(index=self._index_name):
            return 0

        try:
            resp = client.delete_by_query(
                index=self._index_name,
                body={"query": {"term": {"document_name": document_name}}},
                conflicts="proceed",
            )
            deleted = resp.get("deleted", 0)
            if deleted:
                log.info("Deleted %d chunks of '%s'", deleted, document_name)
            return deleted
        except Exception:
            log.warning("Failed to delete '%s'", document_name, exc_info=True)
            return 0

    # ------------------------------------------------------------------
    # Ask (context + LLM)
    # ------------------------------------------------------------------

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
