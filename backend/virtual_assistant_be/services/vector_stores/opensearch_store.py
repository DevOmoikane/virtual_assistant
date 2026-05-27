from __future__ import annotations

import logging

import numpy as np
from opensearchpy import OpenSearch, helpers

from virtual_assistant_be.core.config import settings
from virtual_assistant_be.services.vector_stores.base import VectorStore
from virtual_assistant_be.services.rag_service import chunk_text
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


class OpenSearchStore(VectorStore):
    def __init__(self) -> None:
        super().__init__()
        self._client: OpenSearch | None = None
        self._index_name = settings.opensearch_index

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

    def ingest(self, text: str, source: str) -> int:
        client = self._get_client()
        if client is None:
            return 0
        if not self._ensure_index():
            return 0

        chunks = chunk_text(text)
        if not chunks:
            return 0

        embeds = self.embed(chunks)
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

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        with Timer("rag.retrieve"):
            client = self._get_client()
            if client is None:
                return self._fallback_retrieve(query, k)

            if not self._ensure_index():
                return []

            q_embed = self.embed(query)
            if isinstance(q_embed, list) and len(q_embed) > 0 and not isinstance(q_embed[0], (int, float)):
                q_embed = q_embed[0]
            if isinstance(q_embed, np.ndarray):
                q_embed = q_embed.tolist()

            text_body = {
                "query": {"match": {"text": query}},
                "size": k,
            }

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
        log.warning("OpenSearch unavailable, falling back to keyword retrieval")
        try:
            from virtual_assistant_be.services.memory_service import MemoryService
            mem = MemoryService()
            return mem.search(query, k=k)
        except Exception:
            return []

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
