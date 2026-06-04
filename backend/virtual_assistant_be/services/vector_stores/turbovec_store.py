from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from virtual_assistant_be.core.config import settings
from virtual_assistant_be.services.rag_service import chunk_text
from virtual_assistant_be.services.vector_stores.base import VectorStore

log = logging.getLogger(__name__)

_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"))


class TurbovecStore(VectorStore):
    def __init__(self, db_path: str | None = None) -> None:
        super().__init__()
        from turbovec import IdMapIndex

        self.index = IdMapIndex(dim=768, bit_width=4)

        self._db_path = db_path or os.path.join(_DATA_DIR, "turbovec.db")
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._init_db()

        self._bm25: BM25Okapi | None = None
        self._bm25_docs: list[str] = []
        self._bm25_ids: list[int] = []
        self._populate_cache()

    # ------------------------------------------------------------------
    # SQLite metadata store
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_name TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                vec_id INTEGER UNIQUE NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_doc_name ON chunks(doc_name)
        """)
        self._conn.commit()

    def _populate_cache(self) -> None:
        rows = self._conn.execute(
            "SELECT vec_id, text FROM chunks ORDER BY id"
        ).fetchall()
        if rows:
            self._bm25_ids = [r[0] for r in rows]
            self._bm25_docs = [r[1] for r in rows]
            self._bm25 = BM25Okapi(self._bm25_docs)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, text: str, source: str) -> int:
        chunks = chunk_text(text)
        if not chunks:
            return 0

        embeds = self.embed(chunks)
        if not embeds:
            return 0

        ids = np.array(
            [int(uuid.uuid4().int & 0x7FFFFFFFFFFFFFFF) for _ in chunks],
            dtype=np.uint64,
        )

        vecs = np.array(embeds, dtype=np.float32)
        self.index.add_with_ids(vecs, ids)

        cursor = self._conn.cursor()
        for i, (chunk, vid) in enumerate(zip(chunks, ids.tolist())):
            cursor.execute(
                "INSERT INTO chunks (doc_name, chunk_index, text, vec_id) VALUES (?, ?, ?, ?)",
                (source, i, chunk, vid),
            )
        self._conn.commit()

        self._bm25_docs.extend(chunks)
        self._bm25_ids.extend(ids.tolist())
        self._bm25 = BM25Okapi(self._bm25_docs)

        log.info("Ingested %d chunks from '%s' via turbovec", len(chunks), source)
        return len(chunks)

    # ------------------------------------------------------------------
    # Retrieval (hybrid: BM25 → vector rerank via allowlist)
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        if self._bm25 is None or len(self._bm25_ids) == 0:
            return []

        bm25_scores = self._bm25.get_scores(query.split())
        top_n = min(k * 10, len(self._bm25_ids))
        top_indices = np.argsort(bm25_scores)[-top_n:][::-1]
        candidate_ids = np.array(
            [self._bm25_ids[i] for i in top_indices if bm25_scores[i] > 0],
            dtype=np.uint64,
        )

        if len(candidate_ids) == 0:
            return []

        q_embed = self.embed(query)
        if isinstance(q_embed, list) and len(q_embed) > 0 and not isinstance(q_embed[0], (int, float)):
            q_embed = q_embed[0]
        if isinstance(q_embed, np.ndarray):
            q_embed = q_embed.tolist()
        q_vec = np.array([q_embed], dtype=np.float32)

        scores, ids = self.index.search(q_vec, k=k, allowlist=candidate_ids)

        if len(ids) == 0 or len(ids[0]) == 0:
            return []

        result_ids = ids[0].tolist()

        placeholders = ",".join("?" for _ in result_ids)
        rows = self._conn.execute(
            f"SELECT vec_id, text FROM chunks WHERE vec_id IN ({placeholders})",
            result_ids,
        ).fetchall()
        text_map = {row[0]: row[1] for row in rows}

        ordered = []
        for vid in result_ids:
            if vid in text_map:
                ordered.append(text_map[vid])
        return ordered

    # ------------------------------------------------------------------
    # Document management
    # ------------------------------------------------------------------

    def list_documents(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT doc_name, COUNT(*) as cnt FROM chunks GROUP BY doc_name ORDER BY doc_name"
        ).fetchall()
        return [{"name": r[0], "chunks": r[1]} for r in rows]

    def delete(self, document_name: str) -> int:
        rows = self._conn.execute(
            "SELECT vec_id FROM chunks WHERE doc_name = ?", (document_name,)
        ).fetchall()

        if not rows:
            return 0

        for (vid,) in rows:
            self.index.remove(vid)

        self._conn.execute("DELETE FROM chunks WHERE doc_name = ?", (document_name,))
        self._conn.commit()

        self._populate_cache()

        log.info("Deleted %d chunks of '%s' from turbovec", len(rows), document_name)
        return len(rows)
