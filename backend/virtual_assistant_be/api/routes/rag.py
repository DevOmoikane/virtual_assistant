from __future__ import annotations

import os
import tempfile

import requests
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from virtual_assistant_be.core.config import settings
from virtual_assistant_be.services.rag_service import RagService
from virtual_assistant_be.services.memory_service import MemoryService
from virtual_assistant_be.services.file_parser import extract_text, SUPPORTED_EXTENSIONS

router = APIRouter(prefix="/api/rag", tags=["rag"])
tools_router = APIRouter(tags=["tools"])

_HERE = os.path.dirname(__file__)
_TEMPLATE = os.path.join(_HERE, "..", "templates", "rag.html")


@tools_router.get("/tools/rag", response_class=HTMLResponse)
async def rag_tools_page():
    try:
        with open(_TEMPLATE) as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse("<h1>Template not found</h1>", status_code=404)

rag = RagService()
memory = MemoryService()


class IngestRequest(BaseModel):
    text: str
    source: str


class AskRequest(BaseModel):
    query: str


@router.post("/ingest")
async def ingest(req: IngestRequest):
    chunks = rag.ingest(req.text, source=req.source)
    if chunks == 0:
        raise HTTPException(500, "Ingestion failed — is ChromaDB running?")
    return {"chunks": chunks, "source": req.source}


@router.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "upload")[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    try:
        content = await file.read()
        tmp.write(content)
        tmp.close()

        text = extract_text(tmp.name)
        if text is None:
            raise HTTPException(400, f"Failed to extract text from {file.filename}")
        if not text.strip():
            raise HTTPException(400, f"No text content found in {file.filename}")

        chunks = rag.ingest(text, source=file.filename or "upload")
        if chunks == 0:
            raise HTTPException(500, "Ingestion failed — is ChromaDB running?")
        return {"chunks": chunks, "source": file.filename, "type": SUPPORTED_EXTENSIONS[ext]}
    finally:
        os.unlink(tmp.name)


@router.get("/memory/person-count")
async def person_count():
    return {"person_count": memory.get_person_count()}


@router.post("/ask")
async def ask(req: AskRequest):
    memory_docs = memory.search(req.query, k=3)
    doc_docs = rag.retrieve(req.query, k=3)

    all_context = []
    seen = set()
    for d in memory_docs + doc_docs:
        if d not in seen:
            seen.add(d)
            all_context.append(d)

    if not all_context:
        return {"answer": "", "context": []}

    prompt = (
        f"Only answer using the context below. If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{''.join(all_context)}\n\n"
        f"Question: {req.query}"
    )

    try:
        resp = requests.post(
            f"{settings.ollama_url}/api/generate",
            json={"model": settings.ollama_gen_model, "prompt": prompt, "stream": False},
            timeout=30,
        )
        resp.raise_for_status()
        answer = resp.json()["response"]
        return {"answer": answer, "context": all_context}
    except Exception:
        return {"answer": "", "context": all_context}
