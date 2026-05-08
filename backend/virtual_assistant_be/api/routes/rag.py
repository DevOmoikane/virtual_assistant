from __future__ import annotations

import os
import tempfile

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from virtual_assistant_be.services.rag_service import RagService
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


@router.post("/ask")
async def ask(req: AskRequest):
    answer = rag.ask(req.query)
    if not answer:
        docs = rag.retrieve(req.query)
        if not docs:
            return {"answer": "", "context": []}
        return {"answer": "", "context": docs}
    return {"answer": answer, "context": []}
