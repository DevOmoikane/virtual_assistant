from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from virtual_assistant_be.services.rag_service import RagService

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


@router.post("/ask")
async def ask(req: AskRequest):
    answer = rag.ask(req.query)
    if not answer:
        docs = rag.retrieve(req.query)
        if not docs:
            return {"answer": "", "context": []}
        return {"answer": "", "context": docs}
    return {"answer": answer, "context": []}
