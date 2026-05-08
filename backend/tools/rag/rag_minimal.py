import sys
import requests
import uuid

OLLAMA_URL = "http://localhost:11434"
CHROMA_URL = "http://localhost:8000"
EMBED_MODEL = "nomic-embed-text"
GEN_MODEL = "llama3.1"
COLLECTION = "rag_docs"
CHROMA_BASE = f"{CHROMA_URL}/api/v2/tenants/default_tenant/databases/default_database/collections"


def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


def embed(texts):
    single = not isinstance(texts, list)
    if single:
        texts = [texts]

    resp = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
    )
    data = resp.json()
    if "embeddings" in data:
        embeds = data["embeddings"]
    else:
        embeds = []
        for t in texts:
            r = requests.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": t},
            )
            embeds.append(r.json()["embedding"])

    return embeds[0] if single else embeds


def ensure_collection():
    resp = requests.get(CHROMA_BASE)
    for c in resp.json():
        if c["name"] == COLLECTION:
            return c["id"]
    resp = requests.post(
        CHROMA_BASE,
        json={"name": COLLECTION, "metadata": {"hnsw:space": "cosine"}},
    )
    return resp.json()["id"]


def ingest(filepath):
    with open(filepath) as f:
        text = f.read()

    chunks = chunk_text(text)
    ids = [str(uuid.uuid4()) for _ in chunks]
    metas = [{"source": filepath, "chunk": i} for i in range(len(chunks))]
    embeds = embed(chunks)

    collection_id = ensure_collection()
    requests.post(
        f"{CHROMA_BASE}/{collection_id}/add",
        json={"ids": ids, "embeddings": embeds, "metadatas": metas, "documents": chunks},
    )
    print(f"Ingested {len(chunks)} chunks.")


def query(q, k=3):
    collection_id = ensure_collection()
    q_embed = embed(q)
    if not isinstance(q_embed[0], list):
        q_embed = [q_embed]

    resp = requests.post(
        f"{CHROMA_BASE}/{collection_id}/query",
        json={"query_embeddings": q_embed, "n_results": k},
    )
    data = resp.json()
    return data["documents"][0]


def ask(q):
    docs = query(q)
    context = "\n\n".join(docs)

    prompt = f"""Only answer using the context below. If the answer is not in the context, say you don't know.

Context:
{context}

Question: {q}"""

    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": GEN_MODEL, "prompt": prompt, "stream": False},
    )
    return resp.json()["response"]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python rag_minimal.py ingest <file>   -- load a document")
        print("  python rag_minimal.py ask <question>   -- ask a question")
        sys.exit(1)

    if sys.argv[1] == "ingest":
        ingest(sys.argv[2])
    elif sys.argv[1] == "ask":
        print(ask(" ".join(sys.argv[2:])))
