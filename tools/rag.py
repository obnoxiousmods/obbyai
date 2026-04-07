"""
RAG tool backed by ArangoDB with Ollama embeddings.

Collections:
  rag_documents  – chunked document text with embeddings
  rag_sessions   – per-session message summaries

ArangoSearch view:
  rag_view – indexes the `text` field for full-text fallback
"""
import asyncio
import hashlib
import os
import time
from typing import Optional

import httpx
from arango import ArangoClient

ARANGO_URL  = os.getenv("ARANGO_URL",  "http://localhost:8529")
ARANGO_USER = os.getenv("ARANGO_USER", "root")
ARANGO_PASS = os.getenv("ARANGO_PASS", "")
ARANGO_DB   = os.getenv("ARANGO_DB",   "ai_chat_rag")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
OLLAMA_URL  = os.getenv("OLLAMA_LOCAL_URL", "http://127.0.0.1:11434")
CHUNK_SIZE  = 500   # chars per chunk
CHUNK_OVERLAP = 80
TOP_K       = 5     # results to return


# ── ArangoDB setup ─────────────────────────────────────────────────────────────

def _get_db():
    client = ArangoClient(hosts=ARANGO_URL)
    sys_db = client.db("_system", username=ARANGO_USER, password=ARANGO_PASS)
    if not sys_db.has_database(ARANGO_DB):
        sys_db.create_database(ARANGO_DB)
    db = client.db(ARANGO_DB, username=ARANGO_USER, password=ARANGO_PASS)
    # Ensure collections exist
    for col in ("rag_documents", "rag_sessions"):
        if not db.has_collection(col):
            db.create_collection(col)
    # Ensure ArangoSearch view for full-text fallback
    view_names = [v["name"] for v in db.views()]
    if "rag_view" not in view_names:
        db.create_arangosearch_view(
            "rag_view",
            properties={
                "links": {
                    "rag_documents": {
                        "fields": {"text": {"analyzers": ["text_en"]}},
                        "includeAllFields": False,
                    }
                }
            },
        )
    return db


_db_cache: Optional[object] = None

def get_db():
    global _db_cache
    if _db_cache is None:
        _db_cache = _get_db()
    return _db_cache


# ── Embeddings ──────────────────────────────────────────────────────────────────

async def embed(text: str) -> list[float]:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": text},
        )
        data = r.json()
        # Ollama returns {"embeddings": [[...]]} for /api/embed
        embs = data.get("embeddings") or data.get("embedding")
        if isinstance(embs, list) and embs and isinstance(embs[0], list):
            return embs[0]
        return embs or []


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na  = sum(x * x for x in a) ** 0.5
    nb  = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


# ── Chunking ────────────────────────────────────────────────────────────────────

def chunk_text(text: str, source: str = "") -> list[dict]:
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        chunks.append({
            "text": chunk,
            "source": source,
            "chunk_idx": idx,
            "hash": hashlib.sha256(chunk.encode()).hexdigest()[:16],
        })
        start += CHUNK_SIZE - CHUNK_OVERLAP
        idx += 1
    return chunks


# ── Ingest ──────────────────────────────────────────────────────────────────────

async def ingest_document(text: str, source: str = "upload", session_id: str = "") -> dict:
    """Chunk, embed, and store a document. Returns stats."""
    db = get_db()
    col = db.collection("rag_documents")
    chunks = chunk_text(text, source)
    inserted = 0
    skipped = 0
    for chunk in chunks:
        # Skip if already exists (same hash)
        existing = list(db.aql.execute(
            "FOR d IN rag_documents FILTER d.hash == @h LIMIT 1 RETURN d._key",
            bind_vars={"h": chunk["hash"]},
        ))
        if existing:
            skipped += 1
            continue
        emb = await embed(chunk["text"])
        doc = {
            **chunk,
            "embedding": emb,
            "session_id": session_id,
            "ingested_at": time.time(),
        }
        col.insert(doc)
        inserted += 1
    return {"inserted": inserted, "skipped": skipped, "total_chunks": len(chunks), "source": source}


# ── Search ──────────────────────────────────────────────────────────────────────

async def search(query: str, session_id: str = "", top_k: int = TOP_K) -> list[dict]:
    """Embed query, compute cosine similarity against stored docs, return top-k."""
    db = get_db()
    query_emb = await embed(query)
    if not query_emb:
        # Fall back to full-text search
        return _fulltext_search(db, query, top_k)

    # Fetch all docs with embeddings (filter by session if given)
    aql = "FOR d IN rag_documents FILTER d.embedding != null RETURN d"
    cursor = db.aql.execute(aql)
    scored = []
    for doc in cursor:
        emb = doc.get("embedding", [])
        score = cosine_similarity(query_emb, emb)
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {"text": d["text"], "source": d.get("source", ""), "score": round(s, 4)}
        for s, d in scored[:top_k]
        if s > 0.3
    ]


def _fulltext_search(db, query: str, top_k: int) -> list[dict]:
    aql = """
    FOR d IN rag_view
      SEARCH ANALYZER(d.text IN TOKENS(@q, 'text_en'), 'text_en')
      SORT BM25(d) DESC LIMIT @k RETURN d
    """
    cursor = db.aql.execute(aql, bind_vars={"q": query, "k": top_k})
    return [{"text": d["text"], "source": d.get("source", ""), "score": 0.0} for d in cursor]


# ── Store conversation message ──────────────────────────────────────────────────

async def store_message(role: str, content: str, session_id: str):
    if not content or not session_id:
        return
    db = get_db()
    emb = await embed(content[:500])  # embed first 500 chars
    db.collection("rag_sessions").insert({
        "role": role,
        "content": content[:2000],
        "embedding": emb,
        "session_id": session_id,
        "ts": time.time(),
    })


async def search_conversation_history(query: str, session_id: str, top_k: int = 3) -> list[dict]:
    db = get_db()
    query_emb = await embed(query)
    if not query_emb:
        return []
    aql = "FOR d IN rag_sessions FILTER d.session_id == @sid AND d.embedding != null RETURN d"
    cursor = db.aql.execute(aql, bind_vars={"sid": session_id})
    scored = []
    for doc in cursor:
        score = cosine_similarity(query_emb, doc.get("embedding", []))
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {"role": d["role"], "text": d["content"][:300], "score": round(s, 4)}
        for s, d in scored[:top_k]
        if s > 0.4
    ]


# ── Tool spec for Ollama ────────────────────────────────────────────────────────

TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "rag_search",
        "description": (
            "Search the local knowledge base (uploaded documents, files, and past conversation context) "
            "for relevant information. Use this when the user asks about something they may have previously "
            "uploaded or shared, or to recall prior context."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for in the knowledge base.",
                }
            },
            "required": ["query"],
        },
    },
}


async def run_tool(query: str, session_id: str = "") -> dict:
    results = await search(query, session_id)
    history = await search_conversation_history(query, session_id)
    return {
        "query": query,
        "documents": results,
        "conversation_history": history,
    }
