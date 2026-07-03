"""
api/main.py  —  FastAPI RAG query service (Free APIs: Gemini + sentence-transformers)

Endpoints:
  POST /ask           -> answer a question (JSON)
  GET  /documents     -> list indexed documents
  GET  /health        -> health + model info
"""

import os
from typing import List

import chromadb
from chromadb.config import Settings
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH     = os.getenv("CHROMA_PATH", "./data/chroma_db")
COLLECTION  = "documents"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL   = "gemini-flash-latest"
TOP_K       = int(os.getenv("TOP_K", "5"))
MAX_TOKENS  = 1024

# ── Init models ───────────────────────────────────────────────────────────────
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
llm      = genai.GenerativeModel(LLM_MODEL)
embedder = SentenceTransformer(EMBED_MODEL)

# ── ChromaDB ──────────────────────────────────────────────────────────────────
chroma = chromadb.PersistentClient(
    path=DB_PATH, settings=Settings(anonymized_telemetry=False))
try:
    collection = chroma.get_collection(COLLECTION)
except Exception:
    collection = None

# ── Schemas ───────────────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000,
                          example="What are the key findings?")
    top_k: int = Field(TOP_K, ge=1, le=20)

class Source(BaseModel):
    source: str
    chunk_index: int
    excerpt: str

class AskResponse(BaseModel):
    question: str
    answer: str
    sources: List[Source]
    model: str

# ── Core functions ────────────────────────────────────────────────────────────
def embed_query(question: str) -> list:
    return embedder.encode([question])[0].tolist()

def retrieve_chunks(embedding: list, top_k: int) -> list:
    if collection is None:
        raise HTTPException(status_code=503,
            detail="No documents ingested yet. Run: python pipeline.py --docs ./data/docs/")
    results = collection.query(
        query_embeddings=[embedding], n_results=top_k,
        include=["documents", "metadatas", "distances"])
    return [
        {"text": doc, "source": meta.get("source", "unknown"),
         "chunk_index": meta.get("chunk_index", 0), "distance": dist}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0])
    ]

def build_prompt(question: str, chunks: list) -> str:
    context = "\n\n---\n\n".join(
        f"[Source {i}: {c['source']}, chunk {c['chunk_index']}]\n{c['text']}"
        for i, c in enumerate(chunks, 1))
    return f"""You are a precise document assistant. Answer using ONLY the context below.
If the answer is not in the context, say "I couldn't find that in the provided documents."
Always cite which source(s) you used.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="RAG Document Q&A API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"])

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    embedding = embed_query(req.question)
    chunks    = retrieve_chunks(embedding, req.top_k)
    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant chunks found.")
    prompt   = build_prompt(req.question, chunks)
    response = llm.generate_content(prompt)
    answer   = response.text
    sources  = [Source(source=c["source"], chunk_index=c["chunk_index"],
                       excerpt=c["text"][:200] + "...") for c in chunks]
    return AskResponse(question=req.question, answer=answer,
                       sources=sources, model=LLM_MODEL)

@app.get("/documents")
def list_documents():
    if collection is None:
        return {"documents": [], "total_chunks": 0}
    metas   = collection.get(include=["metadatas"])["metadatas"]
    sources = {}
    for m in metas:
        src = m.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1
    return {"documents": [{"name": k, "chunks": v} for k, v in sources.items()],
            "total_chunks": sum(sources.values())}

@app.get("/health")
def health():
    return {"status": "ok", "embed_model": EMBED_MODEL, "llm_model": LLM_MODEL}
