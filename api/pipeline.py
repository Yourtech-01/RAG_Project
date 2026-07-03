"""
pipeline.py
Ingestion pipeline: PDF/TXT/MD -> chunks -> embeddings -> ChromaDB
Uses sentence-transformers (free, local, no API key needed)

Usage:
    python pipeline.py --docs ./data/docs/
    python pipeline.py --docs ./data/docs/ --reset
"""

import argparse, hashlib, pathlib, re
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 50
CHARS_PER_TOK = 4
EMBED_MODEL   = "all-MiniLM-L6-v2"
COLLECTION    = "documents"
DB_PATH       = "./data/chroma_db"
BATCH_SIZE    = 100

# ── Load embedder once ────────────────────────────────────────────────────────
print("[pipeline] Loading embedding model (first run downloads ~80MB)...")
_embedder = SentenceTransformer(EMBED_MODEL)
print(f"[pipeline] Embedding model loaded: {EMBED_MODEL}")

@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    metadata: dict

# ── Text extraction ───────────────────────────────────────────────────────────
def extract_text(path: pathlib.Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz
        doc   = fitz.open(str(path))
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text("text")
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = re.sub(r'[ \t]+', ' ', text)
            pages.append(f"[Page {i+1}]\n{text.strip()}")
        doc.close()
        return "\n\n".join(pages)
    if suffix not in {".txt", ".md"}:
        raise ValueError(f"Unsupported file type: {suffix}")
    return path.read_text(encoding="utf-8", errors="replace")

# ── Chunking ──────────────────────────────────────────────────────────────────
def chunk_text(text: str, doc_id: str, metadata: dict) -> list:
    char_size    = CHUNK_SIZE * CHARS_PER_TOK
    char_overlap = CHUNK_OVERLAP * CHARS_PER_TOK
    sentences    = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) <= char_size:
            current += (" " if current else "") + sentence
        else:
            if current:
                chunks.append(current.strip())
            overlap_text = chunks[-1][-char_overlap:] if chunks else ""
            current = (overlap_text + " " + sentence).strip()
    if current.strip():
        chunks.append(current.strip())
    return [
        Chunk(doc_id=doc_id,
              chunk_id=f"{doc_id}_chunk_{i:04d}",
              text=c,
              metadata={**metadata, "chunk_index": i, "doc_id": doc_id})
        for i, c in enumerate(chunks) if len(c.strip()) > 50
    ]

# ── Embedding ─────────────────────────────────────────────────────────────────
def embed_batch(texts: list) -> list:
    """Embed a batch of texts using local sentence-transformers. No API key needed."""
    return _embedder.encode(texts, show_progress_bar=False).tolist()

# ── Ingestion ─────────────────────────────────────────────────────────────────
def ingest(docs_dir: str, reset: bool = False) -> dict:
    docs_path = pathlib.Path(docs_dir)
    if not docs_path.exists():
        print(f"[pipeline] ERROR: Directory not found: {docs_dir}")
        return {"docs": 0, "chunks": 0, "skipped": 0}

    # Setup ChromaDB
    pathlib.Path(DB_PATH).mkdir(parents=True, exist_ok=True)
    chroma = chromadb.PersistentClient(
        path=DB_PATH, settings=Settings(anonymized_telemetry=False))
    if reset:
        try:
            chroma.delete_collection(COLLECTION)
            print("[pipeline] Existing collection deleted.")
        except Exception:
            pass
    collection = chroma.get_or_create_collection(
        name=COLLECTION, metadata={"hnsw:space": "cosine"})

    # Find documents
    supported = {".pdf", ".txt", ".md"}
    files     = [f for f in docs_path.rglob("*") if f.suffix.lower() in supported]
    print(f"[pipeline] Found {len(files)} document(s) in {docs_dir}")

    stats = {"docs": 0, "chunks": 0, "skipped": 0}

    for file_path in files:
        doc_id = hashlib.md5(file_path.name.encode()).hexdigest()[:12]

        # Skip already ingested docs
        existing = collection.get(where={"doc_id": doc_id}, limit=1)
        if existing["ids"]:
            print(f"  Skipping {file_path.name} (already ingested)")
            stats["skipped"] += 1
            continue

        print(f"  Processing: {file_path.name}")
        try:
            text   = extract_text(file_path)
            chunks = chunk_text(text, doc_id, {"source": file_path.name})
        except Exception as e:
            print(f"  ERROR extracting {file_path.name}: {e}")
            continue

        print(f"    → {len(chunks)} chunks")

        # Embed and store in batches
        for i in range(0, len(chunks), BATCH_SIZE):
            batch      = chunks[i:i + BATCH_SIZE]
            embeddings = embed_batch([c.text for c in batch])
            collection.add(
                ids        =[c.chunk_id  for c in batch],
                embeddings =embeddings,
                documents  =[c.text      for c in batch],
                metadatas  =[c.metadata  for c in batch],
            )

        stats["docs"]   += 1
        stats["chunks"] += len(chunks)

    print(f"\n[pipeline] Done: {stats['docs']} docs, "
          f"{stats['chunks']} chunks, {stats['skipped']} skipped.")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB")
    parser.add_argument("--docs",  default="./data/docs/",
                        help="Directory containing PDF/TXT/MD files")
    parser.add_argument("--reset", action="store_true",
                        help="Delete existing collection before ingesting")
    args = parser.parse_args()
    ingest(args.docs, reset=args.reset)
