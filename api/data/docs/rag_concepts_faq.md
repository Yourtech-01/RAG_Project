# RAG Concepts — Frequently Asked Questions

**What is Retrieval-Augmented Generation?**
Retrieval-Augmented Generation, or RAG, is a technique where a language
model's answer is grounded in text retrieved from an external knowledge
base at query time, rather than relying only on what the model memorized
during training. This reduces hallucination and lets the system answer
questions about documents it was never trained on.

**What is a vector database used for?**
A vector database stores numerical embeddings of text chunks and supports
fast nearest-neighbor search. When a user asks a question, the question is
embedded into the same vector space, and the database returns the chunks
whose embeddings are closest to the question's embedding.

**Why chunk documents instead of embedding the whole document?**
Long documents exceed the context window of most embedding and generation
models, and embedding an entire document produces a single vector that
blurs together many unrelated topics. Chunking into smaller overlapping
pieces preserves local context while keeping each chunk focused enough for
accurate retrieval.

**What does chunk overlap do?**
Overlap between consecutive chunks prevents a sentence or idea that spans a
chunk boundary from being split awkwardly, which would otherwise reduce
retrieval accuracy for that content.

**How is answer quality evaluated in a RAG system?**
Common metrics include faithfulness (does the answer only use facts present
in the retrieved context), answer relevancy (does the answer address the
question asked), and context precision or recall (are the retrieved chunks
actually useful for answering the question).
