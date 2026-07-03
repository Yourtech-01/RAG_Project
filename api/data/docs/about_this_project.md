# RAG Document Q&A System — Project Overview

This project is a Retrieval-Augmented Generation (RAG) system that answers
questions about a collection of documents. It combines local embeddings with
a free-tier large language model so it can run without any paid API keys
other than a free Google Gemini key.

## Architecture

The system has three main components:

1. **Ingestion pipeline (pipeline.py)** — reads PDF, TXT, and Markdown files,
   splits them into overlapping chunks of about 512 tokens, embeds each chunk
   with the local `all-MiniLM-L6-v2` sentence-transformers model, and stores
   the vectors in a persistent ChromaDB collection.
2. **Query API (main.py)** — a FastAPI service exposing `/ask`, `/documents`,
   and `/health` endpoints. On a question, it embeds the query, retrieves the
   top-k most similar chunks from ChromaDB, builds a grounded prompt, and
   sends it to Gemini 1.5 Flash for the final answer.
3. **Chat UI (app.py)** — a Streamlit interface that lets a user ask
   questions and see which source chunks were used to generate each answer.

## Why these choices

Embeddings are computed locally with sentence-transformers so there is no
per-query embedding cost. Gemini 1.5 Flash was chosen as the generation
model because it has a generous free tier and low latency, which makes the
system practical to run as a live demo without incurring hosting costs.

## Evaluation

The project includes an evaluation script (evaluate.py) that uses Gemini
itself as a judge model to score faithfulness, answer relevancy, and context
quality against a small set of test questions, producing a JSON and HTML
report.
