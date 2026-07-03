# Deploying the RAG Document Q&A demo (free hosting)

Two services, two hosts — same pattern as your other projects:

| Service | What it is | Where it lives |
|---|---|---|
| `api/` | FastAPI + ChromaDB + Gemini | Render (free web service) |
| `ui/`  | Streamlit chat UI | Streamlit Community Cloud |

The repo already contains everything needed: `render.yaml` for one-click
Render setup, a `start.sh` that auto-ingests two sample docs on boot so the
demo works immediately, and pinned CPU-only `torch` so the build fits
Render's free 512 MB instance.

---

## 0. Get a free Gemini API key

Go to https://aistudio.google.com/apikey, sign in, and create a key. Free
tier is enough for a demo.

## 1. Push this folder to GitHub

```bash
cd rag_project
git init
git add .
git commit -m "RAG Document Q&A - deploy-ready"
git branch -M main
git remote add origin https://github.com/Yourtech-01/rag-document-qa.git
git push -u origin main
```

(Create the empty repo on GitHub first, or use `gh repo create` if you have
the GitHub CLI.)

## 2. Deploy the API on Render

1. Go to https://dashboard.render.com → **New** → **Blueprint**.
2. Connect the GitHub repo. Render will read `render.yaml` automatically and
   propose a web service named `rag-doc-qa-api` with root dir `api/`.
3. When prompted, paste your `GEMINI_API_KEY` as the secret env var.
4. Click **Apply**. First build takes ~5–8 minutes (downloading the CPU
   torch wheel + the embedding model).
5. Once live, note the URL, e.g. `https://rag-doc-qa-api.onrender.com`.
6. Sanity check: open `https://rag-doc-qa-api.onrender.com/health` — you
   should see `{"status":"ok",...}`. Then check
   `https://rag-doc-qa-api.onrender.com/documents` — it should list the two
   bundled sample docs (ingested automatically by `start.sh`).

**If the build fails or the service crashes with an out-of-memory error:**
Render's free tier is 512 MB RAM, and `sentence-transformers` + `torch` is
close to that ceiling. If it OOMs, upgrade just this one service to the
**Starter** plan ($7/mo, 512 MB → more headroom and no cold sleep) — you can
downgrade again after the demo. Everything else in this guide is unaffected.

## 3. Deploy the UI on Streamlit Community Cloud

1. Go to https://share.streamlit.io → **New app**.
2. Pick the same GitHub repo, set:
   - **Main file path:** `ui/app.py`
3. Before/after deploying, open **App settings → Secrets** and paste:
   ```toml
   API_URL = "https://rag-doc-qa-api.onrender.com"
   ```
   (use the exact Render URL from step 2, no trailing slash).
4. Deploy. Your live demo URL will look like
   `https://your-app-name.streamlit.app`.

## 4. Before a live demo

Render's free instance sleeps after 15 minutes of no traffic and takes
~30–60s to wake up on the next request. **Open your API's `/health` URL
about a minute before you demo** to warm it up, so the first question in
front of a recruiter doesn't hang.

## 5. Add more documents (optional)

The demo ships with two short sample docs so it works out of the box. To
demo with your own PDFs instead:

```bash
cd api
python pipeline.py --docs ./data/docs/ --reset   # run locally with your own files
```

Then either commit the new files under `api/data/docs/` (they'll be
re-ingested automatically on every Render boot via `start.sh`), or extend
`main.py` with an upload endpoint if you want live uploads from the UI.

---

## Local testing before you push

```bash
export GEMINI_API_KEY=your_key_here
docker compose up --build
# API:  http://localhost:8000/docs
# UI:   http://localhost:8501
```
