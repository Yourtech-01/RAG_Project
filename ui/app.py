"""
app.py  —  Streamlit chat interface for the RAG Q&A system
Run: streamlit run app.py
"""

import os, requests
import streamlit as st

# Streamlit Community Cloud secrets (Settings -> Secrets) aren't auto-exported
# to os.environ, so check st.secrets first and fall back to an env var for
# local runs (`API_URL=http://localhost:8000 streamlit run app.py`).
API_URL = st.secrets.get("API_URL", os.getenv("API_URL", "http://localhost:8000"))

st.set_page_config(page_title="Document Q&A", page_icon="📄", layout="wide")
st.markdown("""
<style>
.block-container{padding-top:1.5rem}
.src{background:#f0f4ff;border-left:3px solid #4361ee;border-radius:0 6px 6px 0;
     padding:10px 14px;margin:6px 0;font-size:13px}
.src-lbl{font-weight:600;color:#4361ee;font-size:12px;margin-bottom:4px}
.badge{background:#4361ee;color:white;padding:2px 8px;border-radius:10px;
       font-size:11px;font-weight:600}
</style>""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 Document Q&A")
    st.markdown('<span class="badge">gemini-flash-latest</span> &nbsp; '
                '<span class="badge">MiniLM</span>', unsafe_allow_html=True)
    st.markdown("---")

    top_k = st.slider("Sources to retrieve", 1, 10, 5)

    st.markdown("### 📂 Indexed documents")
    try:
        docs = requests.get(f"{API_URL}/documents", timeout=3).json()
        if docs["documents"]:
            for d in docs["documents"]:
                st.markdown(f"**{d['name']}** — {d['chunks']} chunks")
            st.caption(f"Total: {docs['total_chunks']} chunks")
        else:
            st.info("No documents indexed yet.\n\nRun:\n```\npython pipeline.py --docs ./data/docs/\n```")
    except Exception:
        st.warning("⚠️ API not reachable.\nMake sure the FastAPI server is running:\n```\nuvicorn main:app --reload\n```")

    st.markdown("---")
    try:
        h = requests.get(f"{API_URL}/health", timeout=2).json()
        st.caption(f"🤖 LLM: `{h.get('llm_model','N/A')}`")
        st.caption(f"🔢 Embed: `{h.get('embed_model','N/A')}`")
        st.caption("🟢 API connected")
    except Exception:
        st.caption("🔴 API offline")

    st.markdown("---")
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ── Chat ──────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown("# 💬 Ask your documents")
st.caption("Powered by **Gemini 1.5 Flash** (LLM) + **all-MiniLM-L6-v2** (embeddings) + **ChromaDB** (vector store)")

# Welcome message
if not st.session_state.messages:
    st.info("👋 Upload documents to `data/docs/`, run the ingestion pipeline, then ask anything about them!")

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📚 Sources ({len(msg['sources'])} retrieved)"):
                for i, s in enumerate(msg["sources"], 1):
                    st.markdown(
                        f'<div class="src">'
                        f'<div class="src-lbl">Source {i}: {s["source"]} · chunk {s["chunk_index"]}</div>'
                        f'{s["excerpt"]}'
                        f'</div>',
                        unsafe_allow_html=True)

# Input
if q := st.chat_input("Ask anything about your documents..."):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching + generating..."):
            try:
                resp = requests.post(
                    f"{API_URL}/ask",
                    json={"question": q, "top_k": top_k},
                    timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    st.markdown(data["answer"])
                    if data.get("sources"):
                        with st.expander(f"📚 Sources ({len(data['sources'])} retrieved)"):
                            for i, s in enumerate(data["sources"], 1):
                                st.markdown(
                                    f'<div class="src">'
                                    f'<div class="src-lbl">Source {i}: {s["source"]} · chunk {s["chunk_index"]}</div>'
                                    f'{s["excerpt"]}'
                                    f'</div>',
                                    unsafe_allow_html=True)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": data["answer"],
                        "sources": data.get("sources", [])
                    })
                elif resp.status_code == 503:
                    st.error("No documents indexed yet. Run `python pipeline.py --docs ./data/docs/` first.")
                else:
                    st.error(f"API error {resp.status_code}: {resp.text}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Make sure FastAPI is running: `uvicorn main:app --reload`")
            except Exception as e:
                st.error(f"Error: {e}")
