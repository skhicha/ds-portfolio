"""
Streamlit chat frontend for the RAG Q&A Tool.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import os
from typing import Dict, Tuple

import streamlit as st

from src.ingest import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, load_and_chunk
from src.qa import NO_API_KEY_MESSAGE, answer_question, get_default_llm
from src.vectorstore import GeminiEmbeddings, OfflineHashingEmbeddings, OpenAIEmbeddingsProvider, VectorStore

st.set_page_config(page_title="Document Q&A (RAG)", page_icon="\U0001f916", layout="wide")

# --------------------------------------------------------------------------- #
# Styling - restrained design system layered on the theme in
# .streamlit/config.toml. No raw unstyled widget dumps.
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
    :root {
        --color-bg: #F7F6FB;
        --color-surface: #FFFFFF;
        --color-border: #E6E1F5;
        --color-text: #1C2330;
        --color-text-muted: #5B6472;
        --color-primary: #6D28D9;
        --color-primary-soft: #F1EBFC;
        --color-accent: #B45309;
        --color-accent-soft: #FDF2E3;
        --shadow-card: 0 1px 2px rgba(16, 24, 40, 0.04), 0 6px 16px rgba(16, 24, 40, 0.06);
    }

    .block-container { padding-top: 2rem; max-width: 1200px; }

    div[data-testid="stMetric"] {
        background: var(--color-surface);
        border: 1px solid var(--color-border);
        border-radius: 12px;
        padding: 1rem 1.1rem 0.9rem 1.1rem;
        box-shadow: var(--shadow-card);
    }
    div[data-testid="stMetricLabel"] { color: var(--color-text-muted); font-weight: 600; }
    div[data-testid="stMetricValue"] { color: var(--color-primary); }

    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--color-text);
        margin: 0.25rem 0 0.1rem 0;
    }
    .section-caption { color: var(--color-text-muted); font-size: 0.92rem; margin-bottom: 0.9rem; }

    .pill {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 0.35rem;
    }
    .pill-primary { background: var(--color-primary-soft); color: var(--color-primary); }
    .pill-accent { background: var(--color-accent-soft); color: var(--color-accent); }

    .citation-card {
        background: var(--color-surface);
        border: 1px solid var(--color-border);
        border-radius: 10px;
        padding: 0.7rem 0.9rem;
        margin-bottom: 0.6rem;
        box-shadow: var(--shadow-card);
    }
    .citation-source {
        font-weight: 700;
        color: var(--color-primary);
        font-size: 0.85rem;
    }
    .citation-score {
        color: var(--color-text-muted);
        font-size: 0.78rem;
        float: right;
    }
    .citation-text {
        color: var(--color-text);
        font-size: 0.88rem;
        margin-top: 0.3rem;
        line-height: 1.4;
    }

    .callout-warning {
        background: var(--color-accent-soft);
        border: 1px solid #F3DDB0;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        color: #7A4A08;
        font-size: 0.9rem;
        margin-bottom: 0.75rem;
    }

    hr { border-color: var(--color-border); }
    </style>
    """,
    unsafe_allow_html=True,
)


def embedding_backend_label() -> str:
    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        return "Gemini (live)"
    if os.environ.get("OPENAI_API_KEY"):
        return "OpenAI (live)"
    return "Offline TF-IDF (no key needed)"


def build_embedding_provider():
    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        try:
            return GeminiEmbeddings()
        except Exception as exc:  # pragma: no cover - depends on live network/package
            st.warning(f"Falling back to offline embeddings: {exc}")
            return OfflineHashingEmbeddings()
    if os.environ.get("OPENAI_API_KEY"):
        try:
            return OpenAIEmbeddingsProvider()
        except Exception as exc:  # pragma: no cover
            st.warning(f"Falling back to offline embeddings: {exc}")
            return OfflineHashingEmbeddings()
    return OfflineHashingEmbeddings()


@st.cache_resource(show_spinner="Indexing documents...")
def build_index(
    file_bytes_tuple: Tuple[Tuple[str, bytes], ...],
    chunk_size: int,
    chunk_overlap: int,
    backend_label: str,
) -> Tuple[VectorStore, int]:
    sources: Dict[str, bytes] = dict(file_bytes_tuple)
    chunks = load_and_chunk(sources, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    provider = build_embedding_provider()
    store = VectorStore.build(chunks, embedding_provider=provider)
    return store, len(chunks)


def section(title: str, caption: str = "") -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="section-caption">{caption}</div>', unsafe_allow_html=True)


def render_citations(citations) -> None:
    for i, c in enumerate(citations, start=1):
        st.markdown(
            f"""
            <div class="citation-card">
                <span class="citation-source">Chunk {i} &middot; {c.source}</span>
                <span class="citation-score">score {c.score:.3f}</span>
                <div class="citation-text">{c.text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    st.title("\U0001f916 LLM-Powered Document Q&A")
    st.caption(
        "Upload PDFs, retrieve the most relevant chunks with FAISS, and get a grounded, "
        "cited answer. Retrieval works fully offline; generation uses a live LLM when an "
        "API key is provided."
    )

    with st.sidebar:
        st.header("LLM API Key")
        gemini_key = st.text_input("Google Gemini API Key", type="password")
        openai_key = st.text_input("OpenAI API Key", type="password")
        if gemini_key:
            os.environ["GOOGLE_API_KEY"] = gemini_key
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key

        st.header("Chunking")
        chunk_size = st.slider("Chunk size (characters)", 200, 2000, DEFAULT_CHUNK_SIZE, step=50)
        chunk_overlap = st.slider("Chunk overlap (characters)", 0, 400, DEFAULT_CHUNK_OVERLAP, step=10)
        top_k = st.slider("Chunks retrieved per question (k)", 1, 8, 3)

        st.header("Documents")
        uploaded = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)

    if not uploaded:
        st.info("Upload one or more PDF documents in the sidebar to get started.")
        st.caption(
            "No sample is auto-loaded here on purpose - try the bundled PDFs in `data/` "
            "(`sample_insurance_policy.pdf`, `sample_leave_policy.pdf`, `sample_warranty_manual.pdf`)."
        )
        return

    file_bytes_tuple = tuple((f.name, f.getvalue()) for f in uploaded)
    backend_label = embedding_backend_label()
    store, n_chunks = build_index(file_bytes_tuple, chunk_size, chunk_overlap, backend_label)

    if not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        st.markdown(
            f'<div class="callout-warning">{NO_API_KEY_MESSAGE}</div>',
            unsafe_allow_html=True,
        )

    k1, k2, k3 = st.columns(3)
    k1.metric("Documents indexed", f"{len(uploaded)}")
    k2.metric("Chunks indexed", f"{n_chunks}")
    k3.metric("Embedding backend", backend_label)

    st.divider()
    section("Ask a question", "Answers are grounded only in the text retrieved from your documents.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("citations"):
                with st.expander("Source chunks used"):
                    render_citations(msg["citations"])

    if question := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": question, "citations": None})
        with st.chat_message("user"):
            st.write(question)

        with st.spinner("Retrieving and generating..."):
            llm = get_default_llm()
            result = answer_question(store, question, llm=llm, k=top_k)

        with st.chat_message("assistant"):
            st.write(result.answer)
            if result.citations:
                with st.expander("Source chunks used", expanded=False):
                    render_citations(result.citations)

        st.session_state.messages.append(
            {"role": "assistant", "content": result.answer, "citations": result.citations}
        )


if __name__ == "__main__":
    main()
