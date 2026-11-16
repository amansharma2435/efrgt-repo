"""Streamlit interface for the document question-answering system.

Run it from the repository root:

    streamlit run app.py

The root ``app.py`` calls ``main`` in this module. Keeping the widgets here
lets the retrieval code stay free of Streamlit imports.
"""

from __future__ import annotations

import hashlib

import streamlit as st

from docqa.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_TOP_K,
    get_gemini_model,
)
from docqa.embeddings import SentenceTransformerEmbedder
from docqa.errors import DocumentQAError
from docqa.indexing import IndexResult, index_pdf
from docqa.rag_pipeline import NOT_FOUND_MESSAGE, GeminiAnswerGenerator, RAGPipeline
from docqa.vector_database import FaissVectorStore, SearchHit

_SOURCE_PREVIEW = 280


@st.cache_resource(show_spinner="Loading the embedding model. The first run downloads it.")
def load_embedder() -> SentenceTransformerEmbedder:
    """Load MiniLM once per server process. Streamlit reruns do not reload it."""
    return SentenceTransformerEmbedder(DEFAULT_EMBEDDING_MODEL)


def main() -> None:
    st.set_page_config(
        page_title="Document Question Answering",
        page_icon=":page_facing_up:",
        layout="centered",
    )
    _apply_layout()
    _init_state()

    st.title("Document question answering")
    st.caption(
        "Upload a PDF, then ask a question. The app retrieves the three closest "
        "passages and asks Gemini to answer only from those passages."
    )

    _render_sidebar()
    _render_chat()


def _apply_layout() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.4rem; padding-bottom: 4rem;}
        div[data-testid="stChatMessage"] p {line-height: 1.5;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "uploads" not in st.session_state:
        # file digest -> original bytes, so a rebuild does not ask for the file again
        st.session_state.uploads = {}
    if "indexed" not in st.session_state:
        st.session_state.indexed = {}
    if "store" not in st.session_state:
        st.session_state.store = None
    if "chunk_size" not in st.session_state:
        st.session_state.chunk_size = DEFAULT_CHUNK_SIZE
    if "chunk_overlap" not in st.session_state:
        st.session_state.chunk_overlap = DEFAULT_CHUNK_OVERLAP


def _store() -> FaissVectorStore:
    embedder = load_embedder()
    store = st.session_state.store
    if store is None or store.dimension != embedder.dimension:
        store = FaissVectorStore(embedder.dimension)
        st.session_state.store = store
    return store


def _render_sidebar() -> None:
    with st.sidebar:
        st.header("Documents")
        st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_uploader",
            help="Text-based PDFs work. Scanned image PDFs do not, because this app does not run OCR.",
        )
        _ingest_uploads()

        indexed: dict[str, IndexResult] = st.session_state.indexed
        if not indexed:
            st.info("No document is indexed yet.")
        else:
            for name, result in indexed.items():
                empty_note = (
                    f", {result.empty_page_count} empty"
                    if result.empty_page_count
                    else ""
                )
                st.markdown(f"**{name}**")
                st.caption(
                    f"{result.page_count} pages{empty_note}, {result.chunk_count} chunks"
                )
                if st.button("Remove", key=f"remove-{name}"):
                    _store().remove_document(name)
                    st.session_state.indexed.pop(name, None)
                    st.session_state.uploads.pop(name, None)
                    st.rerun()

        st.divider()
        st.header("Settings")
        st.number_input(
            "Chunk size (characters)",
            min_value=100,
            max_value=2000,
            step=50,
            key="chunk_size",
            help="Applies the next time a PDF is indexed. Larger chunks keep more context.",
        )
        st.number_input(
            "Chunk overlap (characters)",
            min_value=0,
            max_value=400,
            step=10,
            key="chunk_overlap",
            help="Repeated text at each chunk boundary, so a cut sentence is not lost.",
        )
        if st.session_state.chunk_overlap >= st.session_state.chunk_size:
            st.error("Overlap must be smaller than the chunk size.")
        if indexed and st.button("Rebuild index with these settings", key="rebuild"):
            _rebuild_index()

        st.caption(f"Retrieval returns the top {DEFAULT_TOP_K} passages.")
        st.caption(f"Embedding model: `{DEFAULT_EMBEDDING_MODEL}`")
        st.caption(f"Gemini model: `{get_gemini_model()}`")
        st.caption(
            "Set GEMINI_API_KEY in the environment or in Streamlit secrets. "
            "The key itself is never shown here."
        )

        if st.button("Clear chat", key="clear_chat"):
            st.session_state.messages = []
            st.rerun()


def _ingest_uploads() -> None:
    """Index new uploads once. The same bytes are not embedded on every rerun."""
    uploaded_files = st.session_state.get("pdf_uploader") or []
    for uploaded in uploaded_files:
        data = uploaded.getvalue()
        digest = hashlib.sha256(data).hexdigest()
        current = st.session_state.uploads.get(uploaded.name)
        already_done = (
            current is not None
            and current["digest"] == digest
            and (uploaded.name in st.session_state.indexed or current.get("failed"))
        )
        if already_done:
            continue
        st.session_state.uploads[uploaded.name] = {
            "name": uploaded.name,
            "data": data,
            "digest": digest,
            "failed": False,
        }
        _index_one(uploaded.name, data)
        st.session_state.uploads[uploaded.name]["failed"] = (
            uploaded.name not in st.session_state.indexed
        )


def _index_one(filename: str, data: bytes) -> None:
    if st.session_state.chunk_overlap >= st.session_state.chunk_size:
        st.error("Fix the chunk settings before indexing.")
        return
    try:
        with st.spinner(f"Indexing {filename}..."):
            result = index_pdf(
                data,
                filename,
                _store(),
                load_embedder(),
                chunk_size=int(st.session_state.chunk_size),
                chunk_overlap=int(st.session_state.chunk_overlap),
            )
    except DocumentQAError as exc:
        st.error(str(exc))
        return
    except Exception:
        st.error(f"'{filename}' could not be indexed. Check that it is a text-based PDF.")
        return

    st.session_state.indexed[result.document_name] = result
    st.success(
        f"Indexed {result.document_name}: {result.page_count} pages, "
        f"{result.chunk_count} chunks."
    )


def _rebuild_index() -> None:
    if st.session_state.chunk_overlap >= st.session_state.chunk_size:
        st.error("Fix the chunk settings before rebuilding.")
        return
    uploads = list(st.session_state.uploads.values())
    if not uploads:
        st.warning("Upload a PDF before rebuilding the index.")
        return
    embedder = load_embedder()
    st.session_state.store = FaissVectorStore(embedder.dimension)
    st.session_state.indexed = {}
    for item in uploads:
        _index_one(item["name"], item["data"])


def _render_chat() -> None:
    indexed = st.session_state.indexed
    if not indexed:
        st.info("Upload a PDF in the sidebar to start. A sample file is in `samples/campus_handbook.pdf`.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                _render_sources(message["sources"])

    question = st.chat_input(
        "Ask a question about the uploaded documents",
        disabled=not indexed,
        key="question_box",
    )
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question, "sources": None})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving passages and asking Gemini..."):
                pipeline = RAGPipeline(
                    store=_store(),
                    embedder=load_embedder(),
                    generator=GeminiAnswerGenerator(),
                    top_k=DEFAULT_TOP_K,
                )
                result = pipeline.answer(question)
        except DocumentQAError as exc:
            st.error(str(exc))
            st.session_state.messages.append(
                {"role": "assistant", "content": str(exc), "sources": None, "is_error": True}
            )
            return
        except Exception:
            fallback = "The question could not be answered. Check the Gemini key and the network, then try again."
            st.error(fallback)
            st.session_state.messages.append(
                {"role": "assistant", "content": fallback, "sources": None, "is_error": True}
            )
            return

        st.markdown(result.answer)
        sources = [_source_payload(hit) for hit in result.sources]
        if sources:
            _render_sources(sources)
        elif result.answer == NOT_FOUND_MESSAGE:
            st.caption("No passages were retrieved because the index is empty.")

    st.session_state.messages.append(
        {"role": "assistant", "content": result.answer, "sources": sources}
    )


def _source_payload(hit: SearchHit) -> dict:
    return {
        "document_name": hit.chunk.document_name,
        "page_number": hit.chunk.page_number,
        "score": hit.score,
        "text": hit.chunk.text,
    }


def _render_sources(sources: list[dict]) -> None:
    st.markdown("**Sources**")
    for source in sources:
        label = (
            f"{source['document_name']} · page {source['page_number']} "
            f"· similarity {source['score']:.2f}"
        )
        with st.expander(label):
            st.write(source["text"])
            if len(source["text"]) > _SOURCE_PREVIEW:
                st.caption("Full retrieved passage.")
