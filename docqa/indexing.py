"""Glue between a PDF upload and the vector index.

This module does not call Gemini. Building the index only needs the local
embedding model, so students can try retrieval before they add an API key.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from docqa.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from docqa.embeddings import SentenceTransformerEmbedder
from docqa.errors import EmptyDocumentError
from docqa.pdf_processing import extract_pages
from docqa.text_chunking import chunk_pages
from docqa.vector_database import FaissVectorStore


@dataclass(frozen=True)
class IndexResult:
    """What the interface reports after a PDF is accepted."""

    document_name: str
    page_count: int
    empty_page_count: int
    chunk_count: int


def index_pdf(
    data: bytes,
    filename: str,
    store: FaissVectorStore,
    embedder: SentenceTransformerEmbedder,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> IndexResult:
    """Extract, chunk, embed, and store one PDF.

    Uploading the same file name again replaces that document's chunks.
    Other documents already in the index are left as they are.
    """
    pages = extract_pages(data, filename)
    document_name = pages[0].document_name
    chunks = chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise EmptyDocumentError(
            f"No selectable text was found in '{document_name}'. "
            "Scanned pages need OCR before this app can read them, and OCR "
            "is not included."
        )

    vectors: np.ndarray = embedder.embed_texts([chunk.text for chunk in chunks])
    store.replace_document(document_name, chunks, vectors)
    empty_pages = sum(1 for page in pages if page.is_empty)
    return IndexResult(
        document_name=document_name,
        page_count=len(pages),
        empty_page_count=empty_pages,
        chunk_count=len(chunks),
    )
