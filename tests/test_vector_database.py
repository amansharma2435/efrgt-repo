"""FAISS retrieval with hand-built vectors and with the real embedding model."""

import numpy as np
import pytest

from docqa.embeddings import SentenceTransformerEmbedder
from docqa.indexing import index_pdf
from docqa.sample_document import build_sample_pdf_bytes
from docqa.text_chunking import TextChunk
from docqa.vector_database import FaissVectorStore


def _unit(values: list[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def test_search_returns_top_3_and_keeps_citation_metadata():
    store = FaissVectorStore(dimension=3)
    chunks = [
        TextChunk("handbook.pdf", 1, 0, "library hours"),
        TextChunk("handbook.pdf", 3, 1, "refund policy"),
        TextChunk("handbook.pdf", 4, 2, "classroom"),
        TextChunk("other.pdf", 2, 3, "unrelated"),
    ]
    vectors = np.vstack(
        [
            _unit([1, 0, 0]),
            _unit([0.2, 1, 0]),
            _unit([0, 0, 1]),
            _unit([0, 1, 0]),
        ]
    )
    store.add(chunks, vectors)

    hits = store.search(_unit([1, 0, 0]), top_k=3)

    assert len(hits) == 3
    assert hits[0].chunk.text == "library hours"
    assert hits[0].chunk.document_name == "handbook.pdf"
    assert hits[0].chunk.page_number == 1
    assert hits[0].score > hits[1].score
    assert {hit.chunk.page_number for hit in hits} == {1, 3, 4}


def test_replace_document_removes_old_chunks():
    store = FaissVectorStore(dimension=2)
    store.add(
        [TextChunk("a.pdf", 1, 0, "old")],
        np.vstack([_unit([1, 0])]),
    )
    store.replace_document(
        "a.pdf",
        [TextChunk("a.pdf", 5, 0, "new")],
        np.vstack([_unit([0, 1])]),
    )

    assert len(store) == 1
    assert store.chunks[0].text == "new"
    assert store.chunks[0].page_number == 5
    hits = store.search(_unit([0, 1]), top_k=3)
    assert len(hits) == 1
    assert hits[0].chunk.text == "new"


def test_empty_store_returns_no_hits():
    store = FaissVectorStore(dimension=4)
    assert store.search(np.ones(4, dtype=np.float32), top_k=3) == []


def test_real_embeddings_retrieve_the_refund_page(embedder: SentenceTransformerEmbedder):
    store = FaissVectorStore(embedder.dimension)
    index_pdf(build_sample_pdf_bytes(), "campus_handbook.pdf", store, embedder)

    hits = store.search(embedder.embed_query("How many days do I have for a full refund?"), top_k=3)

    assert len(hits) == 3
    assert hits[0].chunk.page_number == 3
    assert "14 days" in hits[0].chunk.text
    assert hits[0].chunk.document_name == "campus_handbook.pdf"
    assert hits[0].score > hits[1].score


@pytest.fixture(scope="module")
def embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder()
