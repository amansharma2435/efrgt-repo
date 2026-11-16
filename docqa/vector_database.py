"""FAISS index of document chunks.

The index is an exact inner-product search (``IndexFlatIP``). It is the
right default for a student project: every chunk is compared with the
question, the top results are exact, and there is no extra training step.
With normalised embeddings, the score is cosine similarity.

The index lives in memory for the Streamlit session. Closing the browser
tab drops it. That avoids writing document text to disk.
"""

from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np

from docqa.config import DEFAULT_TOP_K
from docqa.embeddings import l2_normalize
from docqa.text_chunking import TextChunk


@dataclass(frozen=True)
class SearchHit:
    """A retrieved chunk and how similar it is to the question (higher is closer)."""

    chunk: TextChunk
    score: float


class FaissVectorStore:
    """Stores chunks and their vectors, and returns the top matches."""

    def __init__(self, dimension: int):
        if dimension <= 0:
            raise ValueError("The embedding dimension must be a positive integer.")
        self.dimension = int(dimension)
        self.chunks: list[TextChunk] = []
        self._vectors: list[np.ndarray] = []
        self._index = faiss.IndexFlatIP(self.dimension)

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def document_names(self) -> list[str]:
        names: list[str] = []
        for chunk in self.chunks:
            if chunk.document_name not in names:
                names.append(chunk.document_name)
        return names

    def chunks_for(self, document_name: str) -> list[TextChunk]:
        return [chunk for chunk in self.chunks if chunk.document_name == document_name]

    def add(self, chunks: list[TextChunk], vectors: np.ndarray) -> None:
        """Append chunks. Vectors are normalised again so scores stay cosine."""
        self._append(chunks, vectors)
        self._rebuild()

    def remove_document(self, document_name: str) -> int:
        """Drop every chunk that came from ``document_name``. Returns the count."""
        kept = [
            (chunk, vector)
            for chunk, vector in zip(self.chunks, self._vectors)
            if chunk.document_name != document_name
        ]
        removed = len(self.chunks) - len(kept)
        if removed:
            self.chunks = [chunk for chunk, _ in kept]
            self._vectors = [vector for _, vector in kept]
            self._reindex_chunk_ids()
            self._rebuild()
        return removed

    def replace_document(
        self,
        document_name: str,
        chunks: list[TextChunk],
        vectors: np.ndarray,
    ) -> None:
        """Replace any previous chunks for this file, then add the new ones."""
        kept = [
            (chunk, vector)
            for chunk, vector in zip(self.chunks, self._vectors)
            if chunk.document_name != document_name
        ]
        self.chunks = [chunk for chunk, _ in kept]
        self._vectors = [vector for _, vector in kept]
        self._append(chunks, vectors)
        self._rebuild()

    def search(self, query_vector: np.ndarray, top_k: int = DEFAULT_TOP_K) -> list[SearchHit]:
        """Return up to ``top_k`` chunks closest to the question vector."""
        if top_k <= 0:
            raise ValueError("top_k must be at least 1.")
        if len(self.chunks) == 0:
            return []

        query = l2_normalize(np.asarray(query_vector, dtype=np.float32)).reshape(1, -1)
        if query.shape[1] != self.dimension:
            raise ValueError(
                f"Query dimension {query.shape[1]} does not match the index ({self.dimension})."
            )
        k = min(top_k, len(self.chunks))
        scores, indices = self._index.search(np.ascontiguousarray(query), k)
        hits: list[SearchHit] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0 or index >= len(self.chunks):
                continue
            hits.append(SearchHit(chunk=self.chunks[int(index)], score=float(score)))
        return hits

    def _append(self, chunks: list[TextChunk], vectors: np.ndarray) -> None:
        if len(chunks) == 0:
            return
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("Embedding matrix must have shape (number of chunks, dimension).")
        if matrix.shape[0] != len(chunks):
            raise ValueError("Each chunk needs exactly one embedding vector.")
        if matrix.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {matrix.shape[1]} does not match the index ({self.dimension})."
            )
        matrix = l2_normalize(matrix)
        for chunk, row in zip(chunks, matrix):
            self.chunks.append(chunk)
            self._vectors.append(np.asarray(row, dtype=np.float32))
        self._reindex_chunk_ids()

    def _reindex_chunk_ids(self) -> None:
        self.chunks = [
            TextChunk(
                document_name=chunk.document_name,
                page_number=chunk.page_number,
                chunk_index=index,
                text=chunk.text,
            )
            for index, chunk in enumerate(self.chunks)
        ]

    def _rebuild(self) -> None:
        self._index = faiss.IndexFlatIP(self.dimension)
        if not self._vectors:
            return
        matrix = np.vstack(self._vectors).astype(np.float32)
        self._index.add(np.ascontiguousarray(matrix))
