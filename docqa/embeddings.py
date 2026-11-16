"""Turn text into vectors with a Sentence Transformers model.

``all-MiniLM-L6-v2`` is a small, widely used English model. Two sentences
about the same fact end up close together in vector space, which is what
lets the search step run without calling Gemini.

Vectors are L2-normalised. After that, the inner product of two vectors is
their cosine similarity, and FAISS can rank passages with ``IndexFlatIP``.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from docqa.config import DEFAULT_EMBEDDING_MODEL


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to length 1. A zero vector is left unchanged."""
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe = np.maximum(norms, 1e-12)
    return (matrix / safe).astype(np.float32)


class SentenceTransformerEmbedder:
    """Thin wrapper so the rest of the project does not depend on the library API."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        # The first call downloads the model into the Hugging Face cache.
        self._model = SentenceTransformer(model_name)
        if hasattr(self._model, "get_embedding_dimension"):
            self.dimension = int(self._model.get_embedding_dimension())
        else:
            self.dimension = int(self._model.get_sentence_embedding_dimension())

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Return one normalised row per text. Shape: ``(len(texts), dimension)``."""
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Return one normalised vector for the user's question."""
        rows = self.embed_texts([text])
        return rows[0]
