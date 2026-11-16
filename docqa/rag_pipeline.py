"""Retrieval-augmented question answering.

The pipeline does three things for every question:

1. Embed the question with the same model used for the document chunks.
2. Ask FAISS for the top 3 closest chunks.
3. Send those chunks and the question to Gemini, with instructions to answer
   only from that context.

If the index is empty, Gemini is not called. The app says the information
was not found. That is an honest result, not a substitute for an API call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from docqa.config import DEFAULT_TOP_K, get_gemini_api_key, get_gemini_model
from docqa.embeddings import SentenceTransformerEmbedder
from docqa.errors import AnswerGenerationError, MissingAPIKeyError
from docqa.vector_database import FaissVectorStore, SearchHit

NOT_FOUND_MESSAGE = "The information was not found in the document."

_SECRET_PATTERN = re.compile(r"(AIza[0-9A-Za-z\-_]{10,}|sk-[A-Za-z0-9]{10,})")


def redact_secrets(text: str) -> str:
    """Remove strings that look like API keys before anything is shown."""
    return _SECRET_PATTERN.sub("[REDACTED]", text)


def build_prompt(question: str, hits: list[SearchHit]) -> str:
    """Build the instruction Gemini sees. The passages are the only source."""
    if hits:
        blocks = []
        for number, hit in enumerate(hits, start=1):
            chunk = hit.chunk
            blocks.append(
                f"[Passage {number}]\n"
                f"Document: {chunk.document_name}\n"
                f"Page: {chunk.page_number}\n"
                f"Text:\n{chunk.text}"
            )
        context = "\n\n".join(blocks)
    else:
        context = "(no passages were retrieved)"

    return (
        "You are a document question-answering assistant.\n"
        "Answer the question using ONLY the passages in the context below.\n"
        "Do not use outside knowledge, even if you know the answer.\n"
        "If the passages do not contain the answer, reply with exactly:\n"
        f"{NOT_FOUND_MESSAGE}\n"
        "When you use a fact, mention the document name and the page number.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question.strip()}\n\n"
        "Answer:"
    )


class AnswerGenerator(Protocol):
    """Anything that can turn a prompt string into an answer string."""

    def generate(self, prompt: str) -> str:
        """Return the answer text."""


@dataclass
class RAGAnswer:
    """The answer shown in the chat, plus the passages that supported it."""

    question: str
    answer: str
    sources: list[SearchHit]
    used_language_model: bool


class GeminiAnswerGenerator:
    """Calls the Google Gemini API. The key stays inside the client object."""

    def __init__(self, api_key: str | None = None, model: str | None = None, client=None):
        self.model = model or get_gemini_model()
        if client is not None:
            self._client = client
            return
        key = api_key if api_key is not None else get_gemini_api_key()
        if not str(key).strip():
            raise MissingAPIKeyError(
                "The Gemini API key is empty. Set GEMINI_API_KEY and restart the app."
            )
        # Imported here so PDF and retrieval tests do not need the network SDK
        # until a real answer is requested.
        from google import genai

        self._client = genai.Client(api_key=str(key).strip())

    def generate(self, prompt: str) -> str:
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
            )
        except Exception as exc:
            detail = redact_secrets(str(exc)).strip()
            if len(detail) > 300:
                detail = detail[:300] + "..."
            raise AnswerGenerationError(
                "Gemini could not generate an answer. " + (detail or "The request failed.")
            ) from exc

        text = getattr(response, "text", None)
        if text is None or not str(text).strip():
            raise AnswerGenerationError("Gemini returned an empty answer.")
        return str(text).strip()


class RAGPipeline:
    """Connects the embedder, the FAISS store, and an answer generator."""

    def __init__(
        self,
        store: FaissVectorStore,
        embedder: SentenceTransformerEmbedder,
        generator: AnswerGenerator,
        top_k: int = DEFAULT_TOP_K,
    ):
        self.store = store
        self.embedder = embedder
        self.generator = generator
        self.top_k = top_k

    def answer(self, question: str) -> RAGAnswer:
        cleaned = question.strip()
        if not cleaned:
            raise ValueError("Please enter a question.")

        if len(self.store) == 0:
            return RAGAnswer(
                question=cleaned,
                answer=NOT_FOUND_MESSAGE,
                sources=[],
                used_language_model=False,
            )

        query_vector = self.embedder.embed_query(cleaned)
        sources = self.store.search(query_vector, top_k=self.top_k)
        prompt = build_prompt(cleaned, sources)
        generated = self.generator.generate(prompt)
        return RAGAnswer(
            question=cleaned,
            answer=generated,
            sources=sources,
            used_language_model=True,
        )
