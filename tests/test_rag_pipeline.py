"""Answer generation: prompt, retrieval wiring, and the Gemini client wrapper.

These tests do not call the Gemini API. A live call needs GEMINI_API_KEY and
is covered by ``test_live_gemini_answers_from_the_handbook``, which skips
when no key is configured.
"""

import os
from types import SimpleNamespace

import pytest

from docqa.config import get_gemini_api_key
from docqa.embeddings import SentenceTransformerEmbedder
from docqa.errors import AnswerGenerationError, MissingAPIKeyError
from docqa.indexing import index_pdf
from docqa.rag_pipeline import (
    NOT_FOUND_MESSAGE,
    GeminiAnswerGenerator,
    RAGPipeline,
    build_prompt,
    redact_secrets,
)
from docqa.sample_document import REFUND_FACT, build_sample_pdf_bytes
from docqa.vector_database import FaissVectorStore, SearchHit
from docqa.text_chunking import TextChunk


class RecordingGenerator:
    """Test double. It records the prompt and does not talk to Google."""

    def __init__(self):
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if "14 days" in prompt:
            return "A full refund is available within 14 days (campus_handbook.pdf, page 3)."
        return NOT_FOUND_MESSAGE


class FakeModels:
    def __init__(self, text: str | None):
        self.text = text
        self.calls: list[dict] = []

    def generate_content(self, *, model: str, contents: str):
        self.calls.append({"model": model, "contents": contents})
        return SimpleNamespace(text=self.text)


class FakeClient:
    def __init__(self, text: str | None):
        self.models = FakeModels(text)


def test_prompt_tells_gemini_to_stay_inside_the_passages():
    hit = SearchHit(
        chunk=TextChunk("campus_handbook.pdf", 3, 0, REFUND_FACT),
        score=0.82,
    )
    prompt = build_prompt("What is the refund window?", [hit])

    assert "ONLY the passages" in prompt
    assert NOT_FOUND_MESSAGE in prompt
    assert "campus_handbook.pdf" in prompt
    assert "Page: 3" in prompt
    assert "14 days" in prompt
    assert "What is the refund window?" in prompt


def test_pipeline_sends_retrieved_context_and_returns_citations(embedder):
    store = FaissVectorStore(embedder.dimension)
    index_pdf(build_sample_pdf_bytes(), "campus_handbook.pdf", store, embedder)
    generator = RecordingGenerator()
    pipeline = RAGPipeline(store, embedder, generator, top_k=3)

    result = pipeline.answer("What is the full refund window for a student?")

    assert result.used_language_model is True
    assert len(generator.prompts) == 1
    assert "14 days" in generator.prompts[0]
    assert NOT_FOUND_MESSAGE in generator.prompts[0]
    assert "14 days" in result.answer
    assert result.sources[0].chunk.page_number == 3
    assert result.sources[0].chunk.document_name == "campus_handbook.pdf"


def test_pipeline_does_not_call_the_model_when_the_index_is_empty(embedder):
    store = FaissVectorStore(embedder.dimension)
    generator = RecordingGenerator()
    pipeline = RAGPipeline(store, embedder, generator, top_k=3)

    result = pipeline.answer("What is the cafeteria menu?")

    assert result.answer == NOT_FOUND_MESSAGE
    assert result.sources == []
    assert result.used_language_model is False
    assert generator.prompts == []


def test_unrelated_question_still_receives_the_not_found_instruction(embedder):
    """The handbook is small, so all three passages are retrieved.

    A real model must still refuse when none of them answers the question.
    This test double does that refusal. It does not call Gemini.
    """

    class RefusingGenerator:
        def __init__(self):
            self.prompts: list[str] = []

        def generate(self, prompt: str) -> str:
            self.prompts.append(prompt)
            if "cafeteria menu on Mars" in prompt:
                return NOT_FOUND_MESSAGE
            raise AssertionError("The question did not reach the generator.")

    store = FaissVectorStore(embedder.dimension)
    index_pdf(build_sample_pdf_bytes(), "campus_handbook.pdf", store, embedder)
    generator = RefusingGenerator()
    pipeline = RAGPipeline(store, embedder, generator, top_k=3)

    result = pipeline.answer("What is the cafeteria menu on Mars?")

    assert result.used_language_model is True
    assert NOT_FOUND_MESSAGE in generator.prompts[0]
    assert "ONLY the passages" in generator.prompts[0]
    assert result.answer == NOT_FOUND_MESSAGE
    assert len(result.sources) == 3


def test_gemini_wrapper_uses_injected_client_and_does_not_require_a_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    client = FakeClient("The library closes at 9 PM.")
    generator = GeminiAnswerGenerator(model="gemini-2.5-flash", client=client)

    answer = generator.generate("Question: when does the library close?")

    assert answer == "The library closes at 9 PM."
    assert client.models.calls[0]["model"] == "gemini-2.5-flash"
    assert "library close" in client.models.calls[0]["contents"]


def test_gemini_wrapper_rejects_an_empty_model_response():
    generator = GeminiAnswerGenerator(model="gemini-2.5-flash", client=FakeClient("  "))
    with pytest.raises(AnswerGenerationError, match="empty"):
        generator.generate("Question")


def test_missing_api_key_raises_before_any_client_is_built(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError, match="GEMINI_API_KEY"):
        GeminiAnswerGenerator(model="gemini-2.5-flash")


def test_api_key_is_read_from_the_environment_and_not_echoed(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "unit-test-key")
    assert get_gemini_api_key() == "unit-test-key"


def test_secret_looking_text_is_redacted():
    leaked = "request failed for AIzaSyA1234567890abcdefghi"
    redacted = redact_secrets(leaked)
    assert "AIza" not in redacted
    assert "[REDACTED]" in redacted


@pytest.mark.skipif(
    not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
    reason="No Gemini API key is configured, so the live API was not called.",
)
def test_live_gemini_answers_from_the_handbook(embedder):
    """Optional end-to-end check. Runs only when a real key is present."""
    store = FaissVectorStore(embedder.dimension)
    index_pdf(build_sample_pdf_bytes(), "campus_handbook.pdf", store, embedder)
    pipeline = RAGPipeline(store, embedder, GeminiAnswerGenerator(), top_k=3)

    result = pipeline.answer("How many days does a student have to request a full refund?")

    assert result.used_language_model is True
    assert "14" in result.answer
    assert result.sources[0].chunk.page_number == 3


@pytest.fixture(scope="module")
def embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder()
