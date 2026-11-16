"""Split page text into overlapping chunks.

A retrieval system works better on short passages than on a whole PDF. We
split each page on its own, so a chunk never mixes two pages and the page
number on a citation stays correct.

Overlap means the end of one chunk is repeated at the start of the next.
If a sentence is cut at the boundary, the full sentence still sits inside
one of the two chunks.

The splitter itself comes from LangChain
(``RecursiveCharacterTextSplitter``). It prefers to break on paragraphs,
then lines, then spaces, and only then on single characters.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from docqa.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from docqa.pdf_processing import PageText


@dataclass(frozen=True)
class TextChunk:
    """One searchable passage plus the metadata needed for a citation."""

    document_name: str
    page_number: int
    chunk_index: int
    text: str


def _splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    if chunk_size < 100:
        raise ValueError("Chunk size must be at least 100 characters.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be zero or more, and smaller than the chunk size.")
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_pages(
    pages: list[PageText],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[TextChunk]:
    """Turn pages into chunks. Empty pages produce no chunks."""
    splitter = _splitter(chunk_size, chunk_overlap)
    chunks: list[TextChunk] = []
    for page in pages:
        if page.is_empty:
            continue
        pieces = splitter.split_text(page.text)
        for piece in pieces:
            text = piece.strip()
            if not text:
                continue
            chunks.append(
                TextChunk(
                    document_name=page.document_name,
                    page_number=page.page_number,
                    chunk_index=len(chunks),
                    text=text,
                )
            )
    return chunks
