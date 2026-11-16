"""Chunking keeps document name and page number, and skips empty pages."""

from docqa.pdf_processing import PageText
from docqa.sample_document import build_sample_pdf_bytes
from docqa.pdf_processing import extract_pages
from docqa.text_chunking import chunk_pages


def test_chunks_keep_document_and_page_metadata():
    pages = extract_pages(build_sample_pdf_bytes(), "campus_handbook.pdf")
    chunks = chunk_pages(pages, chunk_size=500, chunk_overlap=50)

    assert chunks, "The handbook should produce at least one chunk."
    assert all(chunk.document_name == "campus_handbook.pdf" for chunk in chunks)
    # Page 2 is blank, so no chunk should cite it.
    assert 2 not in {chunk.page_number for chunk in chunks}
    pages_present = {chunk.page_number for chunk in chunks}
    assert pages_present == {1, 3, 4}
    assert chunks[0].chunk_index == 0
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_long_page_splits_into_overlapping_chunks():
    letter_block = "abcdefghij" * 30  # 300 characters, no spaces
    pages = [PageText("notes.pdf", page_number=4, text=letter_block)]

    chunks = chunk_pages(pages, chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(chunk.page_number == 4 for chunk in chunks)
    assert all(chunk.document_name == "notes.pdf" for chunk in chunks)
    assert all(len(chunk.text) <= 100 for chunk in chunks)
    # The boundary is repeated, so the tail of one chunk starts the next.
    assert chunks[1].text.startswith(chunks[0].text[-20:])


def test_empty_page_produces_no_chunk_and_does_not_merge_pages():
    pages = [
        PageText("a.pdf", 1, "Only on page one. " * 5),
        PageText("a.pdf", 2, "   \n"),
        PageText("a.pdf", 3, "Only on page three."),
    ]
    chunks = chunk_pages(pages, chunk_size=200, chunk_overlap=20)

    assert [chunk.page_number for chunk in chunks] == [1, 3]
    assert "page three" not in chunks[0].text
    assert "page one" not in chunks[1].text


def test_overlap_must_be_smaller_than_chunk_size():
    pages = [PageText("a.pdf", 1, "Some text " * 40)]
    try:
        chunk_pages(pages, chunk_size=100, chunk_overlap=100)
    except ValueError as exc:
        assert "overlap" in str(exc).lower()
    else:
        raise AssertionError("Invalid overlap should be rejected.")
