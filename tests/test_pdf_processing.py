"""PDF extraction: valid files, empty pages, and rejected uploads."""

from pathlib import Path

import pymupdf
import pytest

from docqa.errors import InvalidPDFError
from docqa.pdf_processing import extract_pages, validate_pdf_bytes
from docqa.sample_document import (
    CLASS_FACT,
    LIBRARY_FACT,
    REFUND_FACT,
    build_sample_pdf_bytes,
)


def _flat(text: str) -> str:
    """Ignore line wraps inserted by the PDF text box."""
    return " ".join(text.split())


def test_sample_pdf_keeps_page_numbers_and_empty_page():
    pages = extract_pages(build_sample_pdf_bytes(), "notes/campus_handbook.pdf")

    assert [page.page_number for page in pages] == [1, 2, 3, 4]
    assert pages[0].document_name == "campus_handbook.pdf"
    assert LIBRARY_FACT in _flat(pages[0].text)
    assert pages[1].is_empty
    assert REFUND_FACT in _flat(pages[2].text)
    assert CLASS_FACT in _flat(pages[3].text)
    assert pages[2].page_number == 3


def test_checked_in_sample_pdf_is_readable():
    sample = Path(__file__).resolve().parents[1] / "samples" / "campus_handbook.pdf"
    pages = extract_pages(sample.read_bytes(), sample.name)

    assert [page.page_number for page in pages] == [1, 2, 3, 4]
    assert pages[1].is_empty
    assert "14 days" in _flat(pages[2].text)


def test_empty_upload_and_non_pdf_are_rejected():
    with pytest.raises(InvalidPDFError, match="empty"):
        validate_pdf_bytes(b"", "notes.pdf")

    with pytest.raises(InvalidPDFError, match="not a PDF"):
        extract_pages(b"this is plain text", "essay.pdf")


def test_damaged_pdf_header_is_rejected():
    # The header looks real, but the body is not a PDF PyMuPDF can parse.
    with pytest.raises(InvalidPDFError, match="could not be opened"):
        extract_pages(b"%PDF-1.4\nthis is not a real pdf object", "broken.pdf")


def test_password_protected_pdf_is_rejected():
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Hidden library hours.")
    encrypted = document.tobytes(
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        user_pw="secret",
        owner_pw="owner-secret",
    )
    document.close()

    with pytest.raises(InvalidPDFError, match="password-protected"):
        extract_pages(encrypted, "locked.pdf")
