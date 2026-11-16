"""Extract text from a PDF, one page at a time.

PyMuPDF (imported as ``fitz``) reads the text layer of a PDF. A scanned page
is just a picture, so this module returns an empty string for it instead of
guessing words. Page numbers are 1-based, matching the numbers printed by a
PDF reader.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pymupdf

from docqa.errors import InvalidPDFError

_PDF_MAGIC = b"%PDF"


@dataclass(frozen=True)
class PageText:
    """Text of a single PDF page, including pages that have no text."""

    document_name: str
    page_number: int
    text: str

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


def safe_document_name(filename: str) -> str:
    """Keep only the file name, so a path cannot be shown in the interface."""
    cleaned = os.path.basename((filename or "").replace("\\", "/")).strip()
    if not cleaned:
        return "document.pdf"
    return cleaned


def validate_pdf_bytes(data: bytes, filename: str = "document.pdf") -> str:
    """Reject empty files, non-PDFs, and oversized uploads by content.

    The caller passes the raw bytes. The extension is not trusted: a renamed
    text file still fails the ``%PDF`` header check.
    """
    document_name = safe_document_name(filename)
    if not data:
        raise InvalidPDFError(f"'{document_name}' is empty.")
    if len(data) > 25 * 1024 * 1024:
        raise InvalidPDFError(
            f"'{document_name}' is larger than 25 MB. Upload a smaller PDF."
        )
    header = data[:1024].lstrip()
    if not header.startswith(_PDF_MAGIC):
        raise InvalidPDFError(
            f"'{document_name}' is not a PDF file. Upload a file that starts "
            "with a real PDF header."
        )
    return document_name


def clean_page_text(text: str) -> str:
    """Drop null bytes and trailing spaces while keeping paragraph breaks."""
    without_nulls = text.replace("\x00", "")
    lines = [line.rstrip() for line in without_nulls.splitlines()]
    return "\n".join(lines).strip()


def extract_pages(data: bytes, filename: str) -> list[PageText]:
    """Return every page of the PDF, in order.

    Empty pages are kept in the result so the page count stays honest. The
    chunker skips them later. Password-protected files are rejected because
    PyMuPDF would otherwise return blank text and look like a scan.
    """
    document_name = validate_pdf_bytes(data, filename)
    try:
        document = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise InvalidPDFError(
            f"'{document_name}' could not be opened. It may be damaged."
        ) from exc

    try:
        if document.needs_pass:
            raise InvalidPDFError(
                f"'{document_name}' is password-protected. "
                "Remove the password and upload it again."
            )
        if document.page_count == 0:
            raise InvalidPDFError(f"'{document_name}' has no pages.")

        pages: list[PageText] = []
        for index, page in enumerate(document, start=1):
            raw = page.get_text("text") or ""
            pages.append(
                PageText(
                    document_name=document_name,
                    page_number=index,
                    text=clean_page_text(raw),
                )
            )
        return pages
    except InvalidPDFError:
        raise
    except Exception as exc:
        raise InvalidPDFError(
            f"Text could not be read from '{document_name}'."
        ) from exc
    finally:
        document.close()
