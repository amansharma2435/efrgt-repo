"""Build the small handbook PDF used by the tests and the README demo.

The facts are deliberately on different pages so a retrieval test can check
that the right page comes back:

- page 1: library hours
- page 2: intentionally blank
- page 3: refund policy
- page 4: classroom schedule
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

SAMPLE_FILENAME = "campus_handbook.pdf"

LIBRARY_FACT = (
    "The central library opens at 8 AM and closes at 9 PM on weekdays. "
    "On Sundays the library remains closed."
)
REFUND_FACT = (
    "A student may request a full refund within 14 days of enrollment. "
    "After 14 days the tuition fee is non-refundable."
)
CLASS_FACT = (
    "Professor Mehta teaches Data Structures in Room 204 on Mondays at 10 AM."
)


def build_sample_pdf_bytes() -> bytes:
    """Create the handbook in memory and return the PDF bytes."""
    document = pymupdf.open()
    try:
        _write_page(
            document,
            "Campus Handbook\n\nLibrary hours\n\n" + LIBRARY_FACT,
        )
        document.new_page()  # page 2 is intentionally empty
        _write_page(
            document,
            "Refund policy\n\n" + REFUND_FACT,
        )
        _write_page(
            document,
            "Course schedule\n\n" + CLASS_FACT,
        )
        return document.tobytes()
    finally:
        document.close()


def write_sample_pdf(path: Path) -> Path:
    """Write the handbook to ``path`` and return that path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(build_sample_pdf_bytes())
    return path


def _write_page(document: pymupdf.Document, text: str) -> None:
    """Draw the page as short lines so a phrase is not split mid-word."""
    page = document.new_page()
    y = 72
    for paragraph in text.split("\n"):
        line = ""
        words = paragraph.split()
        if not words:
            y += 12
            continue
        for word in words:
            candidate = word if not line else f"{line} {word}"
            if len(candidate) > 78 and line:
                page.insert_text((72, y), line, fontsize=12, fontname="helv")
                y += 18
                line = word
            else:
                line = candidate
        page.insert_text((72, y), line, fontsize=12, fontname="helv")
        y += 28
