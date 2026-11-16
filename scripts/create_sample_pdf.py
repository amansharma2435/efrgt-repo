"""Write samples/campus_handbook.pdf for manual trials in the web app.

Usage:
    python scripts/create_sample_pdf.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docqa.sample_document import SAMPLE_FILENAME, write_sample_pdf


def main() -> None:
    destination = Path(__file__).resolve().parents[1] / "samples" / SAMPLE_FILENAME
    write_sample_pdf(destination)
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
