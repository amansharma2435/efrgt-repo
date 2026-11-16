"""Settings and secret lookup.

The Gemini key is read from the process environment or, when the Streamlit
server is actually running, from ``.streamlit/secrets.toml``. Nothing in this
module prints the key.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from docqa.errors import MissingAPIKeyError

# Load a local .env file if the student created one. Existing environment
# variables win, so a deployed host can set the key without a file.
load_dotenv()

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_CHUNK_SIZE = 700
DEFAULT_CHUNK_OVERLAP = 120
DEFAULT_TOP_K = 3
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

_KEY_NAMES = ("GEMINI_API_KEY", "GOOGLE_API_KEY")


def _running_inside_streamlit() -> bool:
    """True only during a Streamlit script run, not during unit tests."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        return False
    try:
        return get_script_run_ctx() is not None
    except Exception:
        return False


def _read_streamlit_secret(name: str) -> str | None:
    if not _running_inside_streamlit():
        return None
    try:
        import streamlit as st

        if name not in st.secrets:
            return None
        value = str(st.secrets[name]).strip()
    except Exception:
        return None
    return value or None


def get_gemini_api_key() -> str:
    """Return the configured Gemini API key.

    Lookup order:
    1. Environment variable ``GEMINI_API_KEY``
    2. Environment variable ``GOOGLE_API_KEY``
    3. The same names in Streamlit secrets, when the app is running
    """
    for name in _KEY_NAMES:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    for name in _KEY_NAMES:
        value = _read_streamlit_secret(name)
        if value:
            return value
    raise MissingAPIKeyError(
        "No Gemini API key was found. Add GEMINI_API_KEY to a .env file "
        "or to .streamlit/secrets.toml, then restart the app. "
        "Indexing a PDF does not need a key. Asking a question does."
    )


def get_gemini_model() -> str:
    """Model id sent to the Gemini API. Override with GEMINI_MODEL."""
    value = os.environ.get("GEMINI_MODEL", "").strip()
    if value:
        return value
    secret = _read_streamlit_secret("GEMINI_MODEL")
    if secret:
        return secret
    return DEFAULT_GEMINI_MODEL
