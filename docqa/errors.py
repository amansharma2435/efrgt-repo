"""Errors that are safe to show to a user.

Messages in this module describe what went wrong. They must never include an
API key, a file path from the server, or the raw document dump.
"""


class DocumentQAError(Exception):
    """Base class for expected failures in this project."""


class InvalidPDFError(DocumentQAError):
    """The upload is missing, corrupt, encrypted, or not a PDF."""


class EmptyDocumentError(DocumentQAError):
    """The PDF opened, but it has no text we can search."""


class MissingAPIKeyError(DocumentQAError):
    """Gemini was asked to answer, but no API key is configured."""


class AnswerGenerationError(DocumentQAError):
    """Gemini was called and did not return a usable answer."""
