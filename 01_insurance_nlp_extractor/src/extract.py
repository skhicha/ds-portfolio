"""PDF text extraction for insurance documents.

Wraps pdfplumber so the rest of the pipeline never has to deal with the
quirks of opening a PDF from a path, raw bytes, or a file-like object (e.g.
Streamlit's ``UploadedFile``).
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import pdfplumber

PdfSource = Union[str, Path, bytes, io.IOBase]


class PdfExtractionError(Exception):
    """Raised when a PDF cannot be opened or parsed at all."""


def extract_text(source: PdfSource) -> str:
    """Extract all text from a PDF, concatenating every page with a space.

    Accepts a file path (str/Path), raw bytes, or any file-like object with
    a ``.read()`` method (e.g. an uploaded Streamlit file). Pages that yield
    no text (scanned/image-only pages) contribute an empty string rather
    than raising.

    Raises ``PdfExtractionError`` if the source cannot be parsed as a PDF at
    all (e.g. malformed/corrupted bytes, not a PDF).
    """
    if isinstance(source, bytes):
        source = io.BytesIO(source)

    try:
        with pdfplumber.open(source) as pdf:
            return " ".join(page.extract_text() or "" for page in pdf.pages)
    except Exception as exc:  # pdfplumber/pdfminer raise several exception types
        raise PdfExtractionError(f"Could not extract text from PDF: {exc}") from exc


def extract_pages(source: PdfSource) -> list[str]:
    """Extract text per page, returning a list of page strings (may be empty)."""
    if isinstance(source, bytes):
        source = io.BytesIO(source)

    try:
        with pdfplumber.open(source) as pdf:
            return [page.extract_text() or "" for page in pdf.pages]
    except Exception as exc:
        raise PdfExtractionError(f"Could not extract text from PDF: {exc}") from exc
