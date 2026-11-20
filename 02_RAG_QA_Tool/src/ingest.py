"""PDF text extraction and chunking for the RAG Q&A pipeline.

Two steps, kept intentionally separate and dependency-light:
1. ``extract_text_from_pdf`` pulls raw text out of a PDF with pdfplumber.
2. ``chunk_text`` splits that text into overlapping chunks with LangChain's
   ``RecursiveCharacterTextSplitter`` so each chunk is small enough to embed
   and retrieve independently, while overlap keeps context from being cut
   mid-sentence at chunk boundaries.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import BinaryIO, Dict, List, Union

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100

PDFSource = Union[str, bytes, BinaryIO]


def extract_text_from_pdf(source: PDFSource) -> str:
    """Extract and concatenate text from every page of a PDF.

    Accepts a filesystem path, raw bytes, or any file-like object pdfplumber
    can open (e.g. a Streamlit ``UploadedFile`` or ``io.BytesIO``). Pages with
    no extractable text (scanned images, etc.) contribute an empty string
    rather than raising.
    """
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    with pdfplumber.open(source) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n".join(pages).strip()


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """Split ``text`` into overlapping chunks.

    Returns an empty list for empty/whitespace-only text instead of a list
    containing one empty chunk.
    """
    if not text or not text.strip():
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)


@dataclass(frozen=True)
class Chunk:
    """A single retrievable unit: a piece of text tagged with its origin."""

    text: str
    source: str
    chunk_index: int


def load_and_chunk(
    sources: Dict[str, PDFSource],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Chunk]:
    """Extract text from each named PDF source and split into tagged chunks.

    ``sources`` maps a display name (e.g. an uploaded filename) to a PDF
    source (path, bytes, or file-like). Documents that extract to no text
    contribute zero chunks rather than raising.
    """
    chunks: List[Chunk] = []
    for name, source in sources.items():
        text = extract_text_from_pdf(source)
        for i, piece in enumerate(chunk_text(text, chunk_size, chunk_overlap)):
            chunks.append(Chunk(text=piece, source=name, chunk_index=i))
    return chunks
