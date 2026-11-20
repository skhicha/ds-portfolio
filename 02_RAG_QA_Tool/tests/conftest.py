from __future__ import annotations

import os

import pytest

from src.ingest import Chunk, extract_text_from_pdf, load_and_chunk
from src.qa import FakeLLM
from src.vectorstore import OfflineHashingEmbeddings, VectorStore

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

SAMPLE_PDFS = {
    "sample_insurance_policy.pdf": os.path.join(DATA_DIR, "sample_insurance_policy.pdf"),
    "sample_leave_policy.pdf": os.path.join(DATA_DIR, "sample_leave_policy.pdf"),
    "sample_warranty_manual.pdf": os.path.join(DATA_DIR, "sample_warranty_manual.pdf"),
}


@pytest.fixture()
def sample_pdf_paths():
    return dict(SAMPLE_PDFS)


@pytest.fixture()
def sample_texts(sample_pdf_paths):
    return {name: extract_text_from_pdf(path) for name, path in sample_pdf_paths.items()}


@pytest.fixture()
def sample_chunks(sample_pdf_paths):
    return load_and_chunk(sample_pdf_paths)


@pytest.fixture()
def built_vectorstore(sample_chunks):
    return VectorStore.build(sample_chunks, embedding_provider=OfflineHashingEmbeddings())


@pytest.fixture()
def fake_llm():
    return FakeLLM(fixed_answer="This is a mocked answer.")


@pytest.fixture()
def no_api_keys(monkeypatch):
    """Ensure no LLM/embedding API key leaks into a test from the real environment."""
    for var in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    return None


def make_chunks(*texts: str, source: str = "doc.pdf") -> list[Chunk]:
    return [Chunk(text=t, source=source, chunk_index=i) for i, t in enumerate(texts)]
