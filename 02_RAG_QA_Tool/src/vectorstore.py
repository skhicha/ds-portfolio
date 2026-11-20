"""Embedding providers and a FAISS-backed vector store.

Two embedding backends share one interface (``fit_embed_documents`` /
``embed_query``):

- ``OfflineHashingEmbeddings`` - a deterministic, fully local TF-IDF based
  embedding. No network call, no model download, no API key. This is what
  the test suite uses, and what the app falls back to when no LLM API key
  is configured, so the retrieval half of the pipeline is always
  demonstrable offline.
- ``GeminiEmbeddings`` / ``OpenAIEmbeddingsProvider`` - thin wrappers around
  the real hosted embedding APIs, imported lazily so this module (and
  anything that imports it, including the test suite) never requires those
  packages or network access just to be imported.

``get_default_embedding_provider`` picks a real provider when a matching API
key is present in the environment, and the offline fallback otherwise.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Tuple

import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .ingest import Chunk


class EmbeddingProvider(Protocol):
    """Minimal interface every embedding backend implements."""

    def fit_embed_documents(self, texts: List[str]) -> np.ndarray:
        ...

    def embed_query(self, text: str) -> np.ndarray:
        ...


def _normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize rows so inner product == cosine similarity in FAISS."""
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class OfflineHashingEmbeddings:
    """Deterministic, offline TF-IDF embedding.

    Fit once on the document corpus at index-build time, then queries are
    transformed with that same fitted vocabulary. Safe for tests (no
    randomness beyond what the input text implies) and for local demos
    without any API key.
    """

    def __init__(self, max_features: int = 2048) -> None:
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_features=max_features,
        )
        self._fitted = False

    def fit_embed_documents(self, texts: List[str]) -> np.ndarray:
        if not texts:
            self._fitted = False
            return np.zeros((0, 1), dtype="float32")
        matrix = self.vectorizer.fit_transform(texts).toarray().astype("float32")
        self._fitted = True
        return _normalize(matrix)

    def embed_query(self, text: str) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("fit_embed_documents() must be called before embed_query().")
        vec = self.vectorizer.transform([text]).toarray().astype("float32")
        return _normalize(vec)[0]

    @property
    def dimension(self) -> int:
        return len(self.vectorizer.vocabulary_) if self._fitted else 0


class GeminiEmbeddings:
    """Wraps ``langchain_google_genai.GoogleGenerativeAIEmbeddings``."""

    def __init__(self, model: str = "models/gemini-embedding-001") -> None:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        self._embeddings = GoogleGenerativeAIEmbeddings(model=model)

    def fit_embed_documents(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype="float32")
        vectors = np.array(self._embeddings.embed_documents(texts), dtype="float32")
        return _normalize(vectors)

    def embed_query(self, text: str) -> np.ndarray:
        vector = np.array(self._embeddings.embed_query(text), dtype="float32")
        return _normalize(vector.reshape(1, -1))[0]


class OpenAIEmbeddingsProvider:
    """Wraps the raw OpenAI embeddings endpoint (no langchain dependency)."""

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        from openai import OpenAI

        self._client = OpenAI()
        self._model = model

    def fit_embed_documents(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype="float32")
        resp = self._client.embeddings.create(model=self._model, input=texts)
        vectors = np.array([d.embedding for d in resp.data], dtype="float32")
        return _normalize(vectors)

    def embed_query(self, text: str) -> np.ndarray:
        resp = self._client.embeddings.create(model=self._model, input=[text])
        vector = np.array(resp.data[0].embedding, dtype="float32")
        return _normalize(vector.reshape(1, -1))[0]


def get_default_embedding_provider() -> EmbeddingProvider:
    """Choose a real embedding backend if a matching API key is set, else fall back."""
    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        if os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
        return GeminiEmbeddings()
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIEmbeddingsProvider()
    return OfflineHashingEmbeddings()


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


@dataclass
class VectorStore:
    """A FAISS index over embedded chunks, with the chunk metadata alongside it."""

    chunks: List[Chunk] = field(default_factory=list)
    embedding_provider: Optional[EmbeddingProvider] = None
    _index: Optional["faiss.Index"] = field(default=None, repr=False)

    @classmethod
    def build(cls, chunks: List[Chunk], embedding_provider: Optional[EmbeddingProvider] = None) -> "VectorStore":
        """Embed every chunk and build a FAISS inner-product index over them."""
        provider = embedding_provider or get_default_embedding_provider()
        store = cls(chunks=list(chunks), embedding_provider=provider)
        if not chunks:
            store._index = None
            return store
        vectors = provider.fit_embed_documents([c.text for c in chunks])
        dim = vectors.shape[1] if vectors.ndim == 2 and vectors.shape[0] else 1
        index = faiss.IndexFlatIP(dim)
        if vectors.shape[0]:
            index.add(vectors)
        store._index = index
        return store

    def similarity_search(self, query: str, k: int = 3) -> List[SearchResult]:
        """Return the top-k chunks most similar to ``query``, best first."""
        if not self.chunks or self._index is None or self._index.ntotal == 0:
            return []
        query_vec = self.embedding_provider.embed_query(query).reshape(1, -1).astype("float32")
        k = min(k, self._index.ntotal)
        scores, indices = self._index.search(query_vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append(SearchResult(chunk=self.chunks[idx], score=float(score)))
        return results

    def __len__(self) -> int:
        return len(self.chunks)
