"""Retrieval + generation chain: ask a question, get a cited, grounded answer.

The LLM call is behind a one-method interface (``LLMProvider.generate``) so
tests can inject a fake/deterministic implementation instead of a real
Gemini/OpenAI call. ``answer_question`` is the single function both the
Streamlit app and the test suite drive: given a built ``VectorStore`` and a
question, it retrieves the top-k chunks, builds the grounding prompt, calls
the LLM, and returns the answer together with the exact source chunks used
(citations), so the UI can always show its receipts.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol

from .vectorstore import SearchResult, VectorStore

PROMPT_TEMPLATE = (
    "You are a precise document assistant. Answer ONLY from the context below.\n"
    "If the answer is not present, say 'Not found in document.'\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)

NO_API_KEY_MESSAGE = (
    "No live LLM configured. Set GOOGLE_API_KEY (or GEMINI_API_KEY) or "
    "OPENAI_API_KEY in your environment to get a generated answer. The "
    "retrieved source chunks below are real and still useful on their own."
)

NO_CONTEXT_MESSAGE = "Not found in document."


class LLMProvider(Protocol):
    """Minimal interface every LLM backend implements."""

    def generate(self, prompt: str) -> str:
        ...


class GeminiLLM:
    """Wraps ``langchain_google_genai.ChatGoogleGenerativeAI``."""

    def __init__(self, model: str = "models/gemini-2.5-flash", temperature: float = 0.0) -> None:
        from langchain_google_genai import ChatGoogleGenerativeAI

        self._llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)

    def generate(self, prompt: str) -> str:
        return self._llm.invoke(prompt).content


class OpenAILLM:
    """Wraps the raw OpenAI chat completions endpoint (no langchain dependency)."""

    def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0.1) -> None:
        from openai import OpenAI

        self._client = OpenAI()
        self._model = model
        self._temperature = temperature

    def generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
            max_tokens=500,
        )
        return response.choices[0].message.content


class NoKeyLLM:
    """Fallback used when no LLM API key is configured. Never calls the network."""

    def generate(self, prompt: str) -> str:  # noqa: ARG002 - interface parity
        return NO_API_KEY_MESSAGE


class FakeLLM:
    """Deterministic test double. Wraps a plain function of (prompt) -> str."""

    def __init__(self, fn: Optional[Callable[[str], str]] = None, fixed_answer: Optional[str] = None) -> None:
        if fn is not None and fixed_answer is not None:
            raise ValueError("Pass either fn or fixed_answer, not both.")
        self._fn = fn
        self._fixed_answer = fixed_answer
        self.calls: List[str] = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        if self._fn is not None:
            return self._fn(prompt)
        return self._fixed_answer if self._fixed_answer is not None else "mock answer"


def get_default_llm() -> LLMProvider:
    """Choose a real LLM backend if a matching API key is set, else a safe no-op."""
    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        if os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
        return GeminiLLM()
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAILLM()
    return NoKeyLLM()


def format_context(results: List[SearchResult]) -> str:
    """Render retrieved chunks as a numbered, source-tagged context block."""
    parts = []
    for i, r in enumerate(results, start=1):
        parts.append(f"[Chunk {i} - {r.chunk.source}]\n{r.chunk.text}")
    return "\n\n".join(parts)


@dataclass
class Citation:
    source: str
    chunk_index: int
    text: str
    score: float


@dataclass
class Answer:
    question: str
    answer: str
    citations: List[Citation] = field(default_factory=list)


def answer_question(
    vectorstore: VectorStore,
    question: str,
    llm: Optional[LLMProvider] = None,
    k: int = 3,
) -> Answer:
    """Retrieve top-k chunks for ``question``, generate a grounded answer, cite sources.

    If retrieval finds nothing (empty store, or a query with zero relevant
    chunks returned by the index), the LLM is not called at all and the
    canned "not found" message is returned with no citations - this keeps
    the function safe to call with an empty/unbuilt vector store and avoids
    wasting an LLM call when there is nothing to ground an answer in.
    """
    results = vectorstore.similarity_search(question, k=k)
    if not results:
        return Answer(question=question, answer=NO_CONTEXT_MESSAGE, citations=[])

    llm = llm or get_default_llm()
    context = format_context(results)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    generated = llm.generate(prompt)

    citations = [
        Citation(source=r.chunk.source, chunk_index=r.chunk.chunk_index, text=r.chunk.text, score=r.score)
        for r in results
    ]
    return Answer(question=question, answer=generated, citations=citations)
