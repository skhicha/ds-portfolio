"""End-to-end pipeline tests against the real sample PDFs.

These exercise ingest -> chunk -> embed (offline, deterministic) -> FAISS
retrieve -> (mocked) generate, with no network access and no API key,
matching how the test suite is documented to run in the README.
"""
from __future__ import annotations

from src.ingest import load_and_chunk
from src.qa import FakeLLM, answer_question
from src.vectorstore import OfflineHashingEmbeddings, VectorStore


def test_real_pdfs_produce_chunks_for_every_document(sample_pdf_paths):
    chunks = load_and_chunk(sample_pdf_paths)
    assert len(chunks) >= len(sample_pdf_paths)
    sources = {c.source for c in chunks}
    assert sources == set(sample_pdf_paths.keys())


def test_retrieval_prefers_the_document_the_question_is_about(sample_pdf_paths):
    chunks = load_and_chunk(sample_pdf_paths)
    store = VectorStore.build(chunks, embedding_provider=OfflineHashingEmbeddings())

    leave_results = store.similarity_search("How many days of annual leave do employees get?", k=1)
    warranty_results = store.similarity_search("What does the laptop warranty cover?", k=1)

    assert leave_results[0].chunk.source == "sample_leave_policy.pdf"
    assert warranty_results[0].chunk.source == "sample_warranty_manual.pdf"


def test_full_pipeline_end_to_end_with_mocked_generation(sample_pdf_paths):
    chunks = load_and_chunk(sample_pdf_paths)
    store = VectorStore.build(chunks, embedding_provider=OfflineHashingEmbeddings())

    llm = FakeLLM(fixed_answer="Employees get 24 days of paid annual leave per year.")
    result = answer_question(store, "How many days of annual leave are employees entitled to?", llm=llm, k=3)

    assert result.answer == "Employees get 24 days of paid annual leave per year."
    assert len(result.citations) == 3
    assert any(c.source == "sample_leave_policy.pdf" for c in result.citations)
    assert len(llm.calls) == 1


def test_question_with_no_relevant_chunks_in_an_unrelated_single_doc_store(sample_pdf_paths):
    # Build a store from only the warranty doc, then ask about leave policy.
    # Retrieval still returns the (irrelevant) top-k warranty chunks rather
    # than nothing - FAISS always returns its k nearest neighbors - but the
    # pipeline must not crash, and the mocked LLM still receives a prompt.
    single_doc = {"sample_warranty_manual.pdf": sample_pdf_paths["sample_warranty_manual.pdf"]}
    chunks = load_and_chunk(single_doc)
    store = VectorStore.build(chunks, embedding_provider=OfflineHashingEmbeddings())

    llm = FakeLLM(fixed_answer="Not found in document.")
    result = answer_question(store, "How many vacation days do I get per year?", llm=llm, k=2)

    assert result.answer == "Not found in document."
    assert all(c.source == "sample_warranty_manual.pdf" for c in result.citations)
