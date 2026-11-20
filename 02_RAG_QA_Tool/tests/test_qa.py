from __future__ import annotations

from src.ingest import Chunk
from src.qa import (
    NO_API_KEY_MESSAGE,
    NO_CONTEXT_MESSAGE,
    FakeLLM,
    NoKeyLLM,
    answer_question,
    format_context,
    get_default_llm,
)
from src.vectorstore import OfflineHashingEmbeddings, SearchResult, VectorStore


def make_store(*texts, source="doc.pdf"):
    chunks = [Chunk(text=t, source=source, chunk_index=i) for i, t in enumerate(texts)]
    return VectorStore.build(chunks, embedding_provider=OfflineHashingEmbeddings())


class TestFormatContext:
    def test_numbers_and_tags_each_chunk_by_source(self):
        results = [
            SearchResult(chunk=Chunk(text="first chunk", source="a.pdf", chunk_index=0), score=0.9),
            SearchResult(chunk=Chunk(text="second chunk", source="b.pdf", chunk_index=2), score=0.5),
        ]
        rendered = format_context(results)
        assert "[Chunk 1 - a.pdf]" in rendered
        assert "[Chunk 2 - b.pdf]" in rendered
        assert "first chunk" in rendered
        assert "second chunk" in rendered

    def test_empty_results_renders_empty_string(self):
        assert format_context([]) == ""


class TestAnswerQuestion:
    def test_mocked_llm_end_to_end_with_citations(self):
        store = make_store(
            "Annual leave is 24 days per calendar year for permanent employees.",
            "The warranty period is three years from the date of purchase.",
        )
        llm = FakeLLM(fixed_answer="Employees get 24 days of annual leave.")
        result = answer_question(store, "How many days of annual leave are employees entitled to?", llm=llm, k=1)

        assert result.answer == "Employees get 24 days of annual leave."
        assert len(result.citations) == 1
        assert "leave" in result.citations[0].text.lower()
        assert len(llm.calls) == 1
        assert "Context:" in llm.calls[0]
        assert result.question.startswith("How many days")

    def test_llm_receives_prompt_built_from_retrieved_context(self):
        store = make_store("The warranty covers manufacturing defects for three years.")
        captured = {}

        def fn(prompt: str) -> str:
            captured["prompt"] = prompt
            return "three years"

        llm = FakeLLM(fn=fn)
        result = answer_question(store, "How long is the warranty?", llm=llm, k=1)

        assert "warranty covers manufacturing defects" in captured["prompt"]
        assert "How long is the warranty?" in captured["prompt"]
        assert result.answer == "three years"

    def test_empty_vectorstore_short_circuits_without_calling_llm(self):
        store = VectorStore.build([], embedding_provider=OfflineHashingEmbeddings())
        llm = FakeLLM(fixed_answer="should never be returned")
        result = answer_question(store, "anything?", llm=llm, k=3)

        assert result.answer == NO_CONTEXT_MESSAGE
        assert result.citations == []
        assert llm.calls == []  # LLM must not be called when there is nothing to ground on

    def test_very_short_chunks_do_not_crash(self):
        store = make_store("ok", "warranty", "a")
        llm = FakeLLM(fixed_answer="fine")
        result = answer_question(store, "warranty?", llm=llm, k=3)
        assert isinstance(result.answer, str)


class TestNoKeyLLM:
    def test_generate_returns_informative_message_without_network(self):
        llm = NoKeyLLM()
        assert llm.generate("any prompt") == NO_API_KEY_MESSAGE

    def test_get_default_llm_falls_back_when_no_keys_set(self, no_api_keys):
        llm = get_default_llm()
        assert isinstance(llm, NoKeyLLM)

    def test_answer_question_without_api_key_is_informative_not_a_crash(self, no_api_keys):
        store = make_store("The deductible is $500 per claim.")
        result = answer_question(store, "What is the deductible?", k=1)
        assert result.answer == NO_API_KEY_MESSAGE
        assert len(result.citations) == 1
