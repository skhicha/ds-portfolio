from __future__ import annotations

from src.ingest import Chunk
from src.vectorstore import OfflineHashingEmbeddings, VectorStore


def make_chunks(*texts, source="doc.pdf"):
    return [Chunk(text=t, source=source, chunk_index=i) for i, t in enumerate(texts)]


class TestOfflineHashingEmbeddings:
    def test_fit_embed_documents_shape(self):
        emb = OfflineHashingEmbeddings()
        vectors = emb.fit_embed_documents(["cats and dogs", "dogs and cats", "stock market prices"])
        assert vectors.shape[0] == 3
        assert vectors.shape[1] == emb.dimension
        assert emb.dimension > 0

    def test_embed_query_matches_document_dimension(self):
        emb = OfflineHashingEmbeddings()
        emb.fit_embed_documents(["annual leave policy", "warranty coverage terms"])
        query_vec = emb.embed_query("leave policy")
        assert query_vec.shape == (emb.dimension,)

    def test_embed_query_before_fit_raises(self):
        emb = OfflineHashingEmbeddings()
        try:
            emb.embed_query("anything")
            assert False, "expected RuntimeError"
        except RuntimeError:
            pass

    def test_fit_on_empty_corpus_is_safe(self):
        emb = OfflineHashingEmbeddings()
        vectors = emb.fit_embed_documents([])
        assert vectors.shape[0] == 0
        assert emb.dimension == 0

    def test_vectors_are_unit_normalized(self):
        emb = OfflineHashingEmbeddings()
        vectors = emb.fit_embed_documents(["the quick brown fox", "jumps over the lazy dog"])
        import numpy as np

        norms = np.linalg.norm(vectors, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)


class TestVectorStore:
    def test_build_and_search_returns_most_relevant_chunk_first(self):
        chunks = make_chunks(
            "The annual leave policy allows 24 days of paid leave per year.",
            "The warranty covers defects in materials and workmanship for three years.",
            "Health insurance covers hospitalization and outpatient treatment.",
        )
        store = VectorStore.build(chunks, embedding_provider=OfflineHashingEmbeddings())
        results = store.similarity_search("how many days of paid leave do employees get", k=1)
        assert len(results) == 1
        assert "leave" in results[0].chunk.text.lower()

    def test_search_orders_by_descending_score(self):
        chunks = make_chunks(
            "warranty warranty warranty coverage period three years",
            "completely unrelated text about kitchen recipes and baking bread",
        )
        store = VectorStore.build(chunks, embedding_provider=OfflineHashingEmbeddings())
        results = store.similarity_search("warranty coverage period", k=2)
        assert len(results) == 2
        assert results[0].score >= results[1].score

    def test_k_larger_than_available_chunks_is_clipped_not_crashed(self):
        chunks = make_chunks("only one chunk here about insurance premiums")
        store = VectorStore.build(chunks, embedding_provider=OfflineHashingEmbeddings())
        results = store.similarity_search("insurance premium", k=50)
        assert len(results) == 1

    def test_empty_chunk_list_builds_empty_store(self):
        store = VectorStore.build([], embedding_provider=OfflineHashingEmbeddings())
        assert len(store) == 0
        assert store.similarity_search("anything", k=3) == []

    def test_very_short_chunks_do_not_crash(self):
        chunks = make_chunks("ok", "a", "warranty", "")
        store = VectorStore.build(chunks, embedding_provider=OfflineHashingEmbeddings())
        results = store.similarity_search("warranty", k=4)
        assert isinstance(results, list)
