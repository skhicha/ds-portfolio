from __future__ import annotations

import pytest

from src.ingest import chunk_text, extract_text_from_pdf, load_and_chunk


class TestExtractTextFromPdf:
    def test_extracts_nonempty_text_from_each_sample(self, sample_pdf_paths):
        for name, path in sample_pdf_paths.items():
            text = extract_text_from_pdf(path)
            assert isinstance(text, str)
            assert len(text) > 200, f"{name} extracted suspiciously little text"

    def test_extracted_text_contains_expected_keywords(self, sample_pdf_paths):
        insurance_text = extract_text_from_pdf(sample_pdf_paths["sample_insurance_policy.pdf"])
        leave_text = extract_text_from_pdf(sample_pdf_paths["sample_leave_policy.pdf"])
        warranty_text = extract_text_from_pdf(sample_pdf_paths["sample_warranty_manual.pdf"])

        assert "coverage" in insurance_text.lower()
        assert "leave" in leave_text.lower()
        assert "warranty" in warranty_text.lower()

    def test_accepts_bytes_source(self, sample_pdf_paths):
        path = sample_pdf_paths["sample_leave_policy.pdf"]
        with open(path, "rb") as f:
            raw = f.read()
        text_from_bytes = extract_text_from_pdf(raw)
        text_from_path = extract_text_from_pdf(path)
        assert text_from_bytes == text_from_path

    def test_malformed_pdf_raises(self, tmp_path):
        bad_pdf = tmp_path / "not_really_a_pdf.pdf"
        bad_pdf.write_bytes(b"this is not pdf content at all")
        with pytest.raises(Exception):
            extract_text_from_pdf(str(bad_pdf))


class TestChunkText:
    def test_empty_text_returns_no_chunks(self):
        assert chunk_text("") == []
        assert chunk_text("   \n\t  ") == []

    def test_short_text_returns_single_chunk(self):
        chunks = chunk_text("A short sentence.", chunk_size=800, chunk_overlap=100)
        assert chunks == ["A short sentence."]

    def test_chunks_respect_max_size_with_some_tolerance(self):
        long_text = ("Energy consumption forecasting is a time series problem. " * 100).strip()
        chunks = chunk_text(long_text, chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1
        # the recursive splitter prefers natural boundaries and can slightly
        # exceed chunk_size when no good split point exists nearby
        assert all(len(c) <= 260 for c in chunks)

    def test_consecutive_chunks_overlap_exactly(self):
        # No spaces/newlines forces character-level splitting, where the
        # splitter's overlap behavior is exact and easy to assert on.
        text = "".join(str(i % 10) for i in range(300))
        chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)
        assert len(chunks) > 1
        for a, b in zip(chunks, chunks[1:]):
            assert a[-10:] == b[:10]


class TestLoadAndChunk:
    def test_real_sample_pdfs_produce_tagged_chunks(self, sample_pdf_paths):
        chunks = load_and_chunk(sample_pdf_paths)
        assert len(chunks) > 0
        sources = {c.source for c in chunks}
        assert sources == set(sample_pdf_paths.keys())
        for c in chunks:
            assert c.text.strip() != ""
            assert c.chunk_index >= 0

    def test_chunk_indices_are_sequential_per_source(self, sample_pdf_paths):
        chunks = load_and_chunk(sample_pdf_paths)
        by_source: dict[str, list[int]] = {}
        for c in chunks:
            by_source.setdefault(c.source, []).append(c.chunk_index)
        for source, indices in by_source.items():
            assert indices == list(range(len(indices))), f"non-sequential indices for {source}"

    def test_empty_source_dict_returns_no_chunks(self):
        assert load_and_chunk({}) == []
