import io

import pytest

from src.extract import PdfExtractionError, extract_pages, extract_text


class TestExtractTextRealPdfs:
    def test_extract_text_returns_nonempty_string_for_each_sample(self, sample_pdf_paths):
        for name, path in sample_pdf_paths.items():
            text = extract_text(path)
            assert isinstance(text, str)
            assert len(text.strip()) > 100, f"{name} PDF yielded suspiciously little text"

    def test_health_pdf_contains_expected_keywords(self, sample_pdf_paths):
        text = extract_text(sample_pdf_paths["health"])
        assert "Policy" in text
        assert "Premium" in text
        assert "Star Health" in text

    def test_extract_text_accepts_bytes(self, sample_pdf_paths):
        with open(sample_pdf_paths["motor"], "rb") as f:
            raw = f.read()
        text_from_bytes = extract_text(raw)
        text_from_path = extract_text(sample_pdf_paths["motor"])
        assert text_from_bytes == text_from_path

    def test_extract_text_accepts_file_like_object(self, sample_pdf_paths):
        with open(sample_pdf_paths["home"], "rb") as f:
            buf = io.BytesIO(f.read())
        text = extract_text(buf)
        assert "Bajaj" in text

    def test_extract_pages_returns_list_of_strings(self, sample_pdf_paths):
        pages = extract_pages(sample_pdf_paths["health"])
        assert isinstance(pages, list)
        assert len(pages) >= 1
        assert all(isinstance(p, str) for p in pages)


class TestExtractTextEdgeCases:
    def test_malformed_pdf_bytes_raises_extraction_error(self):
        with pytest.raises(PdfExtractionError):
            extract_text(b"this is not a pdf file at all, just plain text bytes")

    def test_empty_bytes_raises_extraction_error(self):
        with pytest.raises(PdfExtractionError):
            extract_text(b"")

    def test_nonexistent_path_raises_extraction_error(self):
        with pytest.raises(PdfExtractionError):
            extract_text("/tmp/this/path/does/not/exist/nope.pdf")
