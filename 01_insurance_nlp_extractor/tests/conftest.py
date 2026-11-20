import os

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SAMPLE_PDFS = {
    "health": os.path.join(PROJECT_ROOT, "data", "sample_health_insurance.pdf"),
    "home": os.path.join(PROJECT_ROOT, "data", "sample_home_insurance.pdf"),
    "motor": os.path.join(PROJECT_ROOT, "data", "sample_motor_insurance.pdf"),
}


@pytest.fixture(scope="session")
def sample_pdf_paths():
    return SAMPLE_PDFS


@pytest.fixture(scope="session")
def sample_texts(sample_pdf_paths):
    """Real text extracted from the 3 bundled sample PDFs, keyed by name."""
    from src.extract import extract_text

    return {name: extract_text(path) for name, path in sample_pdf_paths.items()}


@pytest.fixture
def health_insurance_text():
    return """
    Star Health and Allied Insurance Co. Ltd.
    Policy No. SHHI-2024-HP-004821
    Sum Insured INR 5,00,000
    Basic Premium INR 12,450
    GST (18%) INR 2,241
    Total Premium INR 14,691
    Start Date 20-01-2024
    End Date 19-01-2025
    """


@pytest.fixture
def nlp_model():
    """Load the spaCy model once per test session (expensive to construct)."""
    import spacy

    return spacy.load("en_core_web_sm")
