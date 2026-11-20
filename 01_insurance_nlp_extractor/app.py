"""
Streamlit dashboard for the Insurance Document NLP Extractor.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from src.extract import PdfExtractionError, extract_text
from src.patterns import NOT_FOUND, extract_fields, extract_regex_contexts
from src.schema_match import validate_extracted_fields

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(PROJECT_ROOT, "data")
SAMPLE_FILES = {
    "Health insurance sample": "sample_health_insurance.pdf",
    "Home insurance sample": "sample_home_insurance.pdf",
    "Motor insurance sample": "sample_motor_insurance.pdf",
}

FIELD_LABELS = {
    "policy_number": "Policy Number",
    "insurer": "Insurer",
    "premium": "Total Premium",
    "start_date": "Coverage Start",
    "end_date": "Coverage End",
    "sum_insured": "Sum Insured",
}
FIELD_ORDER = ["policy_number", "insurer", "premium", "start_date", "end_date", "sum_insured"]

st.set_page_config(
    page_title="Insurance NLP Extractor",
    page_icon="📄",
    layout="wide",
)

# --------------------------------------------------------------------------- #
# Styling - a small design system layered on the Streamlit theme set in
# .streamlit/config.toml. No raw unstyled dumps: every surface below is
# either a themed widget or an explicitly styled container.
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
    :root {
        --color-bg: #F6F8FB;
        --color-surface: #FFFFFF;
        --color-border: #E3E8F0;
        --color-text: #1A2233;
        --color-text-muted: #5C6679;
        --color-primary: #2749D6;
        --color-primary-soft: #E9EDFC;
        --color-good: #0F766E;
        --color-good-soft: #E6F7F4;
        --color-warn: #B45309;
        --color-warn-soft: #FDF2E3;
        --shadow-card: 0 1px 2px rgba(16, 24, 40, 0.04), 0 6px 16px rgba(16, 24, 40, 0.07);
    }

    ::selection { background: var(--color-primary-soft); color: var(--color-primary); }

    .block-container { padding-top: 2rem; max-width: 1280px; }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: var(--color-surface);
        border: 1px solid var(--color-border);
        border-radius: 12px;
        padding: 1rem 1.1rem 0.9rem 1.1rem;
        box-shadow: var(--shadow-card);
    }
    div[data-testid="stMetricLabel"] { color: var(--color-text-muted); font-weight: 600; }
    div[data-testid="stMetricValue"] { color: var(--color-primary); font-size: 1.35rem; }

    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--color-text);
        margin: 1.4rem 0 0.15rem 0;
    }
    .section-caption { color: var(--color-text-muted); font-size: 0.92rem; margin-bottom: 0.9rem; }

    .pill {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 600;
        margin-left: 0.4rem;
    }
    .pill-good { background: var(--color-good-soft); color: var(--color-good); }
    .pill-warn { background: var(--color-warn-soft); color: var(--color-warn); }

    .doc-card {
        background: var(--color-surface);
        border: 1px solid var(--color-border);
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        box-shadow: var(--shadow-card);
        margin-bottom: 1rem;
    }
    .doc-card h4 { margin: 0 0 0.6rem 0; color: var(--color-text); font-size: 1rem; }
    .field-row {
        display: flex;
        justify-content: space-between;
        padding: 0.4rem 0;
        border-bottom: 1px dashed var(--color-border);
        font-size: 0.92rem;
    }
    .field-row:last-child { border-bottom: none; }
    .field-label { color: var(--color-text-muted); font-weight: 600; }
    .field-value { color: var(--color-text); text-align: right; }

    .empty-state {
        text-align: center;
        padding: 3.2rem 1rem;
        color: var(--color-text-muted);
        border: 1px dashed var(--color-border);
        border-radius: 14px;
        background: var(--color-surface);
    }

    a, button, [tabindex] { outline-color: var(--color-primary) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Insurance Document NLP Extractor")
st.caption(
    "Upload insurance PDFs to pull structured fields with regex + spaCy NER, "
    "cross-checked against a TF-IDF schema matcher."
)

with st.sidebar:
    st.markdown("### Source")
    use_samples = st.checkbox("Use bundled sample PDFs", value=False)
    uploaded_files = st.file_uploader(
        "Upload one or more insurance PDFs", type="pdf", accept_multiple_files=True
    )
    st.markdown("---")
    st.markdown(
        "**Pipeline**\n\n"
        "1. `pdfplumber` extracts raw text\n"
        "2. Regex patterns pull policy number, premium, dates, sum insured\n"
        "3. spaCy NER (`en_core_web_sm`) pulls the insurer\n"
        "4. TF-IDF + cosine similarity cross-checks each value against a "
        "master field schema"
    )


def _load_sources():
    sources = []
    if use_samples:
        for label, fname in SAMPLE_FILES.items():
            path = os.path.join(SAMPLE_DIR, fname)
            if os.path.exists(path):
                sources.append((label, path))
    if uploaded_files:
        for f in uploaded_files:
            sources.append((f.name, f))
    return sources


sources = _load_sources()

if not sources:
    st.markdown(
        """
        <div class="empty-state">
            <strong>No documents loaded yet.</strong><br/>
            Upload a PDF in the sidebar, or check "Use bundled sample PDFs" to try it instantly.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

all_results = []
agreement_scores = []

for label, source in sources:
    with st.spinner(f"Processing {label}..."):
        try:
            text = extract_text(source)
        except PdfExtractionError as exc:
            st.error(f"**{label}**: could not read this PDF ({exc}). Skipping.")
            continue

        fields = extract_fields(text)
        contexts = extract_regex_contexts(text)
        validation = validate_extracted_fields(fields, contexts)

        row = {"document": label, **fields}
        all_results.append(row)

        found = sum(1 for f in FIELD_ORDER if fields.get(f, NOT_FOUND) != NOT_FOUND)
        agrees = sum(1 for v in validation.values() if v["agrees"])
        agreement_scores.append(agrees / len(FIELD_ORDER))

        st.markdown(
            f"""
            <div class="doc-card">
                <h4>{label}
                    <span class="pill pill-good">{found}/{len(FIELD_ORDER)} fields found</span>
                    <span class="pill pill-warn">{agrees}/{len(FIELD_ORDER)} schema-verified</span>
                </h4>
            """,
            unsafe_allow_html=True,
        )
        rows_html = ""
        for field in FIELD_ORDER:
            value = fields.get(field, NOT_FOUND)
            check = validation.get(field, {})
            badge = ""
            if value != NOT_FOUND:
                badge = (
                    ' <span class="pill pill-good">verified</span>'
                    if check.get("agrees")
                    else f' <span class="pill pill-warn">confidence {check.get("score", 0):.2f}</span>'
                )
            rows_html += (
                f'<div class="field-row"><span class="field-label">'
                f"{FIELD_LABELS[field]}</span><span class=\"field-value\">{value}{badge}</span></div>"
            )
        st.markdown(rows_html + "</div>", unsafe_allow_html=True)

if all_results:
    st.markdown('<div class="section-title">Aggregated Results</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">One row per document, ready to export.</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Documents processed", len(all_results))
    avg_found = sum(
        sum(1 for f in FIELD_ORDER if r.get(f, NOT_FOUND) != NOT_FOUND) for r in all_results
    ) / (len(all_results) * len(FIELD_ORDER))
    col2.metric("Field extraction rate", f"{avg_found * 100:.0f}%")
    col3.metric(
        "Avg. schema agreement",
        f"{(sum(agreement_scores) / len(agreement_scores)) * 100:.0f}%" if agreement_scores else "—",
    )

    df = pd.DataFrame(all_results)[
        ["document"] + FIELD_ORDER
    ].rename(columns=FIELD_LABELS)
    st.dataframe(df, use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False)
    st.download_button("Download CSV", csv, "extracted_insurance_data.csv", "text/csv")
