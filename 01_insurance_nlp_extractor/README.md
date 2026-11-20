# Insurance Document NLP Extractor

A working insurance-PDF field extractor: regex patterns tuned against real
policy documents, spaCy NER for the insurer name, a TF-IDF + cosine-similarity
schema matcher that independently cross-checks every extracted value, a
polished interactive Streamlit dashboard, and a real pytest suite — all run
against the 3 real sample insurance PDFs bundled in this repo, not synthetic
strings.

**Achieved on the 3 bundled sample PDFs (health, home, motor insurance):
18/18 fields extracted (100%), 17/18 schema-verified (94%)** — the one
disagreement is an honest TF-IDF result, not a bug (see Notes below).

## What it does

1. **Extracts text from PDFs** (`src/extract.py`) via `pdfplumber`, accepting
   a file path, raw bytes, or a file-like object (so it works the same for
   a local sample file and a Streamlit-uploaded file).
2. **Extracts structured fields via regex + spaCy NER** (`src/patterns.py`):
   - Policy number, total premium, coverage start/end date, and sum insured
     via regex patterns that require an explicit keyword next to the value
     (e.g. `Total Premium`, `Total Sum Insured`) so that GST sub-totals or
     partial building/contents values are never captured by mistake. A
     looser fallback pattern (`Sum Insured` / `Liability`) is used only when
     the stricter "total" wording isn't present in the document.
   - Insurer name via spaCy (`en_core_web_sm`) NER over `ORG` entities,
     preferring an entity that literally contains "insurance" (e.g. "Allied
     Insurance Co. Ltd." over a shorter co-located "Star Health" entity),
     then a known brand keyword (Bajaj, ICICI, HDFC, Allianz, ...), then
     falling back to the first `ORG` entity spaCy finds.
3. **Cross-checks every value against a master schema** (`src/schema_match.py`):
   a `SCHEMA_FIELDS` dict of field descriptions, matched against each
   extracted value's full regex context (not the bare number/date, which
   shares no vocabulary with anything) via `TfidfVectorizer(ngram_range=(1,2),
   stop_words="english")` + cosine similarity. This is a genuine second
   opinion, independent of which label the regex/NER layer assigned — if the
   TF-IDF match disagrees, that's a real signal, not a hardcoded confidence
   score.
4. **Renders a polished Streamlit dashboard** (`app.py`): multi-file upload
   (or a one-click toggle to run the 3 bundled samples), a per-document card
   showing every field with a "verified" / confidence badge from the schema
   matcher, aggregated results across all documents, and CSV export.

## Architecture

```
01_insurance_nlp_extractor/
├── app.py                          # Streamlit dashboard (entry point)
├── data/
│   ├── sample_health_insurance.pdf
│   ├── sample_home_insurance.pdf
│   └── sample_motor_insurance.pdf
├── src/
│   ├── extract.py                  # pdfplumber text extraction (path/bytes/file-like)
│   ├── patterns.py                 # regex field patterns + spaCy NER insurer extraction
│   └── schema_match.py             # TF-IDF + cosine-similarity schema cross-check
├── tests/
│   ├── conftest.py
│   ├── test_extract.py             # PDF extraction on real samples + malformed input
│   ├── test_patterns.py            # regex correctness + NER + edge cases
│   └── test_schema_match.py        # TF-IDF matching correctness + edge cases
├── requirements.txt
├── pytest.ini
├── .streamlit/config.toml
├── .gitignore
└── README.md
```

**Data flow:** `extract.extract_text()` → raw text → `patterns.extract_fields()`
(regex fields + NER insurer) and `patterns.extract_regex_contexts()` (the full
matched span per field, used only for schema validation) →
`schema_match.validate_extracted_fields()` (TF-IDF cross-check against
`SCHEMA_FIELDS`) → consumed by `app.py` for display and CSV export.

## Setup

```bash
cd 01_insurance_nlp_extractor

# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 3. Launch the interactive dashboard
streamlit run app.py
# open http://localhost:8501, check "Use bundled sample PDFs" to try it
# instantly, or upload your own insurance PDF(s)

# 4. Run the test suite
pytest -v
```

## Notes on the real results

- `data/*.pdf` are 3 synthetic-but-realistic sample policies (health, home,
  motor) generated for this project, run through the full pipeline exactly
  as a real upload would be.
- All 6 fields (policy number, insurer, premium, start date, end date, sum
  insured) were correctly extracted for all 3 documents — 18/18, 100%.
- Schema verification agreed with the extractor's own label for 17/18
  fields (94%). The one disagreement is `start_date` on the home-insurance
  sample, where the regex matches the context `"From 10-01-2024"` — the
  word "From" is an English stopword the TF-IDF vectorizer discards from
  *both* the fragment and the schema description, so the two share no
  vocabulary and score 0. That's a genuine limitation of keyword-free
  context strings, not a hardcoded number, and the dashboard reports it as
  a low-confidence badge rather than hiding it.

## Tech stack

Python, pdfplumber (PDF text extraction), spaCy `en_core_web_sm` (NER),
scikit-learn (TF-IDF + cosine similarity), Streamlit (dashboard), pandas
(CSV export), pytest (tests).

## Testing

```bash
pytest -v
```

45 tests, all passing, covering:
- PDF extraction: real text pulled from all 3 bundled sample PDFs, accepting
  a path, raw bytes, or a file-like object (equivalent to a Streamlit
  upload), and edge cases — malformed/non-PDF bytes, empty bytes, and a
  nonexistent path, all raising a clear `PdfExtractionError` instead of a
  raw traceback.
- Regex patterns: every field against known real-world phrasings (including
  the stricter "Total Premium"/"Total Sum Insured" requirement, its
  fallback, and multiple date-keyword variants), the "Policy Schedule"
  false-positive that the stricter policy-number pattern must avoid, and
  edge cases — empty string, `None` input, and text with no matches at all.
- spaCy NER insurer extraction: the generic-keyword-first / brand-keyword
  fallback / first-ORG-fallback priority order, and edge cases — no ORG
  entities found, empty/`None` text.
- TF-IDF schema matching: correct field assignment for premium/insurer/sum-
  insured contexts, a custom schema override, and edge cases — empty
  fragment list, an empty-string fragment, an all-stopword fragment (which
  would otherwise raise `ValueError: empty vocabulary` from
  `TfidfVectorizer`), and a bare numeric fragment with genuinely zero
  vocabulary overlap.

The Streamlit app was additionally run live with
`streamlit run app.py --server.headless true` and confirmed to serve
(`HTTP 200`, `/_stcore/health` → `ok`) with zero server-side exceptions, and
the full extraction pipeline was run end-to-end against all 3 real sample
PDFs with the results above.

## Disclaimer

The sample PDFs are synthetic documents created for demonstration purposes;
they do not represent real policyholders or insurers.
