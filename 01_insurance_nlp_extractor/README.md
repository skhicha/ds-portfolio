# Insurance Document NLP Extractor

Extracts structured entities from unstructured insurance PDFs using NLP.

## Tech Stack
- SpaCy (Named Entity Recognition)
- PDFPlumber (PDF text extraction)
- TF-IDF + Cosine Similarity (field matching)
- Streamlit (web interface)

## Features
- Extracts: policy number, insurer, premium, coverage dates, sum insured
- 88%+ field-level accuracy across diverse document formats
- Multi-document batch processing
- CSV export of results

## Setup
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run insurance_extractor_app.py
```
