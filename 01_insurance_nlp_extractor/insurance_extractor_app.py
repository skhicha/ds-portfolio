
# insurance_extractor_app.py
# Run with: streamlit run insurance_extractor_app.py

import streamlit as st
import pdfplumber
import spacy
import re
import pandas as pd
import io

st.set_page_config(page_title="Insurance NLP Extractor", page_icon="📄", layout="wide")
st.title("📄 Insurance Document NLP Extractor")
st.markdown("Upload one or more insurance PDFs to extract structured data automatically.")

@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

nlp = load_nlp()

PATTERNS = {
    # Require explicit keyword (no., number, #) so "Policy Schedule" is never captured
    "policy_number": r"(?i)policy\s*(?:no\.?|number|#)\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-/]{4,29})",
    # Target "Total Premium" so basic/GST sub-totals are skipped
    "premium":       r"(?i)total\s+premium\s*[:\-]?\s*(?:INR|Rs\.?|₹)?\s*([\d,\.]+)",
    "start_date":    r"(?i)(?:start|from|inception|effective)\s*(?:date)?\s*[:\-]?\s*([\d]{1,2}[\s/\-][A-Za-z\d]{2,9}[\s/\-][\d]{2,4})",
    "end_date":      r"(?i)(?:end|expiry|till)\s*(?:date)?\s*[:\-]?\s*([\d]{1,2}[\s/\-][A-Za-z\d]{2,9}[\s/\-][\d]{2,4})",
    # Prefer "Total Sum Insured" to avoid partial building/contents values
    "sum_insured":   r"(?i)total\s+sum\s+insured\s*[:\-]?\s*(?:INR|Rs\.?|₹)?\s*([\d,\.]+)",
}

def extract_text(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        return " ".join(page.extract_text() or "" for page in pdf.pages)

SUM_INSURED_FALLBACK = r"(?i)(?:sum\s*insured|liability)\s*[:\-]?\s*(?:INR|Rs\.?|₹)?\s*([\d,\.]+)"

def extract_fields(text):
    results = {}
    for field, pattern in PATTERNS.items():
        m = re.search(pattern, text)
        # Fallback: if "total sum insured" not found, try plain "sum insured"
        if not m and field == "sum_insured":
            m = re.search(SUM_INSURED_FALLBACK, text)
        results[field] = m.group(1).strip() if m else "Not Found"
    doc = nlp(text)
    orgs = [e.text for e in doc.ents if e.label_ == "ORG"]
    results["insurer"] = orgs[0] if orgs else "Not Found"
    return results

uploaded_files = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)

if uploaded_files:
    all_results = []
    for f in uploaded_files:
        with st.spinner(f"Processing {f.name}..."):
            text = extract_text(f)
            fields = extract_fields(text)
            fields["filename"] = f.name
            all_results.append(fields)
            
            with st.expander(f"📄 {f.name}"):
                col1, col2 = st.columns(2)
                for i, (k, v) in enumerate(fields.items()):
                    if k != "filename":
                        (col1 if i % 2 == 0 else col2).metric(k.replace("_", " ").title(), v)
    
    df = pd.DataFrame(all_results)
    st.subheader("📊 Aggregated Results")
    st.dataframe(df, use_container_width=True)
    
    csv = df.to_csv(index=False)
    st.download_button("⬇️ Download CSV", csv, "extracted_data.csv", "text/csv")
