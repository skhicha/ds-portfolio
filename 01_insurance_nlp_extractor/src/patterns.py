"""Regex + spaCy NER field extraction for insurance documents.

Two extraction strategies, combined by ``extract_fields``:

1. **Regex patterns** for structured fields that reliably carry a keyword
   right next to the value (policy number, premium, dates, sum insured).
   Patterns are deliberately strict (e.g. "Total Premium", "Total Sum
   Insured") so that GST/basic sub-totals or partial building/contents
   values are never captured instead of the figure that actually matters;
   a looser fallback is used for sum insured when the stricter "total"
   wording isn't present in the document.
2. **spaCy NER** for the insurer name, which has no reliable keyword
   anchor and varies in wording/position across documents. ORG entities
   that contain a known insurance-industry keyword are preferred over the
   first ORG found, since the first ORG spaCy finds is not always the
   underwriting insurer (it can be a government body like "IRDAI" or a
   bank name mentioned in passing).
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional

import spacy

# Require an explicit keyword ("no.", "number", "#") so that headers like
# "Policy Schedule" or "Policy Details" are never captured as the number.
PATTERNS: dict[str, str] = {
    "policy_number": r"(?i)policy\s*(?:no\.?|number|#)\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-/]{4,29})",
    # Target "Total Premium" specifically so basic/GST sub-totals are skipped.
    "premium": r"(?i)total\s+premium\s*[:\-]?\s*(?:INR|Rs\.?|₹)?\s*([\d,\.]+)",
    "start_date": r"(?i)(?:start|from|inception|effective)\s*(?:date)?\s*[:\-]?\s*([\d]{1,2}[\s/\-][A-Za-z\d]{2,9}[\s/\-][\d]{2,4})",
    "end_date": r"(?i)(?:end|expiry|till)\s*(?:date)?\s*[:\-]?\s*([\d]{1,2}[\s/\-][A-Za-z\d]{2,9}[\s/\-][\d]{2,4})",
    # Prefer "Total Sum Insured" to avoid partial building/contents values.
    "sum_insured": r"(?i)total\s+sum\s+insured\s*[:\-]?\s*(?:INR|Rs\.?|₹)?\s*([\d,\.]+)",
}

# Used only when the stricter "total sum insured" wording isn't present.
SUM_INSURED_FALLBACK = r"(?i)(?:sum\s*insured|liability)\s*[:\-]?\s*(?:INR|Rs\.?|₹)?\s*([\d,\.]+)"

NOT_FOUND = "Not Found"

# Generic words that unambiguously mark an ORG entity as *the insurer*
# itself, checked first so e.g. "Allied Insurance Co. Ltd." beats a shorter
# brand-only entity like "Star Health" that spaCy split off separately.
GENERIC_INSURER_KEYWORDS = ["insurance", "insurer", "assurance"]

# Known insurer brand names, checked only if no ORG matched a generic
# keyword above. An ORG entity containing one of these is preferred over an
# arbitrary first ORG match (e.g. "IRDAI" or a payment processor mentioned
# in the fine print).
BRAND_INSURER_KEYWORDS = [
    "ergo", "bajaj", "hdfc", "icici", "tata", "new india", "star health",
    "lombard", "allianz", "oriental", "united india", "future generali",
    "reliance general",
]


@lru_cache(maxsize=1)
def _load_nlp():
    """Load the spaCy English model once per process."""
    return spacy.load("en_core_web_sm")


def extract_regex_fields(text: str) -> dict[str, str]:
    """Run every regex pattern in ``PATTERNS`` against ``text``.

    Returns a dict with the same keys as ``PATTERNS``, each mapped to the
    captured value or ``"Not Found"``. Falls back to the looser sum-insured
    pattern only when the stricter "total sum insured" pattern misses.
    """
    text = text or ""
    results: dict[str, str] = {}
    for field, pattern in PATTERNS.items():
        match = re.search(pattern, text)
        if not match and field == "sum_insured":
            match = re.search(SUM_INSURED_FALLBACK, text)
        results[field] = match.group(1).strip() if match else NOT_FOUND
    return results


def extract_regex_contexts(text: str) -> dict[str, str]:
    """Like ``extract_regex_fields``, but returns the *full matched span*
    (keyword + value, e.g. ``"Total Premium INR 14,691"``) instead of just
    the captured value, for each field that matched.

    This context -- rather than a bare number or date -- is what should be
    fed to ``src.schema_match.match_schema``: a bare value like ``"14,691"``
    shares no vocabulary with the schema's field descriptions and always
    scores 0, whereas the surrounding keywords ("Total Premium") do.
    Fields with no match map to ``""``.
    """
    text = text or ""
    contexts: dict[str, str] = {}
    for field, pattern in PATTERNS.items():
        match = re.search(pattern, text)
        if not match and field == "sum_insured":
            match = re.search(SUM_INSURED_FALLBACK, text)
        contexts[field] = match.group(0).strip() if match else ""
    return contexts


def extract_insurer(text: str, nlp=None) -> str:
    """Extract the insurer name via spaCy NER over ORG entities.

    Prefers an ORG entity containing a generic insurer word ("insurance",
    "insurer", "assurance"); if none of those exist, falls back to an ORG
    containing a known brand name; if still nothing, falls back to the
    first ORG entity found; returns ``"Not Found"`` if spaCy finds no ORG
    entities at all.
    """
    text = text or ""
    if not text.strip():
        return NOT_FOUND

    nlp = nlp or _load_nlp()
    doc = nlp(text)
    orgs = [ent.text.strip() for ent in doc.ents if ent.label_ == "ORG" and ent.text.strip()]
    if not orgs:
        return NOT_FOUND

    for org in orgs:
        if any(keyword in org.lower() for keyword in GENERIC_INSURER_KEYWORDS):
            return org
    for org in orgs:
        if any(keyword in org.lower() for keyword in BRAND_INSURER_KEYWORDS):
            return org
    return orgs[0]


def extract_fields(text: str, nlp=None) -> dict[str, str]:
    """Run the full field-extraction pipeline: regex fields + NER insurer.

    Returns a single flat dict: the 5 regex fields plus ``"insurer"``.
    """
    results = extract_regex_fields(text)
    results["insurer"] = extract_insurer(text, nlp=nlp)
    return results
