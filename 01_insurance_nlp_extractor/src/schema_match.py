"""TF-IDF + cosine-similarity schema matching.

The regex/NER layer (``src.patterns``) extracts values under fixed, known
keys (``policy_number``, ``premium``, ...). This module provides an
independent, second opinion: given a short text fragment, which field in a
*master schema* of field descriptions does it most resemble? It is used in
the app as a confidence/validation signal -- if the fragment extracted for
"premium" actually looks (by TF-IDF content) more like a "sum_insured"
fragment than a "premium" fragment, that's a useful red flag even though
the regex technically matched.

This has no notion of which extractor produced a fragment; it only ever
sees text.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Master schema: field name -> a short description of what that field means,
# written in the vocabulary a real extracted fragment for that field would
# plausibly share (so TF-IDF has something to latch onto).
SCHEMA_FIELDS: dict[str, str] = {
    "policy_number": "policy number identification certificate reference ID alphanumeric code",
    "insurer_name": "insurance company insurer underwriter provider organization name",
    "premium": "premium amount payable due payment annual monthly quarterly total rupees",
    "start_date": "coverage commencement inception start from effective date period",
    "end_date": "coverage expiry end till until maturity date period termination",
    "sum_insured": "sum insured coverage amount limit liability maximum benefit value",
}

# Maps the keys produced by src.patterns.extract_fields to schema field names
# (they differ only for the insurer field: "insurer" -> "insurer_name").
EXTRACTOR_TO_SCHEMA_FIELD: dict[str, str] = {
    "policy_number": "policy_number",
    "insurer": "insurer_name",
    "premium": "premium",
    "start_date": "start_date",
    "end_date": "end_date",
    "sum_insured": "sum_insured",
}


def match_schema(
    fragments: list[str], schema_fields: dict[str, str] | None = None
) -> list[dict]:
    """Match each fragment to its best-scoring schema field.

    Returns a list (parallel to ``fragments``) of
    ``{"fragment": str, "field": str | None, "score": float}``.
    ``field`` is ``None`` and ``score`` is ``0.0`` when no similarity at all
    could be computed (e.g. an empty/whitespace-only fragment, or a
    fragment made entirely of English stopwords that TF-IDF discards).

    Edge cases handled explicitly: empty ``fragments`` list returns ``[]``;
    a fragment list where every fragment is empty/stopword-only never
    raises (TfidfVectorizer would otherwise raise ``ValueError: empty
    vocabulary`` in that case).
    """
    schema_fields = schema_fields or SCHEMA_FIELDS
    if not fragments:
        return []

    schema_names = list(schema_fields.keys())
    schema_texts = list(schema_fields.values())
    all_texts = schema_texts + list(fragments)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    try:
        tfidf_matrix = vectorizer.fit_transform(all_texts)
    except ValueError:
        # Empty vocabulary: every document was empty or pure stopwords.
        return [{"fragment": f, "field": None, "score": 0.0} for f in fragments]

    schema_vectors = tfidf_matrix[: len(schema_texts)]
    fragment_vectors = tfidf_matrix[len(schema_texts) :]
    similarity = np.nan_to_num(cosine_similarity(fragment_vectors, schema_vectors))

    results = []
    for i, fragment in enumerate(fragments):
        row = similarity[i]
        if not np.any(row):
            results.append({"fragment": fragment, "field": None, "score": 0.0})
            continue
        best_idx = int(np.argmax(row))
        results.append(
            {"fragment": fragment, "field": schema_names[best_idx], "score": float(row[best_idx])}
        )
    return results


def validate_extracted_fields(
    fields: dict[str, str], contexts: dict[str, str] | None = None
) -> dict[str, dict]:
    """Cross-check each extracted field's value against the master schema.

    ``fields`` is the dict produced by ``src.patterns.extract_fields``
    (keys: policy_number, premium, start_date, end_date, sum_insured,
    insurer). For every field whose value isn't "Not Found", runs the
    surrounding **context** (full matched span, e.g. "Total Premium INR
    14,691") through ``match_schema`` and reports whether the TF-IDF
    schema match agrees with the label the regex/NER extractor originally
    assigned. A bare numeric/date value shares no vocabulary with the
    schema descriptions and would always score 0, so ``contexts`` (from
    ``src.patterns.extract_regex_contexts``) should be passed whenever
    available; fields without a context (e.g. ``insurer``, which has no
    regex match) fall back to the plain value.

    Returns a dict keyed by the same field names, each mapping to
    ``{"value": str, "predicted_field": str | None, "score": float,
    "agrees": bool}``.
    """
    contexts = contexts or {}
    present = {k: v for k, v in fields.items() if v and v != "Not Found"}
    if not present:
        return {
            k: {"value": v, "predicted_field": None, "score": 0.0, "agrees": False}
            for k, v in fields.items()
        }

    keys = list(present.keys())
    texts_to_match = [contexts.get(k) or present[k] for k in keys]
    matches = match_schema(texts_to_match)

    validation: dict[str, dict] = {}
    for key, match in zip(keys, matches):
        expected_schema_field = EXTRACTOR_TO_SCHEMA_FIELD.get(key)
        validation[key] = {
            "value": present[key],
            "predicted_field": match["field"],
            "score": match["score"],
            "agrees": match["field"] == expected_schema_field,
        }

    for key, value in fields.items():
        if key not in validation:
            validation[key] = {
                "value": value,
                "predicted_field": None,
                "score": 0.0,
                "agrees": False,
            }
    return validation
