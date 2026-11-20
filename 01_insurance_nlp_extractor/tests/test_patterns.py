from src.patterns import (
    NOT_FOUND,
    extract_fields,
    extract_insurer,
    extract_regex_contexts,
    extract_regex_fields,
)


class TestRegexKnownStrings:
    def test_policy_number_with_no_dot(self):
        result = extract_regex_fields("Policy No. SHHI-2024-HP-004821")
        assert result["policy_number"] == "SHHI-2024-HP-004821"

    def test_policy_number_with_word_number(self):
        result = extract_regex_fields("Policy Number ICICILMB/MPC/2024/007345")
        assert result["policy_number"] == "ICICILMB/MPC/2024/007345"

    def test_policy_number_with_hash(self):
        result = extract_regex_fields("Policy # ABC12345")
        assert result["policy_number"] == "ABC12345"

    def test_policy_schedule_header_is_not_captured_as_number(self):
        # "Policy Schedule" must never be mistaken for a policy number -
        # this is exactly the false positive the stricter keyword avoids.
        result = extract_regex_fields("Policy Schedule\nSome unrelated content here.")
        assert result["policy_number"] == NOT_FOUND

    def test_premium_requires_total_premium_keyword(self):
        text = "Basic Premium INR 12,450\nGST (18%) INR 2,241\nTotal Premium INR 14,691"
        result = extract_regex_fields(text)
        assert result["premium"] == "14,691"

    def test_premium_not_found_without_total_keyword(self):
        # Only a basic/sub-total premium line, no "Total Premium" anywhere.
        text = "Basic Premium INR 12,450\nGST (18%) INR 2,241"
        result = extract_regex_fields(text)
        assert result["premium"] == NOT_FOUND

    def test_start_date_variants(self):
        assert extract_regex_fields("Start Date 20-01-2024")["start_date"] == "20-01-2024"
        assert extract_regex_fields("Effective From 10-01-2024")["start_date"] == "10-01-2024"
        assert extract_regex_fields("Inception Date 05-03-2024")["start_date"] == "05-03-2024"

    def test_end_date_variants(self):
        assert extract_regex_fields("End Date 19-01-2025")["end_date"] == "19-01-2025"
        assert extract_regex_fields("Valid Till 09-01-2025")["end_date"] == "09-01-2025"
        assert extract_regex_fields("Expiry Date 04-03-2025")["end_date"] == "04-03-2025"

    def test_sum_insured_requires_total_when_present(self):
        text = "Building Sum Insured INR 45,00,000\nTotal Sum Insured INR 53,00,000"
        result = extract_regex_fields(text)
        assert result["sum_insured"] == "53,00,000"

    def test_sum_insured_fallback_when_no_total_wording(self):
        text = "Sum Insured INR 5,85,000"
        result = extract_regex_fields(text)
        assert result["sum_insured"] == "5,85,000"

    def test_sum_insured_fallback_liability_keyword(self):
        text = "Maximum Liability INR 2,00,000"
        result = extract_regex_fields(text)
        assert result["sum_insured"] == "2,00,000"


class TestRegexEdgeCases:
    def test_empty_string_returns_not_found_for_all_fields(self):
        result = extract_regex_fields("")
        assert all(v == NOT_FOUND for v in result.values())

    def test_none_input_does_not_crash(self):
        result = extract_regex_fields(None)
        assert all(v == NOT_FOUND for v in result.values())

    def test_text_with_no_matches_at_all(self):
        result = extract_regex_fields("The quick brown fox jumps over the lazy dog.")
        assert all(v == NOT_FOUND for v in result.values())

    def test_extract_regex_contexts_returns_full_span(self):
        text = "Total Premium INR 14,691"
        contexts = extract_regex_contexts(text)
        assert contexts["premium"] == "Total Premium INR 14,691"

    def test_extract_regex_contexts_empty_for_missing_field(self):
        contexts = extract_regex_contexts("nothing relevant here")
        assert contexts["premium"] == ""


class TestInsurerExtraction:
    def test_prefers_generic_insurance_keyword_org(self, nlp_model):
        text = (
            "Star Health and Allied Insurance Co. Ltd.\n"
            "Individual Health Insurance Policy\nPolicy No. SHHI-2024-HP-004821"
        )
        insurer = extract_insurer(text, nlp=nlp_model)
        assert "insurance" in insurer.lower() or "Star Health" in insurer

    def test_brand_keyword_fallback(self, nlp_model):
        # No ORG here contains the literal word "insurance" -> must fall
        # back to the brand-name keyword list ("bajaj") rather than
        # returning Not Found.
        text = "Policy issued by Bajaj Allianz Corporation on behalf of the policyholder."
        insurer = extract_insurer(text, nlp=nlp_model)
        assert insurer == "Bajaj Allianz Corporation"

    def test_no_org_entities_returns_not_found(self, nlp_model):
        insurer = extract_insurer("12345 67890 00000", nlp=nlp_model)
        assert insurer == NOT_FOUND

    def test_empty_text_returns_not_found(self, nlp_model):
        assert extract_insurer("", nlp=nlp_model) == NOT_FOUND
        assert extract_insurer(None, nlp=nlp_model) == NOT_FOUND


class TestExtractFieldsIntegration:
    def test_combines_regex_and_insurer(self, health_insurance_text, nlp_model):
        result = extract_fields(health_insurance_text, nlp=nlp_model)
        assert result["policy_number"] == "SHHI-2024-HP-004821"
        assert result["premium"] == "14,691"
        assert result["start_date"] == "20-01-2024"
        assert result["end_date"] == "19-01-2025"
        assert result["sum_insured"] == "5,00,000"
        assert result["insurer"] != NOT_FOUND

    def test_extract_fields_on_all_sample_pdfs_finds_every_field(self, sample_texts, nlp_model):
        for name, text in sample_texts.items():
            result = extract_fields(text, nlp=nlp_model)
            not_found_fields = [k for k, v in result.items() if v == NOT_FOUND]
            assert not not_found_fields, f"{name}: fields not found: {not_found_fields}"
