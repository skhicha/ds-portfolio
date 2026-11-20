from src.schema_match import SCHEMA_FIELDS, match_schema, validate_extracted_fields


class TestMatchSchemaCorrectness:
    def test_premium_context_matches_premium_field(self):
        results = match_schema(["Total Premium payable INR 14,691 annual payment"])
        assert results[0]["field"] == "premium"
        assert results[0]["score"] > 0

    def test_insurer_context_matches_insurer_name_field(self):
        results = match_schema(["Bajaj Allianz General Insurance Company underwriter"])
        assert results[0]["field"] == "insurer_name"

    def test_sum_insured_context_matches_sum_insured_field(self):
        results = match_schema(["Total Sum Insured coverage amount maximum benefit"])
        assert results[0]["field"] == "sum_insured"

    def test_multiple_fragments_parallel_results(self):
        fragments = [
            "Total Premium INR 14,691 payable annual",
            "Sum Insured maximum liability benefit amount",
        ]
        results = match_schema(fragments)
        assert len(results) == 2
        assert results[0]["fragment"] == fragments[0]
        assert results[1]["fragment"] == fragments[1]

    def test_custom_schema_is_respected(self):
        tiny_schema = {"a": "apple banana fruit", "b": "car truck vehicle"}
        results = match_schema(["I drove my truck to the market"], schema_fields=tiny_schema)
        assert results[0]["field"] == "b"

    def test_all_default_schema_fields_present(self):
        expected = {
            "policy_number",
            "insurer_name",
            "premium",
            "start_date",
            "end_date",
            "sum_insured",
        }
        assert set(SCHEMA_FIELDS.keys()) == expected


class TestMatchSchemaEdgeCases:
    def test_empty_fragments_list_returns_empty(self):
        assert match_schema([]) == []

    def test_empty_string_fragment_returns_none_field(self):
        results = match_schema([""])
        assert results[0]["field"] is None
        assert results[0]["score"] == 0.0

    def test_all_stopword_fragment_does_not_crash(self):
        # "the a an of" are all English stopwords -> empty vocabulary after
        # filtering; must not raise ValueError from TfidfVectorizer.
        results = match_schema(["the a an of"])
        assert results[0]["field"] is None

    def test_bare_numeric_fragment_has_no_vocabulary_overlap(self):
        # A bare captured value like "14,691" shares no words with any
        # schema description -> legitimately scores 0/None, unlike a
        # fragment that includes its surrounding keyword context.
        results = match_schema(["14,691"])
        assert results[0]["score"] == 0.0

    def test_mixed_empty_and_real_fragments(self):
        results = match_schema(["", "Total Premium payable INR 500 annual"])
        assert results[0]["field"] is None
        assert results[1]["field"] == "premium"


class TestValidateExtractedFields:
    def test_agreement_when_context_matches_label(self):
        fields = {"premium": "14,691", "sum_insured": "Not Found"}
        contexts = {"premium": "Total Premium INR 14,691"}
        validation = validate_extracted_fields(fields, contexts)
        assert validation["premium"]["agrees"] is True
        assert validation["sum_insured"]["agrees"] is False
        assert validation["sum_insured"]["predicted_field"] is None

    def test_no_fields_found_returns_all_disagree(self):
        fields = {"premium": "Not Found", "sum_insured": "Not Found"}
        validation = validate_extracted_fields(fields)
        assert all(v["agrees"] is False for v in validation.values())
        assert all(v["score"] == 0.0 for v in validation.values())

    def test_falls_back_to_bare_value_when_no_context_given(self):
        fields = {"insurer": "Bajaj Allianz General Insurance Co. Ltd."}
        validation = validate_extracted_fields(fields)
        assert validation["insurer"]["predicted_field"] == "insurer_name"
        assert validation["insurer"]["agrees"] is True

    def test_every_input_field_present_in_output(self):
        fields = {
            "policy_number": "ABC123",
            "premium": "Not Found",
            "insurer": "Not Found",
        }
        validation = validate_extracted_fields(fields, {"policy_number": "Policy No. ABC123"})
        assert set(validation.keys()) == set(fields.keys())
