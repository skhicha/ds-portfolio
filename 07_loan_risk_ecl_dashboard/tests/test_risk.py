"""
Tests for src/risk.py: delinquency bucketing, IFRS 9 / Ind AS 109 staging,
roll-rate (transition matrix) computation, PD derivation, and the ECL
formula itself (PD * LGD * EAD), including edge cases at bucket/stage
boundaries.
"""

import numpy as np
import pandas as pd
import pytest

from src.risk import (
    BUCKET_ORDER,
    bucket_from_dpd,
    stage_from_dpd,
    compute_transition_matrix,
    pd_from_transition_matrix,
    lifetime_pd_from_transition_matrix,
    compute_ecl,
    ecl_summary,
)


# ---------------------------------------------------------------------------
# Delinquency bucketing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "dpd, expected_bucket",
    [
        (0, "Current"),
        (1, "1-29"),
        (29, "1-29"),
        (30, "30-59"),
        (59, "30-59"),
        (60, "60-89"),
        (89, "60-89"),
        (90, "90+"),
        (400, "90+"),
    ],
)
def test_bucket_from_dpd_boundaries(dpd, expected_bucket):
    assert bucket_from_dpd(dpd) == expected_bucket


def test_bucket_from_dpd_rejects_negative():
    with pytest.raises(ValueError):
        bucket_from_dpd(-1)


def test_bucket_from_dpd_rejects_null():
    with pytest.raises(ValueError):
        bucket_from_dpd(None)


# ---------------------------------------------------------------------------
# IFRS 9 / Ind AS 109 staging
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "dpd, expected_stage",
    [
        (0, 1),
        (29, 1),
        (30, 2),   # rebuttable-presumption SICR threshold
        (89, 2),
        (90, 3),   # credit-impaired / default threshold
        (200, 3),
    ],
)
def test_stage_from_dpd_boundaries(dpd, expected_stage):
    assert stage_from_dpd(dpd) == expected_stage


def test_stage_from_dpd_sicr_overlay_upgrades_stage1_to_stage2():
    # Without the qualitative overlay, 15 DPD is Stage 1.
    assert stage_from_dpd(15, sicr_flag=False) == 1
    # With a SICR flag (e.g. from the early-warning model), it becomes Stage 2.
    assert stage_from_dpd(15, sicr_flag=True) == 2


def test_stage_from_dpd_sicr_overlay_does_not_downgrade_stage3():
    assert stage_from_dpd(120, sicr_flag=True) == 3


def test_stage_from_dpd_rejects_negative():
    with pytest.raises(ValueError):
        stage_from_dpd(-5)


# ---------------------------------------------------------------------------
# Roll-rate / transition matrix
# ---------------------------------------------------------------------------

def test_compute_transition_matrix_exact_probabilities():
    # Construct a tiny, hand-countable set of transitions:
    # From Current: 8 stay Current, 2 roll to 1-29 -> P(Current->Current)=0.8, P(Current->1-29)=0.2
    pairs = pd.DataFrame(
        {
            "bucket_from": ["Current"] * 10 + ["1-29"] * 4,
            "bucket_to": ["Current"] * 8 + ["1-29"] * 2 + ["Current"] * 1 + ["30-59"] * 3,
        }
    )
    matrix = compute_transition_matrix(pairs)

    assert list(matrix.index) == BUCKET_ORDER
    assert list(matrix.columns) == BUCKET_ORDER
    assert matrix.loc["Current", "Current"] == pytest.approx(0.8)
    assert matrix.loc["Current", "1-29"] == pytest.approx(0.2)
    assert matrix.loc["1-29", "Current"] == pytest.approx(0.25)
    assert matrix.loc["1-29", "30-59"] == pytest.approx(0.75)

    # Rows with observed data (plus the always-enforced 90+ absorbing row)
    # must sum to 1 (valid stochastic rows); buckets with no observed
    # outgoing transitions in this tiny fixture (30-59, 60-89) legitimately
    # sum to 0 rather than being fabricated.
    for bucket in ["Current", "1-29", "90+"]:
        assert matrix.loc[bucket].sum() == pytest.approx(1.0)


def test_compute_transition_matrix_treats_90plus_as_absorbing_when_no_data():
    pairs = pd.DataFrame({"bucket_from": ["Current"], "bucket_to": ["1-29"]})
    matrix = compute_transition_matrix(pairs)
    assert matrix.loc["90+", "90+"] == pytest.approx(1.0)
    assert matrix.loc["90+"].sum() == pytest.approx(1.0)


def test_compute_transition_matrix_rejects_empty_input():
    with pytest.raises(ValueError):
        compute_transition_matrix(pd.DataFrame({"bucket_from": [], "bucket_to": []}))


def _two_state_matrix(p_default: float) -> pd.DataFrame:
    """Helper: a transition matrix where every non-default bucket has the
    same flat probability `p_default` of jumping straight to 90+ each
    month, and otherwise stays put -- lets us hand-verify PD formulas
    against the closed-form geometric distribution."""
    m = pd.DataFrame(0.0, index=BUCKET_ORDER, columns=BUCKET_ORDER)
    for b in BUCKET_ORDER:
        if b == "90+":
            m.loc[b, b] = 1.0
        else:
            m.loc[b, b] = 1 - p_default
            m.loc[b, "90+"] = p_default
    return m


def test_pd_from_transition_matrix_matches_closed_form_geometric():
    p = 0.02
    matrix = _two_state_matrix(p)
    pd_12m = pd_from_transition_matrix(matrix, horizon_months=12)
    # P(absorbed within 12 months) = 1 - (1-p)^12 for a simple stay/absorb chain.
    expected = 1 - (1 - p) ** 12
    for bucket in ["Current", "1-29", "30-59", "60-89"]:
        assert pd_12m[bucket] == pytest.approx(expected, rel=1e-6)
    assert pd_12m["90+"] == pytest.approx(1.0)


def test_pd_from_transition_matrix_zero_horizon_is_identity():
    matrix = _two_state_matrix(0.05)
    pd_0m = pd_from_transition_matrix(matrix, horizon_months=0)
    # Raising to the 0th power gives the identity matrix: PD(0 months) is 1
    # only for a loan already in the default bucket, else 0.
    assert pd_0m["Current"] == pytest.approx(0.0)
    assert pd_0m["90+"] == pytest.approx(1.0)


def test_lifetime_pd_matches_geometric_absorption_probability():
    p = 0.10
    matrix = _two_state_matrix(p)
    lifetime = lifetime_pd_from_transition_matrix(matrix)
    # For a chain that can only stay or jump to the absorbing state, the
    # lifetime absorption probability from any transient state is 1.
    for bucket in ["Current", "1-29", "30-59", "60-89"]:
        assert lifetime[bucket] == pytest.approx(1.0, rel=1e-6)
    assert lifetime["90+"] == pytest.approx(1.0)


def test_lifetime_pd_is_at_least_12_month_pd():
    """Lifetime PD can never be lower than the 12-month PD for the same
    starting bucket (cumulative probability is monotonically non-decreasing
    in the horizon)."""
    rng = np.random.default_rng(0)
    counts = pd.DataFrame(
        rng.integers(1, 50, size=(5, 5)), index=BUCKET_ORDER, columns=BUCKET_ORDER
    )
    pairs_rows = []
    for i, from_b in enumerate(BUCKET_ORDER):
        for j, to_b in enumerate(BUCKET_ORDER):
            pairs_rows += [{"bucket_from": from_b, "bucket_to": to_b}] * int(counts.iloc[i, j])
    pairs = pd.DataFrame(pairs_rows)
    matrix = compute_transition_matrix(pairs)

    pd_12m = pd_from_transition_matrix(matrix, horizon_months=12)
    pd_life = lifetime_pd_from_transition_matrix(matrix)
    for bucket in BUCKET_ORDER:
        assert pd_life[bucket] >= pd_12m[bucket] - 1e-9


# ---------------------------------------------------------------------------
# ECL formula
# ---------------------------------------------------------------------------

@pytest.fixture
def toy_matrix():
    return _two_state_matrix(0.03)


def test_ecl_formula_arithmetic_is_pd_times_lgd_times_ead(toy_matrix):
    portfolio = pd.DataFrame(
        {
            "loan_id": ["A", "B", "C"],
            "product": ["mortgage", "personal_loan", "credit_card"],
            "delinquency_bucket": ["Current", "1-29", "90+"],
            "stage": [1, 2, 3],
            "outstanding_balance": [100_000.0, 5_000.0, 2_000.0],
        }
    )
    lgd_map = {"mortgage": 0.35, "personal_loan": 0.65, "credit_card": 0.80}
    result = compute_ecl(portfolio, toy_matrix, lgd_by_product=lgd_map)

    for _, row in result.iterrows():
        assert row["ecl"] == pytest.approx(row["pd_applied"] * row["lgd_applied"] * row["ead"])

    # Stage 3 loan: PD must be exactly 1.0 (already credit-impaired/default).
    stage3_row = result.loc[result["loan_id"] == "C"].iloc[0]
    assert stage3_row["pd_applied"] == pytest.approx(1.0)
    assert stage3_row["ecl"] == pytest.approx(0.80 * 2_000.0)

    # Stage 1 loan uses the 12-month PD, which must be strictly between 0 and 1
    # here (non-trivial roll-rate matrix).
    stage1_row = result.loc[result["loan_id"] == "A"].iloc[0]
    assert 0 < stage1_row["pd_applied"] < 1

    # LGD assumptions are applied per-product exactly as configured.
    assert stage3_row["lgd_applied"] == pytest.approx(0.80)


def test_ecl_unknown_product_falls_back_to_default_lgd(toy_matrix):
    portfolio = pd.DataFrame(
        {
            "loan_id": ["Z"],
            "product": ["unknown_product"],
            "delinquency_bucket": ["Current"],
            "stage": [1],
            "outstanding_balance": [1_000.0],
        }
    )
    result = compute_ecl(portfolio, toy_matrix, lgd_by_product={}, default_lgd=0.5)
    assert result.iloc[0]["lgd_applied"] == pytest.approx(0.5)


def test_ecl_higher_lgd_assumption_increases_ecl_linearly(toy_matrix):
    portfolio = pd.DataFrame(
        {
            "loan_id": ["A"],
            "product": ["personal_loan"],
            "delinquency_bucket": ["30-59"],
            "stage": [2],
            "outstanding_balance": [10_000.0],
        }
    )
    low = compute_ecl(portfolio, toy_matrix, lgd_by_product={"personal_loan": 0.3})
    high = compute_ecl(portfolio, toy_matrix, lgd_by_product={"personal_loan": 0.6})
    assert high.iloc[0]["ecl"] == pytest.approx(2 * low.iloc[0]["ecl"])


def test_ecl_zero_balance_gives_zero_ecl(toy_matrix):
    portfolio = pd.DataFrame(
        {
            "loan_id": ["A"],
            "product": ["mortgage"],
            "delinquency_bucket": ["Current"],
            "stage": [1],
            "outstanding_balance": [0.0],
        }
    )
    result = compute_ecl(portfolio, toy_matrix)
    assert result.iloc[0]["ecl"] == pytest.approx(0.0)


def test_ecl_summary_aggregation(toy_matrix):
    portfolio = pd.DataFrame(
        {
            "loan_id": ["A", "B"],
            "product": ["mortgage", "mortgage"],
            "delinquency_bucket": ["Current", "Current"],
            "stage": [1, 1],
            "outstanding_balance": [100_000.0, 200_000.0],
        }
    )
    ecl_df = compute_ecl(portfolio, toy_matrix)
    summary = ecl_summary(ecl_df, by=["product"])
    assert summary.iloc[0]["loan_count"] == 2
    assert summary.iloc[0]["total_ead"] == pytest.approx(300_000.0)
    assert summary.iloc[0]["total_ecl"] == pytest.approx(ecl_df["ecl"].sum())


def test_real_generated_portfolio_produces_sane_ecl(small_loan_book):
    """Integration-style check: run the full pipeline (generate -> bucket ->
    transition matrix -> PD -> ECL) end to end on real generated data and
    assert the results are sane (non-negative, bounded, non-trivial)."""
    loans_df, snapshots_df = small_loan_book

    # Build roll-rate pairs the same way sql/roll_rate_pairs.sql does.
    snaps = snapshots_df.sort_values(["loan_id", "months_on_book"])
    snaps["next_bucket"] = snaps.groupby("loan_id")["delinquency_bucket"].shift(-1)
    snaps["next_month"] = snaps.groupby("loan_id")["months_on_book"].shift(-1)
    pairs = snaps.dropna(subset=["next_bucket"])
    pairs = pairs[pairs["next_month"] == pairs["months_on_book"] + 1]
    pairs = pairs.rename(columns={"delinquency_bucket": "bucket_from", "next_bucket": "bucket_to"})

    matrix = compute_transition_matrix(pairs)

    # Point-in-time portfolio: last snapshot per loan.
    latest = snapshots_df.sort_values("months_on_book").groupby("loan_id").tail(1)
    portfolio = latest.merge(loans_df, on="loan_id")

    ecl_df = compute_ecl(portfolio, matrix)
    assert (ecl_df["ecl"] >= 0).all()
    assert (ecl_df["pd_applied"] >= 0).all()
    assert (ecl_df["pd_applied"] <= 1).all()
    assert ecl_df["ecl"].sum() > 0
    # Stage 3 loans' PD must be exactly 1.
    assert (ecl_df.loc[ecl_df["stage"] == 3, "pd_applied"] == 1.0).all()
