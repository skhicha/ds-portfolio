import pandas as pd

from src.generate_data import (
    PRACTICE_SETTINGS,
    REGIONS,
    SPECIALTIES,
    generate_dataset,
)

EXPECTED_COLUMNS = [
    "physician_id", "specialty", "region", "practice_setting",
    "years_in_practice", "patient_panel_size", "pct_patients_65_plus",
    "pct_female_patients", "avg_comorbidity_score",
    "total_market_trx_monthly", "company_trx_monthly", "company_nrx_monthly",
    "company_rx_value_monthly", "market_share_pct", "digital_engagement_score",
    "rep_visits_last_12mo", "samples_distributed_last_12mo",
    "peer_influence_score", "years_prescribing_company_product",
]


def test_default_shape_and_columns():
    df = generate_dataset(n_physicians=500, seed=42)
    assert len(df) == 500
    assert list(df.columns) == EXPECTED_COLUMNS


def test_reproducible_with_same_seed():
    df1 = generate_dataset(n_physicians=200, seed=42)
    df2 = generate_dataset(n_physicians=200, seed=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_different_seed_gives_different_data():
    df1 = generate_dataset(n_physicians=200, seed=1)
    df2 = generate_dataset(n_physicians=200, seed=2)
    assert not df1["company_rx_value_monthly"].equals(df2["company_rx_value_monthly"])


def test_physician_ids_are_unique():
    df = generate_dataset(n_physicians=300, seed=5)
    assert df["physician_id"].is_unique
    assert df["physician_id"].str.startswith("PHY").all()


def test_categorical_columns_within_expected_domains():
    df = generate_dataset(n_physicians=400, seed=3)
    assert set(df["specialty"].unique()) <= set(SPECIALTIES)
    assert set(df["region"].unique()) <= set(REGIONS)
    assert set(df["practice_setting"].unique()) <= set(PRACTICE_SETTINGS)


def test_numeric_ranges_are_realistic():
    df = generate_dataset(n_physicians=1000, seed=9)
    assert (df["years_in_practice"] >= 1).all() and (df["years_in_practice"] <= 45).all()
    assert (df["patient_panel_size"] > 0).all()
    assert (df["pct_patients_65_plus"].between(0, 1)).all()
    assert (df["pct_female_patients"].between(0, 1)).all()
    assert (df["market_share_pct"].between(0, 1)).all()
    assert (df["total_market_trx_monthly"] > 0).all()
    assert (df["company_trx_monthly"] >= 0).all()
    assert (df["company_nrx_monthly"] >= 0).all()
    assert (df["digital_engagement_score"].between(0, 100)).all()
    assert (df["peer_influence_score"].between(0, 100)).all()
    assert (df["rep_visits_last_12mo"] >= 0).all()
    assert (df["samples_distributed_last_12mo"] >= 0).all()


def test_company_trx_never_exceeds_total_market_trx():
    df = generate_dataset(n_physicians=1000, seed=11)
    # market_share_pct is capped at 0.90, so company volume should never
    # exceed total market volume for the same physician.
    assert (df["company_trx_monthly"] <= df["total_market_trx_monthly"] + 1e-6).all()


def test_tenure_never_exceeds_years_in_practice():
    df = generate_dataset(n_physicians=1000, seed=13)
    assert (df["years_prescribing_company_product"] <= df["years_in_practice"] + 1e-6).all()


def test_no_missing_values():
    df = generate_dataset(n_physicians=300, seed=17)
    assert df.isna().sum().sum() == 0


def test_zero_physicians_returns_empty_with_correct_columns():
    df = generate_dataset(n_physicians=0, seed=1)
    assert len(df) == 0
    assert list(df.columns) == EXPECTED_COLUMNS


def test_single_physician_does_not_crash():
    df = generate_dataset(n_physicians=1, seed=1)
    assert len(df) == 1
    assert df.isna().sum().sum() == 0
