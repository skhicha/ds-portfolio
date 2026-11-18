import numpy as np
import pandas as pd
import pytest

from src.generate_data import generate_dataset
from src.segment import engineer_features, scale_features, segment
from src.targeting import (
    assign_deciles,
    build_target_list,
    compute_opportunity_score,
    decile_silhouette_score,
    decile_summary,
)


# --------------------------------------------------------------------------- #
# Opportunity score
# --------------------------------------------------------------------------- #
def test_compute_opportunity_score_is_bounded(synthetic_df):
    score = compute_opportunity_score(synthetic_df)
    assert len(score) == len(synthetic_df)
    assert (score >= 0).all()
    assert (score <= 1).all()


def test_compute_opportunity_score_empty_input():
    score = compute_opportunity_score(pd.DataFrame())
    assert len(score) == 0


def test_compute_opportunity_score_constant_column_does_not_divide_by_zero():
    df = pd.DataFrame({
        "total_market_trx_monthly": [100.0] * 10,
        "company_trx_monthly": [50.0] * 10,
        "company_rx_value_monthly": [1000.0] * 10,
        "digital_engagement_score": [50.0] * 10,
        "peer_influence_score": [50.0] * 10,
    })
    score = compute_opportunity_score(df)
    assert np.isfinite(score).all()
    assert (score == 0).all()  # every normalized signal is constant -> 0 contribution


# --------------------------------------------------------------------------- #
# Decile assignment
# --------------------------------------------------------------------------- #
def test_assign_deciles_are_integers_in_1_to_10_range():
    score = pd.Series(np.linspace(0, 1, 100))
    deciles = assign_deciles(score)
    assert set(deciles.unique()) == set(range(1, 11))
    assert deciles.between(1, 10).all()


def test_assign_deciles_counts_are_even_for_round_n():
    score = pd.Series(np.linspace(0, 1, 100))
    deciles = assign_deciles(score)
    counts = deciles.value_counts()
    assert (counts == 10).all()


def test_assign_deciles_ordering_decile_1_has_highest_scores():
    score = pd.Series(np.linspace(0, 1, 100))
    deciles = assign_deciles(score)
    means_by_decile = score.groupby(deciles).mean()
    means_sorted = means_by_decile.sort_index()
    # decile 1 (highest priority) should have the highest mean score,
    # decreasing monotonically through decile 10.
    assert list(means_sorted) == sorted(means_sorted, reverse=True)


def test_assign_deciles_handles_ties_without_crashing():
    score = pd.Series([0.5] * 50)
    deciles = assign_deciles(score)
    assert deciles.between(1, 10).all()
    assert len(deciles) == 50


def test_assign_deciles_empty_input():
    deciles = assign_deciles(pd.Series(dtype=float))
    assert len(deciles) == 0


def test_assign_deciles_single_row():
    deciles = assign_deciles(pd.Series([0.7]))
    assert list(deciles) == [1]


def test_assign_deciles_small_n_still_bounded():
    score = pd.Series([0.9, 0.1, 0.5, 0.3, 0.7])
    deciles = assign_deciles(score)
    assert deciles.between(1, 10).all()
    # highest score gets the lowest (best) decile number
    assert deciles.loc[score.idxmax()] <= deciles.loc[score.idxmin()]


# --------------------------------------------------------------------------- #
# Decile silhouette validation
# --------------------------------------------------------------------------- #
def test_decile_silhouette_score_runs_on_synthetic_data(synthetic_df):
    result = segment(synthetic_df, k=4)
    targets = build_target_list(result.df, cluster_name_map=result.cluster_labels)
    scaled, _ = scale_features(engineer_features(result.df))
    sil = decile_silhouette_score(scaled, targets["decile"])
    assert -1 <= sil <= 1


def test_decile_silhouette_score_empty_input_returns_nan():
    sil = decile_silhouette_score(np.empty((0, 3)), pd.Series(dtype=int))
    assert np.isnan(sil)


def test_decile_silhouette_score_mismatched_length_returns_nan():
    sil = decile_silhouette_score(np.zeros((5, 3)), pd.Series([1, 2, 3]))
    assert np.isnan(sil)


# --------------------------------------------------------------------------- #
# Target list / decile summary
# --------------------------------------------------------------------------- #
def test_build_target_list_is_ranked_and_complete(synthetic_df):
    result = segment(synthetic_df, k=4)
    targets = build_target_list(result.df, cluster_name_map=result.cluster_labels)
    assert len(targets) == len(synthetic_df)
    assert list(targets["rank"]) == list(range(1, len(targets) + 1))
    # ranked descending by score
    assert targets["opportunity_score"].is_monotonic_decreasing
    assert set(targets["decile"].unique()) <= set(range(1, 11))


def test_build_target_list_empty_input():
    targets = build_target_list(pd.DataFrame())
    assert len(targets) == 0
    assert "rank" in targets.columns


def test_decile_summary_counts_and_ordering(synthetic_df):
    result = segment(synthetic_df, k=4)
    targets = build_target_list(result.df, cluster_name_map=result.cluster_labels)
    dsum = decile_summary(targets)
    assert dsum["physician_count"].sum() == len(targets)
    assert list(dsum["decile"]) == sorted(dsum["decile"])
    # avg opportunity score should decrease (or stay flat) across deciles
    scores = dsum["avg_opportunity_score"].to_list()
    assert all(scores[i] >= scores[i + 1] - 1e-9 for i in range(len(scores) - 1))


def test_decile_summary_empty_input():
    dsum = decile_summary(pd.DataFrame())
    assert len(dsum) == 0
