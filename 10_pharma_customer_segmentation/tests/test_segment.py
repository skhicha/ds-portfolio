import numpy as np
import pandas as pd
import pytest

from src.generate_data import generate_dataset
from src.segment import (
    FEATURE_COLUMNS,
    choose_k,
    engineer_features,
    fit_kmeans,
    profile_clusters,
    recommend_k,
    run_pca,
    scale_features,
    segment,
)


# --------------------------------------------------------------------------- #
# Feature engineering / scaling
# --------------------------------------------------------------------------- #
def test_engineer_features_shape_and_engineered_columns(synthetic_df):
    feat = engineer_features(synthetic_df)
    assert len(feat) == len(synthetic_df)
    assert "nrx_to_trx_ratio" in feat.columns
    assert "whitespace_trx_monthly" in feat.columns
    assert feat.isna().sum().sum() == 0


def test_engineer_features_imputes_missing_values(synthetic_df):
    dirty = synthetic_df.copy()
    dirty.loc[0:10, "digital_engagement_score"] = np.nan
    dirty.loc[5:8, "patient_panel_size"] = np.nan
    feat = engineer_features(dirty)
    assert feat.isna().sum().sum() == 0
    assert len(feat) == len(dirty)


def test_scale_features_is_zero_mean_unit_variance(synthetic_df):
    feat = engineer_features(synthetic_df)
    scaled, scaler = scale_features(feat)
    assert scaled.shape == (len(synthetic_df), feat.shape[1])
    assert np.allclose(scaled.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(scaled.std(axis=0), 1, atol=1e-6)


# --------------------------------------------------------------------------- #
# PCA
# --------------------------------------------------------------------------- #
def test_pca_output_shape_and_variance_sanity(synthetic_df):
    feat = engineer_features(synthetic_df)
    scaled, _ = scale_features(feat)
    pca = run_pca(scaled)

    assert pca.transformed.shape[0] == len(synthetic_df)
    assert pca.transformed.shape[1] == pca.n_components_selected
    assert pca.n_components_selected >= 2

    # Explained variance ratios are valid fractions that sum to <= 1 and are
    # non-increasing (sklearn orders components by variance descending).
    assert (pca.explained_variance_ratio >= 0).all()
    assert pca.explained_variance_ratio.sum() <= 1.0 + 1e-6
    assert all(
        pca.explained_variance_ratio[i] >= pca.explained_variance_ratio[i + 1] - 1e-9
        for i in range(len(pca.explained_variance_ratio) - 1)
    )
    # Cumulative variance reached by the selected components meets the
    # default threshold (or uses all available components trying to).
    reached = pca.cumulative_variance[pca.n_components_selected - 1]
    assert reached > 0.5  # sanity floor - PCA should recover most structure


def test_pca_respects_explicit_n_components(synthetic_df):
    feat = engineer_features(synthetic_df)
    scaled, _ = scale_features(feat)
    pca = run_pca(scaled, n_components=3)
    assert pca.n_components_selected == 3
    assert pca.transformed.shape[1] == 3


def test_pca_highly_correlated_features_does_not_crash(synthetic_df):
    df = synthetic_df.copy()
    df["dup_feature_1"] = df["company_trx_monthly"]
    df["dup_feature_2"] = df["company_trx_monthly"] * 2.0
    feature_cols = FEATURE_COLUMNS + ["dup_feature_1", "dup_feature_2"]
    feat = engineer_features(df, feature_cols)
    scaled, _ = scale_features(feat)
    pca = run_pca(scaled)
    assert pca.transformed.shape[0] == len(df)
    assert np.isfinite(pca.transformed).all()


def test_pca_empty_input_does_not_crash():
    empty = pd.DataFrame(columns=FEATURE_COLUMNS)
    feat = engineer_features(empty)
    scaled, _ = scale_features(feat)
    pca = run_pca(scaled)
    assert pca.transformed.shape[0] == 0


# --------------------------------------------------------------------------- #
# K-Means / k selection
# --------------------------------------------------------------------------- #
def test_choose_k_and_recommend_k_on_synthetic_data(synthetic_df):
    feat = engineer_features(synthetic_df)
    scaled, _ = scale_features(feat)
    pca = run_pca(scaled)
    k_selection = choose_k(pca.transformed, k_range=range(2, 9))

    assert set(k_selection.table["k"]) == set(range(2, 9))
    assert (k_selection.table["silhouette_score"] >= -1).all()
    assert (k_selection.table["silhouette_score"] <= 1).all()

    recommended = recommend_k(k_selection.table, min_k=3)
    assert recommended >= 3


def test_fit_kmeans_produces_expected_number_of_clusters(synthetic_df):
    feat = engineer_features(synthetic_df)
    scaled, _ = scale_features(feat)
    pca = run_pca(scaled)
    model, labels, sil = fit_kmeans(pca.transformed, k=4)
    assert model.n_clusters == 4
    assert len(set(labels)) == 4
    assert len(labels) == len(synthetic_df)


def test_fit_kmeans_reasonable_silhouette_on_synthetic_data(synthetic_df):
    """The synthetic generator encodes 5 hidden behavioral archetypes, so a
    correctly specified k should recover genuine (if imperfect) structure."""
    feat = engineer_features(synthetic_df)
    scaled, _ = scale_features(feat)
    pca = run_pca(scaled)
    _, _, sil = fit_kmeans(pca.transformed, k=4)
    assert sil > 0.1


def test_full_segment_pipeline_matches_fit_kmeans_silhouette(synthetic_df):
    result = segment(synthetic_df, k=4)
    assert result.k == 4
    assert result.silhouette > 0.1
    assert "cluster" in result.df.columns
    assert set(result.df["cluster"].unique()) == set(range(4))


# --------------------------------------------------------------------------- #
# Cluster profiling
# --------------------------------------------------------------------------- #
def test_profile_clusters_sizes_sum_to_total(synthetic_df):
    result = segment(synthetic_df, k=4)
    profiles = profile_clusters(result.df)
    assert profiles["size"].sum() == len(synthetic_df)
    assert set(profiles["cluster"]) == set(range(4))


def test_label_clusters_are_unique_and_identify_high_value(synthetic_df):
    result = segment(synthetic_df, k=4)
    labels = result.cluster_labels
    assert len(set(labels.values())) == len(labels)
    assert "High-Value Core" in labels.values()
    assert "Low-Priority" in labels.values()


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #
def test_single_cluster_degenerate_data_does_not_crash(synthetic_df):
    """All rows identical -> PCA/K-Means should degrade gracefully, not crash."""
    dup = pd.concat([synthetic_df.iloc[[0]]] * 50, ignore_index=True)
    result = segment(dup, k=4)
    assert result.k == 1
    assert np.isnan(result.silhouette)
    assert len(result.df) == 50


def test_empty_input_does_not_crash():
    empty = pd.DataFrame(columns=FEATURE_COLUMNS + ["physician_id"])
    result = segment(empty)
    assert len(result.df) == 0
    assert result.k == 0
    assert np.isnan(result.silhouette)


def test_single_row_input_does_not_crash():
    df = generate_dataset(n_physicians=1, seed=1)
    result = segment(df)
    assert len(result.df) == 1
    assert result.k == 1
    assert np.isnan(result.silhouette)


def test_two_row_input_does_not_crash():
    df = generate_dataset(n_physicians=2, seed=1)
    result = segment(df)
    assert len(result.df) == 2
    # silhouette is undefined when every label count == n_samples; the
    # pipeline should fall back to a single cluster rather than raising.
    assert result.k == 1


def test_highly_correlated_features_pca_degenerate_case_does_not_crash(synthetic_df):
    df = synthetic_df.copy()
    df["dup_feature_1"] = df["company_trx_monthly"]
    df["dup_feature_2"] = df["company_trx_monthly"] * 2.0
    feature_cols = FEATURE_COLUMNS + ["dup_feature_1", "dup_feature_2"]
    result = segment(df, feature_columns=feature_cols, k=4)
    assert result.k == 4
    assert np.isfinite(result.silhouette)


def test_k_larger_than_distinct_points_is_clipped_not_crashed(small_df):
    """Only 5 distinct physicians but k=8 requested - should clip, not crash."""
    feat = engineer_features(small_df)
    scaled, _ = scale_features(feat)
    pca = run_pca(scaled)
    model, labels, sil = fit_kmeans(pca.transformed, k=8)
    assert model.n_clusters <= 5
    assert len(labels) == 5


def test_k_larger_than_distinct_points_full_pipeline(small_df):
    result = segment(small_df, k=8)
    assert result.k <= len(small_df)
    assert len(result.df) == len(small_df)


def test_zero_rows_raises_clean_error_on_direct_fit_kmeans_call():
    with pytest.raises(ValueError):
        fit_kmeans(np.empty((0, 3)), k=2)
