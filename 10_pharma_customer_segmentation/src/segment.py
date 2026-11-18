"""
Core customer-segmentation pipeline: feature engineering, scaling, PCA
(dimensionality reduction), and K-Means clustering with a principled method
for choosing k (elbow / inertia + silhouette score), plus cluster profiling.

Everything here operates on plain pandas/numpy/scikit-learn objects so it can
be reused identically from the CLI, the test suite, and the Streamlit app.

Design notes:
    - Clustering is performed on the PCA-reduced feature space (not the raw
      scaled features) - this is the conventional "PCA then K-Means" workflow
      and also makes the elbow/silhouette search faster and less sensitive to
      noise in any single feature.
    - All functions are defensive about small/degenerate inputs (0-2 rows,
      a requested k larger than the number of distinct points, zero-variance
      columns, fully duplicated rows) so the Streamlit app never crashes on
      an edge case; see tests/test_segment.py for the exact contracts.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Numeric columns used as clustering features. All come directly from
# src.generate_data's output schema (or are engineered from it below).
FEATURE_COLUMNS = [
    "years_in_practice",
    "patient_panel_size",
    "pct_patients_65_plus",
    "avg_comorbidity_score",
    "total_market_trx_monthly",
    "company_trx_monthly",
    "company_nrx_monthly",
    "company_rx_value_monthly",
    "market_share_pct",
    "digital_engagement_score",
    "rep_visits_last_12mo",
    "samples_distributed_last_12mo",
    "peer_influence_score",
    "years_prescribing_company_product",
]

DEFAULT_K_RANGE = range(2, 9)
DEFAULT_VARIANCE_THRESHOLD = 0.85
RANDOM_STATE = 42


@dataclass
class PCAResult:
    model: PCA
    transformed: np.ndarray  # (n_samples, n_components_selected)
    explained_variance_ratio: np.ndarray  # full spectrum, one entry per available component
    cumulative_variance: np.ndarray  # same length as explained_variance_ratio
    n_components_selected: int
    loadings: pd.DataFrame  # (n_components_selected, n_features) component loadings


@dataclass
class KSelectionResult:
    table: pd.DataFrame  # columns: k, inertia, silhouette_score
    best_k: int


@dataclass
class SegmentationResult:
    df: pd.DataFrame  # original rows + "cluster" + "pc1"/"pc2" (+ more PCs if selected)
    feature_matrix: pd.DataFrame  # engineered features actually used for clustering
    scaler: StandardScaler
    pca: PCAResult
    kmeans: KMeans
    k: int
    silhouette: float
    k_selection: KSelectionResult
    cluster_profiles: pd.DataFrame
    cluster_labels: dict = field(default_factory=dict)


def engineer_features(df: pd.DataFrame, feature_columns: Iterable[str] = FEATURE_COLUMNS) -> pd.DataFrame:
    """
    Select + engineer the numeric feature matrix used for clustering.

    Adds two ratio features that capture behavior PCA otherwise has to infer
    indirectly from raw volumes:
        - nrx_to_trx_ratio: new-patient-start intensity relative to total
          company volume (high = growing prescriber, low = maintenance-only).
        - whitespace_trx_monthly: total market Rx the physician writes that
          is NOT currently captured by the company product (growth headroom).

    Missing values are median-imputed per column so a partially-missing
    upload never crashes the pipeline. Returns an empty-but-correctly-shaped
    DataFrame when `df` has no rows.
    """
    cols = list(feature_columns)
    base = df.reindex(columns=cols).apply(pd.to_numeric, errors="coerce").copy()

    if len(df) == 0:
        engineered = base.copy()
        engineered["nrx_to_trx_ratio"] = pd.Series(dtype=float)
        engineered["whitespace_trx_monthly"] = pd.Series(dtype=float)
        return engineered

    trx = df.get("company_trx_monthly", pd.Series(0.0, index=df.index)).astype(float)
    nrx = df.get("company_nrx_monthly", pd.Series(0.0, index=df.index)).astype(float)
    total_market = df.get("total_market_trx_monthly", pd.Series(0.0, index=df.index)).astype(float)

    nrx_to_trx_ratio = np.where(trx > 0, nrx / trx.replace(0, np.nan), 0.0)
    whitespace = (total_market - trx).clip(lower=0)

    engineered = base.copy()
    engineered["nrx_to_trx_ratio"] = pd.Series(nrx_to_trx_ratio, index=df.index).fillna(0.0)
    engineered["whitespace_trx_monthly"] = whitespace

    # Median-impute any remaining NaNs (missing source columns, bad uploads).
    for col in engineered.columns:
        if engineered[col].isna().any():
            median = engineered[col].median()
            fill_value = 0.0 if pd.isna(median) else median
            engineered[col] = engineered[col].fillna(fill_value)

    return engineered


def scale_features(feature_df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Standardize features to zero mean / unit variance. Handles n==0/1 rows safely."""
    scaler = StandardScaler()
    if len(feature_df) == 0:
        return np.empty((0, feature_df.shape[1])), scaler
    scaled = scaler.fit_transform(feature_df.to_numpy(dtype=float))
    return scaled, scaler


def run_pca(
    scaled: np.ndarray,
    n_components: Optional[int] = None,
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
) -> PCAResult:
    """
    Fit PCA on standardized features.

    If `n_components` is None, selects the smallest number of components
    whose cumulative explained variance reaches `variance_threshold`
    (minimum 2, for visualization), capped by the data's actual rank so
    degenerate/highly-correlated inputs never raise.
    """
    n_samples, n_features = scaled.shape
    max_components = max(1, min(n_samples, n_features)) if n_samples > 0 else 1

    if n_samples == 0:
        empty_model = PCA(n_components=1)
        return PCAResult(
            model=empty_model,
            transformed=np.empty((0, 1)),
            explained_variance_ratio=np.array([]),
            cumulative_variance=np.array([]),
            n_components_selected=1,
            loadings=pd.DataFrame(),
        )

    if n_samples == 1:
        # A single sample has zero variance by definition; PCA's variance
        # ratio is mathematically undefined (0/0). Short-circuit instead of
        # letting sklearn emit a divide-by-zero RuntimeWarning.
        chosen = max(1, min(n_components or 1, max_components))
        pca_model = PCA(n_components=chosen, random_state=RANDOM_STATE)
        with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            pca_model.fit(np.vstack([scaled, np.zeros_like(scaled)]))  # fake 2nd point so fit succeeds
        transformed = np.zeros((1, chosen))
        return PCAResult(
            model=pca_model,
            transformed=transformed,
            explained_variance_ratio=np.zeros(chosen),
            cumulative_variance=np.zeros(chosen),
            n_components_selected=chosen,
            loadings=pd.DataFrame(pca_model.components_),
        )

    full_pca = PCA(n_components=max_components, random_state=RANDOM_STATE)
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        full_pca.fit(scaled)
        variance_ratio = np.nan_to_num(full_pca.explained_variance_ratio_, nan=0.0)
    cumulative = np.cumsum(variance_ratio)

    if n_components is not None:
        chosen = max(1, min(n_components, max_components))
    else:
        reach = np.argmax(cumulative >= variance_threshold) + 1 if cumulative.size else 1
        chosen = max(1, min(2, max_components)) if max_components < 2 else max(2, min(int(reach), max_components))
        chosen = min(chosen, max_components)

    pca_model = PCA(n_components=chosen, random_state=RANDOM_STATE)
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        transformed = pca_model.fit_transform(scaled)
    loadings = pd.DataFrame(pca_model.components_)

    return PCAResult(
        model=pca_model,
        transformed=transformed,
        explained_variance_ratio=variance_ratio,
        cumulative_variance=cumulative,
        n_components_selected=chosen,
        loadings=loadings,
    )


def _n_unique_rows(X: np.ndarray) -> int:
    if X.shape[0] == 0:
        return 0
    return len(np.unique(np.round(X, 8), axis=0))


def choose_k(
    transformed: np.ndarray,
    k_range: Iterable[int] = DEFAULT_K_RANGE,
    random_state: int = RANDOM_STATE,
) -> KSelectionResult:
    """
    Elbow (inertia) + silhouette-score search over `k_range`.

    Only k values that are feasible for the data (2 <= k <= min(n_samples,
    n_unique_rows)) are evaluated. The best k is the one maximizing silhouette
    score (ties broken by the smaller k). If no k is feasible (e.g. fewer
    than 2 distinct points), returns an empty table and best_k=1.
    """
    n_unique = _n_unique_rows(transformed)
    n_samples = transformed.shape[0]
    # silhouette_score requires 2 <= n_labels <= n_samples - 1.
    feasible_ks = [k for k in k_range if 2 <= k <= min(n_samples - 1, n_unique)]

    rows = []
    for k in feasible_ks:
        model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = model.fit_predict(transformed)
        if len(set(labels)) < 2:
            sil = float("nan")
        else:
            sil = float(silhouette_score(transformed, labels))
        rows.append({"k": k, "inertia": float(model.inertia_), "silhouette_score": sil})

    table = pd.DataFrame(rows, columns=["k", "inertia", "silhouette_score"])
    if table.empty or table["silhouette_score"].notna().sum() == 0:
        return KSelectionResult(table=table, best_k=1 if n_unique < 2 else min(2, n_unique))

    best_row = table.loc[table["silhouette_score"].idxmax()]
    return KSelectionResult(table=table, best_k=int(best_row["k"]))


def recommend_k(k_selection_table: pd.DataFrame, min_k: int = 3) -> int:
    """
    Pick a recommended k from a `choose_k` search table.

    Silhouette score is maximized globally at k=2 for almost any dataset
    with one well-separated supercluster, which is statistically correct
    but not actionable as a "customer segmentation" (two buckets isn't a
    targeting model). We therefore recommend the silhouette-maximizing k
    within the business-meaningful range `k >= min_k`, falling back to the
    unrestricted global maximum if nothing in that range is feasible.
    """
    if k_selection_table.empty:
        return 1
    valid = k_selection_table[k_selection_table["silhouette_score"].notna()]
    if valid.empty:
        return int(k_selection_table.iloc[0]["k"])

    constrained = valid[valid["k"] >= min_k]
    pool = constrained if not constrained.empty else valid
    best_row = pool.loc[pool["silhouette_score"].idxmax()]
    return int(best_row["k"])


def fit_kmeans(
    transformed: np.ndarray,
    k: int,
    random_state: int = RANDOM_STATE,
) -> tuple[KMeans, np.ndarray, float]:
    """
    Fit K-Means with `k` clusters, clipping `k` down to the number of
    distinct points when the request is infeasible (so the app never
    crashes when a user asks for more clusters than the data supports).

    Returns (fitted_model, labels, silhouette_score). silhouette_score is
    NaN when fewer than 2 distinct cluster labels result (e.g. k clipped
    to 1, or all points identical).
    """
    n_samples = transformed.shape[0]
    if n_samples == 0:
        raise ValueError("Cannot fit K-Means on zero rows.")

    n_unique = _n_unique_rows(transformed)
    effective_k = max(1, min(k, n_samples, max(n_unique, 1)))

    model = KMeans(n_clusters=effective_k, n_init=10, random_state=random_state)
    labels = model.fit_predict(transformed)

    n_labels = len(set(labels))
    # silhouette_score requires 2 <= n_labels <= n_samples - 1.
    if n_labels < 2 or n_labels > n_samples - 1:
        sil = float("nan")
    else:
        sil = float(silhouette_score(transformed, labels))

    return model, labels, sil


def profile_clusters(df_with_clusters: pd.DataFrame, feature_columns: Iterable[str] = FEATURE_COLUMNS) -> pd.DataFrame:
    """Per-cluster mean of each feature + cluster size, sorted by cluster id."""
    cols = [c for c in feature_columns if c in df_with_clusters.columns]
    grouped = df_with_clusters.groupby("cluster")[cols].mean()
    grouped.insert(0, "size", df_with_clusters.groupby("cluster").size())
    grouped.insert(1, "pct_of_total", (grouped["size"] / len(df_with_clusters) * 100).round(1) if len(df_with_clusters) else 0.0)
    return grouped.reset_index()


def label_clusters(cluster_profiles: pd.DataFrame, value_col: str = "company_rx_value_monthly") -> dict:
    """
    Assign a human-readable tier label per cluster, ranked by `value_col`
    (descending). Purely descriptive - does not affect the clustering itself.
    """
    if value_col not in cluster_profiles.columns or cluster_profiles.empty:
        return {row["cluster"]: f"Segment {row['cluster']}" for _, row in cluster_profiles.iterrows()}

    ranked = cluster_profiles.sort_values(value_col, ascending=False).reset_index(drop=True)
    labels = {}
    n = len(ranked)
    for i, row in ranked.iterrows():
        if n == 1:
            labels[row["cluster"]] = "All Physicians"
        elif i == 0:
            labels[row["cluster"]] = "High-Value Core"
        elif i == n - 1:
            labels[row["cluster"]] = "Low-Priority"
        else:
            labels[row["cluster"]] = f"Mid-Tier {i}"
    return labels


def segment(
    df: pd.DataFrame,
    feature_columns: Iterable[str] = FEATURE_COLUMNS,
    k: Optional[int] = None,
    k_range: Iterable[int] = DEFAULT_K_RANGE,
    pca_variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
    pca_n_components: Optional[int] = None,
    min_actionable_k: int = 3,
    random_state: int = RANDOM_STATE,
) -> SegmentationResult:
    """
    End-to-end segmentation pipeline: engineer features -> scale -> PCA ->
    choose k (elbow + silhouette, unless `k` is given explicitly) ->
    K-Means -> cluster profiling.

    Safe for 0, 1, or a handful of rows: with fewer than 2 distinct points
    every physician is assigned to cluster 0 and silhouette is NaN.
    """
    feature_df = engineer_features(df, feature_columns)
    scaled, scaler = scale_features(feature_df)
    pca_result = run_pca(scaled, n_components=pca_n_components, variance_threshold=pca_variance_threshold)

    if len(df) == 0:
        empty = df.copy()
        empty["cluster"] = pd.Series(dtype=int)
        return SegmentationResult(
            df=empty,
            feature_matrix=feature_df,
            scaler=scaler,
            pca=pca_result,
            kmeans=KMeans(n_clusters=1),
            k=0,
            silhouette=float("nan"),
            k_selection=KSelectionResult(table=pd.DataFrame(columns=["k", "inertia", "silhouette_score"]), best_k=0),
            cluster_profiles=pd.DataFrame(),
            cluster_labels={},
        )

    k_selection = choose_k(pca_result.transformed, k_range=k_range, random_state=random_state)
    chosen_k = k if k is not None else recommend_k(k_selection.table, min_k=min_actionable_k)
    kmeans_model, labels, sil = fit_kmeans(pca_result.transformed, chosen_k, random_state=random_state)

    out = df.copy()
    out["cluster"] = labels
    for i in range(pca_result.transformed.shape[1]):
        out[f"pc{i + 1}"] = pca_result.transformed[:, i]

    profiles = profile_clusters(out, feature_columns)
    labels_map = label_clusters(profiles)

    return SegmentationResult(
        df=out,
        feature_matrix=feature_df,
        scaler=scaler,
        pca=pca_result,
        kmeans=kmeans_model,
        k=int(kmeans_model.n_clusters),
        silhouette=sil,
        k_selection=k_selection,
        cluster_profiles=profiles,
        cluster_labels=labels_map,
    )


def main() -> None:
    import os

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "data", "physicians.csv")
    df = pd.read_csv(data_path)
    result = segment(df)
    print(f"Chosen k = {result.k} | Silhouette score = {result.silhouette:.3f}")
    print(f"PCA components used: {result.pca.n_components_selected} "
          f"(cumulative variance explained: {result.pca.cumulative_variance[result.pca.n_components_selected - 1]:.1%})")
    print("\nK selection search:")
    print(result.k_selection.table)
    print("\nCluster profiles:")
    print(result.cluster_profiles[["cluster", "size", "pct_of_total", "company_rx_value_monthly", "market_share_pct"]])
    print("\nCluster labels:", result.cluster_labels)


if __name__ == "__main__":
    main()
