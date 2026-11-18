"""
Decile-based physician targeting model.

Scores every physician on a composite "commercial opportunity" score built
from four normalized signals (whitespace volume, current product value,
digital engagement, and peer/KOL influence), ranks them, and assigns deciles
1-10 (1 = highest-priority 10%, 10 = lowest-priority 10%) using a rank-based
split that is always well-defined - even with ties, fewer than 10 physicians,
or a single physician.

The targeting model is validated two ways:
    1. The K-Means silhouette score from `src.segment.segment()` (does the
       underlying behavioral segmentation hold together).
    2. `decile_silhouette_score` here (do the deciles THIS model assigns also
       correspond to reasonably separated groups in feature space, or is the
       opportunity score cutting across segment structure arbitrarily).
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

DEFAULT_SCORE_WEIGHTS = {
    "whitespace": 0.40,
    "value": 0.25,
    "engagement": 0.20,
    "peer_influence": 0.15,
}


def _minmax(series: pd.Series) -> pd.Series:
    """Scale a numeric series to [0, 1]. Returns all-zeros for empty/constant input."""
    if len(series) == 0:
        return series.astype(float)
    lo, hi = series.min(), series.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return pd.Series(0.0, index=series.index)
    return (series - lo) / (hi - lo)


def compute_opportunity_score(
    df: pd.DataFrame,
    weights: Optional[dict] = None,
) -> pd.Series:
    """
    Composite 0-1 targeting score per physician:
        whitespace (unmet market demand) + current value + digital
        engagement propensity + peer/KOL influence, min-max normalized and
        blended with `weights` (defaults to DEFAULT_SCORE_WEIGHTS).

    Returns an empty Series for an empty DataFrame.
    """
    w = weights or DEFAULT_SCORE_WEIGHTS
    if len(df) == 0:
        return pd.Series(dtype=float)

    total_market = df.get("total_market_trx_monthly", pd.Series(0.0, index=df.index)).astype(float)
    company_trx = df.get("company_trx_monthly", pd.Series(0.0, index=df.index)).astype(float)
    whitespace = (total_market - company_trx).clip(lower=0)

    value = df.get("company_rx_value_monthly", pd.Series(0.0, index=df.index)).astype(float)
    engagement = df.get("digital_engagement_score", pd.Series(0.0, index=df.index)).astype(float)
    peer = df.get("peer_influence_score", pd.Series(0.0, index=df.index)).astype(float)

    score = (
        w.get("whitespace", 0.0) * _minmax(whitespace)
        + w.get("value", 0.0) * _minmax(value)
        + w.get("engagement", 0.0) * _minmax(engagement)
        + w.get("peer_influence", 0.0) * _minmax(peer)
    )
    return score


def assign_deciles(score: pd.Series) -> pd.Series:
    """
    Rank-based decile assignment: decile 1 = top-scoring 10% of physicians,
    decile 10 = bottom-scoring 10%. Always returns integers in [1, 10]
    (for n < 10, not every decile will be populated) and is tie-safe (uses
    rank with method="first", never raises on duplicate scores, unlike a
    naive `pd.qcut` on raw score values).
    """
    n = len(score)
    if n == 0:
        return pd.Series(dtype="Int64")
    if n == 1:
        return pd.Series([1], index=score.index, dtype="Int64")

    ranks = score.rank(method="first", ascending=False)  # 1 = highest score
    deciles = np.ceil(ranks / n * 10).astype(int).clip(1, 10)
    return pd.Series(deciles, index=score.index, dtype="Int64")


def decile_silhouette_score(feature_matrix_scaled: np.ndarray, deciles: pd.Series) -> float:
    """
    Secondary validation metric: silhouette score treating each decile as a
    cluster label over the (already scaled) feature matrix. NaN when fewer
    than 2 populated deciles or too few rows for a valid silhouette.
    """
    n_samples = feature_matrix_scaled.shape[0]
    if n_samples == 0 or len(deciles) != n_samples:
        return float("nan")

    labels = deciles.to_numpy()
    n_labels = len(set(labels))
    if n_labels < 2 or n_labels > n_samples - 1:
        return float("nan")
    return float(silhouette_score(feature_matrix_scaled, labels))


def build_target_list(
    df_with_clusters: pd.DataFrame,
    weights: Optional[dict] = None,
    cluster_label_col: str = "cluster",
    cluster_name_map: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Build the ranked, decile-scored target list consumed by the app and the
    Excel report.

    Columns: rank, physician_id, specialty, region, segment (if cluster
    columns + name map available), opportunity_score, decile,
    whitespace_trx_monthly, company_rx_value_monthly.
    """
    if len(df_with_clusters) == 0:
        return pd.DataFrame(columns=[
            "rank", "physician_id", "specialty", "region", "segment",
            "opportunity_score", "decile", "whitespace_trx_monthly", "company_rx_value_monthly",
        ])

    score = compute_opportunity_score(df_with_clusters, weights=weights)
    deciles = assign_deciles(score)

    id_cols = [c for c in ["physician_id", "specialty", "region"] if c in df_with_clusters.columns]
    out = df_with_clusters[id_cols].copy()

    if cluster_label_col in df_with_clusters.columns:
        cluster_vals = df_with_clusters[cluster_label_col]
        if cluster_name_map:
            out["segment"] = cluster_vals.map(cluster_name_map).fillna(cluster_vals.astype(str))
        else:
            out["segment"] = cluster_vals

    total_market = df_with_clusters.get("total_market_trx_monthly", pd.Series(0.0, index=df_with_clusters.index))
    company_trx = df_with_clusters.get("company_trx_monthly", pd.Series(0.0, index=df_with_clusters.index))

    out["opportunity_score"] = score.round(4)
    out["decile"] = deciles
    out["whitespace_trx_monthly"] = (total_market.astype(float) - company_trx.astype(float)).clip(lower=0).round(1)
    out["company_rx_value_monthly"] = df_with_clusters.get(
        "company_rx_value_monthly", pd.Series(0.0, index=df_with_clusters.index)
    ).round(2)

    out = out.sort_values("opportunity_score", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def decile_summary(target_list: pd.DataFrame) -> pd.DataFrame:
    """Per-decile rollup: physician count, avg score, avg whitespace, avg value."""
    if target_list.empty:
        return pd.DataFrame(columns=["decile", "physician_count", "avg_opportunity_score",
                                      "avg_whitespace_trx_monthly", "avg_company_rx_value_monthly"])
    grouped = target_list.groupby("decile").agg(
        physician_count=("rank", "count"),
        avg_opportunity_score=("opportunity_score", "mean"),
        avg_whitespace_trx_monthly=("whitespace_trx_monthly", "mean"),
        avg_company_rx_value_monthly=("company_rx_value_monthly", "mean"),
    ).reset_index()
    return grouped.sort_values("decile").reset_index(drop=True)


def main() -> None:
    import os
    from src.segment import segment

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "data", "physicians.csv")
    df = pd.read_csv(data_path)

    result = segment(df)
    targets = build_target_list(result.df, cluster_name_map=result.cluster_labels)
    sil = decile_silhouette_score(
        result.scaler.transform(result.feature_matrix.to_numpy(dtype=float)), targets["decile"]
    )

    print(f"Clustering silhouette score: {result.silhouette:.3f}")
    print(f"Decile-group silhouette score: {sil:.3f}")
    print("\nTop 10 targets:")
    print(targets.head(10).to_string(index=False))
    print("\nDecile summary:")
    print(decile_summary(targets).to_string(index=False))


if __name__ == "__main__":
    main()
