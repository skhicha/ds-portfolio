"""
Feature engineering for the house price prediction pipeline: missing-value
treatment, IQR-based outlier handling, and the log-transform applied to the
regression target (with its inverse for converting predictions back to
dollars).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def log_transform_target(y: pd.Series) -> pd.Series:
    """Log-transform a strictly-positive target series. Stabilizes variance
    for right-skewed price data and is what every model in this project is
    actually trained on."""
    if (y <= 0).any():
        raise ValueError("log_transform_target requires strictly positive values")
    return np.log(y)


def inverse_log_transform(y_log) -> np.ndarray:
    """Inverse of log_transform_target: exp() back to the original (dollar) scale."""
    return np.exp(np.asarray(y_log, dtype=float))


def handle_missing_values(df: pd.DataFrame, target_col: str | None = None) -> pd.DataFrame:
    """
    Fill missing values: numeric columns with their median, categorical
    columns with their mode. The target column (if given) is left untouched
    -- we never impute the thing we're trying to predict.
    """
    df = df.copy()
    for col in df.columns:
        if col == target_col:
            continue
        if df[col].isnull().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            mode = df[col].mode()
            df[col] = df[col].fillna(mode.iloc[0] if len(mode) else "Unknown")
    return df


def handle_outliers_iqr(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    multiplier: float = 1.5,
) -> pd.DataFrame:
    """
    Clip (winsorize) numeric columns to the IQR-based [Q1 - k*IQR, Q3 + k*IQR]
    fence instead of dropping rows, so row count -- and alignment with any
    accompanying target series -- is always preserved.
    """
    df = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df


def engineer_features(
    df: pd.DataFrame,
    target_col: str,
    outlier_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Full feature engineering pipeline: missing values, then IQR outlier clipping
    on the numeric feature columns (never on the target)."""
    df = handle_missing_values(df, target_col=target_col)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target_col]
    if outlier_columns is not None:
        numeric_cols = [c for c in numeric_cols if c in outlier_columns]
    df = handle_outliers_iqr(df, columns=numeric_cols)
    return df
