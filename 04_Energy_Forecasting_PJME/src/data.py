"""Loading, cleaning, stationarity testing, and feature engineering for the
PJME hourly energy consumption series.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

TARGET_COL = "PJME_MW"

DEFAULT_LAGS: tuple[int, ...] = (1, 2, 3, 24, 48, 168)
DEFAULT_ROLLING_WINDOWS: tuple[int, ...] = (24, 168)


def load_raw(path: str) -> pd.DataFrame:
    """Load the raw PJME hourly CSV and return a cleaned, hourly-indexed frame.

    Steps: parse the Datetime column, sort chronologically, drop duplicate
    timestamps, set Datetime as the index, resample to a strict hourly grid,
    and interpolate (time-weighted) to fill the small number of missing
    DST clock-change hours.
    """
    df = pd.read_csv(path)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime").drop_duplicates(subset="Datetime")
    df = df.set_index("Datetime")
    df = df.resample("h").interpolate(method="time")
    return df


@dataclass
class ADFResult:
    name: str
    statistic: float
    pvalue: float
    critical_values: dict
    is_stationary: bool
    n_obs: int = 0


def adf_test(series: pd.Series, name: str = "series", alpha: float = 0.05) -> ADFResult:
    """Run the Augmented Dickey-Fuller test for stationarity.

    H0: the series has a unit root (non-stationary).
    If p-value < alpha, we reject H0 -> the series is stationary.
    """
    clean = series.dropna()
    if len(clean) < 3:
        # Not enough data to run the test meaningfully; report as inconclusive
        # (treated as non-stationary) rather than crashing.
        return ADFResult(
            name=name, statistic=float("nan"), pvalue=float("nan"),
            critical_values={}, is_stationary=False, n_obs=len(clean),
        )
    stat, pvalue, _, nobs, crit, _ = adfuller(clean, result_object=False)
    return ADFResult(
        name=name,
        statistic=float(stat),
        pvalue=float(pvalue),
        critical_values={k: float(v) for k, v in crit.items()},
        is_stationary=bool(pvalue < alpha),
        n_obs=int(nobs),
    )


def add_time_features(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    lags: Sequence[int] = DEFAULT_LAGS,
    rolling_windows: Sequence[int] = DEFAULT_ROLLING_WINDOWS,
    dropna: bool = True,
) -> pd.DataFrame:
    """Convert a time-indexed series into a supervised-learning feature frame.

    Three feature families, all leakage-safe (every feature at time t uses
    only information available strictly before or at t, and rolling stats
    are computed on a series shifted by 1 so they never include the current
    target value):

    1. Calendar features: hour, day-of-week, month, quarter, day-of-year,
       week-of-year, is_weekend, plus sin/cos cyclical encodings of hour and
       month (so e.g. hour 23 and hour 0 are seen as adjacent).
    2. Lag features: the target value `lag` hours ago, for each of `lags`.
    3. Rolling-window statistics (mean, std) computed on the target series
       shifted by 1 hour, so the window for row t covers [t-w, t-1] and never
       touches the value being predicted.
    """
    out = df.copy()
    idx = out.index

    out["hour"] = idx.hour
    out["dayofweek"] = idx.day_of_week
    out["month"] = idx.month
    out["quarter"] = idx.quarter
    out["dayofyear"] = idx.day_of_year
    out["weekofyear"] = idx.isocalendar().week.astype(int).to_numpy()
    out["is_weekend"] = (idx.day_of_week >= 5).astype(int)

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)

    for lag in lags:
        out[f"lag_{lag}h"] = out[target_col].shift(lag)

    shifted = out[target_col].shift(1)
    for window in rolling_windows:
        out[f"roll_mean_{window}h"] = shifted.rolling(window).mean()
        out[f"roll_std_{window}h"] = shifted.rolling(window).std()

    if dropna:
        out = out.dropna()
    return out


def time_based_split(
    df_features: pd.DataFrame, target_col: str = TARGET_COL, test_frac: float = 0.2
):
    """Split a feature frame chronologically (no shuffling) into train/test.

    Returns (X_train, y_train, X_test, y_test).
    """
    if len(df_features) == 0:
        empty_x = df_features.drop(columns=[target_col], errors="ignore")
        empty_y = pd.Series(dtype=float)
        return empty_x, empty_y, empty_x.copy(), empty_y.copy()

    split_idx = int(len(df_features) * (1 - test_frac))
    split_idx = max(1, min(split_idx, len(df_features) - 1)) if len(df_features) > 1 else len(df_features)

    feature_cols = [c for c in df_features.columns if c != target_col]
    train = df_features.iloc[:split_idx]
    test = df_features.iloc[split_idx:]

    return train[feature_cols], train[target_col], test[feature_cols], test[target_col]


def resample_daily_window(series: pd.Series, start: str, end: str) -> pd.Series:
    """Resample an hourly series to daily means within [start, end]."""
    return series[start:end].resample("D").mean()
