"""
Data loading for the house price prediction pipeline.

The original notebook/resume target was the Ames Housing dataset. At execution
time the notebook tried to pull a real Ames CSV from a public GitHub mirror
and, when that failed (no network, mirror gone, etc.), fell back to
`sklearn.datasets.fetch_california_housing` -- a free, no-download-required,
no-license-friction stand-in with a similar regression shape (continuous
numeric features -> a continuous home-value target). We keep that same
fallback behavior here and are upfront about it: this project trains on
California housing data, not real Ames Housing data.
"""
from __future__ import annotations

import pandas as pd
import requests
from sklearn.datasets import fetch_california_housing

AMES_URL = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/AmesHousing.csv"
TARGET_COLUMN = "MedHouseVal"
# California housing's MedHouseVal is in units of $100,000s; scale to dollars
# so the target is in the same realistic house-price range Ames would be in.
PRICE_SCALE = 100_000


def load_raw_data(prefer_ames: bool = False, timeout: float = 5.0) -> tuple[pd.DataFrame, str]:
    """
    Load the training dataset.

    When `prefer_ames` is True, attempts to download the real Ames Housing
    CSV first; on any failure (network, parsing, timeout) falls back to the
    bundled California Housing dataset. Default is `prefer_ames=False`
    because this project's committed, reproducible behavior is the
    California Housing fallback -- not a dataset that depends on a third
    party mirror staying online.

    Returns (dataframe, target_column_name).
    """
    if prefer_ames:
        try:
            resp = requests.get(AMES_URL, timeout=timeout)
            resp.raise_for_status()
            from io import StringIO

            df = pd.read_csv(StringIO(resp.text))
            if "SalePrice" in df.columns:
                return df, "SalePrice"
        except Exception:
            pass  # fall through to California Housing

    housing = fetch_california_housing(as_frame=True)
    df = housing.frame.copy()
    df.columns = [c.replace(" ", "_") for c in df.columns]
    df[TARGET_COLUMN] = df[TARGET_COLUMN] * PRICE_SCALE
    return df, TARGET_COLUMN


def load_dataset() -> tuple[pd.DataFrame, str]:
    """Convenience wrapper: load_raw_data() with the project's default settings."""
    return load_raw_data(prefer_ames=False)
