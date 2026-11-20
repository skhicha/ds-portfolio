import os

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "PJME_hourly.csv")


@pytest.fixture
def hourly_series():
    """~6 weeks of synthetic hourly data with a clean daily sine pattern plus
    small noise, so lag/rolling features are predictable by construction.
    """
    rng = np.random.RandomState(0)
    n = 24 * 42  # 42 days
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    hour_of_day = idx.hour.to_numpy()
    values = 1000 + 200 * np.sin(2 * np.pi * hour_of_day / 24) + rng.normal(0, 5, size=n)
    df = pd.DataFrame({"PJME_MW": values}, index=idx)
    df.index.name = "Datetime"
    return df


@pytest.fixture
def tiny_hourly_series():
    """A handful of hourly rows - too short for lag_168h, used for edge cases."""
    idx = pd.date_range("2020-01-01", periods=10, freq="h")
    values = np.linspace(100, 109, 10)
    df = pd.DataFrame({"PJME_MW": values}, index=idx)
    df.index.name = "Datetime"
    return df


@pytest.fixture
def white_noise_series():
    """Stationary by construction: pure Gaussian white noise."""
    rng = np.random.RandomState(1)
    idx = pd.date_range("2020-01-01", periods=500, freq="D")
    return pd.Series(rng.normal(0, 1, size=500), index=idx)


@pytest.fixture
def random_walk_series():
    """Non-stationary by construction: cumulative sum of white noise (unit root)."""
    rng = np.random.RandomState(2)
    idx = pd.date_range("2020-01-01", periods=500, freq="D")
    steps = rng.normal(0, 1, size=500)
    return pd.Series(np.cumsum(steps), index=idx)


@pytest.fixture
def raw_csv_path(tmp_path):
    """A small raw PJME-shaped CSV with a duplicate row and a missing hour,
    to exercise load_raw's dedup/resample/interpolate behavior.
    """
    rows = [
        ("2020-01-01 00:00:00", 100.0),
        ("2020-01-01 01:00:00", 110.0),
        ("2020-01-01 01:00:00", 110.0),  # duplicate timestamp
        # 2020-01-01 02:00:00 missing on purpose -> should be interpolated
        ("2020-01-01 03:00:00", 130.0),
        ("2020-01-01 04:00:00", 140.0),
    ]
    path = tmp_path / "mini_pjme.csv"
    with open(path, "w") as f:
        f.write("Datetime,PJME_MW\n")
        for ts, val in rows:
            f.write(f"{ts},{val}\n")
    return str(path)


@pytest.fixture(scope="session")
def real_data_available():
    return os.path.exists(REAL_DATA_PATH)
