import numpy as np
import pandas as pd

from src.data import adf_test


def test_adf_detects_stationary_white_noise(white_noise_series):
    result = adf_test(white_noise_series, name="white_noise")
    assert result.is_stationary is True
    assert result.pvalue < 0.05
    assert np.isfinite(result.statistic)
    assert set(result.critical_values.keys()) == {"1%", "5%", "10%"}


def test_adf_detects_nonstationary_random_walk(random_walk_series):
    result = adf_test(random_walk_series, name="random_walk")
    assert result.is_stationary is False
    assert result.pvalue >= 0.05


def test_adf_on_differenced_random_walk_becomes_stationary(random_walk_series):
    # First-differencing a random walk recovers the underlying white noise.
    diffed = random_walk_series.diff().dropna()
    result = adf_test(diffed, name="diffed_random_walk")
    assert result.is_stationary is True


def test_adf_handles_too_short_series_without_crashing():
    tiny = pd.Series([1.0, 2.0])
    result = adf_test(tiny, name="tiny")
    assert result.is_stationary is False
    assert np.isnan(result.statistic)


def test_adf_handles_all_nan_series():
    series = pd.Series([np.nan, np.nan, np.nan])
    result = adf_test(series, name="all_nan")
    assert result.n_obs == 0
    assert result.is_stationary is False
