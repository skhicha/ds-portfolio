import numpy as np
import pandas as pd
import pytest

from src.data import TARGET_COL, add_time_features, load_raw, resample_daily_window, time_based_split
from src.evaluate import mape, rmse
from src.models import forecast_sarima, forecast_xgboost, train_sarima, train_xgboost


def test_xgboost_train_forecast_shapes_and_no_nans(hourly_series):
    feat = add_time_features(hourly_series)
    X_train, y_train, X_test, y_test = time_based_split(feat, test_frac=0.2)

    model = train_xgboost(
        X_train, y_train, X_test, y_test,
        n_estimators=20, early_stopping_rounds=5,
    )
    result = forecast_xgboost(model, X_test, y_test)

    assert result.y_pred.shape == result.y_true.shape
    assert len(result.y_pred) == len(X_test)
    assert not np.isnan(result.y_pred).any()
    assert "feature_importance" in result.extra
    assert len(result.extra["feature_importance"]) == X_test.shape[1]
    # a near-periodic daily sine series should be forecastable reasonably well
    assert rmse(result.y_true, result.y_pred) < 200


def test_xgboost_handles_empty_test_set(hourly_series):
    feat = add_time_features(hourly_series)
    feature_cols = [c for c in feat.columns if c != TARGET_COL]
    X_train, y_train = feat[feature_cols], feat[TARGET_COL]
    X_test, y_test = X_train.iloc[0:0], y_train.iloc[0:0]

    model = train_xgboost(X_train, y_train, X_test, y_test, n_estimators=10, early_stopping_rounds=None)
    result = forecast_xgboost(model, X_test, y_test)
    assert len(result.y_pred) == 0
    assert len(result.y_true) == 0


def test_xgboost_small_train_set_skips_early_stopping_without_crashing(hourly_series):
    feat = add_time_features(hourly_series)
    feature_cols = [c for c in feat.columns if c != TARGET_COL]
    small = feat.iloc[:30]
    X_train, y_train = small[feature_cols].iloc[:20], small[TARGET_COL].iloc[:20]
    X_test, y_test = small[feature_cols].iloc[20:], small[TARGET_COL].iloc[20:]

    # early_stopping_rounds=50 > len(X_train) -> should be auto-disabled, not crash
    model = train_xgboost(X_train, y_train, X_test, y_test, n_estimators=10, early_stopping_rounds=50)
    result = forecast_xgboost(model, X_test, y_test)
    assert not np.isnan(result.y_pred).any()


def test_sarima_train_forecast_shapes_and_no_nans():
    rng = np.random.RandomState(3)
    idx = pd.date_range("2020-01-01", periods=120, freq="D")
    values = 1000 + 50 * np.sin(2 * np.pi * np.arange(120) / 7) + rng.normal(0, 2, 120)
    series = pd.Series(values, index=idx)
    train, test = series.iloc[:100], series.iloc[100:]

    fit = train_sarima(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    result = forecast_sarima(fit, test)

    assert result.y_pred.shape == (len(test),)
    assert not np.isnan(result.y_pred).any()
    assert "ci_lower" in result.extra and "ci_upper" in result.extra
    assert np.all(result.extra["ci_upper"] >= result.extra["ci_lower"])
    # a strongly periodic weekly series should be forecastable with low MAPE
    assert mape(result.y_true, result.y_pred) < 15


def test_sarima_handles_empty_test_series():
    idx = pd.date_range("2020-01-01", periods=60, freq="D")
    series = pd.Series(np.random.RandomState(4).normal(100, 5, 60), index=idx)
    fit = train_sarima(series)
    empty_test = series.iloc[0:0]
    result = forecast_sarima(fit, empty_test)
    assert len(result.y_pred) == 0


@pytest.mark.slow
def test_end_to_end_on_real_data_slice(real_data_available):
    if not real_data_available:
        pytest.skip("Real PJME_hourly.csv not available in this environment.")
    import os

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "PJME_hourly.csv")
    df = load_raw(path)
    # Use a recent, reasonably sized slice so the test stays fast.
    df_slice = df.iloc[-5000:]
    feat = add_time_features(df_slice)
    X_train, y_train, X_test, y_test = time_based_split(feat, test_frac=0.2)

    model = train_xgboost(X_train, y_train, X_test, y_test, n_estimators=50, early_stopping_rounds=10)
    result = forecast_xgboost(model, X_test, y_test)

    assert len(result.y_pred) == len(X_test)
    assert not np.isnan(result.y_pred).any()
    real_mape = mape(result.y_true, result.y_pred)
    assert real_mape < 20  # sane upper bound, not a tight assertion on real data
