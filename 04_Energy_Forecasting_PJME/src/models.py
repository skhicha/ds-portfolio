"""SARIMA and XGBoost model training/forecasting with a shared result shape."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor


@dataclass
class ForecastResult:
    """Common output shape for both models so app.py can compare them uniformly."""

    model_name: str
    y_true: np.ndarray
    y_pred: np.ndarray
    index: pd.Index
    extra: dict = None  # e.g. {"feature_importance": Series} for XGBoost


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    early_stopping_rounds: Optional[int] = 50,
    random_state: int = 42,
) -> XGBRegressor:
    """Train an XGBoost regressor with early stopping on the test set.

    Early stopping is disabled automatically when there are too few rows
    (e.g. tiny synthetic test fixtures) to avoid XGBoost raising on a
    degenerate eval set.
    """
    use_early_stopping = (
        early_stopping_rounds is not None
        and len(X_test) > 0
        and len(X_train) > early_stopping_rounds
    )
    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        early_stopping_rounds=early_stopping_rounds if use_early_stopping else None,
        eval_metric="rmse",
        random_state=random_state,
        verbosity=0,
        n_jobs=-1,
    )
    fit_kwargs = {}
    if len(X_test) > 0:
        fit_kwargs["eval_set"] = [(X_test, y_test)]
        fit_kwargs["verbose"] = False
    model.fit(X_train, y_train, **fit_kwargs)
    return model


def forecast_xgboost(model: XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> ForecastResult:
    if len(X_test) == 0:
        return ForecastResult(
            model_name="XGBoost",
            y_true=np.array([]),
            y_pred=np.array([]),
            index=X_test.index,
            extra={"feature_importance": pd.Series(dtype=float)},
        )
    y_pred = model.predict(X_test)
    importances = pd.Series(model.feature_importances_, index=X_test.columns).sort_values(ascending=False)
    return ForecastResult(
        model_name="XGBoost",
        y_true=y_test.to_numpy(dtype=float),
        y_pred=np.asarray(y_pred, dtype=float),
        index=X_test.index,
        extra={"feature_importance": importances},
    )


def train_sarima(
    series: pd.Series,
    order: tuple = (1, 1, 1),
    seasonal_order: tuple = (1, 1, 1, 7),
):
    """Fit a SARIMAX model. Intended for a daily-resampled subset (hourly
    SARIMA over the full multi-year series is impractically slow for an
    interactive tool), per the original notebook's scoping.
    """
    model = SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def forecast_sarima(sarima_fit, test_series: pd.Series, alpha: float = 0.05) -> ForecastResult:
    if len(test_series) == 0:
        return ForecastResult(model_name="SARIMA", y_true=np.array([]), y_pred=np.array([]), index=test_series.index)
    forecast = sarima_fit.get_forecast(steps=len(test_series))
    y_pred = forecast.predicted_mean.to_numpy(dtype=float)
    ci = forecast.conf_int(alpha=alpha)
    return ForecastResult(
        model_name="SARIMA",
        y_true=test_series.to_numpy(dtype=float),
        y_pred=y_pred,
        index=test_series.index,
        extra={
            "ci_lower": ci.iloc[:, 0].to_numpy(dtype=float),
            "ci_upper": ci.iloc[:, 1].to_numpy(dtype=float),
        },
    )
