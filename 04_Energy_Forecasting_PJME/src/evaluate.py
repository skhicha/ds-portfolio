"""MAPE/RMSE scoring and residual diagnostics, shared by app.py and tests."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def mape(y_true, y_pred) -> float:
    """Mean Absolute Percentage Error, safe against zero/near-zero actuals.

    Rows where the actual value is exactly 0 are excluded from the average
    (dividing by zero is undefined, not "0% error"). If every actual value
    is 0, returns NaN rather than raising.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return float("nan")
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    pct_errors = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    return float(np.mean(pct_errors) * 100.0)


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


@dataclass
class ResidualDiagnostics:
    residuals: np.ndarray
    std: float
    ci_lower: np.ndarray
    ci_upper: np.ndarray


def residual_diagnostics(y_true, y_pred, z: float = 1.96) -> ResidualDiagnostics:
    """Residuals plus a symmetric confidence band around the forecast,
    derived from the residual standard deviation (y_pred +/- z * std).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred
    std = float(np.std(residuals)) if residuals.size else 0.0
    ci_lower = y_pred - z * std
    ci_upper = y_pred + z * std
    return ResidualDiagnostics(residuals=residuals, std=std, ci_lower=ci_lower, ci_upper=ci_upper)
