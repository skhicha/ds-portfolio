import numpy as np

from src.evaluate import mape, residual_diagnostics, rmse


def test_mape_basic_known_value():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 300.0])
    # errors: 10/100=10%, 10/200=5%, 0/300=0% -> mean 5%
    assert np.isclose(mape(y_true, y_pred), 5.0)


def test_mape_zero_actuals_are_excluded_not_crashing():
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert np.isnan(mape(y_true, y_pred))


def test_mape_mixed_zero_and_nonzero_actuals():
    y_true = np.array([0.0, 100.0])
    y_pred = np.array([5.0, 90.0])
    # only the second row counts: |100-90|/100 = 10%
    assert np.isclose(mape(y_true, y_pred), 10.0)


def test_mape_empty_input_returns_nan():
    assert np.isnan(mape(np.array([]), np.array([])))


def test_mape_perfect_predictions_is_zero():
    y = np.array([10.0, 20.0, 30.0])
    assert np.isclose(mape(y, y), 0.0)


def test_rmse_known_value():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([3.0, 4.0])
    # sqrt(mean(9, 16)) = sqrt(12.5)
    assert np.isclose(rmse(y_true, y_pred), np.sqrt(12.5))


def test_rmse_empty_input_returns_nan():
    assert np.isnan(rmse(np.array([]), np.array([])))


def test_residual_diagnostics_shapes_and_band_direction():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([9.0, 22.0, 29.0, 41.0])
    diag = residual_diagnostics(y_true, y_pred, z=1.96)
    assert diag.residuals.shape == y_true.shape
    assert diag.ci_lower.shape == y_true.shape
    assert diag.ci_upper.shape == y_true.shape
    assert np.all(diag.ci_upper >= diag.ci_lower)
    assert diag.std >= 0


def test_residual_diagnostics_zero_error_gives_zero_width_band():
    y = np.array([1.0, 2.0, 3.0])
    diag = residual_diagnostics(y, y)
    assert diag.std == 0.0
    assert np.allclose(diag.ci_lower, y)
    assert np.allclose(diag.ci_upper, y)
