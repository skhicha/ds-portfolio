import numpy as np
import pandas as pd
import pytest

from src.mmm import (
    DEFAULT_CHANNELS,
    contribution_summary,
    fit_mmm,
    simulate_budget_scenario,
    compare_scenarios,
)

R2_SANITY_THRESHOLD = 0.7


@pytest.fixture(scope="module")
def fitted_result(synthetic_df):
    return fit_mmm(synthetic_df)


def test_fit_returns_expected_channels(fitted_result):
    assert fitted_result.channels == DEFAULT_CHANNELS


def test_model_r2_exceeds_sanity_threshold(fitted_result):
    assert fitted_result.r2 > R2_SANITY_THRESHOLD, (
        f"Expected R^2 > {R2_SANITY_THRESHOLD}, got {fitted_result.r2:.3f}"
    )


def test_model_r2_is_high_quality(fitted_result):
    """The data generator is specifically constructed so a correctly specified
    model can recover R^2 above ~0.9; verify that headline claim empirically."""
    assert fitted_result.r2 > 0.9


def test_fitted_values_same_length_as_actuals(fitted_result):
    assert len(fitted_result.fitted_values) == len(fitted_result.actuals)


def test_mae_and_mape_are_finite_and_positive(fitted_result):
    assert fitted_result.mae > 0
    assert np.isfinite(fitted_result.mae)
    assert fitted_result.mape > 0
    assert np.isfinite(fitted_result.mape)


def test_channel_params_within_valid_ranges(fitted_result):
    for ch, params in fitted_result.channel_params.items():
        assert 0.0 <= params.decay < 1.0
        assert params.gamma > 0


def test_contribution_summary_sums_close_to_total_sales(fitted_result):
    summary = contribution_summary(fitted_result)
    total_from_components = summary["total_contribution"].sum()
    total_actual_sales_proxy = fitted_result.fitted_values.sum()
    # Contributions are decomposed from the *fitted* values, so they should
    # reconcile with the sum of fitted sales almost exactly.
    assert total_from_components == pytest.approx(total_actual_sales_proxy, rel=1e-6)


def test_contribution_summary_shares_sum_to_one(fitted_result):
    summary = contribution_summary(fitted_result)
    assert summary["share_of_sales"].sum() == pytest.approx(1.0, rel=1e-6)


def test_contribution_summary_includes_baseline_and_all_channels(fitted_result):
    summary = contribution_summary(fitted_result)
    components = set(summary["component"])
    assert "baseline" in components
    for ch in DEFAULT_CHANNELS:
        assert ch in components


def test_budget_scenario_returns_expected_keys(fitted_result):
    sim = simulate_budget_scenario(
        fitted_result, {"tv_spend": 15000, "digital_spend": 8000, "promotions_spend": 2000}
    )
    for key in ("predicted_weekly_sales", "predicted_trajectory", "total_weekly_budget",
                "channel_contributions", "baseline_level", "implied_roi"):
        assert key in sim
    assert sim["total_weekly_budget"] == pytest.approx(25000)
    assert sim["predicted_weekly_sales"] > 0


def test_budget_scenario_more_spend_increases_predicted_sales(fitted_result):
    low = simulate_budget_scenario(fitted_result, {"tv_spend": 2000, "digital_spend": 1000, "promotions_spend": 0})
    high = simulate_budget_scenario(fitted_result, {"tv_spend": 20000, "digital_spend": 15000, "promotions_spend": 10000})
    assert high["predicted_weekly_sales"] > low["predicted_weekly_sales"]


def test_budget_scenario_zero_spend_equals_baseline(fitted_result):
    sim = simulate_budget_scenario(fitted_result, {"tv_spend": 0, "digital_spend": 0, "promotions_spend": 0})
    assert sim["predicted_weekly_sales"] == pytest.approx(sim["baseline_level"], rel=1e-6)


def test_budget_scenario_rejects_unknown_channel(fitted_result):
    with pytest.raises(ValueError):
        simulate_budget_scenario(fitted_result, {"radio_spend": 1000})


def test_compare_scenarios_produces_one_row_per_scenario(fitted_result):
    scenarios = {
        "current": {"tv_spend": 16000, "digital_spend": 7000, "promotions_spend": 3000},
        "tv_heavy": {"tv_spend": 30000, "digital_spend": 5000, "promotions_spend": 1000},
    }
    table = compare_scenarios(fitted_result, scenarios)
    assert len(table) == 2
    assert set(table["scenario"]) == {"current", "tv_heavy"}
    assert (table["predicted_weekly_sales"] > 0).all()
