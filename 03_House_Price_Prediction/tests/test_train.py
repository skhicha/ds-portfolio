import numpy as np
import pandas as pd
import pytest

from src.train import BASELINE_MODELS, evaluate_log_model, prepare_data, run_training


@pytest.fixture()
def quick_df(synthetic_housing_df: pd.DataFrame) -> pd.DataFrame:
    """A slightly larger synthetic frame so cross-validated search has
    enough rows per fold, but still tiny/fast relative to the real dataset."""
    rng = np.random.RandomState(0)
    base = synthetic_housing_df
    frames = [base.sample(n=len(base), replace=True, random_state=i) for i in range(3)]
    df = pd.concat(frames, ignore_index=True)
    # add a little independent noise so rows aren't exact duplicates
    df["MedInc"] = df["MedInc"] + rng.normal(0, 0.01, len(df))
    return df


class TestPrepareData:
    def test_returns_log_target_and_matching_feature_names(self, synthetic_housing_df):
        X, y_log, feature_names = prepare_data(synthetic_housing_df, target_col="MedHouseVal")
        assert "MedHouseVal" not in X.columns
        assert list(X.columns) == feature_names
        assert len(X) == len(y_log) == len(synthetic_housing_df)
        # Target really is log-scale: exponentiating should look like the original.
        np.testing.assert_allclose(
            np.exp(y_log.to_numpy()), synthetic_housing_df["MedHouseVal"].to_numpy(), rtol=1e-6
        )

    def test_handles_missing_values_before_split(self, synthetic_housing_df):
        X, _, _ = prepare_data(synthetic_housing_df, target_col="MedHouseVal")
        assert X.isnull().sum().sum() == 0


class TestBaselineModels:
    def test_all_four_models_present(self):
        assert set(BASELINE_MODELS.keys()) == {
            "Linear Regression",
            "Ridge Regression",
            "Random Forest",
            "XGBoost",
        }

    def test_each_model_fits_and_evaluates(self, synthetic_housing_df):
        X, y_log, _ = prepare_data(synthetic_housing_df, target_col="MedHouseVal")
        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y_log.iloc[:split], y_log.iloc[split:]

        for name, factory in BASELINE_MODELS.items():
            model = factory()
            model.fit(X_train, y_train)
            rmse, r2, mae = evaluate_log_model(model, X_test, y_test)
            assert rmse > 0
            assert mae > 0
            assert np.isfinite(r2)


class TestRunTraining:
    def test_full_pipeline_runs_and_produces_sane_metrics(self, quick_df):
        report, best_rf = run_training(quick_df, target_col="MedHouseVal")

        assert len(report.baseline_results) == 4
        for r in report.baseline_results:
            assert r.rmse > 0
            assert r.mae > 0

        assert report.best_baseline_name in {m.name for m in report.baseline_results}
        assert report.tuned_rmse > 0
        assert report.tuned_mae > 0
        assert isinstance(report.best_params, dict) and len(report.best_params) > 0
        assert hasattr(best_rf, "predict")
        assert hasattr(best_rf, "feature_importances_")

        # A model trained on data this informative (MedInc strongly drives
        # the synthetic target) should beat a RMSE-of-everything-is-the-mean
        # baseline by a wide margin -- a loose but meaningful sanity bound.
        naive_rmse = float(np.std(quick_df["MedHouseVal"]))
        assert report.tuned_rmse < naive_rmse
