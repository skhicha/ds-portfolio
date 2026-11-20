import numpy as np
import pandas as pd
import pytest

from src.features import (
    engineer_features,
    handle_missing_values,
    handle_outliers_iqr,
    inverse_log_transform,
    log_transform_target,
)


class TestLogTransform:
    def test_round_trip_is_identity(self):
        y = pd.Series([10_000.0, 250_000.0, 1_000_000.0])
        y_log = log_transform_target(y)
        recovered = inverse_log_transform(y_log)
        np.testing.assert_allclose(recovered, y.to_numpy(), rtol=1e-9)

    def test_log_of_known_value(self):
        y = pd.Series([np.e * 100_000])
        y_log = log_transform_target(y)
        assert np.isclose(y_log.iloc[0], np.log(np.e * 100_000))

    def test_rejects_non_positive_values(self):
        with pytest.raises(ValueError):
            log_transform_target(pd.Series([100.0, 0.0, 50.0]))
        with pytest.raises(ValueError):
            log_transform_target(pd.Series([100.0, -5.0]))

    def test_inverse_accepts_array_or_series(self):
        arr = np.array([np.log(100.0), np.log(200.0)])
        result = inverse_log_transform(arr)
        np.testing.assert_allclose(result, [100.0, 200.0])


class TestMissingValues:
    def test_fills_numeric_with_median(self):
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0]})
        out = handle_missing_values(df)
        assert out["a"].isnull().sum() == 0
        assert out.loc[2, "a"] == df["a"].median()

    def test_fills_categorical_with_mode(self):
        df = pd.DataFrame({"cat": ["x", "x", None, "y"]})
        out = handle_missing_values(df)
        assert out["cat"].isnull().sum() == 0
        assert out.loc[2, "cat"] == "x"

    def test_never_imputes_target_column(self):
        df = pd.DataFrame({"a": [1.0, np.nan], "target": [100.0, np.nan]})
        out = handle_missing_values(df, target_col="target")
        assert out["a"].isnull().sum() == 0
        assert out["target"].isnull().sum() == 1  # left untouched

    def test_no_missing_values_is_a_no_op(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        out = handle_missing_values(df)
        pd.testing.assert_frame_equal(out, df)


class TestOutlierHandling:
    def test_clips_extreme_outlier(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 1000]})
        out = handle_outliers_iqr(df, columns=["a"])
        assert out["a"].max() < 1000
        # Row count preserved -- we clip, never drop.
        assert len(out) == len(df)

    def test_preserves_row_count_and_alignment(self):
        df = pd.DataFrame({"a": [1, 2, 3, 100], "b": ["w", "x", "y", "z"]})
        out = handle_outliers_iqr(df, columns=["a"])
        assert len(out) == 4
        assert list(out["b"]) == ["w", "x", "y", "z"]

    def test_constant_column_is_untouched(self):
        df = pd.DataFrame({"a": [5.0, 5.0, 5.0, 5.0]})
        out = handle_outliers_iqr(df, columns=["a"])
        assert (out["a"] == 5.0).all()

    def test_defaults_to_all_numeric_columns(self):
        df = pd.DataFrame({"a": [1, 2, 3, 1000], "b": ["x", "y", "z", "w"]})
        out = handle_outliers_iqr(df)
        assert out["a"].max() < 1000
        assert list(out["b"]) == ["x", "y", "z", "w"]


class TestEngineerFeatures:
    def test_full_pipeline_removes_missing_and_clips_outliers(self, synthetic_housing_df):
        out = engineer_features(synthetic_housing_df, target_col="MedHouseVal")
        assert out.drop(columns=["MedHouseVal"]).isnull().sum().sum() == 0
        # The planted 500,000 Population outlier should be clipped down.
        assert out["Population"].max() < 500_000
        assert len(out) == len(synthetic_housing_df)

    def test_target_column_values_unchanged(self, synthetic_housing_df):
        out = engineer_features(synthetic_housing_df, target_col="MedHouseVal")
        np.testing.assert_allclose(
            out["MedHouseVal"].to_numpy(), synthetic_housing_df["MedHouseVal"].to_numpy()
        )

    def test_restricting_outlier_columns(self, synthetic_housing_df):
        out = engineer_features(
            synthetic_housing_df, target_col="MedHouseVal", outlier_columns=["MedInc"]
        )
        # Population outlier untouched since it wasn't in outlier_columns.
        assert out["Population"].max() == pytest.approx(500_000)
