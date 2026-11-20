import os

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.predict import build_input_frame, load_model, predict_price


@pytest.fixture()
def toy_model_and_features(tiny_feature_names, synthetic_housing_df):
    """A real (tiny, fast) fitted RandomForestRegressor on the log target,
    so predict_price exercises actual model.predict + feature_importances_."""
    from src.features import engineer_features, log_transform_target

    df = engineer_features(synthetic_housing_df, target_col="MedHouseVal")
    X = df[tiny_feature_names]
    y_log = log_transform_target(df["MedHouseVal"])
    model = RandomForestRegressor(n_estimators=20, random_state=0)
    model.fit(X, y_log)
    return model, tiny_feature_names


class TestBuildInputFrame:
    def test_complete_input_preserves_values(self, tiny_feature_names):
        features = {name: float(i + 1) for i, name in enumerate(tiny_feature_names)}
        df = build_input_frame(features, tiny_feature_names)
        assert list(df.columns) == tiny_feature_names
        assert len(df) == 1
        for name in tiny_feature_names:
            assert df.iloc[0][name] == features[name]

    def test_missing_features_get_sane_defaults(self, tiny_feature_names):
        partial = {"MedInc": 5.0}
        df = build_input_frame(partial, tiny_feature_names)
        assert df.iloc[0]["MedInc"] == 5.0
        # Every other feature got filled, not left NaN.
        assert df.isnull().sum().sum() == 0

    def test_empty_input_does_not_crash(self, tiny_feature_names):
        df = build_input_frame({}, tiny_feature_names)
        assert len(df) == 1
        assert df.isnull().sum().sum() == 0

    def test_garbage_value_falls_back_to_default_instead_of_crashing(self, tiny_feature_names):
        df = build_input_frame({"MedInc": "not-a-number"}, tiny_feature_names)
        assert np.isfinite(df.iloc[0]["MedInc"])

    def test_out_of_range_value_passes_through(self, tiny_feature_names):
        # predict_price never clips -- out-of-range inputs are allowed through
        # to the model rather than silently altered.
        df = build_input_frame({"Latitude": 9999.0}, tiny_feature_names)
        assert df.iloc[0]["Latitude"] == 9999.0


class TestPredictPrice:
    def test_returns_sane_shape_and_positive_price(self, toy_model_and_features):
        model, feature_names = toy_model_and_features
        result = predict_price(model, feature_names, {"MedInc": 4.0, "HouseAge": 20.0})
        assert "predicted_price" in result
        assert result["predicted_price"] > 0
        assert isinstance(result["predicted_price"], float)
        assert set(result["input"].keys()) == set(feature_names)

    def test_feature_importances_present_and_aligned(self, toy_model_and_features):
        model, feature_names = toy_model_and_features
        result = predict_price(model, feature_names, {})
        assert result["feature_importances"] is not None
        assert set(result["feature_importances"].keys()) == set(feature_names)
        assert all(v >= 0 for v in result["feature_importances"].values())

    def test_model_without_feature_importances_returns_none(self, toy_model_and_features):
        from sklearn.linear_model import LinearRegression

        _, feature_names = toy_model_and_features
        X = pd.DataFrame(
            np.random.RandomState(0).rand(20, len(feature_names)), columns=feature_names
        )
        y = np.random.RandomState(0).rand(20)
        linreg = LinearRegression().fit(X, y)
        result = predict_price(linreg, feature_names, {})
        assert result["feature_importances"] is None

    def test_handles_fully_missing_input(self, toy_model_and_features):
        model, feature_names = toy_model_and_features
        result = predict_price(model, feature_names, {})
        assert np.isfinite(result["predicted_price"])


class TestLoadModel:
    def test_raises_clear_error_when_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_model(
                model_path=str(tmp_path / "nope.pkl"),
                feature_names_path=str(tmp_path / "nope2.pkl"),
            )

    def test_loads_real_artifacts_if_present(self):
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(here, "models", "house_price_model.pkl")
        features_path = os.path.join(here, "models", "feature_names.pkl")
        if not (os.path.exists(model_path) and os.path.exists(features_path)):
            pytest.skip("Trained artifacts not present; run `python -m src.train` first.")
        model, feature_names = load_model(model_path, features_path)
        assert hasattr(model, "predict")
        assert isinstance(feature_names, list) and len(feature_names) > 0
