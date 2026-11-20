"""Inference helpers for the tuned house price model."""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd

from src.features import inverse_log_transform

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "house_price_model.pkl")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.pkl")

# Sane fallback defaults for the 8 California-housing features, used when a
# caller omits a feature. Values are roughly the dataset medians.
DEFAULT_FEATURE_VALUES = {
    "MedInc": 3.53,
    "HouseAge": 29.0,
    "AveRooms": 5.23,
    "AveBedrms": 1.05,
    "Population": 1166.0,
    "AveOccup": 2.82,
    "Latitude": 34.26,
    "Longitude": -118.49,
}

# (min, max) slider bounds covering the 1st-99th percentile of the training
# data for each feature -- wide enough to be useful, narrow enough to avoid
# the long tail of extreme outlier census tracts dominating the slider.
FEATURE_RANGES = {
    "MedInc": (0.5, 15.0),
    "HouseAge": (1.0, 52.0),
    "AveRooms": (1.0, 12.0),
    "AveBedrms": (0.5, 3.0),
    "Population": (3.0, 6000.0),
    "AveOccup": (1.0, 6.0),
    "Latitude": (32.5, 42.0),
    "Longitude": (-124.5, -114.0),
}


def load_model(model_path: str = MODEL_PATH, feature_names_path: str = FEATURE_NAMES_PATH):
    """Load the tuned model and its feature name list from disk."""
    if not os.path.exists(model_path) or not os.path.exists(feature_names_path):
        raise FileNotFoundError(
            f"Model artifacts not found at {model_path} / {feature_names_path}. "
            "Run `python -m src.train` first."
        )
    model = joblib.load(model_path)
    feature_names = joblib.load(feature_names_path)
    return model, feature_names


def build_input_frame(features: dict, feature_names: list[str]) -> pd.DataFrame:
    """
    Build a single-row DataFrame in the model's expected column order from a
    (possibly partial) dict of feature values. Missing features are filled
    with a sane default rather than raising. Values are coerced to float;
    non-numeric/garbage values fall back to the default too, so this never
    crashes on bad input.
    """
    row = {}
    for name in feature_names:
        if name in features:
            try:
                row[name] = float(features[name])
            except (TypeError, ValueError):
                row[name] = DEFAULT_FEATURE_VALUES.get(name, 0.0)
        else:
            row[name] = DEFAULT_FEATURE_VALUES.get(name, 0.0)
    return pd.DataFrame([row], columns=feature_names)


def predict_price(model, feature_names: list[str], features: dict) -> dict:
    """
    Predict a house price (in dollars) from a dict of feature values.
    Returns a dict with the predicted price, the input frame actually used,
    and feature importances (if the model exposes them) aligned to
    feature_names.
    """
    input_df = build_input_frame(features, feature_names)
    log_pred = model.predict(input_df)[0]
    price = float(inverse_log_transform(log_pred))

    importances = None
    if hasattr(model, "feature_importances_"):
        importances = dict(zip(feature_names, np.asarray(model.feature_importances_, dtype=float).tolist()))

    return {
        "predicted_price": price,
        "input": input_df.iloc[0].to_dict(),
        "feature_importances": importances,
    }
