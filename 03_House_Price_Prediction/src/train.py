"""
Model training/comparison for the house price prediction pipeline.

Trains Linear Regression, Ridge, Random Forest, and XGBoost on the
log-transformed target, evaluates all four on the original (dollar) scale,
then runs a RandomizedSearchCV hyperparameter search on Random Forest (the
notebook's best baseline) and saves the tuned model + feature names.

Run as a script:
    python -m src.train
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBRegressor

from src.data import load_dataset
from src.features import engineer_features, inverse_log_transform, log_transform_target

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "house_price_model.pkl")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
RANDOM_STATE = 42

BASELINE_MODELS = {
    "Linear Regression": lambda: LinearRegression(),
    "Ridge Regression": lambda: Ridge(alpha=1.0),
    "Random Forest": lambda: RandomForestRegressor(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
    ),
    "XGBoost": lambda: XGBRegressor(
        n_estimators=100, learning_rate=0.1, random_state=RANDOM_STATE, verbosity=0
    ),
}

RF_PARAM_DISTRIBUTIONS = {
    "n_estimators": [50, 100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.5],
}


@dataclass
class ModelResult:
    name: str
    rmse: float
    r2: float
    mae: float
    train_seconds: float


@dataclass
class TrainingReport:
    baseline_results: list[ModelResult]
    best_baseline_name: str
    tuned_rmse: float
    tuned_r2: float
    tuned_mae: float
    best_params: dict
    feature_names: list[str] = field(default_factory=list)


def evaluate_log_model(model, X_test: pd.DataFrame, y_test_log: pd.Series) -> tuple[float, float, float]:
    """Evaluate a model trained on a log target, converting back to the
    original (dollar) scale before computing metrics."""
    y_pred_log = model.predict(X_test)
    y_pred = inverse_log_transform(y_pred_log)
    y_true = inverse_log_transform(y_test_log)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    return rmse, r2, mae


def prepare_data(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    df_processed = engineer_features(df, target_col=target_col)
    X = df_processed.drop(columns=[target_col])
    y_log = log_transform_target(df_processed[target_col])
    return X, y_log, X.columns.tolist()


def run_training(df: pd.DataFrame | None = None, target_col: str | None = None) -> TrainingReport:
    """Run the full train/compare/tune pipeline and return a TrainingReport.
    Does not touch disk -- see `main()` for persisting the tuned model."""
    if df is None:
        df, target_col = load_dataset()
    assert target_col is not None

    X, y_log, feature_names = prepare_data(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=RANDOM_STATE
    )

    baseline_results: list[ModelResult] = []
    fitted = {}
    for name, factory in BASELINE_MODELS.items():
        model = factory()
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start
        rmse, r2, mae = evaluate_log_model(model, X_test, y_test)
        baseline_results.append(ModelResult(name=name, rmse=rmse, r2=r2, mae=mae, train_seconds=elapsed))
        fitted[name] = model

    best_baseline_name = min(baseline_results, key=lambda r: r.rmse).name

    rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    search = RandomizedSearchCV(
        rf,
        RF_PARAM_DISTRIBUTIONS,
        n_iter=20,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    search.fit(X_train, y_train)
    best_rf = search.best_estimator_
    tuned_rmse, tuned_r2, tuned_mae = evaluate_log_model(best_rf, X_test, y_test)

    return TrainingReport(
        baseline_results=baseline_results,
        best_baseline_name=best_baseline_name,
        tuned_rmse=tuned_rmse,
        tuned_r2=tuned_r2,
        tuned_mae=tuned_mae,
        best_params=search.best_params_,
        feature_names=feature_names,
    ), best_rf


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)

    df, target_col = load_dataset()
    print(f"Loaded dataset: shape={df.shape}, target={target_col}")

    report, best_rf = run_training(df, target_col)

    print("\n=== Baseline model comparison (log target, metrics on $ scale) ===")
    for r in report.baseline_results:
        print(f"  {r.name:<18} RMSE=${r.rmse:,.0f}  R2={r.r2:.4f}  MAE=${r.mae:,.0f}  ({r.train_seconds:.2f}s)")

    print(f"\nBest baseline: {report.best_baseline_name}")
    print("\n=== RandomizedSearchCV tuning (Random Forest, 20 iters, 5-fold CV) ===")
    print(f"  Best params: {report.best_params}")
    print(f"  Tuned RMSE=${report.tuned_rmse:,.0f}  R2={report.tuned_r2:.4f}  MAE=${report.tuned_mae:,.0f}")

    joblib.dump(best_rf, MODEL_PATH)
    joblib.dump(report.feature_names, FEATURE_NAMES_PATH)

    metrics = {
        "baseline_results": [r.__dict__ for r in report.baseline_results],
        "best_baseline_name": report.best_baseline_name,
        "tuned_rmse": report.tuned_rmse,
        "tuned_r2": report.tuned_r2,
        "tuned_mae": report.tuned_mae,
        "best_params": report.best_params,
        "feature_names": report.feature_names,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved tuned model to {MODEL_PATH}")
    print(f"Saved feature names to {FEATURE_NAMES_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
