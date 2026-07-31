"""
model.py
A simple, honest scikit-learn early-warning model.

Goal: for loans that are *not yet* clearly bad (Current or 1-29 DPD),
predict the probability that they will roll into default (90+ DPD) within
the next EARLY_WARNING_HORIZON_MONTHS months, using only information that
would genuinely be available at scoring time (loan attributes + current
delinquency state), and label them using genuine forward-looking outcomes
observed in the snapshot panel -- not a hardcoded score.

Labelling logic (survival-style, to avoid look-ahead leakage on censored
loans):
  * label = 1  if the loan's recorded default_date falls within the
               horizon window after the observation snapshot.
  * label = 0  if the loan was observed for at least `horizon` more months
               after the snapshot without defaulting (a genuine negative),
               OR the loan closed (paid off) within the window without
               defaulting.
  * row dropped (censored, unknown outcome) if the panel simply ends
               (as_of_date reached) before the horizon has elapsed and the
               loan hasn't defaulted -- we don't know what would have
               happened next, so it is excluded from training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

EARLY_WARNING_HORIZON_MONTHS = 6

NUMERIC_FEATURES = [
    "dpd",
    "months_on_book",
    "credit_score",
    "interest_rate",
    "utilization",
    "balance_to_principal",
]
CATEGORICAL_FEATURES = ["product", "geography", "channel"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
LABEL_COLUMN = "early_warning_label"


@dataclass
class EarlyWarningModel:
    pipeline: Pipeline
    auc: float
    precision: float
    recall: float
    n_train: int
    n_test: int
    positive_rate: float


def build_training_frame(
    snapshots: pd.DataFrame,
    loans: pd.DataFrame,
    horizon_months: int = EARLY_WARNING_HORIZON_MONTHS,
) -> pd.DataFrame:
    """
    Construct the labelled training frame described above from the raw
    loan_snapshots + loans tables.
    """
    snap = snapshots.merge(loans, on="loan_id", suffixes=("", "_loan"))
    snap["snapshot_date"] = pd.to_datetime(snap["snapshot_date"])
    snap["default_date"] = pd.to_datetime(snap["default_date"])

    # Only score loans that are not already clearly bad -- the point of an
    # *early* warning system is to flag risk before it is obvious.
    eligible = snap[snap["delinquency_bucket"].isin(["Current", "1-29"])].copy()

    last_obs_date = snap.groupby("loan_id")["snapshot_date"].transform("max")
    eligible["last_obs_date"] = last_obs_date
    eligible["window_end"] = eligible["snapshot_date"] + pd.DateOffset(months=horizon_months)

    will_default_in_window = (
        eligible["default_date"].notna()
        & (eligible["default_date"] > eligible["snapshot_date"])
        & (eligible["default_date"] <= eligible["window_end"])
    )

    fully_observed_negative = (eligible["last_obs_date"] >= eligible["window_end"]) & (
        ~will_default_in_window
    )
    # Loans that closed (paid off) strictly within the window without ever
    # defaulting are legitimate negatives too, even if last_obs_date < window_end.
    closed_negative = (eligible["closed_flag"] == 1) & (~will_default_in_window)

    keep_mask = will_default_in_window | fully_observed_negative | closed_negative
    frame = eligible[keep_mask].copy()
    frame[LABEL_COLUMN] = will_default_in_window[keep_mask].astype(int)

    frame["utilization"] = frame["outstanding_balance"] / frame["principal"].replace(0, np.nan)
    frame["balance_to_principal"] = frame["utilization"]
    frame[NUMERIC_FEATURES] = frame[NUMERIC_FEATURES].fillna(0.0)

    return frame[["loan_id", "snapshot_date", LABEL_COLUMN] + FEATURE_COLUMNS]


def train_early_warning_model(
    training_frame: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
) -> EarlyWarningModel:
    """Fit a logistic-regression early-warning classifier and report holdout metrics."""
    if training_frame[LABEL_COLUMN].nunique() < 2:
        raise ValueError(
            "Training frame must contain both classes (defaulted and non-defaulted "
            "examples) to fit a classifier."
        )

    x = training_frame[FEATURE_COLUMNS]
    y = training_frame[LABEL_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "classifier",
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)

    proba_test = pipeline.predict_proba(x_test)[:, 1]
    pred_test = (proba_test >= 0.5).astype(int)

    auc = roc_auc_score(y_test, proba_test) if y_test.nunique() > 1 else float("nan")
    precision = precision_score(y_test, pred_test, zero_division=0)
    recall = recall_score(y_test, pred_test, zero_division=0)

    return EarlyWarningModel(
        pipeline=pipeline,
        auc=float(auc),
        precision=float(precision),
        recall=float(recall),
        n_train=len(x_train),
        n_test=len(x_test),
        positive_rate=float(y.mean()),
    )


def score_portfolio(model: EarlyWarningModel, current_portfolio: pd.DataFrame) -> pd.DataFrame:
    """
    Score a point-in-time portfolio (one row per live loan, with the same
    feature columns as the training frame) with the fitted pipeline.
    Returns the input frame with an added `early_warning_score` column
    (probability of rolling to default within the horizon).
    """
    frame = current_portfolio.copy()
    frame["utilization"] = frame["outstanding_balance"] / frame["principal"].replace(0, np.nan)
    frame["balance_to_principal"] = frame["utilization"]
    frame[NUMERIC_FEATURES] = frame[NUMERIC_FEATURES].fillna(0.0)
    frame["early_warning_score"] = model.pipeline.predict_proba(frame[FEATURE_COLUMNS])[:, 1]
    return frame
