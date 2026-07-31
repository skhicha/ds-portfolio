"""
Tests for src/model.py: the scikit-learn early-warning classifier. These
checks focus on the label-construction logic (no look-ahead leakage,
correct censoring rules) and on the model actually training and producing
a sane, better-than-random holdout AUC on real generated data.
"""

import pandas as pd
import pytest

from src.model import (
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    build_training_frame,
    train_early_warning_model,
    score_portfolio,
)


def test_build_training_frame_only_includes_current_or_1_29(medium_loan_book):
    loans_df, snapshots_df = medium_loan_book
    frame = build_training_frame(snapshots_df, loans_df)
    assert len(frame) > 0

    snaps_for_merge = snapshots_df[["loan_id", "snapshot_date", "delinquency_bucket"]].copy()
    snaps_for_merge["snapshot_date"] = pd.to_datetime(snaps_for_merge["snapshot_date"])
    merged = frame.merge(snaps_for_merge, on=["loan_id", "snapshot_date"])
    assert len(merged) == len(frame)
    assert set(merged["delinquency_bucket"].unique()) <= {"Current", "1-29"}


def test_build_training_frame_has_both_classes(medium_loan_book):
    loans_df, snapshots_df = medium_loan_book
    frame = build_training_frame(snapshots_df, loans_df)
    assert set(frame[LABEL_COLUMN].unique()) == {0, 1}
    assert frame[LABEL_COLUMN].mean() > 0
    assert frame[LABEL_COLUMN].mean() < 1


def test_build_training_frame_excludes_censored_rows(medium_loan_book):
    """A row observed less than the horizon before the portfolio's as-of
    date, whose loan never defaulted and never closed, is censored (we
    don't know the true 6-month-forward outcome) and must be dropped."""
    loans_df, snapshots_df = medium_loan_book
    frame = build_training_frame(snapshots_df, loans_df)

    snap = snapshots_df.merge(loans_df, on="loan_id")
    snap["snapshot_date"] = pd.to_datetime(snap["snapshot_date"])
    last_obs = snap.groupby("loan_id")["snapshot_date"].transform("max")
    snap["last_obs_date"] = last_obs
    snap["window_end"] = snap["snapshot_date"] + pd.DateOffset(months=6)

    censored_candidates = snap[
        (snap["delinquency_bucket"].isin(["Current", "1-29"]))
        & (snap["last_obs_date"] < snap["window_end"])
        & (snap["closed_flag"] == 0)
        & (snap["default_flag"] == 0)
    ]
    included_keys = set(zip(frame["loan_id"], frame["snapshot_date"]))
    for _, row in censored_candidates.iterrows():
        assert (row["loan_id"], row["snapshot_date"]) not in included_keys


def test_train_early_warning_model_runs_and_beats_random(medium_loan_book):
    loans_df, snapshots_df = medium_loan_book
    frame = build_training_frame(snapshots_df, loans_df)
    model = train_early_warning_model(frame, test_size=0.3, random_state=0)

    assert model.n_train > 0
    assert model.n_test > 0
    assert 0.0 <= model.precision <= 1.0
    assert 0.0 <= model.recall <= 1.0
    # A logistic regression on features that are genuinely predictive of
    # default (credit score, DPD, rate, product/geography risk) should beat
    # a coin flip on holdout data.
    assert model.auc > 0.55


def test_train_early_warning_model_rejects_single_class_frame():
    frame = pd.DataFrame(
        {
            LABEL_COLUMN: [0, 0, 0, 0],
            "dpd": [0, 0, 0, 0],
            "months_on_book": [1, 2, 3, 4],
            "credit_score": [700, 700, 700, 700],
            "interest_rate": [0.1, 0.1, 0.1, 0.1],
            "utilization": [0.5, 0.5, 0.5, 0.5],
            "balance_to_principal": [0.5, 0.5, 0.5, 0.5],
            "product": ["auto_loan"] * 4,
            "geography": ["North"] * 4,
            "channel": ["branch"] * 4,
        }
    )
    with pytest.raises(ValueError):
        train_early_warning_model(frame)


def test_score_portfolio_returns_probabilities_in_unit_interval(medium_loan_book):
    loans_df, snapshots_df = medium_loan_book
    frame = build_training_frame(snapshots_df, loans_df)
    model = train_early_warning_model(frame, test_size=0.3, random_state=0)

    # score_portfolio expects a *raw* point-in-time portfolio (the shape you
    # would get from sql/latest_snapshot_per_loan.sql), not the already
    # feature-engineered training frame -- build that here the same way the
    # dashboard does: latest snapshot per loan, joined to loan attributes.
    latest = snapshots_df.sort_values("months_on_book").groupby("loan_id").tail(1)
    live = latest.merge(loans_df, on="loan_id")
    live = live[live["delinquency_bucket"].isin(["Current", "1-29"])]

    scored = score_portfolio(model, live)
    assert "early_warning_score" in scored.columns
    assert scored["early_warning_score"].between(0, 1).all()
