"""
Tests for src/etl.py: schema creation, row counts, reproducibility and
basic data-quality invariants of the synthetic loan-book generator.
"""

import sqlite3
from datetime import date

import pandas as pd

from src.etl import build_database, generate_loan_book, _quarter_label


def test_generate_loan_book_row_counts():
    loans_df, snapshots_df = generate_loan_book(n_loans=250, as_of_date=date(2026, 6, 30), seed=1)
    assert len(loans_df) == 250
    # Every loan has at least one snapshot (the origination-month snapshot).
    assert snapshots_df["loan_id"].nunique() == 250
    assert len(snapshots_df) >= 250


def test_generate_loan_book_is_reproducible_with_same_seed():
    loans_a, snaps_a = generate_loan_book(n_loans=150, as_of_date=date(2026, 6, 30), seed=99)
    loans_b, snaps_b = generate_loan_book(n_loans=150, as_of_date=date(2026, 6, 30), seed=99)
    pd.testing.assert_frame_equal(loans_a, loans_b)
    pd.testing.assert_frame_equal(snaps_a, snaps_b)


def test_generate_loan_book_different_seeds_differ():
    loans_a, _ = generate_loan_book(n_loans=150, as_of_date=date(2026, 6, 30), seed=1)
    loans_b, _ = generate_loan_book(n_loans=150, as_of_date=date(2026, 6, 30), seed=2)
    assert not loans_a["credit_score"].equals(loans_b["credit_score"])


def test_loans_schema_and_domains(small_loan_book):
    loans_df, _ = small_loan_book
    expected_cols = {
        "loan_id", "product", "geography", "origination_date", "vintage_quarter",
        "term_months", "principal", "interest_rate", "credit_score",
        "borrower_income", "channel", "closed_flag", "default_flag", "default_date",
    }
    assert expected_cols.issubset(set(loans_df.columns))
    assert loans_df["loan_id"].is_unique
    assert set(loans_df["product"].unique()) <= {"personal_loan", "auto_loan", "mortgage", "credit_card"}
    assert set(loans_df["geography"].unique()) <= {"North", "South", "East", "West", "Central"}
    assert loans_df["credit_score"].between(300, 850).all()
    assert loans_df["principal"].gt(0).all()
    assert loans_df["interest_rate"].between(0, 1).all()
    assert set(loans_df["default_flag"].unique()) <= {0, 1}
    assert set(loans_df["closed_flag"].unique()) <= {0, 1}
    # A loan cannot simultaneously be closed (paid off) and defaulted.
    assert not ((loans_df["closed_flag"] == 1) & (loans_df["default_flag"] == 1)).any()
    # default_date is populated iff default_flag is set.
    assert (loans_df.loc[loans_df["default_flag"] == 1, "default_date"].notna()).all()
    assert (loans_df.loc[loans_df["default_flag"] == 0, "default_date"].isna()).all()


def test_snapshots_schema_and_domains(small_loan_book):
    _, snapshots_df = small_loan_book
    expected_cols = {
        "loan_id", "snapshot_date", "months_on_book", "outstanding_balance",
        "dpd", "delinquency_bucket", "stage",
    }
    assert expected_cols.issubset(set(snapshots_df.columns))
    assert snapshots_df["outstanding_balance"].ge(0).all()
    assert snapshots_df["dpd"].ge(0).all()
    assert set(snapshots_df["delinquency_bucket"].unique()) <= {"Current", "1-29", "30-59", "60-89", "90+"}
    assert set(snapshots_df["stage"].unique()) <= {1, 2, 3}
    assert snapshots_df["months_on_book"].ge(0).all()


def test_snapshots_are_monotonic_in_months_on_book_per_loan(small_loan_book):
    """Within a single loan, months_on_book must strictly increase snapshot
    to snapshot (no duplicated or reordered months)."""
    _, snapshots_df = small_loan_book
    for _, group in snapshots_df.groupby("loan_id"):
        months = group.sort_values("snapshot_date")["months_on_book"].to_numpy()
        assert (months[1:] > months[:-1]).all()


def test_loan_stops_simulating_after_default(small_loan_book):
    """Once a loan reaches the 90+ bucket, no further snapshots should be
    recorded for it (absorbing state / charge-off)."""
    loans_df, snapshots_df = small_loan_book
    defaulted_ids = loans_df.loc[loans_df["default_flag"] == 1, "loan_id"]
    for loan_id in defaulted_ids:
        loan_snaps = snapshots_df[snapshots_df["loan_id"] == loan_id].sort_values("months_on_book")
        assert loan_snaps.iloc[-1]["delinquency_bucket"] == "90+"


def test_quarter_label():
    assert _quarter_label(date(2023, 1, 15)) == "2023Q1"
    assert _quarter_label(date(2023, 4, 1)) == "2023Q2"
    assert _quarter_label(date(2023, 9, 30)) == "2023Q3"
    assert _quarter_label(date(2023, 12, 31)) == "2023Q4"


def test_build_database_creates_expected_tables_and_counts(tmp_path):
    db_path = tmp_path / "test_loan_book.db"
    summary = build_database(db_path=db_path, n_loans=120, as_of_date=date(2026, 6, 30), seed=3)

    assert summary["n_loans"] == 120
    assert db_path.exists()

    conn = sqlite3.connect(str(db_path))
    try:
        loans_count = conn.execute("SELECT COUNT(*) FROM loans").fetchone()[0]
        snapshots_count = conn.execute("SELECT COUNT(*) FROM loan_snapshots").fetchone()[0]
        assert loans_count == 120
        assert snapshots_count == summary["n_snapshots"]

        # The view created alongside the schema should be queryable.
        view_rows = conn.execute("SELECT COUNT(*) FROM vw_delinquency_bucket_summary").fetchone()[0]
        assert view_rows > 0
    finally:
        conn.close()


def test_build_database_is_idempotent_on_rerun(tmp_path):
    db_path = tmp_path / "rerun.db"
    build_database(db_path=db_path, n_loans=80, as_of_date=date(2026, 6, 30), seed=5)
    summary_2 = build_database(db_path=db_path, n_loans=80, as_of_date=date(2026, 6, 30), seed=5)
    conn = sqlite3.connect(str(db_path))
    try:
        loans_count = conn.execute("SELECT COUNT(*) FROM loans").fetchone()[0]
        # Rerunning should drop and recreate, not append/duplicate.
        assert loans_count == 80 == summary_2["n_loans"]
    finally:
        conn.close()
