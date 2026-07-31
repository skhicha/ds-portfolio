"""
Shared pytest fixtures. Adds the project root to sys.path so tests can
`import src...` regardless of how pytest is invoked, and provides a small,
fast, deterministic synthetic loan book for tests that need real (but
small) generated data rather than hand-built fixtures.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

from src.etl import generate_loan_book  # noqa: E402
from datetime import date  # noqa: E402


@pytest.fixture(scope="session")
def small_loan_book():
    """A small (a few hundred loans), fast, deterministic synthetic book
    shared across tests in this session, generated with the same code path
    used to build the real database."""
    loans_df, snapshots_df = generate_loan_book(n_loans=400, as_of_date=date(2026, 6, 30), seed=7)
    return loans_df, snapshots_df


@pytest.fixture(scope="session")
def medium_loan_book():
    """A larger synthetic book (enough defaulted loans across the full
    5-year window to give the early-warning classifier both classes with
    reasonable support) shared across the ML tests in this session."""
    loans_df, snapshots_df = generate_loan_book(n_loans=1800, as_of_date=date(2026, 6, 30), seed=11)
    return loans_df, snapshots_df
