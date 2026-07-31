import os
import sys

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.generate_data import generate  # noqa: E402


@pytest.fixture(scope="session")
def small_dataset():
    """A small, fast, reproducible dataset for unit tests."""
    reps_df, territories_df = generate(num_reps=12, num_territories=8, seed=7)
    return reps_df, territories_df


@pytest.fixture(scope="session")
def full_dataset():
    """A dataset matching the project's intended default scale (20-40 reps)."""
    reps_df, territories_df = generate(num_reps=28, num_territories=20, seed=42)
    return reps_df, territories_df
