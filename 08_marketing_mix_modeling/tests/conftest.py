import pandas as pd
import pytest

from src.generate_data import generate_dataset


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    """A deterministic synthetic weekly sales dataset for tests."""
    return generate_dataset(n_weeks=156, seed=42)
