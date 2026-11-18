import pandas as pd
import pytest

from src.generate_data import generate_dataset


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    """A deterministic synthetic physician dataset for tests."""
    return generate_dataset(n_physicians=500, seed=42)


@pytest.fixture(scope="session")
def small_df() -> pd.DataFrame:
    """A tiny (n=5) dataset for small-sample edge cases."""
    return generate_dataset(n_physicians=5, seed=7)
