import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def synthetic_housing_df() -> pd.DataFrame:
    """A small, deterministic synthetic dataset shaped like the real one
    (8 numeric features + a strictly-positive target), with a couple of
    planted missing values and an obvious outlier -- fast enough to use in
    every test without downloading the real California Housing data."""
    rng = np.random.RandomState(42)
    n = 200
    df = pd.DataFrame(
        {
            "MedInc": rng.uniform(1, 10, n),
            "HouseAge": rng.uniform(1, 50, n),
            "AveRooms": rng.uniform(2, 8, n),
            "AveBedrms": rng.uniform(0.8, 1.5, n),
            "Population": rng.uniform(100, 3000, n),
            "AveOccup": rng.uniform(1, 5, n),
            "Latitude": rng.uniform(33, 41, n),
            "Longitude": rng.uniform(-124, -115, n),
        }
    )
    # Target: a noisy linear-ish combination, guaranteed positive.
    df["MedHouseVal"] = (
        20_000
        + df["MedInc"] * 30_000
        - df["HouseAge"] * 200
        + rng.normal(0, 5_000, n)
    ).clip(lower=10_000)

    # Plant missing values in a numeric column.
    df.loc[df.index[:5], "AveRooms"] = np.nan

    # Plant an obvious outlier.
    df.loc[df.index[0], "Population"] = 500_000

    return df


@pytest.fixture()
def tiny_feature_names() -> list[str]:
    return ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
