import numpy as np

from src.generate_data import generate


def test_shapes_and_ranges():
    reps_df, territories_df = generate(num_reps=20, num_territories=15, seed=1)
    assert len(reps_df) == 20
    assert len(territories_df) == 15

    assert (reps_df["max_capacity_fte"] > 0).all()
    assert (reps_df["annual_cost"] > 0).all()
    assert (reps_df["productivity_multiplier"] > 0).all()

    assert (territories_df["required_fte"] > 0).all()
    assert (territories_df["potential_revenue"] > 0).all()
    assert territories_df["min_coverage_pct"].between(0, 1).all()


def test_current_territory_ids_are_valid():
    reps_df, territories_df = generate(num_reps=25, num_territories=18, seed=3)
    valid_ids = set(territories_df["territory_id"])
    assert set(reps_df["current_territory_id"]).issubset(valid_ids)


def test_reproducible_with_same_seed():
    reps1, terr1 = generate(num_reps=10, num_territories=6, seed=99)
    reps2, terr2 = generate(num_reps=10, num_territories=6, seed=99)
    assert reps1.equals(reps2)
    assert terr1.equals(terr2)


def test_default_scale_is_in_spec_range():
    reps_df, territories_df = generate(num_reps=28, num_territories=20, seed=42)
    assert 20 <= len(reps_df) <= 40
    assert 15 <= len(territories_df) <= 25
