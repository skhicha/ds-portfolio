import numpy as np
import pytest

from src.optimize import solve_allocation, evaluate_allocation
from src.baseline import status_quo_allocation, round_robin_allocation

TOL = 1e-4


def test_solver_converges(small_dataset):
    reps_df, territories_df = small_dataset
    result = solve_allocation(reps_df, territories_df)
    assert result.success is True
    assert result.status == "optimal"


def test_solver_converges_full_scale(full_dataset):
    reps_df, territories_df = full_dataset
    result = solve_allocation(reps_df, territories_df)
    assert result.success is True


def test_rep_capacity_not_exceeded(full_dataset):
    reps_df, territories_df = full_dataset
    result = solve_allocation(reps_df, territories_df)
    assert result.success

    allocated_per_rep = result.x.sum(axis=1)
    capacity = reps_df["max_capacity_fte"].to_numpy(dtype=float)
    assert np.all(allocated_per_rep <= capacity + TOL), (
        f"Rep capacity exceeded: max overage = {(allocated_per_rep - capacity).max()}"
    )


def test_territory_min_coverage_met(full_dataset):
    reps_df, territories_df = full_dataset
    result = solve_allocation(reps_df, territories_df)
    assert result.success

    min_coverage = territories_df["min_coverage_pct"].to_numpy(dtype=float)
    assert np.all(result.y >= min_coverage - TOL), (
        f"Min coverage violated: min slack = {(result.y - min_coverage).min()}"
    )
    # y should never exceed 1.0 (100% revenue capture cap)
    assert np.all(result.y <= 1.0 + TOL)


def test_coverage_linkage_holds(full_dataset):
    """y[t] must not exceed the actual supplied FTE / required FTE for that
    territory -- i.e. the solver can't claim revenue coverage it didn't earn."""
    reps_df, territories_df = full_dataset
    result = solve_allocation(reps_df, territories_df)
    assert result.success

    productivity = reps_df["productivity_multiplier"].to_numpy(dtype=float)
    required_fte = territories_df["required_fte"].to_numpy(dtype=float)
    supplied = (productivity[:, None] * result.x).sum(axis=0)

    assert np.all(required_fte * result.y <= supplied + TOL)


def test_objective_matches_manual_recomputation(full_dataset):
    """The solver-reported objective value must match an independent manual
    recomputation of net revenue (revenue - cost) from the returned x, y."""
    reps_df, territories_df = full_dataset
    result = solve_allocation(reps_df, territories_df)
    assert result.success

    revenue = territories_df["potential_revenue"].to_numpy(dtype=float)
    cost = reps_df["annual_cost"].to_numpy(dtype=float)

    manual_total_revenue = float((revenue * result.y).sum())
    manual_total_cost = float((cost[:, None] * result.x).sum())
    manual_net_revenue = manual_total_revenue - manual_total_cost

    assert manual_total_revenue == pytest.approx(result.total_revenue, rel=1e-6)
    assert manual_total_cost == pytest.approx(result.total_cost, rel=1e-6)
    assert manual_net_revenue == pytest.approx(result.objective_value, rel=1e-6)

    # Cross-check against evaluate_allocation's independent computation path too.
    metrics = evaluate_allocation(result.x, reps_df, territories_df)
    # achieved_revenue in evaluate_allocation is derived from coverage computed
    # directly from x (not y), so it should match the LP's y-based revenue
    # (the LP pushes y to exactly match supplied/required at the optimum,
    # except where capped by the y<=1 bound with slack capacity).
    assert metrics["total_revenue"] <= manual_total_revenue + TOL


def test_no_negative_allocations(full_dataset):
    reps_df, territories_df = full_dataset
    result = solve_allocation(reps_df, territories_df)
    assert result.success
    assert np.all(result.x >= -TOL)
    assert np.all(result.y >= -TOL)


def test_optimizer_beats_or_matches_baseline_net_revenue(full_dataset):
    """The whole point of the LP: it should never do worse than a naive
    manual baseline, since the baseline allocation is itself LP-feasible
    only if it happens to satisfy min-coverage (often it won't) -- but even
    scored on the same KPI basis, the optimizer's net revenue should be >= baseline."""
    reps_df, territories_df = full_dataset
    result = solve_allocation(reps_df, territories_df)
    assert result.success

    opt_metrics = evaluate_allocation(result.x, reps_df, territories_df)
    baseline_x = status_quo_allocation(reps_df, territories_df)
    baseline_metrics = evaluate_allocation(baseline_x, reps_df, territories_df)

    assert opt_metrics["net_revenue"] >= baseline_metrics["net_revenue"] - TOL


def test_budget_constraint_is_respected(full_dataset):
    reps_df, territories_df = full_dataset
    unconstrained = solve_allocation(reps_df, territories_df)
    assert unconstrained.success

    # Set a budget below the unconstrained cost to verify it actually binds.
    tight_budget = unconstrained.total_cost * 0.7
    constrained = solve_allocation(reps_df, territories_df, budget=tight_budget)

    if constrained.success:
        assert constrained.total_cost <= tight_budget + TOL
        assert constrained.objective_value <= unconstrained.objective_value + TOL
    else:
        # If min-coverage requirements make the tight budget infeasible,
        # that itself is a valid, meaningful outcome to assert on.
        assert constrained.status == "infeasible_or_error"


def test_baselines_respect_rep_capacity(full_dataset):
    reps_df, territories_df = full_dataset
    capacity = reps_df["max_capacity_fte"].to_numpy(dtype=float)

    for alloc_fn in (status_quo_allocation, round_robin_allocation):
        x = alloc_fn(reps_df, territories_df)
        allocated = x.sum(axis=1)
        assert np.all(allocated <= capacity + TOL)
        assert np.all(x >= 0)
