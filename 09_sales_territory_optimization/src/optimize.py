"""LP allocation engine for the Sales Force Optimization project.

Formulation
-----------
Decision variables (all continuous):
    x[r, t] in [0, cap_r]   -- fraction of rep r's FTE capacity allocated to
                               territory t (fractional rep-time allocation,
                               not a hard 0/1 assignment).
    y[t]    in [min_cov_t, 1] -- fraction of territory t's revenue potential
                               that is actually captured ("coverage").

Objective (maximize net revenue -> minimize its negation for linprog):
    maximize   sum_t potential_revenue[t] * y[t]
             - sum_r sum_t annual_cost[r] * x[r, t]

    i.e. revenue captured across all territories, minus the cost of the
    rep-time actually deployed to earn it.

Constraints:
    1. Coverage linkage (per territory t):
           required_fte[t] * y[t] - sum_r productivity[r] * x[r, t] <= 0
       y[t] cannot exceed the fraction of required workload actually
       supplied by the reps assigned to territory t.

    2. Minimum coverage (hard, per territory t), enforced via the variable
       bound  y[t] >= min_coverage_pct[t]  -- every territory must receive
       at least its contractually required minimum service level.

    3. Rep capacity (per rep r):
           sum_t x[r, t] <= max_capacity_fte[r]
       a rep cannot be allocated beyond their full-time-equivalent capacity.

    4. Optional total budget constraint:
           sum_r sum_t annual_cost[r] * x[r, t] <= budget
       (off by default; pass `budget=` to solve_allocation to enable).

    5. Bounds: 0 <= x[r, t] <= max_capacity_fte[r]; min_coverage_pct[t] <= y[t] <= 1.

Solved with scipy.optimize.linprog (HiGHS solver), which is an exact LP
solver (not a greedy heuristic) operating on the full A_ub/bounds matrices
built below.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linprog

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_data(reps_path: str | None = None, territories_path: str | None = None):
    reps_path = reps_path or os.path.join(DATA_DIR, "reps.csv")
    territories_path = territories_path or os.path.join(DATA_DIR, "territories.csv")
    reps_df = pd.read_csv(reps_path)
    territories_df = pd.read_csv(territories_path)
    return reps_df, territories_df


@dataclass
class OptimizationResult:
    x: np.ndarray                 # shape (R, T) allocation fractions
    y: np.ndarray                 # shape (T,) coverage fractions
    success: bool
    status: str
    scipy_message: str
    objective_value: float        # net revenue = total_revenue - total_cost
    total_revenue: float
    total_cost: float
    reps_df: pd.DataFrame = field(repr=False)
    territories_df: pd.DataFrame = field(repr=False)
    budget: float | None = None


def _variable_index(num_reps: int, num_territories: int):
    """Return helper functions mapping (r, t) -> flat index and t -> y index."""
    def x_idx(r: int, t: int) -> int:
        return r * num_territories + t

    n_x = num_reps * num_territories

    def y_idx(t: int) -> int:
        return n_x + t

    return x_idx, y_idx, n_x


def solve_allocation(reps_df: pd.DataFrame, territories_df: pd.DataFrame,
                      budget: float | None = None) -> OptimizationResult:
    """Build and solve the LP described in the module docstring."""
    reps_df = reps_df.reset_index(drop=True)
    territories_df = territories_df.reset_index(drop=True)

    R = len(reps_df)
    T = len(territories_df)
    x_idx, y_idx, n_x = _variable_index(R, T)
    n_vars = n_x + T

    cost = reps_df["annual_cost"].to_numpy(dtype=float)
    capacity = reps_df["max_capacity_fte"].to_numpy(dtype=float)
    productivity = reps_df["productivity_multiplier"].to_numpy(dtype=float)

    revenue = territories_df["potential_revenue"].to_numpy(dtype=float)
    required_fte = territories_df["required_fte"].to_numpy(dtype=float)
    min_coverage = territories_df["min_coverage_pct"].to_numpy(dtype=float)

    # ---- Objective: minimize -(revenue*y - cost*x)  ------------------
    c = np.zeros(n_vars)
    for r in range(R):
        for t in range(T):
            c[x_idx(r, t)] = cost[r]          # + cost[r] * x[r,t]  (minimized)
    for t in range(T):
        c[y_idx(t)] = -revenue[t]             # - revenue[t] * y[t] (minimized)

    A_ub_rows = []
    b_ub = []

    # ---- Constraint 1: coverage linkage per territory -----------------
    # required_fte[t]*y[t] - sum_r productivity[r]*x[r,t] <= 0
    for t in range(T):
        row = np.zeros(n_vars)
        row[y_idx(t)] = required_fte[t]
        for r in range(R):
            row[x_idx(r, t)] = -productivity[r]
        A_ub_rows.append(row)
        b_ub.append(0.0)

    # ---- Constraint 3: rep capacity ------------------------------------
    # sum_t x[r,t] <= capacity[r]
    for r in range(R):
        row = np.zeros(n_vars)
        for t in range(T):
            row[x_idx(r, t)] = 1.0
        A_ub_rows.append(row)
        b_ub.append(capacity[r])

    # ---- Constraint 4 (optional): total budget -------------------------
    if budget is not None:
        row = np.zeros(n_vars)
        for r in range(R):
            for t in range(T):
                row[x_idx(r, t)] = cost[r]
        A_ub_rows.append(row)
        b_ub.append(budget)

    A_ub = np.vstack(A_ub_rows)
    b_ub = np.array(b_ub)

    # ---- Bounds ----------------------------------------------------------
    bounds = [None] * n_vars
    for r in range(R):
        for t in range(T):
            bounds[x_idx(r, t)] = (0.0, float(capacity[r]))
    for t in range(T):
        bounds[y_idx(t)] = (float(min_coverage[t]), 1.0)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if not res.success:
        return OptimizationResult(
            x=np.zeros((R, T)), y=np.zeros(T), success=False, status="infeasible_or_error",
            scipy_message=res.message, objective_value=float("nan"),
            total_revenue=float("nan"), total_cost=float("nan"),
            reps_df=reps_df, territories_df=territories_df, budget=budget,
        )

    x = np.array([[res.x[x_idx(r, t)] for t in range(T)] for r in range(R)])
    y = np.array([res.x[y_idx(t)] for t in range(T)])

    total_revenue = float((revenue * y).sum())
    total_cost = float((cost[:, None] * x).sum())
    objective_value = total_revenue - total_cost

    return OptimizationResult(
        x=x, y=y, success=True, status="optimal", scipy_message=res.message,
        objective_value=objective_value, total_revenue=total_revenue, total_cost=total_cost,
        reps_df=reps_df, territories_df=territories_df, budget=budget,
    )


def evaluate_allocation(x: np.ndarray, reps_df: pd.DataFrame, territories_df: pd.DataFrame) -> dict:
    """Compute KPI metrics for an arbitrary allocation matrix x (R, T).

    This is used both to sanity-check the LP's own solution and to score
    baseline allocations on an apples-to-apples basis. Coverage here is a
    *derived* metric (min(1, supplied/required)) and does NOT assume any
    minimum-coverage constraint was honored -- that lets it be used to
    evaluate baselines that may under-cover some territories.
    """
    reps_df = reps_df.reset_index(drop=True)
    territories_df = territories_df.reset_index(drop=True)

    cost = reps_df["annual_cost"].to_numpy(dtype=float)
    capacity = reps_df["max_capacity_fte"].to_numpy(dtype=float)
    productivity = reps_df["productivity_multiplier"].to_numpy(dtype=float)

    revenue = territories_df["potential_revenue"].to_numpy(dtype=float)
    required_fte = territories_df["required_fte"].to_numpy(dtype=float)
    min_coverage = territories_df["min_coverage_pct"].to_numpy(dtype=float)

    supplied_fte = productivity[:, None] * x  # (R, T) effective FTE supplied
    territory_supplied = supplied_fte.sum(axis=0)  # (T,)
    coverage = np.minimum(1.0, np.divide(
        territory_supplied, required_fte, out=np.zeros_like(territory_supplied), where=required_fte > 0
    ))
    achieved_revenue = revenue * coverage

    rep_allocated = x.sum(axis=1)  # (R,)
    rep_cost = cost * rep_allocated
    utilization = np.divide(rep_allocated, capacity, out=np.zeros_like(rep_allocated), where=capacity > 0)

    total_revenue = float(achieved_revenue.sum())
    total_cost = float(rep_cost.sum())
    net_revenue = total_revenue - total_cost
    meets_min_coverage = coverage >= (min_coverage - 1e-6)

    return {
        "coverage": coverage,
        "achieved_revenue": achieved_revenue,
        "territory_supplied_fte": territory_supplied,
        "meets_min_coverage": meets_min_coverage,
        "rep_allocated_fte": rep_allocated,
        "rep_cost": rep_cost,
        "utilization": utilization,
        "total_revenue": total_revenue,
        "total_cost": total_cost,
        "net_revenue": net_revenue,
        "num_territories_fully_covered": int((coverage >= 0.999).sum()),
        "num_territories_meeting_min": int(meets_min_coverage.sum()),
        "avg_utilization": float(utilization.mean()),
        "avg_coverage": float(coverage.mean()),
    }


def allocation_to_long_df(x: np.ndarray, reps_df: pd.DataFrame, territories_df: pd.DataFrame,
                           threshold: float = 1e-6) -> pd.DataFrame:
    """Convert the (R, T) allocation matrix into a tidy long-form DataFrame,
    keeping only non-trivial (> threshold) rep-territory assignments."""
    reps_df = reps_df.reset_index(drop=True)
    territories_df = territories_df.reset_index(drop=True)
    productivity = reps_df["productivity_multiplier"].to_numpy(dtype=float)
    cost = reps_df["annual_cost"].to_numpy(dtype=float)

    rows = []
    R, T = x.shape
    for r in range(R):
        for t in range(T):
            frac = x[r, t]
            if frac > threshold:
                rows.append({
                    "rep_id": reps_df.loc[r, "rep_id"],
                    "rep_name": reps_df.loc[r, "rep_name"],
                    "territory_id": territories_df.loc[t, "territory_id"],
                    "territory_name": territories_df.loc[t, "territory_name"],
                    "allocated_fte": round(float(frac), 4),
                    "effective_fte_supplied": round(float(frac * productivity[r]), 4),
                    "cost_contribution": round(float(frac * cost[r]), 2),
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["rep_id", "territory_id"]).reset_index(drop=True)
    return df


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Solve the sales force territory allocation LP.")
    parser.add_argument("--reps", type=str, default=None, help="Path to reps.csv")
    parser.add_argument("--territories", type=str, default=None, help="Path to territories.csv")
    parser.add_argument("--budget", type=float, default=None, help="Optional total budget cap.")
    args = parser.parse_args()

    reps_df, territories_df = load_data(args.reps, args.territories)
    result = solve_allocation(reps_df, territories_df, budget=args.budget)

    print(f"Solver status : {result.status} ({result.scipy_message})")
    if not result.success:
        print("No feasible solution found.")
        return

    print(f"Total revenue : ${result.total_revenue:,.2f}")
    print(f"Total cost    : ${result.total_cost:,.2f}")
    print(f"Net revenue   : ${result.objective_value:,.2f}")

    metrics = evaluate_allocation(result.x, reps_df, territories_df)
    print(f"Avg territory coverage : {metrics['avg_coverage'] * 100:.1f}%")
    print(f"Avg rep utilization    : {metrics['avg_utilization'] * 100:.1f}%")
    print(f"Territories fully covered (100%): {metrics['num_territories_fully_covered']} / {len(territories_df)}")


if __name__ == "__main__":
    main()
