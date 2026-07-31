#!/usr/bin/env python3
"""End-to-end pipeline entry point.

Usage:
    python run.py                     # generate data (if missing), solve, report
    python run.py --regenerate        # force-regenerate synthetic data first
    python run.py --budget 2500000    # solve with an optional total budget cap
    python run.py --baseline round_robin   # compare against round-robin baseline instead of status-quo

Runs the full pipeline:
    1. Load (or generate) reps.csv / territories.csv
    2. Solve the LP allocation model
    3. Compute a baseline allocation and compare KPIs
    4. Write output/territory_allocation_report.xlsx
    5. Print a human-readable summary to stdout
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import generate_data
from src.optimize import load_data, solve_allocation, evaluate_allocation
from src.baseline import status_quo_allocation, round_robin_allocation
from src.report import generate_report

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")


def main():
    parser = argparse.ArgumentParser(description="Run the full sales force territory optimization pipeline.")
    parser.add_argument("--regenerate", action="store_true", help="Force regeneration of synthetic input data.")
    parser.add_argument("--reps", type=int, default=28, help="Number of reps to generate if data doesn't exist.")
    parser.add_argument("--territories", type=int, default=20, help="Number of territories to generate if data doesn't exist.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data generation.")
    parser.add_argument("--budget", type=float, default=None, help="Optional total budget cap constraint.")
    parser.add_argument("--baseline", choices=["status_quo", "round_robin"], default="status_quo",
                         help="Which baseline to compare the optimized solution against.")
    parser.add_argument("--out", type=str, default=None, help="Output path for the Excel report.")
    args = parser.parse_args()

    reps_path = os.path.join(DATA_DIR, "reps.csv")
    territories_path = os.path.join(DATA_DIR, "territories.csv")

    if args.regenerate or not (os.path.exists(reps_path) and os.path.exists(territories_path)):
        print(f"Generating synthetic data ({args.reps} reps, {args.territories} territories, seed={args.seed})...")
        reps_df, territories_df = generate_data.generate(args.reps, args.territories, args.seed)
        os.makedirs(DATA_DIR, exist_ok=True)
        reps_df.to_csv(reps_path, index=False)
        territories_df.to_csv(territories_path, index=False)
    else:
        print(f"Loading existing data from {reps_path} and {territories_path}")
        reps_df, territories_df = load_data(reps_path, territories_path)

    print(f"\n{len(reps_df)} reps / {len(territories_df)} territories loaded.")
    print("Solving LP allocation model (scipy.optimize.linprog, HiGHS)...")
    opt_result = solve_allocation(reps_df, territories_df, budget=args.budget)

    if not opt_result.success:
        print(f"FAILED to find a feasible solution: {opt_result.scipy_message}")
        sys.exit(1)

    opt_metrics = evaluate_allocation(opt_result.x, reps_df, territories_df)

    if args.baseline == "status_quo":
        baseline_x = status_quo_allocation(reps_df, territories_df)
        baseline_label = "Status Quo"
    else:
        baseline_x = round_robin_allocation(reps_df, territories_df)
        baseline_label = "Round Robin"

    baseline_metrics = evaluate_allocation(baseline_x, reps_df, territories_df)

    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"Status              : {opt_result.status}")
    print(f"Total revenue       : ${opt_metrics['total_revenue']:,.2f}")
    print(f"Total cost          : ${opt_metrics['total_cost']:,.2f}")
    print(f"Net revenue         : ${opt_metrics['net_revenue']:,.2f}")
    print(f"Avg territory coverage : {opt_metrics['avg_coverage'] * 100:.1f}%")
    print(f"Avg rep utilization    : {opt_metrics['avg_utilization'] * 100:.1f}%")
    print(f"Territories meeting min coverage: {opt_metrics['num_territories_meeting_min']} / {len(territories_df)}")

    print("\n" + "=" * 60)
    print(f"BASELINE ({baseline_label}) RESULTS")
    print("=" * 60)
    print(f"Total revenue       : ${baseline_metrics['total_revenue']:,.2f}")
    print(f"Total cost          : ${baseline_metrics['total_cost']:,.2f}")
    print(f"Net revenue         : ${baseline_metrics['net_revenue']:,.2f}")
    print(f"Avg territory coverage : {baseline_metrics['avg_coverage'] * 100:.1f}%")
    print(f"Avg rep utilization    : {baseline_metrics['avg_utilization'] * 100:.1f}%")
    print(f"Territories meeting min coverage: {baseline_metrics['num_territories_meeting_min']} / {len(territories_df)}")

    if baseline_metrics["net_revenue"] != 0:
        improvement = ((opt_metrics["net_revenue"] - baseline_metrics["net_revenue"])
                        / abs(baseline_metrics["net_revenue"])) * 100.0
        print("\n" + "=" * 60)
        print(f"NET REVENUE IMPROVEMENT vs. {baseline_label} BASELINE: {improvement:+.1f}%")
        print("=" * 60)

    out_path = args.out or os.path.join(OUTPUT_DIR, "territory_allocation_report.xlsx")
    path = generate_report(opt_result, baseline_x, out_path, baseline_label=baseline_label)
    print(f"\nExcel KPI dashboard written to: {path}")


if __name__ == "__main__":
    main()
