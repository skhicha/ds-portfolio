"""Synthetic data generator for the Sales Force Optimization project.

Generates two CSV files:
    data/reps.csv         - sales rep roster with cost / capacity / skill attributes
    data/territories.csv  - territory roster with revenue potential / workload attributes

The data is randomly generated but seeded for reproducibility, and the ranges
are deliberately tuned so that the resulting LP is *feasible but tight*: total
rep supply comfortably covers each territory's minimum required coverage, but
not enough capacity exists to fully (100%) cover every territory at once.
That forces the optimizer to make real trade-offs between high- and
low-revenue territories, which is what makes the allocation problem
non-trivial (a pure round-robin / proportional baseline will do noticeably
worse than the LP-optimal solution).
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

REGIONS = ["Northeast", "Southeast", "Midwest", "Southwest", "West", "Northwest"]
SKILL_LEVELS = ["Junior", "Mid", "Senior"]
SKILL_BASE_PRODUCTIVITY = {"Junior": 0.75, "Mid": 1.00, "Senior": 1.30}
SKILL_BASE_SALARY = {"Junior": 55_000, "Mid": 78_000, "Senior": 105_000}

FIRST_NAMES = [
    "Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Jamie", "Avery",
    "Cameron", "Drew", "Skyler", "Reese", "Quinn", "Rowan", "Emerson", "Dakota",
    "Peyton", "Hayden", "Finley", "Sawyer", "Blake", "Charlie", "Elliot", "Sage",
    "Micah", "Devon", "Kendall", "Marley", "Shawn", "Tatum", "Remy", "Lane",
    "Harper", "Sydney", "Adrian", "Jesse", "Robin", "Kai", "Val", "Noel",
]
LAST_NAMES = [
    "Nguyen", "Smith", "Patel", "Garcia", "Kim", "Johnson", "Brown", "Lee",
    "Martinez", "Davis", "Chen", "Wilson", "Anderson", "Taylor", "Thomas",
    "Moore", "Jackson", "Martin", "White", "Harris", "Clark", "Lewis",
    "Walker", "Young", "King", "Wright", "Scott", "Green", "Baker", "Adams",
]
TERRITORY_ADJECTIVES = [
    "Metro", "Coastal", "Central", "Greater", "Upper", "Lower", "North",
    "South", "East", "West", "Valley", "Bay", "Lakeside", "Highland", "River",
]
TERRITORY_NOUNS = [
    "District", "Region", "Zone", "Corridor", "Basin", "Territory", "Belt",
    "Area", "Sector", "Cluster",
]


@dataclass
class GeneratorConfig:
    num_reps: int = 28
    num_territories: int = 20
    seed: int = 42


def _make_rep_names(rng: np.random.Generator, n: int) -> list[str]:
    first = rng.choice(FIRST_NAMES, size=n, replace=True)
    last = rng.choice(LAST_NAMES, size=n, replace=True)
    names = [f"{f} {l}" for f, l in zip(first, last)]
    # De-duplicate by appending an index if collisions occur.
    seen: dict[str, int] = {}
    unique_names = []
    for nm in names:
        seen[nm] = seen.get(nm, 0) + 1
        unique_names.append(nm if seen[nm] == 1 else f"{nm} {seen[nm]}")
    return unique_names


def _make_territory_names(rng: np.random.Generator, n: int) -> list[str]:
    adjs = rng.choice(TERRITORY_ADJECTIVES, size=n, replace=True)
    nouns = rng.choice(TERRITORY_NOUNS, size=n, replace=True)
    names = [f"{a} {b}" for a, b in zip(adjs, nouns)]
    seen: dict[str, int] = {}
    unique_names = []
    for nm in names:
        seen[nm] = seen.get(nm, 0) + 1
        unique_names.append(nm if seen[nm] == 1 else f"{nm} {seen[nm]}")
    return unique_names


def generate_territories(cfg: GeneratorConfig, rng: np.random.Generator) -> pd.DataFrame:
    n = cfg.num_territories
    territory_ids = [f"T{idx:03d}" for idx in range(1, n + 1)]
    names = _make_territory_names(rng, n)
    regions = rng.choice(REGIONS, size=n, replace=True)

    # Required workload to fully (100%) staff a territory, expressed in
    # full-time-equivalent (FTE) rep-capacity units.
    required_fte = np.round(rng.uniform(0.8, 2.4, size=n), 2)

    # Monthly required customer visits, roughly proportional to required_fte
    # (used as a human-readable operational metric alongside the FTE figure).
    required_visits_per_month = np.round(required_fte * rng.uniform(18, 26, size=n)).astype(int)

    # Annual revenue potential if the territory is fully (100%) staffed.
    # Larger territories (higher required_fte) tend to have higher revenue
    # potential, with meaningful noise so it is not a trivial linear proxy.
    base_revenue_per_fte = rng.uniform(180_000, 420_000, size=n)
    potential_revenue = np.round(required_fte * base_revenue_per_fte, 2)

    # Minimum coverage fraction that MUST be met (hard LP constraint) --
    # represents contractual / compliance-driven minimum service levels.
    min_coverage_pct = np.round(rng.uniform(0.45, 0.80, size=n), 2)

    df = pd.DataFrame({
        "territory_id": territory_ids,
        "territory_name": names,
        "region": regions,
        "required_fte": required_fte,
        "required_visits_per_month": required_visits_per_month,
        "potential_revenue": potential_revenue,
        "min_coverage_pct": min_coverage_pct,
    })
    return df


def generate_reps(cfg: GeneratorConfig, rng: np.random.Generator, territories: pd.DataFrame) -> pd.DataFrame:
    n = cfg.num_reps
    rep_ids = [f"R{idx:03d}" for idx in range(1, n + 1)]
    names = _make_rep_names(rng, n)
    regions = rng.choice(REGIONS, size=n, replace=True)

    skill_level = rng.choice(SKILL_LEVELS, size=n, replace=True, p=[0.35, 0.40, 0.25])
    experience_years = np.array([
        max(0, int(round(rng.normal({"Junior": 1.5, "Mid": 5, "Senior": 10}[s], 1.8))))
        for s in skill_level
    ])

    # Productivity multiplier: how many "required_fte units" of territory
    # workload one FTE of this rep's time can actually cover, driven by
    # skill level with a small experience bonus and individual noise.
    productivity = np.array([
        SKILL_BASE_PRODUCTIVITY[s] + 0.01 * min(exp, 15) + rng.normal(0, 0.05)
        for s, exp in zip(skill_level, experience_years)
    ])
    productivity = np.clip(np.round(productivity, 3), 0.5, 1.8)

    # Annual fully-loaded cost (salary + overhead), skill-driven with noise.
    base_salary = np.array([
        SKILL_BASE_SALARY[s] + 900 * min(exp, 20) + rng.normal(0, 4000)
        for s, exp in zip(skill_level, experience_years)
    ])
    annual_cost = np.round(np.clip(base_salary, 40_000, 160_000), 2)

    # Max available capacity in FTE terms -- most reps are full time (1.0),
    # a minority are part-time.
    max_capacity_fte = rng.choice([1.0, 0.75, 0.5], size=n, p=[0.78, 0.14, 0.08])

    # Each rep currently "belongs" to a territory under the existing manual
    # process (used to build the status-quo baseline allocation). Assigned
    # round-robin across territories in the rep's own region when possible,
    # else at random, so the baseline resembles a plausible legacy setup.
    current_territory_id = []
    for reg in regions:
        candidates = territories.loc[territories["region"] == reg, "territory_id"].tolist()
        if not candidates:
            candidates = territories["territory_id"].tolist()
        current_territory_id.append(rng.choice(candidates))

    df = pd.DataFrame({
        "rep_id": rep_ids,
        "rep_name": names,
        "region": regions,
        "current_territory_id": current_territory_id,
        "skill_level": skill_level,
        "experience_years": experience_years,
        "productivity_multiplier": productivity,
        "annual_cost": annual_cost,
        "max_capacity_fte": max_capacity_fte,
    })
    return df


def generate(num_reps: int = 28, num_territories: int = 20, seed: int = 42):
    cfg = GeneratorConfig(num_reps=num_reps, num_territories=num_territories, seed=seed)
    rng = np.random.default_rng(seed)
    territories_df = generate_territories(cfg, rng)
    reps_df = generate_reps(cfg, rng, territories_df)
    return reps_df, territories_df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic reps/territories data.")
    parser.add_argument("--reps", type=int, default=28, help="Number of sales reps to generate (default 28).")
    parser.add_argument("--territories", type=int, default=20, help="Number of territories to generate (default 20).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory for CSVs (default: <project>/data).")
    args = parser.parse_args()

    outdir = args.outdir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(outdir, exist_ok=True)

    reps_df, territories_df = generate(args.reps, args.territories, args.seed)

    reps_path = os.path.join(outdir, "reps.csv")
    territories_path = os.path.join(outdir, "territories.csv")
    reps_df.to_csv(reps_path, index=False)
    territories_df.to_csv(territories_path, index=False)

    print(f"Generated {len(reps_df)} reps -> {reps_path}")
    print(f"Generated {len(territories_df)} territories -> {territories_path}")
    print(f"Total rep supply (sum capacity*productivity): "
          f"{(reps_df.max_capacity_fte * reps_df.productivity_multiplier).sum():.2f} FTE-equivalent")
    print(f"Total territory demand (sum required_fte): {territories_df.required_fte.sum():.2f} FTE")
    print(f"Total min-coverage demand: "
          f"{(territories_df.required_fte * territories_df.min_coverage_pct).sum():.2f} FTE")


if __name__ == "__main__":
    main()
