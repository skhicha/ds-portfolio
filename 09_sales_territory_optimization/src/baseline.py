"""Naive / manual allocation baselines used to benchmark the LP solution.

Two baselines are implemented:

1. status_quo_allocation -- each rep is assigned 100% of their capacity to
   whatever `current_territory_id` they already have in the source data
   (i.e. "how the business is allocated today, before any optimization").
   This is the primary baseline used for the headline improvement %.

2. round_robin_allocation -- a simple manual heuristic a sales-ops analyst
   might use with no optimization tooling at all: reps are handed out to
   territories in a round-robin cycle (ignoring skill/productivity/revenue
   entirely), each rep allocating their full capacity to a single territory.

Both return an (R, T) allocation matrix in exactly the same shape/semantics
as the LP's `x`, so they can be scored with `optimize.evaluate_allocation`
for a true apples-to-apples comparison against the optimized solution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def status_quo_allocation(reps_df: pd.DataFrame, territories_df: pd.DataFrame) -> np.ndarray:
    reps_df = reps_df.reset_index(drop=True)
    territories_df = territories_df.reset_index(drop=True)
    R = len(reps_df)
    T = len(territories_df)
    x = np.zeros((R, T))

    territory_index = {tid: i for i, tid in enumerate(territories_df["territory_id"])}
    capacity = reps_df["max_capacity_fte"].to_numpy(dtype=float)

    for r in range(R):
        tid = reps_df.loc[r, "current_territory_id"]
        t = territory_index.get(tid)
        if t is not None:
            x[r, t] = capacity[r]
    return x


def round_robin_allocation(reps_df: pd.DataFrame, territories_df: pd.DataFrame) -> np.ndarray:
    reps_df = reps_df.reset_index(drop=True)
    territories_df = territories_df.reset_index(drop=True)
    R = len(reps_df)
    T = len(territories_df)
    x = np.zeros((R, T))
    capacity = reps_df["max_capacity_fte"].to_numpy(dtype=float)

    for r in range(R):
        t = r % T
        x[r, t] = capacity[r]
    return x
