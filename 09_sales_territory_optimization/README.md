# Sales Force Optimization & Territory Allocation Model

A working linear-programming engine that assigns sales reps to territories to
**maximize net revenue** (revenue captured minus rep cost), subject to real
capacity and coverage constraints — plus an Excel KPI dashboard and a
head-to-head comparison against a naive/manual baseline allocation.

Built as a portfolio project to demonstrate applied operations-research /
optimization skills: problem formulation, constrained LP solving with
`scipy.optimize.linprog`, and turning solver output into a business-readable
Excel report.

## What it actually does

1. **Generates synthetic input data** (`src/generate_data.py`) — a roster of
   sales reps (skill level, experience, productivity, cost, capacity, current
   territory) and a roster of territories (revenue potential, workload
   requirement, minimum required coverage, region), sized so the LP is
   non-trivial (default: 28 reps × 20 territories).
2. **Solves a real LP** (`src/optimize.py`) with `scipy.optimize.linprog`
   (HiGHS solver) — not a greedy heuristic — to find the revenue-maximizing
   allocation of rep-time across territories.
3. **Computes a baseline** (`src/baseline.py`) — either the current
   "status-quo" manual assignment already present in the data, or a naive
   round-robin assignment — scored with the *same* KPI model as the
   optimized solution, so the comparison is apples-to-apples.
4. **Generates an Excel KPI dashboard** (`src/report.py`, via `openpyxl`) with
   5 sheets: Summary, Optimized Allocation, Territory KPIs, Rep KPIs, and
   Baseline Comparison.
5. Ships with a **pytest suite** that verifies the solver's solution actually
   satisfies every constraint, that the reported objective matches an
   independent manual recomputation, and that the Excel report has the
   expected structure.
6. Includes an **optional Streamlit viewer** (`app.py`) for interactively
   re-solving with a different budget cap / baseline.

Nothing here is a stubbed or hardcoded output — every number below (revenue,
cost, coverage %, improvement %) came from actually running the pipeline
against the generated `data/*.csv` in this repo.

## LP formulation

**Decision variables** (continuous, i.e. *fractional rep-time allocation*
rather than a hard 0/1 assignment — this reflects that a rep can legitimately
split their week across two or three nearby territories):

- `x[r, t] ∈ [0, capacity_r]` — the fraction of rep `r`'s full-time-equivalent
  (FTE) capacity allocated to territory `t`.
- `y[t] ∈ [min_coverage_t, 1]` — the fraction of territory `t`'s revenue
  potential actually captured ("coverage").

**Objective** — maximize net revenue (`linprog` minimizes, so the engine
minimizes the negation internally):

```
maximize   Σ_t potential_revenue[t] · y[t]   -   Σ_r Σ_t annual_cost[r] · x[r, t]
```

i.e. total revenue actually captured across all territories, minus the cost
of the rep-time deployed to earn it.

**Constraints**:

1. **Coverage linkage** (per territory `t`): a territory's revenue capture
   `y[t]` cannot exceed the fraction of its required workload that is
   actually staffed —
   `required_fte[t] · y[t]  ≤  Σ_r productivity[r] · x[r, t]`
2. **Minimum coverage** (hard constraint, per territory): every territory
   must receive at least its contractual minimum service level —
   enforced as the lower bound `y[t] ≥ min_coverage_pct[t]`.
3. **Rep capacity** (per rep `r`): a rep cannot be allocated beyond their FTE
   capacity — `Σ_t x[r, t] ≤ max_capacity_fte[r]`.
4. **Optional total budget cap**: `Σ_r Σ_t annual_cost[r] · x[r, t] ≤ budget`
   (off by default; enable with `--budget`).

This is solved exactly via `scipy.optimize.linprog(method="highs")` over the
full `A_ub` / bounds matrices built in `src/optimize.py::solve_allocation` —
there is no rounding, greedy assignment, or heuristic fallback in the solve
path itself.

### Why fractional allocation instead of hard 0/1 assignment

A binary rep→territory assignment would require integer variables (MILP),
which `linprog` does not support. Fractional allocation is also a more
realistic model of how field sales actually works (reps commonly split time
across a home territory plus one or two adjacent ones), and it keeps the
problem a *clean, exactly-solvable* LP rather than an approximately-solved
MILP — the tradeoff is documented here rather than papered over.

### Known simplification / not implemented

A hard "max N territories per rep" constraint would require binary indicator
variables (is rep `r` active in territory `t`, yes/no) and turn this into a
MILP. That's out of scope for a pure-LP `linprog` formulation, so it isn't
implemented; the optional **budget cap** constraint (#4 above) is provided
instead, per the spec's "max-territories-per-rep **or** budget constraint"
option.

## Results from the included data

Running `python run.py` against the committed `data/reps.csv` /
`data/territories.csv` (28 reps, 20 territories, seed 42) produces:

| Metric | Optimized (LP) | Baseline (Status Quo) |
|---|---|---|
| Total revenue | $7,646,016.86 | $4,658,009.44 |
| Total cost | $2,054,029.28 | $2,054,029.28 |
| **Net revenue** | **$5,591,987.58** | **$2,603,980.15** |
| Avg territory coverage | 82.8% | 52.9% |
| Avg rep utilization | 100.0% | 100.0% |
| Territories meeting min. coverage | 20 / 20 | 8 / 20 |

**Net revenue improvement over the status-quo baseline: +114.7%**

This number is *not* hardcoded — it's computed at the end of `run.py` /
`src/report.py` from whatever the solver actually returns, and will change if
you regenerate the data with a different seed or size. Cost is identical
between the two because both allocations use each rep at full utilization;
the entire improvement comes from re-routing the *same* rep-hours toward
higher-revenue, higher-productivity-fit territories and meeting minimum
coverage everywhere (the status-quo baseline leaves 12 of 20 territories
under their contractual minimum coverage — the LP leaves zero).

## Project structure

```
sales-force-territory-optimization/
├── app.py                        # optional Streamlit viewer
├── run.py                        # end-to-end CLI pipeline (generate -> solve -> report)
├── requirements.txt
├── data/
│   ├── reps.csv                  # generated rep roster
│   └── territories.csv           # generated territory roster
├── output/
│   └── territory_allocation_report.xlsx   # generated Excel KPI dashboard
├── src/
│   ├── generate_data.py          # synthetic data generator
│   ├── optimize.py               # LP model + solver + KPI evaluation
│   ├── baseline.py               # status-quo / round-robin baselines
│   └── report.py                 # openpyxl Excel dashboard builder
└── tests/
    ├── conftest.py
    ├── test_generate_data.py
    ├── test_optimize.py
    └── test_report.py
```

## Setup

Requires Python 3.10+.

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Generate data (only needed once — `run.py` will also auto-generate it if
`data/*.csv` is missing) and run the full pipeline:

```bash
python -m src.generate_data --reps 28 --territories 20 --seed 42
python run.py
```

This solves the LP, computes the baseline comparison, prints a summary to
stdout, and writes `output/territory_allocation_report.xlsx`.

Useful flags:

```bash
python run.py --regenerate --reps 35 --territories 22 --seed 7   # new random instance
python run.py --budget 1800000                                    # add a total budget cap
python run.py --baseline round_robin                              # compare vs round-robin instead of status-quo
```

Run just the solver or just the report generator directly:

```bash
python -m src.optimize                 # solve + print summary only
python -m src.report                   # solve + write Excel report only
```

### Optional Streamlit viewer

```bash
python -m streamlit run app.py
```

Lets you regenerate data at a different size, toggle a budget cap, switch
the comparison baseline, re-solve, and download the Excel report from the
browser.

## Tests

```bash
pytest tests/ -v
```

21 tests, all passing, covering:

- Data generator produces valid ranges/shapes and is reproducible for a
  given seed.
- The solver converges to `status == "optimal"` at both a small scale and
  the full 28×20 default scale.
- **Rep capacity is never exceeded** in the LP solution.
- **Every territory's coverage meets its minimum requirement.**
- The coverage-linkage constraint actually holds (`y[t]` never exceeds what
  was really staffed).
- The solver's reported objective value matches an **independent manual
  recomputation** of revenue − cost from the returned `x`/`y` (not just
  trusting `res.fun`).
- The optimizer's net revenue is never worse than the baseline's.
- An optional budget cap actually binds when set below the unconstrained
  cost.
- The Excel report file is created, contains the 5 expected sheets, and each
  sheet has the expected columns.

## Tech stack

Python, pandas, numpy, SciPy (`scipy.optimize.linprog`, HiGHS solver),
openpyxl, pytest, Streamlit (optional viewer).
