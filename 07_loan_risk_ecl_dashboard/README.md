# Loan Portfolio Risk & ECL Analytics Dashboard

A working credit-risk analytics stack for a consumer/commercial loan
portfolio: a synthetic loan-book simulator, a parameterised SQL analytics
layer, an Ind AS 109 / IFRS 9-style Expected Credit Loss (ECL) engine built
on empirically-derived roll rates, a scikit-learn early-warning classifier,
and an interactive Streamlit dashboard with a live stress-scenario
simulator.

Every number in the dashboard is computed from the generated data at run
time — there are no hardcoded KRIs, PDs, or ECL figures anywhere in this
repo. Re-run the ETL with a different `--seed` or `--n-loans` and every
downstream chart, table and metric changes accordingly.

## What this project does

1. **Simulates a realistic loan book** (`src/etl.py`): ~3,500 loans across
   four products (personal loan, auto loan, mortgage, credit card) and five
   geographies, each with a monthly panel of delinquency snapshots from
   origination to a portfolio "as of" date. Delinquency evolves via a
   discrete-time Markov process whose transition probabilities depend on
   each loan's simulated credit profile (credit score, rate, product,
   geography, and a simulated macro-economic stress vintage window) — so
   the resulting roll rates, vintage curves and default rates are emergent,
   not scripted.
2. **Loads it into SQLite** (`data/loan_book.db`) via `sql/schema.sql`.
3. **Exposes a reviewable SQL layer** (`sql/*.sql`): parameterised
   portfolio-segmentation and roll-rate queries plus a delinquency-bucket
   aggregation view, all executed with *bound* parameters
   (`pandas.read_sql_query(sql, conn, params={...})`) — never with string
   formatting.
4. **Computes real risk analytics** (`src/risk.py`):
   - Delinquency bucketing (Current / 1-29 / 30-59 / 60-89 / 90+ DPD).
   - Simplified IFRS 9 / Ind AS 109 staging (Stage 1/2/3) based on DPD
     thresholds, with an optional qualitative SICR overlay.
   - An empirical monthly roll-rate (transition) matrix built from the
     panel data.
   - PD derived from that matrix — **not a hardcoded constant** — via
     matrix-power Markov chain math (12-month PD for Stage 1) and an
     absorbing-Markov-chain fundamental-matrix solve, `N = (I-Q)⁻¹`, for
     lifetime PD (Stage 2).
   - ECL = PD × LGD × EAD per loan, with LGD as a configurable,
     product-level assumption (stressable from the dashboard).
5. **Trains a scikit-learn early-warning model** (`src/model.py`): logistic
   regression predicting the probability that a currently-healthy loan
   (Current/1-29 DPD) rolls to default within 6 months, trained on genuine
   forward-looking outcomes from the simulated panel (with proper
   censoring rules — no look-ahead leakage).
6. **Serves an interactive dashboard** (`app.py`): KRIs, a roll-rate
   heatmap, ECL-by-stage charts, product/geography/vintage drill-down
   filters, an early-warning watchlist, and PD/LGD stress-scenario sliders
   that recompute ECL live.

## Project structure

```
loan-portfolio-risk-ecl-dashboard/
├── app.py                       # Streamlit dashboard
├── data/
│   └── loan_book.db             # generated SQLite database (see "Data" below)
├── sql/
│   ├── schema.sql                       # DDL: loans, loan_snapshots tables
│   ├── view_delinquency_buckets.sql     # delinquency-bucket aggregation view
│   ├── latest_snapshot_per_loan.sql     # point-in-time portfolio (parameterised)
│   ├── segment_portfolio.sql            # product/geography/vintage segmentation
│   ├── roll_rate_pairs.sql              # month-over-month bucket transition pairs
│   └── kri_trend.sql                    # KRI time series
├── src/
│   ├── db.py                    # loads/executes .sql files with bound params
│   ├── etl.py                   # synthetic loan-book generator + SQLite loader
│   ├── risk.py                  # bucketing, staging, roll rates, PD, ECL
│   └── model.py                 # scikit-learn early-warning classifier
├── tests/
│   ├── conftest.py
│   ├── test_etl.py              # schema / row counts / reproducibility
│   ├── test_risk.py             # bucketing, staging, roll rates, ECL arithmetic
│   └── test_model.py            # label construction + model training
├── requirements.txt
└── .streamlit/config.toml       # headless-friendly Streamlit defaults
```

## Setup & running it

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

# 2. Install dependencies (single resolver pass, all from PyPI)
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3. (Re)generate the synthetic loan book -> data/loan_book.db
python src/etl.py                   # or: python -m src.etl
#   optional flags: --n-loans 3500 --seed 42 --as-of-date 2026-06-30

# 4. Run the dashboard
python -m streamlit run app.py

# 5. Run the test suite
pytest
```

The dashboard reads `data/loan_book.db`; if it doesn't exist yet, `app.py`
will show a clear message telling you to run `python src/etl.py` first
rather than crashing.

### A note on environments: use pip *or* conda, never both

All the packages in `requirements.txt` that touch native code (`numpy`,
`scipy`, `scikit-learn`, `pyarrow`) must come from a **single installer**.
Mixing a conda-installed `numpy` (built against conda-forge's OpenBLAS)
with a pip-installed `pyarrow` wheel (or vice versa) in the same
environment causes binary ABI mismatches that surface as a segmentation
fault — typically deep inside `pyarrow`/`pandas` Arrow conversion, e.g.
when Streamlit renders an Altair chart via `st.altair_chart`. This isn't a
version-pinning problem; version numbers can match exactly and it will
still crash, because the two binaries were compiled against different
toolchains.

If you use `venv` + `pip` (as above), everything resolves from PyPI wheels
built for the same ABI, and there's no issue. If you prefer conda, install
*all* of `numpy`/`scipy`/`scikit-learn`/`pyarrow` via
`conda install -c conda-forge ...` — don't `pip install --no-deps` any of
them into a conda env afterward.

If `streamlit run app.py` ever segfaults, get a stack trace before trying
anything else:

```bash
python -X faulthandler -m streamlit run app.py
```

The `Extension modules:` line and Python-level traceback in the crash
output will show which native library was executing at the moment of the
crash — that's the one to check for a mixed-installer conflict.

### Data: why the database is committed

This repo **commits a pre-generated `data/loan_book.db`** (rather than
`.gitignore`-ing it) so the dashboard runs immediately after `git clone` +
`pip install` with no extra step. If you want a fresh/different portfolio
(different size, seed, or as-of date), just re-run `python src/etl.py`
with your preferred flags — it drops and rebuilds the database in place —
and commit the regenerated file. `.gitignore` documents this choice
explicitly rather than silently excluding `*.db`.

## Methodology notes / simplifications

This is a portfolio/demo project, not a production model-risk artifact.
Simplifications worth calling out explicitly (the kind of thing a reviewer
would ask about):

- **Staging** uses DPD thresholds only (30 / 90 days), the IFRS 9
  rebuttable-presumption default, plus an optional qualitative SICR
  override hook (`stage_from_dpd(dpd, sicr_flag=...)`) that a real
  implementation would drive from covenant breaches, credit-bureau score
  deterioration, watchlist status, etc.
- **PD** is derived from a single portfolio-wide roll-rate matrix by
  default; the dashboard could easily be extended to compute
  segment-specific matrices (the SQL and `risk.py` functions already accept
  product/geography filters — see `sql/roll_rate_pairs.sql`) but blends
  segments in the base view so PD stays comparable across the drill-down
  filters.
- **LGD** is a configurable assumption per product (industry-typical
  starting points: mortgage 35%, auto 45%, personal loan 65%, credit card
  80%), not modelled from actual recovery/collateral data — there is no
  collateral or recovery-cashflow table in this dataset.
- **EAD** is simply current outstanding balance; no credit-conversion
  factor is applied to undrawn credit-card limits.
- **Amortization** uses a standard fixed-payment schedule for installment
  products and a simplified utilization/payment-rate model for revolving
  credit cards; delinquent balances accrue interest but do not model fees,
  restructuring, or partial cures precisely.
- Once a simulated loan reaches 90+ DPD it is treated as charged off and
  leaves the monthly panel (absorbing state) — real portfolios sometimes
  cure from 90+ via restructuring; this is a deliberate simplification that
  keeps the roll-rate matrix a clean absorbing Markov chain for the PD math.

## Tech stack

Python · SQLite (portable ANSI-style SQL, easily portable to
SQL Server/PostgreSQL) · pandas · scikit-learn · Streamlit · Altair ·
pytest.