"""
etl.py
Synthetic loan-book generator and SQLite loader.

There is no external data download here: this script *simulates* a
realistic-looking consumer/commercial loan portfolio (personal loans, auto
loans, mortgages, credit cards) month by month, using a discrete-time Markov
process over delinquency buckets whose transition probabilities depend on
each loan's simulated credit risk profile (credit score, rate, product,
geography, vintage/macro shock). That means every downstream statistic in
this project -- delinquency rates, roll-rate matrices, PDs, ECL, model AUC
-- is a genuine computation over generated data, not a hardcoded figure.

Run directly to (re)build the database:

    python src/etl.py --n-loans 3500 --seed 42

which drops and recreates data/loan_book.db from sql/schema.sql and loads
freshly-simulated `loans` and `loan_snapshots` tables.
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Allow this module to be run both as `python src/etl.py` (script mode,
# no parent package) and as `python -m src.etl` / `import src.etl`
# (package mode) by making sure the project root is importable either way.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import db
    from src.risk import bucket_from_dpd, stage_from_dpd
else:
    from src import db
    from src.risk import bucket_from_dpd, stage_from_dpd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "loan_book.db"

PRODUCTS = ["personal_loan", "auto_loan", "mortgage", "credit_card"]
GEOGRAPHIES = ["North", "South", "East", "West", "Central"]
CHANNELS = ["branch", "digital", "partner"]

# ---------------------------------------------------------------------------
# Product-level origination parameters
# ---------------------------------------------------------------------------

PRODUCT_TERM_CHOICES = {
    "personal_loan": [24, 36, 48, 60],
    "auto_loan": [36, 48, 60, 72],
    "mortgage": [180, 240, 360],
    # Credit cards are revolving; term_months here is a nominal behavioural
    # life cap rather than a true amortization term.
    "credit_card": [360],
}
PRODUCT_PRINCIPAL_RANGE = {
    "personal_loan": (3_000, 40_000),
    "auto_loan": (8_000, 55_000),
    "mortgage": (80_000, 600_000),
    "credit_card": (1_000, 25_000),  # credit limit
}
PRODUCT_RATE_RANGE = {
    "personal_loan": (0.10, 0.19),
    "auto_loan": (0.045, 0.12),
    "mortgage": (0.035, 0.075),
    "credit_card": (0.16, 0.28),
}
PRODUCT_CREDIT_SCORE_PARAMS = {
    "personal_loan": (660, 70),
    "auto_loan": (680, 65),
    "mortgage": (730, 55),
    "credit_card": (650, 75),
}
# Relative structural propensity to keep worsening once delinquent
# (unsecured / revolving products roll faster than secured installment debt).
PRODUCT_ROLL_MULTIPLIER = {
    "personal_loan": 1.15,
    "auto_loan": 0.95,
    "mortgage": 0.70,
    "credit_card": 1.30,
}
# Baseline relative riskiness used in the risk-score logit (0-1 scale).
PRODUCT_BASE_RISK = {
    "personal_loan": 0.55,
    "auto_loan": 0.40,
    "mortgage": 0.20,
    "credit_card": 0.65,
}
GEOGRAPHY_LOGIT_ADJUSTMENT = {
    "North": 0.00,
    "South": 0.15,
    "East": -0.10,
    "West": 0.05,
    "Central": 0.20,
}

# Simulated macro-economic stress window (e.g. a rate-hike / inflation
# shock) that makes loans originated in this vintage window structurally
# riskier -- this is what produces realistic vintage curves in the
# segmentation views.
MACRO_STRESS_START = date(2022, 10, 1)
MACRO_STRESS_END = date(2023, 9, 30)
MACRO_STRESS_LOGIT_ADJUSTMENT = 0.45

CLOSURE_PROB_PER_MONTH = 0.010  # voluntary prepayment / payoff, Current only
DPD_RANGE_BY_BUCKET = {
    "1-29": (1, 29),
    "30-59": (30, 59),
    "60-89": (60, 89),
    "90+": (90, 150),
}


def _quarter_label(d: date) -> str:
    q = (d.month - 1) // 3 + 1
    return f"{d.year}Q{q}"


def _macro_factor(origination_date: date) -> float:
    if MACRO_STRESS_START <= origination_date <= MACRO_STRESS_END:
        return MACRO_STRESS_LOGIT_ADJUSTMENT
    return 0.0


def _random_origination_date(rng: np.random.Generator, as_of_date: date) -> date:
    earliest = as_of_date - timedelta(days=5 * 365)
    latest = as_of_date - timedelta(days=45)
    span_days = (latest - earliest).days
    offset = int(rng.integers(0, span_days + 1))
    return earliest + timedelta(days=offset)


@dataclass
class LoanSeed:
    """Static, origination-time attributes for a single simulated loan."""

    loan_id: str
    product: str
    geography: str
    channel: str
    origination_date: date
    term_months: int
    principal: float
    interest_rate: float
    credit_score: int
    borrower_income: float
    risk_score: float


def _generate_loan_seeds(n_loans: int, as_of_date: date, rng: np.random.Generator) -> list[LoanSeed]:
    seeds: list[LoanSeed] = []
    for i in range(n_loans):
        product = rng.choice(PRODUCTS)
        geography = rng.choice(GEOGRAPHIES)
        channel = rng.choice(CHANNELS, p=[0.40, 0.45, 0.15])
        origination_date = _random_origination_date(rng, as_of_date)

        term_months = int(rng.choice(PRODUCT_TERM_CHOICES[product]))
        lo, hi = PRODUCT_PRINCIPAL_RANGE[product]
        principal = float(rng.uniform(lo, hi))
        lo_r, hi_r = PRODUCT_RATE_RANGE[product]
        interest_rate = float(rng.uniform(lo_r, hi_r))

        mean_score, std_score = PRODUCT_CREDIT_SCORE_PARAMS[product]
        credit_score = int(np.clip(rng.normal(mean_score, std_score), 300, 850))

        borrower_income = float(np.clip(rng.lognormal(mean=math.log(60_000), sigma=0.5), 15_000, 500_000))

        z = (
            1.6 * (650 - credit_score) / 150.0
            + 3.0 * (interest_rate - 0.12)
            + 1.4 * (PRODUCT_BASE_RISK[product] - 0.5)
            + GEOGRAPHY_LOGIT_ADJUSTMENT[geography]
            + _macro_factor(origination_date)
            + rng.normal(0, 0.35)
        )
        risk_score = float(np.clip(1.0 / (1.0 + math.exp(-z)), 0.02, 0.98))

        seeds.append(
            LoanSeed(
                loan_id=f"LN{i + 1:06d}",
                product=product,
                geography=geography,
                channel=channel,
                origination_date=origination_date,
                term_months=term_months,
                principal=principal,
                interest_rate=interest_rate,
                credit_score=credit_score,
                borrower_income=borrower_income,
                risk_score=risk_score,
            )
        )
    return seeds


def _bucket_transition_probs(bucket_idx: int, risk_score: float, product: str) -> dict[int, float]:
    """Monthly transition probability distribution over bucket indices
    {0: Current, 1: 1-29, 2: 30-59, 3: 60-89, 4: 90+} given the current
    bucket, the loan's latent risk_score in [0, 1], and product-specific
    structural roll propensity."""
    mult = PRODUCT_ROLL_MULTIPLIER[product]
    r = risk_score

    if bucket_idx == 0:
        p_worsen = float(np.clip(0.012 * (0.5 + 1.5 * r) * mult, 0.0, 0.9))
        return {0: 1.0 - p_worsen, 1: p_worsen}
    if bucket_idx == 1:
        p_cure = float(np.clip(0.45 * (1.0 - 0.6 * r), 0.0, 1.0))
        p_worsen = float(np.clip(0.12 * (0.5 + 1.6 * r) * mult, 0.0, 1.0 - p_cure))
        return {0: p_cure, 1: 1.0 - p_cure - p_worsen, 2: p_worsen}
    if bucket_idx == 2:
        p_cure = float(np.clip(0.25 * (1.0 - 0.6 * r), 0.0, 1.0))
        p_worsen = float(np.clip(0.18 * (0.5 + 1.5 * r) * mult, 0.0, 1.0 - p_cure))
        return {0: p_cure, 2: 1.0 - p_cure - p_worsen, 3: p_worsen}
    if bucket_idx == 3:
        p_cure = float(np.clip(0.10 * (1.0 - 0.6 * r), 0.0, 1.0))
        p_worsen = float(np.clip(0.30 * (0.5 + 1.5 * r) * mult, 0.0, 1.0 - p_cure))
        return {0: p_cure, 3: 1.0 - p_cure - p_worsen, 4: p_worsen}
    return {4: 1.0}  # 90+ is absorbing


def _reporting_date(origination_date: date, months_on_book: int) -> date:
    """
    Map a loan's origination date + months-on-book count to a common,
    calendar-aligned *month-end* reporting date shared by every loan in the
    portfolio (e.g. 2024-01-31, 2024-02-29, ...), rather than each loan's own
    origination-day anniversary. Real loan-level servicing panels report on
    a common monthly cycle, and aligning snapshots this way is what makes
    the roll-rate matrix and KRI trend genuinely "month over month" across
    the whole book instead of a smear of loan-specific dates.
    """
    shifted = pd.Timestamp(origination_date) + pd.DateOffset(months=months_on_book)
    return (shifted + pd.offsets.MonthEnd(0)).date()


def _amortized_balance(principal: float, annual_rate: float, term_months: int, payments_made: int) -> float:
    """Standard fixed-payment amortization schedule balance after
    `payments_made` on-time monthly payments."""
    payments_made = min(payments_made, term_months)
    r = annual_rate / 12.0
    if r == 0:
        return max(principal - principal / term_months * payments_made, 0.0)
    payment = principal * r / (1 - (1 + r) ** (-term_months))
    balance = principal * (1 + r) ** payments_made - payment * (((1 + r) ** payments_made - 1) / r)
    return max(balance, 0.0)


def _simulate_loan(seed: LoanSeed, as_of_date: date, rng: np.random.Generator) -> tuple[list[dict], dict]:
    """Simulate one loan's monthly snapshot history from origination to
    as_of_date (or until it charges off / matures / prepays)."""
    is_revolving = seed.product == "credit_card"
    monthly_rate = seed.interest_rate / 12.0

    snapshots: list[dict] = []
    bucket_idx = 0
    payments_made = 0
    months_on_book = 0

    if is_revolving:
        balance = seed.principal * float(rng.uniform(0.20, 0.60))
    else:
        balance = seed.principal

    snapshot_date = _reporting_date(seed.origination_date, 0)
    dpd = 0
    bucket = "Current"
    stage = stage_from_dpd(dpd)
    snapshots.append(
        dict(
            loan_id=seed.loan_id,
            snapshot_date=snapshot_date.isoformat(),
            months_on_book=months_on_book,
            outstanding_balance=round(balance, 2),
            dpd=dpd,
            delinquency_bucket=bucket,
            stage=stage,
        )
    )

    closed_flag = 0
    default_flag = 0
    default_date: Optional[str] = None

    while True:
        months_on_book += 1
        snapshot_date = _reporting_date(seed.origination_date, months_on_book)
        if snapshot_date > as_of_date:
            break

        # Voluntary prepayment / payoff, only possible while Current.
        if bucket_idx == 0 and not is_revolving and payments_made > 0:
            if rng.random() < CLOSURE_PROB_PER_MONTH:
                closed_flag = 1
                break

        # Natural maturity for installment products.
        if not is_revolving and payments_made >= seed.term_months and bucket_idx == 0:
            closed_flag = 1
            break

        probs = _bucket_transition_probs(bucket_idx, seed.risk_score, seed.product)
        options = list(probs.keys())
        weights = np.array(list(probs.values()))
        weights = weights / weights.sum()
        bucket_idx = int(rng.choice(options, p=weights))
        target_bucket = ["Current", "1-29", "30-59", "60-89", "90+"][bucket_idx]

        if target_bucket == "Current":
            dpd = 0
            payments_made += 1
        else:
            lo, hi = DPD_RANGE_BY_BUCKET[target_bucket]
            dpd = int(rng.integers(lo, hi + 1))

        # bucket_from_dpd is the single source of truth for bucket labelling
        # (also used by risk.py and the dashboard); it must agree with the
        # bucket implied by the Markov transition we just sampled.
        bucket = bucket_from_dpd(dpd)
        assert bucket == target_bucket, "DPD sampling out of sync with bucket transition"

        # Balance roll-forward.
        if is_revolving:
            if bucket == "Current":
                payment_rate = float(rng.uniform(0.05, 0.25))
                spend_rate = float(rng.uniform(0.0, 0.15))
                balance = balance * (1 - payment_rate) + seed.principal * spend_rate
                balance = float(np.clip(balance, 0.0, seed.principal))
            else:
                balance = float(np.clip(balance * (1 + monthly_rate), 0.0, seed.principal * 1.05))
        else:
            if bucket == "Current":
                balance = _amortized_balance(seed.principal, seed.interest_rate, seed.term_months, payments_made)
            else:
                balance = balance * (1 + monthly_rate)

        stage = stage_from_dpd(dpd)
        snapshots.append(
            dict(
                loan_id=seed.loan_id,
                snapshot_date=snapshot_date.isoformat(),
                months_on_book=months_on_book,
                outstanding_balance=round(balance, 2),
                dpd=dpd,
                delinquency_bucket=bucket,
                stage=stage,
            )
        )

        if bucket == "90+":
            default_flag = 1
            default_date = snapshot_date.isoformat()
            break

    loan_outcome = dict(
        closed_flag=closed_flag,
        default_flag=default_flag,
        default_date=default_date,
    )
    return snapshots, loan_outcome


def generate_loan_book(
    n_loans: int = 3500,
    as_of_date: date = date(2026, 6, 30),
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate a full synthetic loan book. Returns (loans_df, snapshots_df)."""
    rng = np.random.default_rng(seed)
    loan_seeds = _generate_loan_seeds(n_loans, as_of_date, rng)

    loan_rows = []
    all_snapshots: list[dict] = []

    for seed_loan in loan_seeds:
        snapshots, outcome = _simulate_loan(seed_loan, as_of_date, rng)
        all_snapshots.extend(snapshots)
        loan_rows.append(
            dict(
                loan_id=seed_loan.loan_id,
                product=seed_loan.product,
                geography=seed_loan.geography,
                origination_date=seed_loan.origination_date.isoformat(),
                vintage_quarter=_quarter_label(seed_loan.origination_date),
                term_months=seed_loan.term_months,
                principal=round(seed_loan.principal, 2),
                interest_rate=round(seed_loan.interest_rate, 4),
                credit_score=seed_loan.credit_score,
                borrower_income=round(seed_loan.borrower_income, 2),
                channel=seed_loan.channel,
                closed_flag=outcome["closed_flag"],
                default_flag=outcome["default_flag"],
                default_date=outcome["default_date"],
            )
        )

    loans_df = pd.DataFrame(loan_rows)
    snapshots_df = pd.DataFrame(all_snapshots)
    return loans_df, snapshots_df


def build_database(
    db_path: Path | str = DEFAULT_DB_PATH,
    n_loans: int = 3500,
    as_of_date: date = date(2026, 6, 30),
    seed: int = 42,
) -> dict:
    """Generate a synthetic loan book and (re)load it into a SQLite database.

    Returns a small dict of summary counts, useful for CLI feedback and for
    tests that want to sanity-check the build without re-reading the DB.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    loans_df, snapshots_df = generate_loan_book(n_loans=n_loans, as_of_date=as_of_date, seed=seed)

    conn = sqlite3.connect(str(db_path))
    try:
        db.run_script("schema.sql", conn)
        loans_df.to_sql("loans", conn, if_exists="append", index=False)
        snapshots_df.to_sql("loan_snapshots", conn, if_exists="append", index=False)
        db.run_script("view_delinquency_buckets.sql", conn)
        conn.commit()
    finally:
        conn.close()

    return {
        "db_path": str(db_path),
        "n_loans": len(loans_df),
        "n_snapshots": len(snapshots_df),
        "n_defaulted": int(loans_df["default_flag"].sum()),
        "n_closed": int(loans_df["closed_flag"].sum()),
        "as_of_date": as_of_date.isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the synthetic loan_book.db SQLite database.")
    parser.add_argument("--n-loans", type=int, default=3500, help="Number of loans to simulate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--as-of-date",
        type=str,
        default="2026-06-30",
        help="Portfolio as-of date (ISO format, YYYY-MM-DD).",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=str(DEFAULT_DB_PATH),
        help="Output SQLite database path.",
    )
    args = parser.parse_args()

    as_of = date.fromisoformat(args.as_of_date)
    summary = build_database(db_path=args.db_path, n_loans=args.n_loans, as_of_date=as_of, seed=args.seed)

    print("Loan book ETL complete:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
