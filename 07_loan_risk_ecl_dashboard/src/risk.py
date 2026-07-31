"""
risk.py
Core risk-analytics engine: delinquency bucketing, IFRS 9 / Ind AS 109-style
staging, empirical roll-rate (transition) matrices, PD derivation from those
roll rates via absorbing Markov-chain algebra, and Expected Credit Loss
(ECL = PD * LGD * EAD).

Nothing in this module is a hardcoded output number -- every statistic is
computed from whatever loan/snapshot data is passed in, so re-running the
ETL with a different random seed, date range, or portfolio mix changes the
downstream KRIs, roll rates, PDs and ECL accordingly.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Delinquency buckets & staging
# ---------------------------------------------------------------------------

BUCKET_ORDER = ["Current", "1-29", "30-59", "60-89", "90+"]

#: Default LGD (Loss Given Default) assumptions by product, expressed as a
#: fraction of EAD expected to be lost after default/recoveries. These are
#: *assumptions*, deliberately kept configurable (the Streamlit app exposes
#: them as sliders for stress testing) rather than derived from the data,
#: which mirrors how LGD is typically set by credit policy / collateral
#: haircut tables in a real Ind AS 109 / IFRS 9 model.
DEFAULT_LGD_ASSUMPTIONS = {
    "mortgage": 0.35,       # secured by property -> lower loss severity
    "auto_loan": 0.45,      # secured by vehicle, faster depreciation
    "personal_loan": 0.65,  # unsecured
    "credit_card": 0.80,    # unsecured, revolving, thin recoveries
}

#: Rebuttable-presumption DPD threshold for Stage 2 (significant increase in
#: credit risk) under IFRS 9 / Ind AS 109 is 30 DPD; Stage 3 (credit-impaired
#: / default) is 90 DPD. Kept as module constants so both the ETL loader and
#: the dashboard stay consistent, and so tests can assert against them
#: directly rather than magic numbers.
STAGE2_DPD_THRESHOLD = 30
STAGE3_DPD_THRESHOLD = 90


def bucket_from_dpd(dpd: int) -> str:
    """Map a days-past-due value to a delinquency bucket label."""
    if dpd is None or pd.isna(dpd):
        raise ValueError("dpd must not be null")
    dpd = int(dpd)
    if dpd < 0:
        raise ValueError("dpd must be >= 0")
    if dpd == 0:
        return "Current"
    if dpd <= 29:
        return "1-29"
    if dpd <= 59:
        return "30-59"
    if dpd <= 89:
        return "60-89"
    return "90+"


def stage_from_dpd(dpd: int, sicr_flag: bool = False) -> int:
    """
    Simplified IFRS 9 / Ind AS 109 staging based on days past due, with an
    optional qualitative "significant increase in credit risk" (SICR)
    overlay (e.g. from the early-warning ML model) that can push an
    otherwise-Stage-1 loan into Stage 2 even before it is 30 DPD.

    Stage 1: performing, 0-29 DPD, no SICR flag -> 12-month ECL.
    Stage 2: 30-89 DPD, or SICR-flagged -> lifetime ECL.
    Stage 3: 90+ DPD -> credit-impaired / default -> lifetime ECL, PD = 1.
    """
    if dpd is None or pd.isna(dpd):
        raise ValueError("dpd must not be null")
    dpd = int(dpd)
    if dpd < 0:
        raise ValueError("dpd must be >= 0")
    if dpd >= STAGE3_DPD_THRESHOLD:
        return 3
    if dpd >= STAGE2_DPD_THRESHOLD or sicr_flag:
        return 2
    return 1


def add_bucket_and_stage(df: pd.DataFrame, dpd_col: str = "dpd") -> pd.DataFrame:
    """Vectorised helper: add delinquency_bucket / stage columns to a frame."""
    out = df.copy()
    out["delinquency_bucket"] = out[dpd_col].apply(bucket_from_dpd)
    out["stage"] = out[dpd_col].apply(lambda d: stage_from_dpd(d))
    return out


# ---------------------------------------------------------------------------
# Roll-rate / transition matrix
# ---------------------------------------------------------------------------

def compute_transition_matrix(
    pairs: pd.DataFrame,
    bucket_order: list[str] = BUCKET_ORDER,
    from_col: str = "bucket_from",
    to_col: str = "bucket_to",
) -> pd.DataFrame:
    """
    Build the empirical monthly roll-rate matrix from a table of
    (bucket_from, bucket_to) loan-month transition pairs (see
    sql/roll_rate_pairs.sql). Row *i*, column *j* is the empirical
    probability that a loan in bucket *i* this month is in bucket *j* next
    month. Rows sum to 1.

    "90+" is treated as absorbing: in this dataset, loans are removed from
    the panel once they first reach 90+ DPD (charge-off), so there is no
    empirical bucket_from == '90+' data; we set P(90+ -> 90+) = 1 to reflect
    that a defaulted loan stays in default for PD-estimation purposes.
    """
    if pairs.empty:
        raise ValueError("pairs frame is empty; cannot compute transition matrix")

    counts = pd.crosstab(pairs[from_col], pairs[to_col])
    counts = counts.reindex(index=bucket_order, columns=bucket_order, fill_value=0)

    row_sums = counts.sum(axis=1)
    with np.errstate(invalid="ignore"):
        matrix = counts.div(row_sums.replace(0, np.nan), axis=0)
    matrix = matrix.fillna(0.0)

    # Enforce the absorbing default state whenever we have no observed
    # outgoing transitions from 90+ (the expected case given the ETL design).
    if matrix.loc["90+"].sum() == 0:
        matrix.loc["90+", :] = 0.0
        matrix.loc["90+", "90+"] = 1.0

    matrix.index.name = "bucket_from"
    matrix.columns.name = "bucket_to"
    return matrix


def pd_from_transition_matrix(
    matrix: pd.DataFrame,
    horizon_months: int = 12,
    bucket_order: list[str] = BUCKET_ORDER,
    default_bucket: str = "90+",
) -> pd.Series:
    """
    Cumulative probability of a loan currently in each bucket being in the
    default bucket after ``horizon_months`` months, computed by raising the
    transition matrix to the ``horizon_months``-th power (Chapman-Kolmogorov
    / matrix-power method for Markov chains). This is the 12-month PD used
    for Stage 1 loans.
    """
    m = matrix.reindex(index=bucket_order, columns=bucket_order).fillna(0.0).to_numpy()
    m_power = np.linalg.matrix_power(m, horizon_months)
    default_idx = bucket_order.index(default_bucket)
    return pd.Series(m_power[:, default_idx], index=bucket_order, name=f"pd_{horizon_months}m")


def lifetime_pd_from_transition_matrix(
    matrix: pd.DataFrame,
    bucket_order: list[str] = BUCKET_ORDER,
    default_bucket: str = "90+",
) -> pd.Series:
    """
    Lifetime (full-horizon) probability of ultimately being absorbed into
    the default state, solved analytically for an absorbing Markov chain:

        N = (I - Q)^-1        (fundamental matrix; Q = transient->transient
                                sub-matrix)
        B = N @ R              (R = transient->absorbing sub-matrix)

    B[i] is the probability that a loan currently in transient bucket *i*
    is eventually absorbed into the default state. This is the lifetime PD
    used for Stage 2 (and, trivially, Stage 3, whose loans are already in
    the default bucket, so PD = 1).

    Falls back to a long-horizon matrix-power approximation if (I - Q) is
    singular (degenerate matrices, e.g. all-zero transitions in a tiny
    test fixture).
    """
    order = list(bucket_order)
    m = matrix.reindex(index=order, columns=order).fillna(0.0).to_numpy()
    absorb_idx = order.index(default_bucket)
    transient_idx = [i for i in range(len(order)) if i != absorb_idx]

    q = m[np.ix_(transient_idx, transient_idx)]
    r = m[np.ix_(transient_idx, [absorb_idx])]
    identity = np.eye(len(transient_idx))

    try:
        fundamental = np.linalg.inv(identity - q)
        absorption = fundamental @ r
    except np.linalg.LinAlgError:
        m_power = np.linalg.matrix_power(m, 360)  # 30-year approximation
        absorption = m_power[np.ix_(transient_idx, [absorb_idx])]

    result = pd.Series(1.0, index=order, name="pd_lifetime")
    for pos, idx in enumerate(transient_idx):
        result.iloc[idx] = float(np.clip(absorption[pos, 0], 0.0, 1.0))
    return result


# ---------------------------------------------------------------------------
# Expected Credit Loss
# ---------------------------------------------------------------------------

def compute_ecl(
    portfolio: pd.DataFrame,
    matrix: pd.DataFrame,
    lgd_by_product: Optional[dict] = None,
    default_lgd: float = 0.60,
    pd_horizon_months: int = 12,
    bucket_col: str = "delinquency_bucket",
    stage_col: str = "stage",
    product_col: str = "product",
    ead_col: str = "outstanding_balance",
) -> pd.DataFrame:
    """
    Compute loan-level ECL = PD * LGD * EAD under a simplified Ind AS 109 /
    IFRS 9 framework:

      * Stage 1 loans use the 12-month PD (matrix-power method).
      * Stage 2 loans use the lifetime PD (absorbing Markov-chain method).
      * Stage 3 loans are already in default: PD is set to 1.0 and EAD/LGD
        alone drive the loss estimate.

    ``lgd_by_product`` lets callers override the LGD assumption per product
    (e.g. from stress-scenario sliders in the dashboard); any product not
    present in the dict falls back to ``default_lgd``.

    Returns a copy of ``portfolio`` with pd_12m, pd_lifetime, pd_applied,
    lgd_applied, ead and ecl columns appended.
    """
    lgd_map = dict(DEFAULT_LGD_ASSUMPTIONS)
    if lgd_by_product:
        lgd_map.update(lgd_by_product)

    pd_12m = pd_from_transition_matrix(matrix, horizon_months=pd_horizon_months)
    pd_lifetime = lifetime_pd_from_transition_matrix(matrix)

    out = portfolio.copy()
    out["pd_12m"] = out[bucket_col].map(pd_12m)
    out["pd_lifetime"] = out[bucket_col].map(pd_lifetime)

    conditions = [out[stage_col] == 1, out[stage_col] == 2, out[stage_col] == 3]
    choices = [out["pd_12m"], out["pd_lifetime"], 1.0]
    out["pd_applied"] = np.select(conditions, choices, default=out["pd_12m"])
    out["pd_applied"] = out["pd_applied"].clip(0.0, 1.0)

    out["lgd_applied"] = out[product_col].map(lgd_map).fillna(default_lgd).clip(0.0, 1.0)
    out["ead"] = out[ead_col].clip(lower=0.0)
    out["ecl"] = out["pd_applied"] * out["lgd_applied"] * out["ead"]
    return out


def ecl_summary(ecl_df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Aggregate loan-level ECL output up to a cohort level (e.g. by stage,
    product, geography, vintage_quarter -- any combination of columns
    present in ``ecl_df``)."""
    agg = (
        ecl_df.groupby(by, dropna=False)
        .agg(
            loan_count=("loan_id", "count") if "loan_id" in ecl_df.columns else ("ecl", "count"),
            total_ead=("ead", "sum"),
            total_ecl=("ecl", "sum"),
            avg_pd_applied=("pd_applied", "mean"),
            avg_lgd_applied=("lgd_applied", "mean"),
        )
        .reset_index()
    )
    agg["ecl_coverage_ratio"] = agg["total_ecl"] / agg["total_ead"].replace(0, np.nan)
    return agg
