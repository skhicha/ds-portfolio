"""
Synthetic physician / prescriber dataset generator for the Pharma Customer
Segmentation & Targeting Analytics project.

The dataset models a single branded cardiometabolic drug and the therapeutic
class it competes in. It is NOT pure noise with labels bolted on: each
physician is first assigned one of five hidden behavioral ARCHETYPES (never
written to the output CSV) that drives correlated, realistic values across
prescribing volume, company product share, and HCP engagement. This means a
correctly specified pipeline (src/segment.py) can genuinely recover
meaningful, separable clusters from the data alone - the resulting
silhouette score reported by the app/report is earned, not hardcoded.

Run directly to (re)write data/physicians.csv:

    python -m src.generate_data
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

SPECIALTIES = ["Endocrinology", "Primary Care", "Internal Medicine", "Cardiology", "Nephrology"]
SPECIALTY_WEIGHTS = [0.22, 0.30, 0.20, 0.18, 0.10]

# Relative likelihood a physician of this specialty treats the relevant
# cardiometabolic condition at all, and how large their relevant patient
# population tends to be (used to scale total market Rx volume).
SPECIALTY_RELEVANCE = {
    "Endocrinology": 1.6,
    "Primary Care": 1.1,
    "Internal Medicine": 1.0,
    "Cardiology": 0.9,
    "Nephrology": 0.8,
}

REGIONS = ["Northeast", "Southeast", "Midwest", "Southwest", "West", "Northwest"]
PRACTICE_SETTINGS = ["Private Practice", "Hospital-Affiliated", "Academic Medical Center", "Community Health Clinic"]
PRACTICE_SETTING_WEIGHTS = [0.40, 0.32, 0.13, 0.15]

# Hidden ground-truth behavioral archetypes. Not exported - the segmentation
# pipeline only ever sees the observable columns below and has to rediscover
# this structure itself via PCA + K-Means.
ARCHETYPES = [
    "high_value_loyalist",
    "high_potential_low_share",
    "digital_first_emerging",
    "low_priority_steady",
    "declining_at_risk",
]
ARCHETYPE_WEIGHTS = [0.20, 0.22, 0.18, 0.25, 0.15]

ARCHETYPE_PARAMS = {
    "high_value_loyalist":      dict(trx_mult=1.35, share_mean=0.42, share_std=0.08, engage_mean=72, engage_std=10, visits_mean=11, visits_std=3.0, tenure_mean=7.5, tenure_std=2.5, peer_mean=62, peer_std=14),
    "high_potential_low_share": dict(trx_mult=1.55, share_mean=0.12, share_std=0.05, engage_mean=48, engage_std=12, visits_mean=6,  visits_std=2.5, tenure_mean=2.0, tenure_std=1.3, peer_mean=45, peer_std=16),
    "digital_first_emerging":   dict(trx_mult=0.85, share_mean=0.24, share_std=0.08, engage_mean=81, engage_std=9,  visits_mean=3,  visits_std=1.8, tenure_mean=1.5, tenure_std=1.0, peer_mean=38, peer_std=15),
    "low_priority_steady":      dict(trx_mult=0.55, share_mean=0.18, share_std=0.06, engage_mean=30, engage_std=11, visits_mean=4,  visits_std=2.0, tenure_mean=4.5, tenure_std=2.0, peer_mean=28, peer_std=12),
    "declining_at_risk":        dict(trx_mult=0.95, share_mean=0.15, share_std=0.06, engage_mean=22, engage_std=9,  visits_mean=2,  visits_std=1.5, tenure_mean=8.0, tenure_std=2.5, peer_mean=40, peer_std=16),
}

RX_UNIT_PRICE = 410.0  # average net price per prescription, $ (synthetic)
PREVALENCE_RATE = 0.07  # fraction of a physician's panel relevant to the drug class


def _clip(arr: np.ndarray, lo: float, hi: float | None) -> np.ndarray:
    return np.clip(arr, lo, hi)


def generate_dataset(n_physicians: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Build a synthetic physician/prescriber dataset.

    Returns a DataFrame with one row per physician and columns covering
    specialty/region/practice attributes, patient panel demographics,
    prescribing volume (total market + company product), and HCP engagement
    signals (digital engagement, rep visits, samples, peer influence).

    `n_physicians <= 0` returns an empty DataFrame with the correct columns.
    """
    columns = [
        "physician_id", "specialty", "region", "practice_setting",
        "years_in_practice", "patient_panel_size", "pct_patients_65_plus",
        "pct_female_patients", "avg_comorbidity_score",
        "total_market_trx_monthly", "company_trx_monthly",
        "company_nrx_monthly", "company_rx_value_monthly", "market_share_pct",
        "digital_engagement_score", "rep_visits_last_12mo",
        "samples_distributed_last_12mo", "peer_influence_score",
        "years_prescribing_company_product",
    ]
    if n_physicians <= 0:
        return pd.DataFrame(columns=columns)

    rng = np.random.default_rng(seed)
    n = n_physicians

    specialty = rng.choice(SPECIALTIES, size=n, p=SPECIALTY_WEIGHTS)
    region = rng.choice(REGIONS, size=n)
    practice_setting = rng.choice(PRACTICE_SETTINGS, size=n, p=PRACTICE_SETTING_WEIGHTS)
    archetype = rng.choice(ARCHETYPES, size=n, p=ARCHETYPE_WEIGHTS)

    years_in_practice = _clip(rng.normal(15, 8, n), 1, 45).round(1)
    relevance = np.array([SPECIALTY_RELEVANCE[s] for s in specialty])

    panel_base = rng.normal(1800, 550, n)
    setting_panel_mult = np.array([
        {"Private Practice": 1.0, "Hospital-Affiliated": 1.15,
         "Academic Medical Center": 0.85, "Community Health Clinic": 1.35}[s]
        for s in practice_setting
    ])
    patient_panel_size = _clip(panel_base * setting_panel_mult, 250, 6500).round(0).astype(int)

    pct_patients_65_plus = _clip(0.18 + 0.006 * years_in_practice + rng.normal(0, 0.07, n), 0.03, 0.80)
    pct_female_patients = _clip(rng.normal(0.52, 0.09, n), 0.10, 0.92)
    avg_comorbidity_score = _clip(
        rng.normal(2.3, 0.9, n)
        + np.where(specialty == "Nephrology", 1.0, 0.0)
        + np.where(specialty == "Endocrinology", 0.4, 0.0),
        0.2, 6.5,
    ).round(2)

    # ---- Archetype-driven behavioral parameters (hidden ground truth) ---- #
    def _arch_vec(key: str) -> np.ndarray:
        return np.array([ARCHETYPE_PARAMS[a][key] for a in archetype])

    trx_mult = _arch_vec("trx_mult")
    total_market_trx_monthly = _clip(
        patient_panel_size * PREVALENCE_RATE * relevance * trx_mult * rng.normal(1.0, 0.18, n),
        2, None,
    ).round(1)

    share_mean, share_std = _arch_vec("share_mean"), _arch_vec("share_std")
    market_share_pct = _clip(rng.normal(share_mean, share_std), 0.01, 0.90)
    company_trx_monthly = (total_market_trx_monthly * market_share_pct).round(1)

    nrx_ratio = _clip(rng.normal(0.22, 0.08, n), 0.03, 0.70)
    company_nrx_monthly = (company_trx_monthly * nrx_ratio).round(1)
    company_rx_value_monthly = (company_trx_monthly * RX_UNIT_PRICE * rng.normal(1.0, 0.06, n)).round(2)

    engage_mean, engage_std = _arch_vec("engage_mean"), _arch_vec("engage_std")
    digital_engagement_score = _clip(rng.normal(engage_mean, engage_std), 0, 100).round(1)

    visits_mean, visits_std = _arch_vec("visits_mean"), _arch_vec("visits_std")
    rep_visits_last_12mo = _clip(rng.normal(visits_mean, visits_std), 0, None).round(0).astype(int)
    samples_distributed_last_12mo = _clip(
        rep_visits_last_12mo * rng.normal(3.2, 1.1, n), 0, None
    ).round(0).astype(int)

    peer_mean, peer_std = _arch_vec("peer_mean"), _arch_vec("peer_std")
    peer_influence_score = _clip(rng.normal(peer_mean, peer_std), 0, 100).round(1)

    tenure_mean, tenure_std = _arch_vec("tenure_mean"), _arch_vec("tenure_std")
    years_prescribing_company_product = _clip(
        np.minimum(rng.normal(tenure_mean, tenure_std), years_in_practice), 0, None
    ).round(1)

    physician_id = [f"PHY{idx:05d}" for idx in range(1, n + 1)]

    df = pd.DataFrame({
        "physician_id": physician_id,
        "specialty": specialty,
        "region": region,
        "practice_setting": practice_setting,
        "years_in_practice": years_in_practice,
        "patient_panel_size": patient_panel_size,
        "pct_patients_65_plus": pct_patients_65_plus.round(3),
        "pct_female_patients": pct_female_patients.round(3),
        "avg_comorbidity_score": avg_comorbidity_score,
        "total_market_trx_monthly": total_market_trx_monthly,
        "company_trx_monthly": company_trx_monthly,
        "company_nrx_monthly": company_nrx_monthly,
        "company_rx_value_monthly": company_rx_value_monthly,
        "market_share_pct": market_share_pct.round(4),
        "digital_engagement_score": digital_engagement_score,
        "rep_visits_last_12mo": rep_visits_last_12mo,
        "samples_distributed_last_12mo": samples_distributed_last_12mo,
        "peer_influence_score": peer_influence_score,
        "years_prescribing_company_product": years_prescribing_company_product,
    })
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic physician/prescriber data.")
    parser.add_argument("--n", type=int, default=500, help="Number of physicians to generate (default 500).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory (default: <project>/data).")
    args = parser.parse_args()

    outdir = args.outdir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, "physicians.csv")

    df = generate_dataset(n_physicians=args.n, seed=args.seed)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows of synthetic physician data to {out_path}")
    print(df.head())
    print("\nNumeric summary:")
    print(df.describe())


if __name__ == "__main__":
    main()
