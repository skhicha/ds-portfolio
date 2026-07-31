"""
Synthetic weekly sales data generator for the Marketing Mix Modeling tool.

The generated series is NOT a random walk with a spend column bolted on -
weekly_sales is deliberately constructed as:

    sales(t) = baseline(t) + trend(t) + seasonality(t)
               + sum_channel[ beta_c * saturation(adstock(spend_c(t))) ]
               + noise(t)

using true (hidden) adstock decay rates and saturation curves per channel.
This means a correctly specified model (see src/mmm.py) can genuinely
recover a high R^2 on this data - the fit quality is earned, not hardcoded.

Run directly to (re)write data/weekly_sales.csv:

    python -m src.generate_data
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# "True" data-generating parameters (unknown to the modeling pipeline).
# The pipeline in src/mmm.py re-discovers similar dynamics purely by fitting
# on the observed spend/sales columns - it never reads this dictionary.
# --------------------------------------------------------------------------
TRUE_PARAMS = {
    "tv_spend": {"decay": 0.60, "gamma_frac": 0.55, "beta": 42000.0},
    "digital_spend": {"decay": 0.35, "gamma_frac": 0.50, "beta": 38000.0},
    "promotions_spend": {"decay": 0.15, "gamma_frac": 0.45, "beta": 30000.0},
}

BASELINE_SALES = 55000.0
TREND_PER_WEEK = 25.0
SEASONAL_AMPLITUDE_1 = 9000.0
SEASONAL_AMPLITUDE_2 = 3500.0
NOISE_STD = 3200.0


def _geometric_adstock(spend: np.ndarray, decay: float) -> np.ndarray:
    """Geometric adstock: today's effect = spend + decay * yesterday's effect."""
    out = np.zeros_like(spend, dtype=float)
    out[0] = spend[0]
    for t in range(1, len(spend)):
        out[t] = spend[t] + decay * out[t - 1]
    return out


def _hill_saturation(x: np.ndarray, gamma: float, alpha: float = 1.0) -> np.ndarray:
    """Diminishing-returns saturation curve, bounded in [0, 1)."""
    x = np.clip(x, 0, None)
    return x ** alpha / (x ** alpha + gamma ** alpha)


def generate_weekly_spend(n_weeks: int, rng: np.random.Generator) -> pd.DataFrame:
    """Simulate weekly spend levels per channel with campaign-like bursts."""
    weeks = np.arange(n_weeks)

    # TV: chunky, campaign-style spend with occasional dark (zero-spend) weeks.
    tv_base = rng.uniform(4000, 22000, n_weeks)
    tv_campaign = (rng.random(n_weeks) < 0.35) * rng.uniform(8000, 20000, n_weeks)
    tv_dark = rng.random(n_weeks) < 0.10
    tv_spend = np.where(tv_dark, 0.0, tv_base + tv_campaign)

    # Digital: smoother, always-on spend with a mild upward trend
    # (budgets shifting to digital over time) plus weekly noise.
    digital_spend = (
        6000
        + 12.0 * weeks
        + 3500 * np.sin(2 * np.pi * weeks / 13 + 0.5)
        + rng.normal(0, 1800, n_weeks)
    )
    digital_spend = np.clip(digital_spend, 500, None)

    # Promotions: periodic promo pushes (e.g. every ~6-8 weeks) with a
    # discount-depth-driven spend proxy.
    promo_flag = np.zeros(n_weeks, dtype=bool)
    i = int(rng.integers(2, 6))
    while i < n_weeks:
        promo_flag[i] = True
        i += int(rng.integers(5, 9))
    promotions_spend = np.where(
        promo_flag, rng.uniform(9000, 26000, n_weeks), rng.uniform(0, 1500, n_weeks)
    )

    return pd.DataFrame(
        {
            "tv_spend": np.round(tv_spend, 2),
            "digital_spend": np.round(digital_spend, 2),
            "promotions_spend": np.round(promotions_spend, 2),
        }
    )


def generate_dataset(n_weeks: int = 156, seed: int = 42, start_date: str = "2023-01-02") -> pd.DataFrame:
    """
    Build a synthetic weekly sales dataset (default: 156 weeks ~= 3 years).

    Returns a DataFrame with columns:
        week_start_date, tv_spend, digital_spend, promotions_spend, weekly_sales
    """
    rng = np.random.default_rng(seed)

    dates = pd.date_range(start=start_date, periods=n_weeks, freq="W-MON")
    spend_df = generate_weekly_spend(n_weeks, rng)

    week_of_year = dates.isocalendar().week.to_numpy(dtype=float)
    t = np.arange(n_weeks, dtype=float)

    seasonality = SEASONAL_AMPLITUDE_1 * np.sin(2 * np.pi * week_of_year / 52.0) + (
        SEASONAL_AMPLITUDE_2 * np.cos(4 * np.pi * week_of_year / 52.0)
    )
    trend = BASELINE_SALES + TREND_PER_WEEK * t

    contribution_total = np.zeros(n_weeks)
    channel_contributions = {}
    for channel, params in TRUE_PARAMS.items():
        spend = spend_df[channel].to_numpy(dtype=float)
        adstocked = _geometric_adstock(spend, params["decay"])
        median_pos = np.median(adstocked[adstocked > 0]) if np.any(adstocked > 0) else 1.0
        gamma = params["gamma_frac"] * median_pos
        saturated = _hill_saturation(adstocked, gamma=gamma)
        contrib = params["beta"] * saturated
        channel_contributions[channel] = contrib
        contribution_total += contrib

    noise = rng.normal(0, NOISE_STD, n_weeks)

    weekly_sales = trend + seasonality + contribution_total + noise
    weekly_sales = np.round(np.clip(weekly_sales, 5000, None), 2)

    df = pd.DataFrame(
        {
            "week_start_date": dates.strftime("%Y-%m-%d"),
            "tv_spend": spend_df["tv_spend"],
            "digital_spend": spend_df["digital_spend"],
            "promotions_spend": spend_df["promotions_spend"],
            "weekly_sales": weekly_sales,
        }
    )
    return df


def main() -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "weekly_sales.csv")

    df = generate_dataset()
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows of synthetic weekly data to {out_path}")
    print(df.head())
    print("\nSpend summary:")
    print(df[["tv_spend", "digital_spend", "promotions_spend", "weekly_sales"]].describe())


if __name__ == "__main__":
    main()
