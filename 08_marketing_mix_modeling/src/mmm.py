"""
Core Marketing Mix Modeling pipeline.

Implements:
    - geometric adstock (carryover) transform
    - Hill-style saturation (diminishing returns) transform
    - a Ridge regression model fit on transformed spend + seasonality/trend
      features to predict weekly_sales
    - lightweight hyperparameter search (adstock decay / saturation gamma)
      that maximizes in-sample R^2, so the reported fit quality is genuinely
      earned by optimizing over the transform parameters rather than reusing
      the (unknown-to-the-pipeline) data-generating constants
    - channel contribution decomposition
    - budget scenario simulation

Everything here operates on plain pandas/numpy structures so it can be
reused identically from the CLI, the test suite, and the Streamlit app.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error

DEFAULT_CHANNELS = ["tv_spend", "digital_spend", "promotions_spend"]


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def apply_adstock(spend: np.ndarray, decay_rate: float) -> np.ndarray:
    """
    Geometric adstock / carryover transform.

    adstock[0] = spend[0]
    adstock[t] = spend[t] + decay_rate * adstock[t-1]     for t > 0

    `decay_rate` must be in [0, 1). 0 means no carryover (adstock == spend);
    values close to 1 mean spend effect lingers for many weeks.
    """
    spend = np.asarray(spend, dtype=float)
    if not (0.0 <= decay_rate < 1.0):
        raise ValueError("decay_rate must be in [0, 1)")
    adstocked = np.zeros_like(spend)
    adstocked[0] = spend[0]
    for t in range(1, len(spend)):
        adstocked[t] = spend[t] + decay_rate * adstocked[t - 1]
    return adstocked


def apply_saturation(x: np.ndarray, gamma: float, alpha: float = 1.0) -> np.ndarray:
    """
    Hill-style saturation / diminishing-returns transform, bounded in [0, 1).

        f(x) = x^alpha / (x^alpha + gamma^alpha)

    - Monotonically non-decreasing in x for x >= 0, alpha > 0, gamma > 0.
    - f(0) == 0, f(x) -> 1 as x -> infinity.
    - `gamma` is the half-saturation point: f(gamma) == 0.5.
    - `alpha` == 1 gives a simple concave diminishing-returns curve;
      alpha > 1 gives an S-shaped curve (slow start, then acceleration,
      then diminishing returns) as used in tools like Meta Robyn.
    """
    x = np.asarray(x, dtype=float)
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    if alpha <= 0:
        raise ValueError("alpha must be > 0")
    x_clipped = np.clip(x, 0, None)
    return x_clipped ** alpha / (x_clipped ** alpha + gamma ** alpha)


def transform_channel(spend: np.ndarray, decay: float, gamma: float, alpha: float = 1.0) -> np.ndarray:
    """Adstock then saturate a single channel's raw spend series."""
    return apply_saturation(apply_adstock(spend, decay), gamma=gamma, alpha=alpha)


# --------------------------------------------------------------------------- #
# Seasonality / trend features
# --------------------------------------------------------------------------- #
def build_seasonality_features(dates: pd.Series) -> pd.DataFrame:
    """Fourier-style yearly seasonality terms plus a linear time trend."""
    dates = pd.to_datetime(dates)
    week_of_year = dates.dt.isocalendar().week.to_numpy(dtype=float)
    t = np.arange(len(dates), dtype=float)
    return pd.DataFrame(
        {
            "trend": t,
            "season_sin_1": np.sin(2 * np.pi * week_of_year / 52.0),
            "season_cos_1": np.cos(2 * np.pi * week_of_year / 52.0),
            "season_sin_2": np.sin(4 * np.pi * week_of_year / 52.0),
            "season_cos_2": np.cos(4 * np.pi * week_of_year / 52.0),
        },
        index=dates.index,
    )


# --------------------------------------------------------------------------- #
# Fitted model container
# --------------------------------------------------------------------------- #
@dataclass
class ChannelParams:
    decay: float
    gamma: float
    alpha: float = 1.0


@dataclass
class MMMResult:
    channels: list
    channel_params: Dict[str, ChannelParams]
    ridge_alpha: float
    model: Ridge
    feature_names: list
    coefficients: Dict[str, float]
    intercept: float
    r2: float
    mae: float
    mape: float
    fitted_values: np.ndarray
    actuals: np.ndarray
    dates: pd.Series
    contributions: pd.DataFrame  # per-row contribution of each channel + baseline
    raw_spend: pd.DataFrame

    def summary_dict(self) -> dict:
        return {
            "r2": self.r2,
            "mae": self.mae,
            "mape": self.mape,
            "n_observations": len(self.actuals),
            "channels": self.channels,
            "ridge_alpha": self.ridge_alpha,
            "channel_params": {
                c: {"decay": p.decay, "gamma": p.gamma, "alpha": p.alpha}
                for c, p in self.channel_params.items()
            },
        }


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #
def _build_design_matrix(
    df: pd.DataFrame,
    channels: Iterable[str],
    params: Dict[str, ChannelParams],
) -> pd.DataFrame:
    seasonality = build_seasonality_features(df["week_start_date"])
    transformed = {}
    for ch in channels:
        p = params[ch]
        transformed[ch] = transform_channel(df[ch].to_numpy(), p.decay, p.gamma, p.alpha)
    transformed_df = pd.DataFrame(transformed, index=df.index)
    return pd.concat([transformed_df, seasonality], axis=1)


def _params_from_vector(channels, theta: np.ndarray, gamma_scale: Dict[str, float], alpha: float) -> Dict[str, ChannelParams]:
    params = {}
    n = len(channels)
    decays = np.clip(theta[:n], 0.0, 0.95)
    gamma_fracs = np.clip(theta[n:], 0.05, 3.0)
    for i, ch in enumerate(channels):
        gamma = max(gamma_fracs[i] * gamma_scale[ch], 1e-6)
        params[ch] = ChannelParams(decay=float(decays[i]), gamma=float(gamma), alpha=alpha)
    return params


def fit_mmm(
    df: pd.DataFrame,
    channels: Optional[Iterable[str]] = None,
    sales_col: str = "weekly_sales",
    ridge_alpha: float = 0.1,
    saturation_shape: float = 1.0,
    optimize_transforms: bool = True,
    n_restarts: int = 3,
    random_state: int = 7,
) -> MMMResult:
    """
    Fit the full MMM pipeline: search adstock decay + saturation gamma per
    channel (if optimize_transforms=True) to maximize R^2, transform the
    spend columns, then fit a Ridge regression against weekly_sales.
    """
    channels = list(channels) if channels is not None else DEFAULT_CHANNELS
    df = df.reset_index(drop=True).copy()
    y = df[sales_col].to_numpy(dtype=float)

    gamma_scale = {}
    for ch in channels:
        pos_vals = df[ch].to_numpy(dtype=float)
        pos_vals = pos_vals[pos_vals > 0]
        gamma_scale[ch] = float(np.median(pos_vals)) if len(pos_vals) else 1.0

    def objective(theta: np.ndarray) -> float:
        params = _params_from_vector(channels, theta, gamma_scale, saturation_shape)
        X = _build_design_matrix(df, channels, params)
        model = Ridge(alpha=ridge_alpha)
        model.fit(X.to_numpy(), y)
        preds = model.predict(X.to_numpy())
        sse = np.sum((y - preds) ** 2)
        return sse

    n = len(channels)
    if optimize_transforms:
        rng = np.random.default_rng(random_state)
        best_theta = None
        best_sse = np.inf
        starts = [np.concatenate([np.full(n, 0.4), np.full(n, 0.6)])]
        for _ in range(n_restarts - 1):
            starts.append(
                np.concatenate(
                    [rng.uniform(0.05, 0.85, n), rng.uniform(0.2, 1.5, n)]
                )
            )
        for x0 in starts:
            res = minimize(objective, x0, method="Nelder-Mead",
                            options={"maxiter": 400, "xatol": 1e-3, "fatol": 1.0})
            if res.fun < best_sse:
                best_sse = res.fun
                best_theta = res.x
        theta = best_theta
    else:
        theta = np.concatenate([np.full(n, 0.5), np.full(n, 0.6)])

    channel_params = _params_from_vector(channels, theta, gamma_scale, saturation_shape)

    X = _build_design_matrix(df, channels, channel_params)
    feature_names = list(X.columns)
    model = Ridge(alpha=ridge_alpha)
    model.fit(X.to_numpy(), y)
    fitted = model.predict(X.to_numpy())

    r2 = r2_score(y, fitted)
    mae = mean_absolute_error(y, fitted)
    mape = mean_absolute_percentage_error(y, fitted)

    coefficients = dict(zip(feature_names, model.coef_))

    # Contribution decomposition: coefficient * transformed feature value,
    # per row. Baseline = intercept + trend + seasonality contribution.
    contrib = {}
    for ch in channels:
        contrib[ch] = coefficients[ch] * X[ch].to_numpy()
    baseline = np.full(len(df), model.intercept_)
    for col in ["trend", "season_sin_1", "season_cos_1", "season_sin_2", "season_cos_2"]:
        baseline = baseline + coefficients[col] * X[col].to_numpy()
    contrib["baseline"] = baseline
    contributions = pd.DataFrame(contrib, index=df.index)

    return MMMResult(
        channels=channels,
        channel_params=channel_params,
        ridge_alpha=ridge_alpha,
        model=model,
        feature_names=feature_names,
        coefficients=coefficients,
        intercept=float(model.intercept_),
        r2=float(r2),
        mae=float(mae),
        mape=float(mape),
        fitted_values=fitted,
        actuals=y,
        dates=df["week_start_date"],
        contributions=contributions,
        raw_spend=df[channels].copy(),
    )


# --------------------------------------------------------------------------- #
# Contribution summary
# --------------------------------------------------------------------------- #
def contribution_summary(result: MMMResult) -> pd.DataFrame:
    """Aggregate per-row contributions into a total-and-share-of-sales table."""
    totals = result.contributions.sum(axis=0)
    total_sales = totals.sum()
    rows = []
    for name, value in totals.items():
        rows.append(
            {
                "component": name,
                "total_contribution": float(value),
                "share_of_sales": float(value / total_sales) if total_sales else 0.0,
            }
        )
    summary = pd.DataFrame(rows).sort_values("total_contribution", ascending=False).reset_index(drop=True)
    return summary


# --------------------------------------------------------------------------- #
# Budget scenario simulation
# --------------------------------------------------------------------------- #
def simulate_budget_scenario(
    result: MMMResult,
    weekly_budget: Dict[str, float],
    n_weeks: int = 13,
    ramp_up: bool = True,
) -> Dict[str, float]:
    """
    Predict weekly sales under a proposed *steady-state* weekly budget
    allocation, using the fitted adstock + saturation + Ridge model.

    A constant weekly spend level per channel is simulated for `n_weeks`
    (default 13 = one quarter) so that adstock carryover reaches
    (near) steady state, and the resulting predicted weekly sales in the
    final week (fully warmed up) is returned along with the full trajectory.
    """
    channels = result.channels
    for ch in weekly_budget:
        if ch not in channels:
            raise ValueError(f"Unknown channel '{ch}'. Expected one of {channels}")

    spend_series = {
        ch: np.full(n_weeks, float(weekly_budget.get(ch, 0.0))) for ch in channels
    }

    transformed = {}
    for ch in channels:
        p = result.channel_params[ch]
        transformed[ch] = transform_channel(spend_series[ch], p.decay, p.gamma, p.alpha)

    # Hold the baseline (intercept + trend + seasonality) at its recent
    # average level so the scenario isolates the effect of the budget
    # allocation itself, rather than re-simulating calendar effects.
    baseline_level = float(result.contributions["baseline"].tail(13).mean())

    predicted_trajectory = np.full(n_weeks, baseline_level)
    channel_contribs_final = {}
    for ch in channels:
        contrib = result.coefficients[ch] * transformed[ch]
        predicted_trajectory = predicted_trajectory + contrib
        channel_contribs_final[ch] = float(contrib[-1])

    total_budget = float(sum(weekly_budget.get(ch, 0.0) for ch in channels))
    predicted_sales_steady_state = float(predicted_trajectory[-1])

    return {
        "predicted_weekly_sales": predicted_sales_steady_state,
        "predicted_trajectory": predicted_trajectory.tolist(),
        "total_weekly_budget": total_budget,
        "channel_contributions": channel_contribs_final,
        "baseline_level": baseline_level,
        "implied_roi": (
            sum(channel_contribs_final.values()) / total_budget if total_budget > 0 else 0.0
        ),
    }


def compare_scenarios(result: MMMResult, scenarios: Dict[str, Dict[str, float]], n_weeks: int = 13) -> pd.DataFrame:
    """Run simulate_budget_scenario for several named scenarios and tabulate results."""
    rows = []
    for name, allocation in scenarios.items():
        sim = simulate_budget_scenario(result, allocation, n_weeks=n_weeks)
        row = {"scenario": name, "predicted_weekly_sales": sim["predicted_weekly_sales"],
               "total_weekly_budget": sim["total_weekly_budget"], "implied_roi": sim["implied_roi"]}
        for ch, val in allocation.items():
            row[f"budget_{ch}"] = val
        rows.append(row)
    return pd.DataFrame(rows)
