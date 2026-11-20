"""
Streamlit dashboard for PJME hourly energy consumption forecasting.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.data import (
    TARGET_COL,
    add_time_features,
    adf_test,
    load_raw,
    resample_daily_window,
    time_based_split,
)
from src.evaluate import mape, residual_diagnostics, rmse
from src.models import forecast_sarima, forecast_xgboost, train_sarima, train_xgboost

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "PJME_hourly.csv")

COLOR_ACTUAL = "#0369A1"
COLOR_FORECAST = "#D97706"
COLOR_BAND = "#D97706"
COLOR_RESIDUAL = "#BE123C"

st.set_page_config(page_title="PJME Energy Forecasting", page_icon="⚡", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --color-bg: #F3F7FA;
        --color-surface: #FFFFFF;
        --color-border: #E1E8EF;
        --color-text: #111827;
        --color-text-muted: #5B6472;
        --color-primary: #0369A1;
        --color-primary-soft: #E5F2FA;
        --color-accent: #D97706;
        --color-accent-soft: #FEF3E2;
        --shadow-card: 0 1px 2px rgba(16, 24, 40, 0.04), 0 6px 16px rgba(16, 24, 40, 0.06);
    }

    .block-container { padding-top: 2rem; max-width: 1280px; }

    div[data-testid="stMetric"] {
        background: var(--color-surface);
        border: 1px solid var(--color-border);
        border-radius: 12px;
        padding: 1rem 1.1rem 0.9rem 1.1rem;
        box-shadow: var(--shadow-card);
    }
    div[data-testid="stMetricLabel"] { color: var(--color-text-muted); font-weight: 600; }
    div[data-testid="stMetricValue"] { color: var(--color-primary); }

    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--color-text);
        margin: 0.25rem 0 0.1rem 0;
    }
    .section-caption { color: var(--color-text-muted); font-size: 0.92rem; margin-bottom: 0.9rem; }

    .pill {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .pill-stationary { background: var(--color-primary-soft); color: var(--color-primary); }
    .pill-nonstationary { background: var(--color-accent-soft); color: var(--color-accent); }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--color-border);
        border-radius: 10px;
        box-shadow: var(--shadow-card);
    }

    div[data-testid="stTabs"] button[data-baseweb="tab"] { font-weight: 600; }

    hr { border-color: var(--color-border); }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading and cleaning PJME_hourly.csv...")
def load_data(path: str) -> pd.DataFrame:
    return load_raw(path)


@st.cache_data(show_spinner="Running ADF stationarity tests...")
def run_adf(series: pd.Series):
    raw_result = adf_test(series, name="Raw series")
    diffed = series.diff().dropna()
    diff_result = adf_test(diffed, name="First difference")
    return raw_result, diff_result


@st.cache_resource(show_spinner="Training XGBoost forecaster...")
def run_xgboost(df_window: pd.DataFrame, test_frac: float):
    feat = add_time_features(df_window, TARGET_COL)
    X_train, y_train, X_test, y_test = time_based_split(feat, TARGET_COL, test_frac=test_frac)
    model = train_xgboost(X_train, y_train, X_test, y_test)
    result = forecast_xgboost(model, X_test, y_test)
    return result


@st.cache_resource(show_spinner="Fitting SARIMA baseline on daily data...")
def run_sarima(series: pd.Series, train_start: str, train_end: str, test_start: str, test_end: str):
    train_daily = resample_daily_window(series, train_start, train_end)
    test_daily = resample_daily_window(series, test_start, test_end)
    if len(train_daily) < 14 or len(test_daily) == 0:
        return None
    fit = train_sarima(train_daily, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    return forecast_sarima(fit, test_daily)


def section(title: str, caption: str = "") -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="section-caption">{caption}</div>', unsafe_allow_html=True)


def style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color="#E1E8EF", linewidth=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    st.title("⚡ PJME Hourly Energy Consumption Forecasting")
    st.caption(
        "Time-series forecasting on 16+ years of PJM Interconnection hourly load data: "
        "ADF stationarity testing, lag/rolling/calendar feature engineering, and an "
        "XGBoost vs. SARIMA model comparison with confidence intervals and residual diagnostics."
    )

    if not os.path.exists(DATA_PATH):
        st.error(f"No data at `{DATA_PATH}`.")
        st.stop()

    df = load_data(DATA_PATH)

    with st.sidebar:
        st.header("Forecast Window")
        months_back = st.slider(
            "Months of recent history to use", 3, 36, 13, step=1,
            help="XGBoost trains/tests on this hourly window; SARIMA uses the daily "
                 "resample of the first (months-1) months as train and the final month as test.",
        )
        test_frac = st.slider("XGBoost test fraction (most recent %)", 0.1, 0.4, 0.2, step=0.05)

        window_end = df.index.max()
        window_start = window_end - pd.DateOffset(months=months_back)
        window_start = max(window_start, df.index.min())
        df_window = df[df.index >= window_start]

        st.markdown(f"**{len(df_window):,} hours** in window")
        st.caption(f"{window_start.date()} → {window_end.date()}")

    # ----------------------------------------------------------------- #
    # Stationarity
    # ----------------------------------------------------------------- #
    raw_adf, diff_adf = run_adf(df_window[TARGET_COL])

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Hours in window", f"{len(df_window):,}")
    k2.metric("Mean load", f"{df_window[TARGET_COL].mean():,.0f} MW")
    k3.metric(
        "Raw series",
        "Stationary" if raw_adf.is_stationary else "Non-stationary",
        f"p={raw_adf.pvalue:.4f}",
    )
    k4.metric(
        "1st difference",
        "Stationary" if diff_adf.is_stationary else "Non-stationary",
        f"p={diff_adf.pvalue:.4f}",
    )

    st.divider()

    # ----------------------------------------------------------------- #
    # Run models
    # ----------------------------------------------------------------- #
    xgb_result = run_xgboost(df_window, test_frac)

    sarima_train_end = window_end - pd.DateOffset(months=1)
    sarima_result = run_sarima(
        df_window[TARGET_COL],
        str(window_start.date()), str(sarima_train_end.date()),
        str((sarima_train_end + pd.Timedelta(days=1)).date()), str(window_end.date()),
    )

    # ----------------------------------------------------------------- #
    # Model comparison
    # ----------------------------------------------------------------- #
    section("Model Comparison", "MAPE / RMSE on each model's own held-out test period.")
    rows = []
    if len(xgb_result.y_pred):
        rows.append({
            "Model": "XGBoost (hourly)",
            "MAPE (%)": round(mape(xgb_result.y_true, xgb_result.y_pred), 3),
            "RMSE (MW)": round(rmse(xgb_result.y_true, xgb_result.y_pred), 1),
            "Test points": len(xgb_result.y_pred),
        })
    if sarima_result is not None and len(sarima_result.y_pred):
        rows.append({
            "Model": "SARIMA (daily)",
            "MAPE (%)": round(mape(sarima_result.y_true, sarima_result.y_pred), 3),
            "RMSE (MW)": round(rmse(sarima_result.y_true, sarima_result.y_pred), 1),
            "Test points": len(sarima_result.y_pred),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.info("Not enough data in the selected window to fit either model. Widen the window.")

    st.divider()

    tab_xgb, tab_sarima = st.tabs(["XGBoost (hourly)", "SARIMA (daily baseline)"])

    with tab_xgb:
        if len(xgb_result.y_pred) == 0:
            st.info("Not enough hourly data in this window for an XGBoost test split.")
        else:
            n_plot = min(24 * 14, len(xgb_result.y_pred))
            diag = residual_diagnostics(xgb_result.y_true, xgb_result.y_pred)

            col1, col2 = st.columns([3, 2])
            with col1:
                section("Forecast vs Actual", f"Most recent {n_plot // 24} days of the test period, with a 95% CI.")
                fig, ax = plt.subplots(figsize=(8, 4.2))
                x = xgb_result.index[-n_plot:]
                ax.plot(x, xgb_result.y_true[-n_plot:], label="Actual", color=COLOR_ACTUAL, linewidth=1.3)
                ax.plot(x, xgb_result.y_pred[-n_plot:], label="XGBoost forecast", color=COLOR_FORECAST,
                        linewidth=1.3, linestyle="--")
                ax.fill_between(
                    x, diag.ci_lower[-n_plot:], diag.ci_upper[-n_plot:],
                    alpha=0.18, color=COLOR_BAND, label="95% CI",
                )
                ax.set_ylabel("MW")
                style_axes(ax)
                ax.legend(frameon=False, fontsize=8)
                fig.tight_layout()
                st.pyplot(fig)

            with col2:
                section("Residual Diagnostics", "Actual minus predicted, with ±2σ bands.")
                fig2, ax2 = plt.subplots(figsize=(6, 4.2))
                x = xgb_result.index[-n_plot:]
                ax2.plot(x, diag.residuals[-n_plot:], color=COLOR_RESIDUAL, linewidth=0.8, alpha=0.85)
                ax2.axhline(0, color="#111827", linewidth=0.6)
                ax2.axhline(2 * diag.std, color=COLOR_FORECAST, linestyle="--", linewidth=1, label="±2σ")
                ax2.axhline(-2 * diag.std, color=COLOR_FORECAST, linestyle="--", linewidth=1)
                ax2.set_ylabel("Residual (MW)")
                style_axes(ax2)
                ax2.legend(frameon=False, fontsize=8)
                fig2.tight_layout()
                st.pyplot(fig2)

            section("Feature Importance", "Top drivers of the XGBoost forecast.")
            importances = xgb_result.extra["feature_importance"].head(12)
            fig3, ax3 = plt.subplots(figsize=(9, 4))
            ax3.barh(importances.index[::-1], importances.to_numpy()[::-1], color=COLOR_ACTUAL)
            ax3.set_xlabel("Importance")
            style_axes(ax3)
            fig3.tight_layout()
            st.pyplot(fig3)

    with tab_sarima:
        if sarima_result is None or len(sarima_result.y_pred) == 0:
            st.info("Widen the window to at least ~2 months so SARIMA has a daily train and test period.")
        else:
            section(
                "SARIMA Forecast vs Actual (daily)",
                "Trained on all but the final month of the window, tested on that final month, both resampled to daily means.",
            )
            fig4, ax4 = plt.subplots(figsize=(10, 4.2))
            x = sarima_result.index
            ax4.plot(x, sarima_result.y_true, label="Actual", color=COLOR_ACTUAL, linewidth=1.5, marker="o", markersize=3)
            ax4.plot(x, sarima_result.y_pred, label="SARIMA forecast", color=COLOR_FORECAST, linewidth=1.5,
                      linestyle="--", marker="o", markersize=3)
            ax4.fill_between(
                x, sarima_result.extra["ci_lower"], sarima_result.extra["ci_upper"],
                alpha=0.18, color=COLOR_BAND, label="95% CI",
            )
            ax4.set_ylabel("MW (daily mean)")
            style_axes(ax4)
            ax4.legend(frameon=False, fontsize=8)
            fig4.tight_layout()
            st.pyplot(fig4)

    st.divider()
    st.caption(
        "Data: PJM Interconnection hourly load (PJME_hourly.csv), Jan 2002 – Aug 2018. "
        "Synthetic/demo scoping choices (recent-window default, daily SARIMA subset) are made for "
        "interactivity; see README for the full-history pipeline numbers."
    )


if __name__ == "__main__":
    main()
