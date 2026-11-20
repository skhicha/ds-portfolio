"""
Streamlit dashboard for the House Price Predictor.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.predict import DEFAULT_FEATURE_VALUES, FEATURE_RANGES, load_model, predict_price

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
METRICS_PATH = os.path.join(PROJECT_ROOT, "models", "metrics.json")

FEATURE_LABELS = {
    "MedInc": "Median Income (in $10,000s)",
    "HouseAge": "House Age (years)",
    "AveRooms": "Average Rooms per Household",
    "AveBedrms": "Average Bedrooms per Household",
    "Population": "Block Group Population",
    "AveOccup": "Average Occupancy per Household",
    "Latitude": "Latitude",
    "Longitude": "Longitude",
}

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --color-bg: #FAF8F5;
        --color-surface: #FFFFFF;
        --color-border: #E6E0D6;
        --color-text: #27231D;
        --color-text-muted: #6B6459;
        --color-primary: #8A5A2B;
        --color-primary-soft: #F3E6D6;
        --color-accent: #2F6F4E;
        --color-accent-soft: #E5F1E9;
        --shadow-card: 0 1px 2px rgba(39, 35, 29, 0.05), 0 6px 16px rgba(39, 35, 29, 0.07);
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
    .pill-primary { background: var(--color-primary-soft); color: var(--color-primary); }
    .pill-accent { background: var(--color-accent-soft); color: var(--color-accent); }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--color-border);
        border-radius: 10px;
        box-shadow: var(--shadow-card);
    }

    div.stButton button[kind="primary"] {
        background-color: var(--color-primary);
        border-color: var(--color-primary);
    }

    hr { border-color: var(--color-border); }
    </style>
    """,
    unsafe_allow_html=True,
)


def section(title: str, caption: str = "") -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="section-caption">{caption}</div>', unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_model():
    return load_model()


@st.cache_data(show_spinner=False)
def get_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH) as f:
        return json.load(f)


def main() -> None:
    st.title("🏠 House Price Predictor")
    st.caption(
        "Random Forest regression (RandomizedSearchCV-tuned) trained on a log-transformed "
        "target, predicting California home values from census block-group features."
    )

    try:
        model, feature_names = get_model()
    except FileNotFoundError:
        st.error("No trained model found. Run `python -m src.train` first, then reload this page.")
        st.stop()

    metrics = get_metrics()

    if metrics:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Tuned Model RMSE", f"${metrics['tuned_rmse']:,.0f}")
        k2.metric("R² (held-out)", f"{metrics['tuned_r2']:.3f}")
        k3.metric("MAE (held-out)", f"${metrics['tuned_mae']:,.0f}")
        k4.metric("Best Baseline", metrics["best_baseline_name"])
        st.divider()

    col_inputs, col_result = st.columns([2, 3])

    with col_inputs:
        section("House Features", "Adjust the sliders to describe a census block group.")
        inputs = {}
        for feature in feature_names:
            lo, hi = FEATURE_RANGES.get(feature, (0.0, 10.0))
            default = DEFAULT_FEATURE_VALUES.get(feature, (lo + hi) / 2)
            step = (hi - lo) / 200 if hi > lo else 0.1
            inputs[feature] = st.slider(
                FEATURE_LABELS.get(feature, feature),
                float(lo), float(hi), float(default), step=float(round(step, 3)) or 0.01,
            )
        predict_clicked = st.button("Predict Price", type="primary", use_container_width=True)

    with col_result:
        section("Prediction", "Model output for the feature values on the left.")
        if predict_clicked:
            result = predict_price(model, feature_names, inputs)
            st.metric("Predicted Sale Price", f"${result['predicted_price']:,.0f}")

            importances = result["feature_importances"]
            if importances:
                order = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:10]
                labels = [FEATURE_LABELS.get(k, k) for k, _ in order][::-1]
                values = [v for _, v in order][::-1]

                fig, ax = plt.subplots(figsize=(7, 4))
                ax.barh(labels, values, color="#8A5A2B")
                ax.set_title("Top Feature Importances", fontweight="bold", fontsize=11)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.grid(axis="x", color="#E6E0D6", linewidth=0.8)
                ax.set_axisbelow(True)
                fig.tight_layout()
                st.pyplot(fig, clear_figure=True)
        else:
            st.info("Set the feature values and click **Predict Price**.")

    if metrics:
        with st.expander("Baseline model comparison (held-out RMSE / R² / MAE)"):
            for r in metrics["baseline_results"]:
                st.markdown(
                    f"- **{r['name']}** &mdash; RMSE ${r['rmse']:,.0f} · "
                    f"R² {r['r2']:.3f} · MAE ${r['mae']:,.0f}"
                )
            st.markdown(f"**Tuned Random Forest hyperparameters:** `{metrics['best_params']}`")


if __name__ == "__main__":
    main()
