"""
Streamlit dashboard for the Marketing Mix Modeling & Sales Forecasting Tool.

Run with:
    streamlit run app.py
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.mmm import (
    DEFAULT_CHANNELS,
    contribution_summary,
    fit_mmm,
    simulate_budget_scenario,
)
from src.report import generate_excel_report

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "weekly_sales.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
REPORT_PATH = os.path.join(OUTPUT_DIR, "mmm_report.xlsx")

CHANNEL_LABELS = {
    "tv_spend": "TV Spend",
    "digital_spend": "Digital Spend",
    "promotions_spend": "Promotions Spend",
}

st.set_page_config(
    page_title="Marketing Mix Modeling & Sales Forecasting",
    page_icon="📈",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource(show_spinner="Fitting marketing mix model (adstock + saturation search + Ridge)...")
def get_fitted_model(data_signature: str):
    df = load_data(DATA_PATH)
    return fit_mmm(df)


def main() -> None:
    st.title("📈 Marketing Mix Modeling & Sales Forecasting Tool")
    st.caption(
        "Ridge regression on adstock-transformed, saturation-transformed channel spend "
        "+ seasonality/trend, fit on synthetic weekly sales data."
    )

    if not os.path.exists(DATA_PATH):
        st.error(
            f"Data file not found at `{DATA_PATH}`.\n\n"
            "Generate it first with:  `python -m src.generate_data`"
        )
        st.stop()

    df = load_data(DATA_PATH)
    result = get_fitted_model(data_signature=str(os.path.getmtime(DATA_PATH)))
    summary = contribution_summary(result)

    # ---------------------------------------------------------------- #
    # Top KPI row
    # ---------------------------------------------------------------- #
    total_sales = float(result.actuals.sum())
    total_spend = float(result.raw_spend.sum().sum())
    baseline_total = float(result.contributions["baseline"].sum())
    media_total = total_sales - baseline_total
    blended_roi = media_total / total_spend if total_spend else 0.0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Model R²", f"{result.r2:.3f}")
    k2.metric("MAPE", f"{result.mape * 100:.2f}%")
    k3.metric("Total Sales", f"${total_sales:,.0f}")
    k4.metric("Total Spend", f"${total_spend:,.0f}")
    k5.metric("Blended ROI", f"{blended_roi:.2f}x")

    st.divider()

    # ---------------------------------------------------------------- #
    # Fit chart + contribution chart
    # ---------------------------------------------------------------- #
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Actual vs. Fitted Weekly Sales")
        dates = pd.to_datetime(result.dates)
        fit_df = pd.DataFrame(
            {"Actual": result.actuals, "Fitted": result.fitted_values}, index=dates
        )
        st.line_chart(fit_df)

    with col_right:
        st.subheader("Sales Contribution by Component")
        chart_df = summary.set_index("component")[["total_contribution"]]
        chart_df.columns = ["Contribution"]
        st.bar_chart(chart_df)

    st.dataframe(
        summary.assign(
            total_contribution=lambda d: d["total_contribution"].round(0),
            share_of_sales=lambda d: (d["share_of_sales"] * 100).round(1).astype(str) + "%",
        ).rename(
            columns={
                "component": "Component",
                "total_contribution": "Total Contribution ($)",
                "share_of_sales": "Share of Sales",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Fitted model coefficients & adstock / saturation parameters"):
        coef_df = pd.DataFrame(
            [{"feature": k, "coefficient": v} for k, v in result.coefficients.items()]
        )
        st.dataframe(coef_df, use_container_width=True, hide_index=True)

        params_df = pd.DataFrame(
            [
                {
                    "channel": ch,
                    "adstock_decay": round(p.decay, 3),
                    "saturation_gamma": round(p.gamma, 1),
                    "saturation_alpha": p.alpha,
                }
                for ch, p in result.channel_params.items()
            ]
        )
        st.dataframe(params_df, use_container_width=True, hide_index=True)

    st.divider()

    # ---------------------------------------------------------------- #
    # Budget scenario simulator
    # ---------------------------------------------------------------- #
    st.subheader("🎛️ Interactive Budget Scenario Simulator")
    st.write(
        "Set a proposed **steady-state weekly budget** per channel. The simulator runs the "
        "fitted adstock + saturation + Ridge model forward until carryover stabilizes "
        "(13 weeks) and reports the resulting predicted weekly sales."
    )

    recent_avg = {ch: float(df[ch].tail(13).mean()) for ch in DEFAULT_CHANNELS}

    slider_cols = st.columns(len(DEFAULT_CHANNELS))
    allocation = {}
    for col, ch in zip(slider_cols, DEFAULT_CHANNELS):
        with col:
            max_val = max(float(df[ch].max()) * 1.5, recent_avg[ch] * 3, 1000.0)
            allocation[ch] = st.slider(
                CHANNEL_LABELS.get(ch, ch),
                min_value=0.0,
                max_value=round(max_val, -2),
                value=round(recent_avg[ch], 2),
                step=100.0,
                key=f"slider_{ch}",
            )

    sim = simulate_budget_scenario(result, allocation)
    baseline_sim = simulate_budget_scenario(
        result, {ch: recent_avg[ch] for ch in DEFAULT_CHANNELS}
    )

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Total Weekly Budget", f"${sim['total_weekly_budget']:,.0f}")
    r2.metric(
        "Predicted Weekly Sales",
        f"${sim['predicted_weekly_sales']:,.0f}",
        delta=f"{sim['predicted_weekly_sales'] - baseline_sim['predicted_weekly_sales']:,.0f} vs. recent avg budget",
    )
    r3.metric("Implied Marginal ROI", f"{sim['implied_roi']:.2f}x")
    r4.metric("Baseline (non-media) Sales", f"${sim['baseline_level']:,.0f}")

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(sim["predicted_trajectory"], marker="o", markersize=3, label="Predicted sales trajectory")
    ax.axhline(sim["baseline_level"], color="gray", linestyle="--", linewidth=1, label="Baseline level")
    ax.set_xlabel("Week (from scenario start)")
    ax.set_ylabel("Predicted weekly sales")
    ax.set_title("Predicted sales ramp-up under proposed budget (adstock carryover building up)")
    ax.legend()
    st.pyplot(fig)

    contrib_rows = [
        {"channel": CHANNEL_LABELS.get(ch, ch), "weekly_budget": allocation[ch],
         "predicted_contribution": val}
        for ch, val in sim["channel_contributions"].items()
    ]
    st.dataframe(pd.DataFrame(contrib_rows), use_container_width=True, hide_index=True)

    st.divider()

    # ---------------------------------------------------------------- #
    # Excel export
    # ---------------------------------------------------------------- #
    st.subheader("📊 Export Client Report")
    st.write("Generate a full Excel workbook (KPIs, channel contribution chart, model coefficients).")
    if st.button("Generate & Download Excel Report"):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = generate_excel_report(result, REPORT_PATH)
        with open(path, "rb") as f:
            st.download_button(
                label="⬇️ Download mmm_report.xlsx",
                data=f.read(),
                file_name="mmm_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        st.success(f"Report generated at {path}")


if __name__ == "__main__":
    main()
