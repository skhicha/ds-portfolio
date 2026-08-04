"""Optional lightweight Streamlit viewer for the Sales Force Optimization model.

Lets a user:
  - load the generated reps/territories data (or regenerate it with new sizes)
  - tweak the optional budget constraint
  - choose which baseline to compare against
  - re-solve the LP interactively and see the allocation + KPIs + a download
    link for the Excel report

Run with:
    streamlit run app.py

This is a convenience layer on top of the core engine in `src/` -- all of
the actual optimization logic lives there, not in this file.
"""
from __future__ import annotations

import os
import tempfile

import pandas as pd
import streamlit as st

from src.generate_data import generate
from src.optimize import solve_allocation, evaluate_allocation, allocation_to_long_df
from src.baseline import status_quo_allocation, round_robin_allocation
from src.report import generate_report

st.set_page_config(page_title="Sales Force Territory Optimization", layout="wide")

st.title("Sales Force Optimization & Territory Allocation Model")
st.caption(
    "LP-based rep-to-territory allocation (scipy.optimize.linprog / HiGHS) "
    "maximizing net revenue subject to territory coverage and rep capacity constraints."
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@st.cache_data(show_spinner=False)
def _generate(num_reps, num_territories, seed):
    return generate(num_reps=num_reps, num_territories=num_territories, seed=seed)


def _load_default_or_generate():
    reps_path = os.path.join(DATA_DIR, "reps.csv")
    territories_path = os.path.join(DATA_DIR, "territories.csv")
    if os.path.exists(reps_path) and os.path.exists(territories_path):
        return pd.read_csv(reps_path), pd.read_csv(territories_path)
    return _generate(28, 20, 42)


with st.sidebar:
    st.header("Data & Constraints")

    regen = st.checkbox("Regenerate synthetic data", value=False)
    if regen:
        num_reps = st.slider("Number of reps", 20, 40, 28)
        num_territories = st.slider("Number of territories", 15, 25, 20)
        seed = st.number_input("Random seed", value=42, step=1)
        reps_df, territories_df = _generate(num_reps, num_territories, seed)
    else:
        reps_df, territories_df = _load_default_or_generate()

    st.markdown(f"**{len(reps_df)} reps** / **{len(territories_df)} territories** loaded")

    use_budget = st.checkbox("Apply total budget cap", value=False)
    budget = None
    if use_budget:
        total_cost_cap_default = float(reps_df["annual_cost"].sum())
        budget = st.slider(
            "Budget cap ($)", min_value=0.0, max_value=total_cost_cap_default,
            value=total_cost_cap_default * 0.7, step=10_000.0, format="%.0f",
        )

    baseline_choice = st.radio("Baseline for comparison", ["Status Quo", "Round Robin"], index=0)

    solve_clicked = st.button("Solve LP", type="primary")

data_signature = (len(reps_df), len(territories_df), budget)

if (
    "result" not in st.session_state
    or st.session_state.get("result_signature") != data_signature
    or solve_clicked
):
    with st.spinner("Solving LP with scipy.optimize.linprog (HiGHS)..."):
        result = solve_allocation(reps_df, territories_df, budget=budget)
    st.session_state["result"] = result
    st.session_state["result_signature"] = data_signature
else:
    result = st.session_state["result"]

if not result.success:
    st.error(f"Solver could not find a feasible solution: {result.scipy_message}")
    st.stop()

opt_metrics = evaluate_allocation(result.x, reps_df, territories_df)

if baseline_choice == "Status Quo":
    baseline_x = status_quo_allocation(reps_df, territories_df)
else:
    baseline_x = round_robin_allocation(reps_df, territories_df)
baseline_metrics = evaluate_allocation(baseline_x, reps_df, territories_df)

improvement = None
if baseline_metrics["net_revenue"] != 0:
    improvement = ((opt_metrics["net_revenue"] - baseline_metrics["net_revenue"])
                    / abs(baseline_metrics["net_revenue"])) * 100.0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Net Revenue (Optimized)", f"${opt_metrics['net_revenue']:,.0f}",
            f"{improvement:+.1f}% vs baseline" if improvement is not None else None)
col2.metric("Total Revenue", f"${opt_metrics['total_revenue']:,.0f}")
col3.metric("Total Cost", f"${opt_metrics['total_cost']:,.0f}")
col4.metric("Avg Rep Utilization", f"{opt_metrics['avg_utilization'] * 100:.1f}%")

st.subheader("Territory Coverage")
territory_view = territories_df.copy()
territory_view["coverage_pct"] = (opt_metrics["coverage"] * 100).round(1)
territory_view["achieved_revenue"] = opt_metrics["achieved_revenue"].round(2)
territory_view["meets_min_coverage"] = opt_metrics["meets_min_coverage"]
st.dataframe(
    territory_view[[
        "territory_id", "territory_name", "region", "required_fte",
        "min_coverage_pct", "coverage_pct", "potential_revenue",
        "achieved_revenue", "meets_min_coverage",
    ]],
    use_container_width=True,
)
st.bar_chart(territory_view.set_index("territory_id")[["potential_revenue", "achieved_revenue"]])

st.subheader("Rep Utilization")
rep_view = reps_df.copy()
rep_view["allocated_fte"] = opt_metrics["rep_allocated_fte"].round(3)
rep_view["utilization_pct"] = (opt_metrics["utilization"] * 100).round(1)
rep_view["cost_incurred"] = opt_metrics["rep_cost"].round(2)
st.dataframe(
    rep_view[[
        "rep_id", "rep_name", "region", "skill_level", "productivity_multiplier",
        "max_capacity_fte", "allocated_fte", "utilization_pct", "annual_cost", "cost_incurred",
    ]],
    use_container_width=True,
)

st.subheader("Rep -> Territory Allocation")
alloc_df = allocation_to_long_df(result.x, reps_df, territories_df)
st.dataframe(alloc_df, use_container_width=True)

st.subheader(f"Optimized vs. Baseline ({baseline_choice})")
comparison = pd.DataFrame({
    "Metric": ["Total Revenue", "Total Cost", "Net Revenue", "Avg Coverage %", "Avg Utilization %"],
    "Optimized": [opt_metrics["total_revenue"], opt_metrics["total_cost"], opt_metrics["net_revenue"],
                  opt_metrics["avg_coverage"] * 100, opt_metrics["avg_utilization"] * 100],
    "Baseline": [baseline_metrics["total_revenue"], baseline_metrics["total_cost"], baseline_metrics["net_revenue"],
                 baseline_metrics["avg_coverage"] * 100, baseline_metrics["avg_utilization"] * 100],
})
st.dataframe(comparison, use_container_width=True)

st.subheader("Download Excel KPI Dashboard")
if st.button("Generate Excel Report"):
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        generate_report(result, baseline_x, tmp.name, baseline_label=baseline_choice)
        with open(tmp.name, "rb") as f:
            st.download_button(
                "Download territory_allocation_report.xlsx", data=f.read(),
                file_name="territory_allocation_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
