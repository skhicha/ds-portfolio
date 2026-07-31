"""
app.py
Loan Portfolio Risk & ECL Analytics Dashboard (Streamlit).

Interactive KRIs, segmentation drill-downs, a roll-rate heatmap, a
scikit-learn early-warning watchlist, and a live stress-scenario simulator
that recomputes ECL when you shock PD or LGD assumptions with the sliders.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import db, risk
from src.etl import DEFAULT_DB_PATH
from src.model import (
    build_training_frame,
    train_early_warning_model,
    score_portfolio,
)

st.set_page_config(
    page_title="Loan Portfolio Risk & ECL Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = DEFAULT_DB_PATH


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_loans() -> pd.DataFrame:
    conn = db.get_connection(DB_PATH)
    try:
        return pd.read_sql_query("SELECT * FROM loans", conn)
    finally:
        conn.close()


@st.cache_data(show_spinner=False)
def load_snapshots() -> pd.DataFrame:
    conn = db.get_connection(DB_PATH)
    try:
        return pd.read_sql_query("SELECT * FROM loan_snapshots", conn)
    finally:
        conn.close()


@st.cache_data(show_spinner=False)
def load_point_in_time(as_of_date: str) -> pd.DataFrame:
    """Point-in-time portfolio (one row per live loan) via
    sql/latest_snapshot_per_loan.sql, executed with a bound parameter."""
    return db.run_query("latest_snapshot_per_loan.sql", params={"as_of_date": as_of_date}, db_path=DB_PATH)


@st.cache_data(show_spinner=False)
def load_roll_rate_pairs(product: str | None, geography: str | None) -> pd.DataFrame:
    return db.run_query(
        "roll_rate_pairs.sql", params={"product": product, "geography": geography}, db_path=DB_PATH
    )


@st.cache_data(show_spinner=False)
def load_segment_summary(as_of_date: str, product: str | None, geography: str | None, vintage: str | None) -> pd.DataFrame:
    return db.run_query(
        "segment_portfolio.sql",
        params={"as_of_date": as_of_date, "product": product, "geography": geography, "vintage": vintage},
        db_path=DB_PATH,
    )


@st.cache_data(show_spinner=False)
def load_kri_trend(product: str | None, geography: str | None) -> pd.DataFrame:
    return db.run_query("kri_trend.sql", params={"product": product, "geography": geography}, db_path=DB_PATH)


@st.cache_resource(show_spinner="Training early-warning logistic regression model...")
def get_trained_model():
    loans_df = load_loans()
    snapshots_df = load_snapshots()
    frame = build_training_frame(snapshots_df, loans_df)
    return train_early_warning_model(frame, test_size=0.25, random_state=42), frame


def _none_if_all(value: str) -> str | None:
    return None if value == "All" else value


def apply_pd_shock(ecl_df: pd.DataFrame, pd_shock_multiplier: float) -> pd.DataFrame:
    """Apply a uniform multiplicative shock to the already-derived PD
    (pd_applied) and recompute ECL -- used by the stress-scenario slider.
    Stage 3 loans keep PD = 1.0 (already in default; a PD shock cannot make
    a defaulted loan "more defaulted")."""
    shocked = ecl_df.copy()
    shocked["pd_applied"] = np.where(
        shocked["stage"] == 3,
        1.0,
        np.clip(shocked["pd_applied"] * pd_shock_multiplier, 0.0, 1.0),
    )
    shocked["ecl"] = shocked["pd_applied"] * shocked["lgd_applied"] * shocked["ead"]
    return shocked


# ---------------------------------------------------------------------------
# Load base data
# ---------------------------------------------------------------------------

if not Path(DB_PATH).exists():
    st.error(
        f"Database not found at `{DB_PATH}`.\n\n"
        "Run `python src/etl.py` first to generate the synthetic loan book."
    )
    st.stop()

loans_all = load_loans()
snapshots_all = load_snapshots()
as_of_date = str(pd.to_datetime(snapshots_all["snapshot_date"]).max().date())

st.title("Loan Portfolio Risk & ECL Analytics Dashboard")
st.caption(
    f"Synthetic loan book · as of **{as_of_date}** · {len(loans_all):,} loans · "
    f"{len(snapshots_all):,} monthly snapshots · Ind AS 109 / IFRS 9-style ECL"
)

# ---------------------------------------------------------------------------
# Sidebar: drill-down filters + stress scenario simulator
# ---------------------------------------------------------------------------

st.sidebar.header("Portfolio filters")
product_choice = st.sidebar.selectbox("Product", ["All"] + sorted(loans_all["product"].unique()))
geography_choice = st.sidebar.selectbox("Geography", ["All"] + sorted(loans_all["geography"].unique()))
vintage_choice = st.sidebar.selectbox("Vintage quarter", ["All"] + sorted(loans_all["vintage_quarter"].unique()))

product_param = _none_if_all(product_choice)
geography_param = _none_if_all(geography_choice)
vintage_param = _none_if_all(vintage_choice)

st.sidebar.markdown("---")
st.sidebar.header("Stress-scenario simulator")
st.sidebar.caption("Shock the PD and LGD assumptions and watch ECL recompute live.")
pd_shock = st.sidebar.slider("PD shock multiplier", min_value=0.25, max_value=3.0, value=1.0, step=0.05)

lgd_overrides = {}
with st.sidebar.expander("LGD assumptions by product", expanded=False):
    for product_name, default_lgd in risk.DEFAULT_LGD_ASSUMPTIONS.items():
        lgd_overrides[product_name] = st.slider(
            f"LGD — {product_name.replace('_', ' ').title()}",
            min_value=0.0,
            max_value=1.0,
            value=float(default_lgd),
            step=0.01,
            key=f"lgd_{product_name}",
        )

# ---------------------------------------------------------------------------
# Point-in-time portfolio + roll rates + baseline / stressed ECL
# ---------------------------------------------------------------------------

portfolio = load_point_in_time(as_of_date)
pairs_all = load_roll_rate_pairs(None, None)  # portfolio-wide matrix underpins PD for every segment
transition_matrix = risk.compute_transition_matrix(pairs_all)

filtered_portfolio = portfolio.copy()
if product_param:
    filtered_portfolio = filtered_portfolio[filtered_portfolio["product"] == product_param]
if geography_param:
    filtered_portfolio = filtered_portfolio[filtered_portfolio["geography"] == geography_param]
if vintage_param:
    filtered_portfolio = filtered_portfolio[filtered_portfolio["vintage_quarter"] == vintage_param]

baseline_ecl = risk.compute_ecl(filtered_portfolio, transition_matrix, lgd_by_product=risk.DEFAULT_LGD_ASSUMPTIONS)
scenario_ecl = risk.compute_ecl(filtered_portfolio, transition_matrix, lgd_by_product=lgd_overrides)
scenario_ecl = apply_pd_shock(scenario_ecl, pd_shock)

# ---------------------------------------------------------------------------
# KRI tiles
# ---------------------------------------------------------------------------

total_outstanding = filtered_portfolio["outstanding_balance"].sum()
delinquent_balance = filtered_portfolio.loc[
    filtered_portfolio["delinquency_bucket"] != "Current", "outstanding_balance"
].sum()
delinquent_pct = (delinquent_balance / total_outstanding) if total_outstanding else 0.0
stage_balances = filtered_portfolio.groupby("stage")["outstanding_balance"].sum()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total outstanding (EAD)", f"${total_outstanding:,.0f}")
col2.metric("Delinquent balance %", f"{delinquent_pct * 100:.2f}%")
col3.metric("Stage 2 balance", f"${stage_balances.get(2, 0):,.0f}")
col4.metric("Stage 3 balance", f"${stage_balances.get(3, 0):,.0f}")
col5.metric(
    "Baseline ECL",
    f"${baseline_ecl['ecl'].sum():,.0f}",
    delta=f"{(scenario_ecl['ecl'].sum() - baseline_ecl['ecl'].sum()):,.0f} under stress scenario",
    delta_color="inverse",
)

st.markdown("---")

# ---------------------------------------------------------------------------
# ECL by stage: baseline vs stress
# ---------------------------------------------------------------------------

left, right = st.columns([1, 1])

with left:
    st.subheader("ECL by stage — baseline vs stress scenario")
    baseline_by_stage = risk.ecl_summary(baseline_ecl, by=["stage"]).assign(scenario="Baseline")
    scenario_by_stage = risk.ecl_summary(scenario_ecl, by=["stage"]).assign(scenario="Stress")
    combined = pd.concat([baseline_by_stage, scenario_by_stage], ignore_index=True)
    combined["stage"] = combined["stage"].map({1: "Stage 1", 2: "Stage 2", 3: "Stage 3"})

    chart = (
        alt.Chart(combined)
        .mark_bar()
        .encode(
            x=alt.X("stage:N", title="IFRS 9 / Ind AS 109 stage"),
            y=alt.Y("total_ecl:Q", title="Total ECL ($)"),
            color=alt.Color("scenario:N", title="Scenario"),
            xOffset="scenario:N",
            tooltip=["stage", "scenario", alt.Tooltip("total_ecl:Q", format=",.0f"), "loan_count"],
        )
        .properties(height=350)
    )
    st.altair_chart(chart, use_container_width=True)

    st.dataframe(
        combined[["scenario", "stage", "loan_count", "total_ead", "total_ecl", "ecl_coverage_ratio"]]
        .sort_values(["stage", "scenario"])
        .style.format({"total_ead": "{:,.0f}", "total_ecl": "{:,.0f}", "ecl_coverage_ratio": "{:.2%}"}),
        use_container_width=True,
    )

with right:
    st.subheader("Roll-rate heatmap (empirical monthly transitions)")
    matrix_long = transition_matrix.reset_index().melt(
        id_vars="bucket_from", var_name="bucket_to", value_name="probability"
    )
    heatmap = (
        alt.Chart(matrix_long)
        .mark_rect()
        .encode(
            x=alt.X("bucket_to:N", sort=risk.BUCKET_ORDER, title="Bucket next month"),
            y=alt.Y("bucket_from:N", sort=risk.BUCKET_ORDER, title="Bucket this month"),
            color=alt.Color("probability:Q", scale=alt.Scale(scheme="orangered"), title="P(transition)"),
            tooltip=["bucket_from", "bucket_to", alt.Tooltip("probability:Q", format=".2%")],
        )
        .properties(height=350)
    )
    text = heatmap.mark_text(baseline="middle").encode(
        text=alt.Text("probability:Q", format=".1%"),
        color=alt.condition("datum.probability > 0.5", alt.value("white"), alt.value("black")),
    )
    st.altair_chart(heatmap + text, use_container_width=True)
    st.caption(
        "Whole-portfolio matrix (drill-down filters do not restrict this view, so PD stays "
        "comparable across segments). '90+' is absorbing: once a loan charges off in this "
        "simulation it leaves the active panel."
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# Segmentation drill-down
# ---------------------------------------------------------------------------

st.subheader("Portfolio segmentation (product x geography x vintage)")
segment_df = load_segment_summary(as_of_date, product_param, geography_param, vintage_param)
segment_df = segment_df.assign(
    delinquent_balance_pct=lambda d: d["delinquent_balance"] / d["total_outstanding"].replace(0, np.nan)
)
st.dataframe(
    segment_df.style.format(
        {
            "total_outstanding": "{:,.0f}",
            "total_principal": "{:,.0f}",
            "avg_interest_rate": "{:.2%}",
            "delinquent_balance": "{:,.0f}",
            "delinquent_balance_pct": "{:.2%}",
            "stage3_balance": "{:,.0f}",
        }
    ),
    use_container_width=True,
    height=300,
)

trend_df = load_kri_trend(product_param, geography_param)
trend_df["snapshot_date"] = pd.to_datetime(trend_df["snapshot_date"])
trend_chart = (
    alt.Chart(trend_df)
    .mark_line(point=False)
    .encode(
        x=alt.X("snapshot_date:T", title="Snapshot month"),
        y=alt.Y("delinquent_balance_pct:Q", title="Delinquent balance %", axis=alt.Axis(format="%")),
        tooltip=[alt.Tooltip("snapshot_date:T"), alt.Tooltip("delinquent_balance_pct:Q", format=".2%")],
    )
    .properties(height=250, title="Delinquent balance % over time")
)
st.altair_chart(trend_chart, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Early-warning model
# ---------------------------------------------------------------------------

st.subheader("Early-warning scoring (scikit-learn logistic regression)")
st.caption(
    "Trained on historical outcomes: for loans currently Current/1-29 DPD, predicts the "
    "probability of rolling to 90+ DPD within the next 6 months."
)

ew_model, training_frame = get_trained_model()
m1, m2, m3, m4 = st.columns(4)
m1.metric("Holdout AUC", f"{ew_model.auc:.3f}")
m2.metric("Holdout precision", f"{ew_model.precision:.3f}")
m3.metric("Holdout recall", f"{ew_model.recall:.3f}")
m4.metric("Training positives", f"{ew_model.positive_rate * 100:.1f}%")

live_scoreable = portfolio[portfolio["delinquency_bucket"].isin(["Current", "1-29"])].copy()
if product_param:
    live_scoreable = live_scoreable[live_scoreable["product"] == product_param]
if geography_param:
    live_scoreable = live_scoreable[live_scoreable["geography"] == geography_param]
if vintage_param:
    live_scoreable = live_scoreable[live_scoreable["vintage_quarter"] == vintage_param]

scored = score_portfolio(ew_model, live_scoreable)
watchlist = scored.sort_values("early_warning_score", ascending=False).head(25)

st.markdown("**Top 25 watchlist loans (highest predicted early-warning score)**")
st.dataframe(
    watchlist[
        [
            "loan_id", "product", "geography", "vintage_quarter", "dpd", "delinquency_bucket",
            "credit_score", "interest_rate", "outstanding_balance", "early_warning_score",
        ]
    ].style.format({"interest_rate": "{:.2%}", "outstanding_balance": "{:,.0f}", "early_warning_score": "{:.1%}"}),
    use_container_width=True,
    height=400,
)

st.markdown("---")

# ---------------------------------------------------------------------------
# Loan-level drill-down table
# ---------------------------------------------------------------------------

with st.expander("Loan-level drill-down (filtered point-in-time portfolio)"):
    ecl_for_display = baseline_ecl[
        [
            "loan_id", "product", "geography", "vintage_quarter", "origination_date", "dpd",
            "delinquency_bucket", "stage", "outstanding_balance", "pd_applied", "lgd_applied", "ecl",
        ]
    ].sort_values("ecl", ascending=False)
    st.dataframe(
        ecl_for_display.style.format(
            {
                "outstanding_balance": "{:,.0f}",
                "pd_applied": "{:.2%}",
                "lgd_applied": "{:.2%}",
                "ecl": "{:,.0f}",
            }
        ),
        use_container_width=True,
        height=400,
    )
