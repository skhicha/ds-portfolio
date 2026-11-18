"""
Streamlit dashboard for the Pharma Customer Segmentation & Targeting
Analytics tool.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import io
import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.generate_data import generate_dataset
from src.report import generate_excel_report
from src.segment import FEATURE_COLUMNS, segment
from src.targeting import build_target_list, decile_silhouette_score, decile_summary

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "physicians.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
REPORT_PATH = os.path.join(OUTPUT_DIR, "segmentation_report.xlsx")

CLUSTER_COLORS = ["#0F766E", "#4338CA", "#B45309", "#BE123C", "#475569", "#0E7490", "#7C3AED", "#65A30D"]

st.set_page_config(
    page_title="Pharma Customer Segmentation & Targeting",
    page_icon="🧬",
    layout="wide",
)

# --------------------------------------------------------------------------- #
# Styling - small, restrained design system layered on top of the Streamlit
# theme set in .streamlit/config.toml. No raw unstyled dumps for KPIs/tables;
# everything below is either a themed Streamlit widget or a styled container.
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
    :root {
        --color-bg: #F7F9FB;
        --color-surface: #FFFFFF;
        --color-border: #E4E9F0;
        --color-text: #1C2330;
        --color-text-muted: #5B6472;
        --color-primary: #0F766E;
        --color-primary-soft: #E6F7F4;
        --color-accent: #B45309;
        --color-accent-soft: #FDF2E3;
        --shadow-card: 0 1px 2px rgba(16, 24, 40, 0.04), 0 6px 16px rgba(16, 24, 40, 0.06);
    }

    .block-container { padding-top: 2rem; max-width: 1280px; }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: var(--color-surface);
        border: 1px solid var(--color-border);
        border-radius: 12px;
        padding: 1rem 1.1rem 0.9rem 1.1rem;
        box-shadow: var(--shadow-card);
    }
    div[data-testid="stMetricLabel"] { color: var(--color-text-muted); font-weight: 600; }
    div[data-testid="stMetricValue"] { color: var(--color-primary); }

    /* Section headers */
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--color-text);
        margin: 0.25rem 0 0.1rem 0;
    }
    .section-caption { color: var(--color-text-muted); font-size: 0.92rem; margin-bottom: 0.9rem; }

    /* Segment / decile badge pills */
    .pill {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .pill-primary { background: var(--color-primary-soft); color: var(--color-primary); }
    .pill-accent { background: var(--color-accent-soft); color: var(--color-accent); }

    /* Dataframes: tighter, consistent borders */
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--color-border);
        border-radius: 10px;
        box-shadow: var(--shadow-card);
    }

    /* Download button accent */
    div.stDownloadButton button, div.stButton button[kind="primary"] {
        background-color: var(--color-primary);
        border-color: var(--color-primary);
    }

    hr { border-color: var(--color-border); }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_bundled_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data(show_spinner=False)
def generate_sample_data(n: int, seed: int) -> pd.DataFrame:
    return generate_dataset(n_physicians=n, seed=seed)


@st.cache_data(show_spinner="Running PCA + K-Means segmentation pipeline...")
def run_pipeline(df: pd.DataFrame, k_mode: str, manual_k: int, min_actionable_k: int):
    k = None if k_mode == "Auto (elbow + silhouette)" else manual_k
    result = segment(df, k=k, min_actionable_k=min_actionable_k)
    targets = build_target_list(result.df, cluster_name_map=result.cluster_labels)
    scaled_features = result.scaler.transform(result.feature_matrix.to_numpy(dtype=float)) if len(df) else np.empty((0, len(FEATURE_COLUMNS) + 2))
    dec_sil = decile_silhouette_score(scaled_features, targets["decile"]) if len(targets) else float("nan")
    return result, targets, dec_sil


def section(title: str, caption: str = "") -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="section-caption">{caption}</div>', unsafe_allow_html=True)


def main() -> None:
    st.title("🧬 Pharma Customer Segmentation & Targeting Analytics")
    st.caption(
        "K-Means clustering + PCA on synthetic physician prescription data to identify "
        "high-value customer segments, with a decile-based targeting model validated by silhouette score."
    )

    with st.sidebar:
        st.header("Data")
        source = st.radio(
            "Data source",
            ["Use bundled sample", "Generate new sample", "Upload CSV"],
            index=0,
        )

        if source == "Use bundled sample":
            if not os.path.exists(DATA_PATH):
                st.error(f"No bundled data at `{DATA_PATH}`. Run `python -m src.generate_data` first.")
                st.stop()
            df = load_bundled_data()
        elif source == "Generate new sample":
            n = st.slider("Number of physicians", 5, 2000, 500, step=5)
            seed = st.number_input("Random seed", value=42, step=1)
            df = generate_sample_data(int(n), int(seed))
        else:
            uploaded = st.file_uploader("Upload a physician-level CSV", type=["csv"])
            if uploaded is None:
                st.info("Upload a CSV with the same schema as `data/physicians.csv`, or switch data source.")
                st.stop()
            df = pd.read_csv(uploaded)

        st.markdown(f"**{len(df):,} physicians** loaded")

        st.header("Clustering")
        k_mode = st.radio("Choose k (number of segments)", ["Auto (elbow + silhouette)", "Manual"], index=0)
        manual_k = st.slider("Manual k", 2, 8, 4, disabled=(k_mode != "Manual"))
        min_actionable_k = st.slider(
            "Minimum actionable k (for Auto mode)", 2, 6, 3,
            help="Silhouette score alone often maximizes at k=2. Auto mode restricts the "
                 "search to k >= this value so the recommendation stays commercially actionable.",
        )

    if len(df) == 0:
        st.warning("No rows in the current dataset. Adjust the data source in the sidebar.")
        st.stop()

    result, targets, dec_sil = run_pipeline(df, k_mode, int(manual_k), int(min_actionable_k))

    # ----------------------------------------------------------------- #
    # KPI row
    # ----------------------------------------------------------------- #
    total_market_trx = float(result.df.get("total_market_trx_monthly", pd.Series(dtype=float)).sum())
    total_company_trx = float(result.df.get("company_trx_monthly", pd.Series(dtype=float)).sum())
    total_value = float(result.df.get("company_rx_value_monthly", pd.Series(dtype=float)).sum())
    overall_share = total_company_trx / total_market_trx if total_market_trx else 0.0
    cum_var = (
        result.pca.cumulative_variance[result.pca.n_components_selected - 1]
        if result.pca.n_components_selected and len(result.pca.cumulative_variance)
        else 0.0
    )

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Physicians", f"{len(result.df):,}")
    k2.metric("Segments (k)", f"{result.k}")
    k3.metric("Silhouette", f"{result.silhouette:.3f}" if pd.notna(result.silhouette) else "n/a")
    k4.metric("PCA Variance", f"{cum_var * 100:.1f}%", help=f"{result.pca.n_components_selected} components")
    k5.metric("Market Share", f"{overall_share * 100:.1f}%", f"${total_value / 1000:,.0f}K Rx value/mo")

    st.divider()

    # ----------------------------------------------------------------- #
    # PCA scatter + k-selection chart
    # ----------------------------------------------------------------- #
    col_left, col_right = st.columns([3, 2])

    with col_left:
        section("Segments in PCA Space", "First two principal components, colored by segment.")
        if "pc1" in result.df.columns and "pc2" in result.df.columns and len(result.df) > 0:
            fig, ax = plt.subplots(figsize=(7, 5))
            clusters = sorted(result.df["cluster"].unique())
            for i, c in enumerate(clusters):
                sub = result.df[result.df["cluster"] == c]
                ax.scatter(
                    sub["pc1"], sub["pc2"],
                    s=26, alpha=0.75, linewidths=0,
                    color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                    label=result.cluster_labels.get(c, f"Segment {c}"),
                )
            var_ratio = result.pca.explained_variance_ratio
            ax.set_xlabel(f"PC1 ({var_ratio[0] * 100:.1f}% variance)" if len(var_ratio) > 0 else "PC1")
            ax.set_ylabel(f"PC2 ({var_ratio[1] * 100:.1f}% variance)" if len(var_ratio) > 1 else "PC2")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(color="#E4E9F0", linewidth=0.8)
            ax.set_axisbelow(True)
            ax.legend(frameon=False, fontsize=8, loc="best")
            fig.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Not enough variance in this dataset to render a 2-component PCA scatter.")

    with col_right:
        section("Choosing k", "Inertia (elbow) and silhouette score across candidate k.")
        table = result.k_selection.table
        if not table.empty:
            fig2, ax1 = plt.subplots(figsize=(5.2, 5))
            ax1.plot(table["k"], table["inertia"], color="#475569", marker="o", markersize=4, label="Inertia")
            ax1.set_xlabel("k")
            ax1.set_ylabel("Inertia", color="#475569")
            ax1.spines["top"].set_visible(False)
            ax2 = ax1.twinx()
            ax2.plot(table["k"], table["silhouette_score"], color="#0F766E", marker="o", markersize=4, label="Silhouette score")
            ax2.set_ylabel("Silhouette score", color="#0F766E")
            ax2.spines["top"].set_visible(False)
            ax1.axvline(result.k, color="#B45309", linestyle="--", linewidth=1.2, label=f"Chosen k = {result.k}")
            fig2.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=1, frameon=False, fontsize=8)
            fig2.tight_layout()
            st.pyplot(fig2)
        else:
            st.info("Not enough distinct physicians to search over k.")

    st.divider()

    # ----------------------------------------------------------------- #
    # Cluster profile table
    # ----------------------------------------------------------------- #
    section("Segment Profiles", "Mean feature values per segment.")
    profiles = result.cluster_profiles.copy()
    if not profiles.empty:
        profiles.insert(1, "label", profiles["cluster"].map(result.cluster_labels))
        display_cols = [
            "cluster", "label", "size", "pct_of_total", "company_rx_value_monthly",
            "market_share_pct", "digital_engagement_score", "rep_visits_last_12mo", "peer_influence_score",
        ]
        display_cols = [c for c in display_cols if c in profiles.columns]
        st.dataframe(
            profiles[display_cols].rename(columns={
                "cluster": "Segment #", "label": "Label", "size": "Physicians", "pct_of_total": "% of Total",
                "company_rx_value_monthly": "Avg Rx Value/mo ($)", "market_share_pct": "Avg Market Share",
                "digital_engagement_score": "Avg Digital Engagement", "rep_visits_last_12mo": "Avg Rep Visits/yr",
                "peer_influence_score": "Avg Peer Influence",
            }),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Avg Rx Value/mo ($)": st.column_config.NumberColumn(format="$%.0f"),
                "Avg Market Share": st.column_config.NumberColumn(format="%.3f"),
                "% of Total": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )
    else:
        st.info("No segments to profile for this dataset.")

    st.divider()

    # ----------------------------------------------------------------- #
    # Decile targeting
    # ----------------------------------------------------------------- #
    section(
        "Decile-Based Targeting",
        f"Decile 1 = top 10% highest-priority physicians by composite opportunity score. "
        f"Decile-group silhouette score: {dec_sil:.3f}" if pd.notna(dec_sil) else
        "Decile 1 = top 10% highest-priority physicians by composite opportunity score.",
    )

    dsum = decile_summary(targets)
    if not dsum.empty:
        fig3, ax3 = plt.subplots(figsize=(11, 3.2))
        bar_colors = ["#B45309" if d <= 2 else "#0F766E" for d in dsum["decile"]]
        ax3.bar(dsum["decile"].astype(str), dsum["avg_opportunity_score"], color=bar_colors, width=0.65)
        ax3.set_xlabel("Decile (1 = highest priority)")
        ax3.set_ylabel("Avg Opportunity Score")
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
        ax3.grid(axis="y", color="#E4E9F0", linewidth=0.8)
        ax3.set_axisbelow(True)
        fig3.tight_layout()
        st.pyplot(fig3)

    filt_col1, filt_col2 = st.columns(2)
    with filt_col1:
        decile_choices = sorted(targets["decile"].unique()) if "decile" in targets.columns and not targets.empty else []
        decile_filter = st.multiselect("Filter by decile", decile_choices, default=decile_choices[:3] if decile_choices else [])
    with filt_col2:
        specialty_choices = sorted(targets["specialty"].unique()) if "specialty" in targets.columns and not targets.empty else []
        specialty_filter = st.multiselect("Filter by specialty", specialty_choices, default=[])

    filtered = targets.copy()
    if decile_filter:
        filtered = filtered[filtered["decile"].isin(decile_filter)]
    if specialty_filter:
        filtered = filtered[filtered["specialty"].isin(specialty_filter)]

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
        height=380,
        column_config={
            "opportunity_score": st.column_config.NumberColumn("Opportunity Score", format="%.3f"),
            "company_rx_value_monthly": st.column_config.NumberColumn("Rx Value/mo ($)", format="$%.0f"),
            "whitespace_trx_monthly": st.column_config.NumberColumn("Whitespace Rx/mo", format="%.0f"),
        },
    )

    st.divider()

    # ----------------------------------------------------------------- #
    # Excel export
    # ----------------------------------------------------------------- #
    section("Export Report", "Full Excel workbook: summary KPIs, segment profiles, decile targeting, and the ranked target list.")
    if st.button("Generate & Download Excel Report", type="primary"):
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            path = generate_excel_report(result, targets, dec_sil, tmp.name)
            with open(path, "rb") as f:
                data = f.read()
        st.download_button(
            label="⬇️ Download segmentation_report.xlsx",
            data=data,
            file_name="segmentation_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.success("Report generated.")

    st.caption(
        "Dataset is synthetic and generated for demonstration purposes; it does not represent "
        "real physicians, patients, or prescribing behavior."
    )


if __name__ == "__main__":
    main()
