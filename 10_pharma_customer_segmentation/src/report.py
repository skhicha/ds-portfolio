"""
Automated Excel report generation for the Pharma Customer Segmentation &
Targeting Analytics tool.

Builds a client-deliverable-style workbook (output/segmentation_report.xlsx)
with:
    1. "Summary"            - headline KPIs (physician count, chosen k,
                               silhouette scores, PCA variance explained,
                               market/value totals)
    2. "Cluster Profiles"   - per-segment feature means + a native Excel
                               bar chart of segment value
    3. "Targeting Deciles"  - decile rollup table + a native Excel bar chart
    4. "Target List"        - the full ranked, decile-scored physician list

Usage:
    python -m src.report     # runs the full pipeline on data/physicians.csv
                              # and writes output/segmentation_report.xlsx
"""
from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from src.segment import SegmentationResult
from src.targeting import decile_silhouette_score, decile_summary

HEADER_FILL = PatternFill(start_color="134E4A", end_color="134E4A", fill_type="solid")
HEADER_FONT = Font(color="FFFFFF", bold=True)
TITLE_FONT = Font(size=16, bold=True, color="134E4A")
SUBTITLE_FONT = Font(size=10, italic=True, color="595959")
KPI_LABEL_FONT = Font(bold=True)
KPI_VALUE_FONT = Font(size=14, bold=True, color="134E4A")


def _style_header_row(ws, row_idx: int, col_start: int, n_cols: int) -> None:
    for col_idx in range(col_start, col_start + n_cols):
        cell = ws.cell(row=row_idx, column=col_idx)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center")


def _autofit_columns(ws, widths: dict) -> None:
    for col_letter, width in widths.items():
        ws.column_dimensions[col_letter].width = width


def _write_summary_sheet(
    wb: Workbook,
    result: SegmentationResult,
    targets: pd.DataFrame,
    decile_sil: float,
) -> None:
    ws = wb.active
    ws.title = "Summary"

    ws["B2"] = "Pharma Customer Segmentation & Targeting - Summary Report"
    ws["B2"].font = TITLE_FONT
    ws["B3"] = f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} | {len(result.df)} physicians analyzed"
    ws["B3"].font = SUBTITLE_FONT

    total_market_trx = float(result.df.get("total_market_trx_monthly", pd.Series(dtype=float)).sum())
    total_company_trx = float(result.df.get("company_trx_monthly", pd.Series(dtype=float)).sum())
    total_value = float(result.df.get("company_rx_value_monthly", pd.Series(dtype=float)).sum())
    overall_share = total_company_trx / total_market_trx if total_market_trx else 0.0
    cum_var = (
        result.pca.cumulative_variance[result.pca.n_components_selected - 1]
        if result.pca.n_components_selected and len(result.pca.cumulative_variance)
        else 0.0
    )

    kpis = [
        ("Physicians Analyzed", f"{len(result.df):,}"),
        ("Segments Identified (k)", f"{result.k}"),
        ("Clustering Silhouette Score", f"{result.silhouette:.3f}" if pd.notna(result.silhouette) else "n/a"),
        ("Decile-Group Silhouette Score", f"{decile_sil:.3f}" if pd.notna(decile_sil) else "n/a"),
        ("PCA Components Used", f"{result.pca.n_components_selected}"),
        ("PCA Cumulative Variance Explained", f"{cum_var * 100:.1f}%"),
        ("Total Monthly Market Rx (class)", f"{total_market_trx:,.0f}"),
        ("Total Monthly Company Rx", f"{total_company_trx:,.0f}"),
        ("Overall Market Share", f"{overall_share * 100:.1f}%"),
        ("Total Monthly Company Rx Value", f"${total_value:,.0f}"),
        ("Top-Decile (Decile 1) Physicians", f"{int((targets['decile'] == 1).sum()) if 'decile' in targets else 0}"),
    ]

    start_row = 5
    ws.cell(row=start_row, column=2, value="KPI")
    ws.cell(row=start_row, column=3, value="Value")
    _style_header_row(ws, start_row, 2, 2)

    for i, (label, value) in enumerate(kpis, start=start_row + 1):
        ws.cell(row=i, column=2, value=label).font = KPI_LABEL_FONT
        ws.cell(row=i, column=3, value=value).font = KPI_VALUE_FONT

    ws.cell(
        row=start_row + len(kpis) + 2, column=2,
        value="Segment labels (ranked by mean monthly Rx value):",
    ).font = Font(bold=True)
    label_row = start_row + len(kpis) + 3
    for i, (cluster_id, label) in enumerate(sorted(result.cluster_labels.items(), key=lambda kv: str(kv[1]))):
        ws.cell(row=label_row + i, column=2, value=f"Segment {int(cluster_id)}")
        ws.cell(row=label_row + i, column=3, value=label)

    _autofit_columns(ws, {"A": 2, "B": 38, "C": 22})


def _write_cluster_profiles_sheet(wb: Workbook, result: SegmentationResult) -> None:
    ws = wb.create_sheet("Cluster Profiles")
    ws["B2"] = "Segment Profiles"
    ws["B2"].font = TITLE_FONT
    ws["B3"] = "Mean feature values per segment, from PCA + K-Means clustering."
    ws["B3"].font = SUBTITLE_FONT

    profiles = result.cluster_profiles.copy()
    if "cluster" in profiles.columns:
        profiles.insert(1, "label", profiles["cluster"].map(result.cluster_labels))
    else:
        profiles = pd.DataFrame(columns=["cluster", "label"])

    header_row = 5
    headers = [str(c) for c in profiles.columns]
    for j, h in enumerate(headers):
        ws.cell(row=header_row, column=2 + j, value=h)
    _style_header_row(ws, header_row, 2, len(headers))

    for i, row in enumerate(profiles.itertuples(index=False), start=header_row + 1):
        for j, val in enumerate(row):
            v = round(val, 3) if isinstance(val, float) else val
            ws.cell(row=i, column=2 + j, value=v)

    last_row = header_row + len(profiles)
    value_col_idx = headers.index("company_rx_value_monthly") + 2 if "company_rx_value_monthly" in headers else None
    label_col_idx = headers.index("label") + 2

    if value_col_idx is not None and len(profiles) > 0:
        chart = BarChart()
        chart.type = "col"
        chart.title = "Mean Monthly Rx Value by Segment"
        chart.y_axis.title = "Mean Company Rx Value ($)"
        chart.x_axis.title = "Segment"
        data = Reference(ws, min_col=value_col_idx, min_row=header_row, max_row=last_row)
        cats = Reference(ws, min_col=label_col_idx, min_row=header_row + 1, max_row=last_row)
        chart.add_data(data, titles_from_data=True)
        chart.set_categories(cats)
        chart.width = 20
        chart.height = 11
        ws.add_chart(chart, f"{get_column_letter(len(headers) + 4)}{header_row}")

    widths = {"A": 2}
    for j, h in enumerate(headers):
        widths[get_column_letter(2 + j)] = max(14, len(h) + 2)
    _autofit_columns(ws, widths)


def _write_targeting_deciles_sheet(wb: Workbook, decile_table: pd.DataFrame) -> None:
    ws = wb.create_sheet("Targeting Deciles")
    ws["B2"] = "Decile-Based Targeting Summary"
    ws["B2"].font = TITLE_FONT
    ws["B3"] = "Decile 1 = top 10% highest-priority physicians by composite opportunity score."
    ws["B3"].font = SUBTITLE_FONT

    header_row = 5
    headers = ["Decile", "Physician Count", "Avg Opportunity Score", "Avg Whitespace Rx/mo", "Avg Company Rx Value/mo"]
    for j, h in enumerate(headers):
        ws.cell(row=header_row, column=2 + j, value=h)
    _style_header_row(ws, header_row, 2, len(headers))

    for i, row in enumerate(decile_table.itertuples(index=False), start=header_row + 1):
        ws.cell(row=i, column=2, value=int(row.decile))
        ws.cell(row=i, column=3, value=int(row.physician_count))
        ws.cell(row=i, column=4, value=round(float(row.avg_opportunity_score), 4))
        ws.cell(row=i, column=5, value=round(float(row.avg_whitespace_trx_monthly), 1))
        ws.cell(row=i, column=6, value=round(float(row.avg_company_rx_value_monthly), 2))

    last_row = header_row + len(decile_table)
    if len(decile_table) > 0:
        chart = BarChart()
        chart.type = "col"
        chart.title = "Avg Opportunity Score by Decile"
        chart.y_axis.title = "Avg Opportunity Score"
        chart.x_axis.title = "Decile"
        data = Reference(ws, min_col=4, min_row=header_row, max_row=last_row)
        cats = Reference(ws, min_col=2, min_row=header_row + 1, max_row=last_row)
        chart.add_data(data, titles_from_data=True)
        chart.set_categories(cats)
        chart.width = 20
        chart.height = 11
        ws.add_chart(chart, f"H{header_row}")

    _autofit_columns(ws, {"A": 2, "B": 10, "C": 16, "D": 20, "E": 20, "F": 22})


def _write_target_list_sheet(wb: Workbook, targets: pd.DataFrame) -> None:
    ws = wb.create_sheet("Target List")
    ws["B2"] = "Ranked Physician Target List"
    ws["B2"].font = TITLE_FONT

    header_row = 4
    headers = [str(c) for c in targets.columns]
    for j, h in enumerate(headers):
        ws.cell(row=header_row, column=2 + j, value=h)
    _style_header_row(ws, header_row, 2, len(headers))

    for i, row in enumerate(targets.itertuples(index=False), start=header_row + 1):
        for j, val in enumerate(row):
            v = round(val, 4) if isinstance(val, float) else val
            ws.cell(row=i, column=2 + j, value=v)

    widths = {"A": 2}
    for j, h in enumerate(headers):
        widths[get_column_letter(2 + j)] = max(14, len(h) + 2)
    _autofit_columns(ws, widths)


def generate_excel_report(
    result: SegmentationResult,
    targets: pd.DataFrame,
    decile_sil: float,
    output_path: str,
) -> str:
    """Build the full workbook and save it to `output_path`. Returns the path."""
    decile_table = decile_summary(targets)

    wb = Workbook()
    _write_summary_sheet(wb, result, targets, decile_sil)
    _write_cluster_profiles_sheet(wb, result)
    _write_targeting_deciles_sheet(wb, decile_table)
    _write_target_list_sheet(wb, targets)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    wb.save(output_path)
    return output_path


def main() -> None:
    from src.segment import segment
    from src.targeting import build_target_list

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "data", "physicians.csv")
    output_path = os.path.join(project_root, "output", "segmentation_report.xlsx")

    df = pd.read_csv(data_path)
    result = segment(df)
    targets = build_target_list(result.df, cluster_name_map=result.cluster_labels)
    decile_sil = decile_silhouette_score(
        result.scaler.transform(result.feature_matrix.to_numpy(dtype=float)), targets["decile"]
    )

    path = generate_excel_report(result, targets, decile_sil, output_path)
    print(f"k = {result.k} | silhouette = {result.silhouette:.3f} | decile silhouette = {decile_sil:.3f}")
    print(f"Excel report written to {path}")


if __name__ == "__main__":
    main()
