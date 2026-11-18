import os

import openpyxl
import pandas as pd
import pytest

from src.report import generate_excel_report
from src.segment import engineer_features, scale_features, segment
from src.targeting import build_target_list, decile_silhouette_score

EXPECTED_SHEETS = ["Summary", "Cluster Profiles", "Targeting Deciles", "Target List"]


def _pipeline(df, k=None):
    result = segment(df, k=k)
    targets = build_target_list(result.df, cluster_name_map=result.cluster_labels)
    scaled, _ = scale_features(engineer_features(result.df))
    dec_sil = decile_silhouette_score(scaled, targets["decile"]) if len(targets) else float("nan")
    return result, targets, dec_sil


@pytest.fixture(scope="module")
def report_path(tmp_path_factory, synthetic_df):
    result, targets, dec_sil = _pipeline(synthetic_df, k=4)
    out_dir = tmp_path_factory.mktemp("report_output")
    out_path = os.path.join(str(out_dir), "segmentation_report_test.xlsx")
    generate_excel_report(result, targets, dec_sil, out_path)
    return out_path


def test_report_file_is_created(report_path):
    assert os.path.exists(report_path)
    assert os.path.getsize(report_path) > 0


def test_report_is_valid_xlsx_workbook(report_path):
    wb = openpyxl.load_workbook(report_path)
    assert wb is not None


def test_report_contains_expected_sheets(report_path):
    wb = openpyxl.load_workbook(report_path)
    for sheet_name in EXPECTED_SHEETS:
        assert sheet_name in wb.sheetnames


def test_summary_sheet_has_silhouette_kpi(report_path):
    wb = openpyxl.load_workbook(report_path)
    ws = wb["Summary"]
    all_values = [cell.value for row in ws.iter_rows() for cell in row]
    assert any(v == "Clustering Silhouette Score" for v in all_values)
    assert any(v == "Decile-Group Silhouette Score" for v in all_values)


def test_cluster_profiles_sheet_has_chart(report_path):
    wb = openpyxl.load_workbook(report_path)
    ws = wb["Cluster Profiles"]
    assert len(ws._charts) >= 1


def test_cluster_profiles_sheet_has_expected_columns(report_path):
    wb = openpyxl.load_workbook(report_path)
    ws = wb["Cluster Profiles"]
    header_values = [ws.cell(row=5, column=c).value for c in range(2, 6)]
    assert header_values[0] == "cluster"
    assert header_values[1] == "label"


def test_targeting_deciles_sheet_has_10_rows_and_chart(report_path):
    wb = openpyxl.load_workbook(report_path)
    ws = wb["Targeting Deciles"]
    data_rows = [r for r in ws.iter_rows(min_row=6, min_col=2, max_col=2) if r[0].value is not None]
    assert len(data_rows) == 10
    assert len(ws._charts) >= 1


def test_target_list_sheet_row_count_matches_input(report_path, synthetic_df):
    wb = openpyxl.load_workbook(report_path)
    ws = wb["Target List"]
    header_row = 4
    data_rows = [r for r in ws.iter_rows(min_row=header_row + 1, min_col=2, max_col=2) if r[0].value is not None]
    assert len(data_rows) == len(synthetic_df)


def test_target_list_sheet_has_rank_and_decile_columns(report_path):
    wb = openpyxl.load_workbook(report_path)
    ws = wb["Target List"]
    headers = [ws.cell(row=4, column=c).value for c in range(2, 12)]
    assert "rank" in headers
    assert "decile" in headers
    assert "opportunity_score" in headers


def test_report_generation_on_tiny_dataset_does_not_crash(tmp_path, synthetic_df):
    tiny = synthetic_df.head(5)
    result, targets, dec_sil = _pipeline(tiny)
    out_path = os.path.join(str(tmp_path), "tiny_report.xlsx")
    path = generate_excel_report(result, targets, dec_sil, out_path)
    assert os.path.exists(path)
    wb = openpyxl.load_workbook(path)
    for sheet_name in EXPECTED_SHEETS:
        assert sheet_name in wb.sheetnames


def test_report_generation_on_empty_dataset_does_not_crash(tmp_path):
    empty = pd.DataFrame(columns=synthetic_columns())
    result, targets, dec_sil = _pipeline(empty)
    out_path = os.path.join(str(tmp_path), "empty_report.xlsx")
    path = generate_excel_report(result, targets, dec_sil, out_path)
    assert os.path.exists(path)


def synthetic_columns():
    from src.generate_data import generate_dataset
    return list(generate_dataset(1, 1).columns)
