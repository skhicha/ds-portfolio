from src.data import TARGET_COLUMN, load_dataset, load_raw_data


class TestLoadDataset:
    def test_returns_expected_shape_and_target(self):
        df, target = load_dataset()
        assert target == TARGET_COLUMN == "MedHouseVal"
        assert target in df.columns
        assert len(df) > 1000  # California Housing has ~20,640 rows
        assert len(df.columns) == 9  # 8 features + target

    def test_target_is_strictly_positive_and_realistic_scale(self):
        df, target = load_dataset()
        assert (df[target] > 0).all()
        # Scaled to dollars (MedHouseVal * 100_000), not the raw 0-5-ish units.
        assert df[target].median() > 10_000

    def test_no_missing_values_in_raw_data(self):
        df, _ = load_dataset()
        assert df.isnull().sum().sum() == 0


class TestLoadRawDataFallback:
    def test_ames_url_failure_falls_back_cleanly(self):
        # A deliberately unreachable/invalid-looking timeout forces the Ames
        # download to fail quickly, exercising the fallback path.
        df, target = load_raw_data(prefer_ames=True, timeout=0.001)
        assert target in ("SalePrice", "MedHouseVal")
        assert len(df) > 0
