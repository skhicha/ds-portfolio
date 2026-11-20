import numpy as np
import pandas as pd

from src.data import add_time_features, load_raw, time_based_split


def test_load_raw_dedupes_sorts_and_interpolates(raw_csv_path):
    df = load_raw(raw_csv_path)
    # Hourly grid from 00:00 to 04:00 -> 5 rows, duplicate collapsed
    assert len(df) == 5
    assert df.index.is_monotonic_increasing
    assert df.index.freqstr in ("h", "H") or pd.infer_freq(df.index) in ("h", "H")
    # The missing 02:00 hour should be filled by time-interpolation between
    # 110 (01:00) and 130 (03:00) -> 120
    assert df.loc["2020-01-01 02:00:00", "PJME_MW"] == 120.0


def test_add_time_features_calendar_matches_known_dates(hourly_series):
    feat = add_time_features(hourly_series, dropna=False)
    first_ts = hourly_series.index[0]  # 2020-01-01 00:00 is a Wednesday
    row0 = feat.iloc[0]
    assert row0["hour"] == 0
    assert row0["dayofweek"] == first_ts.day_of_week
    assert row0["month"] == 1
    assert row0["is_weekend"] in (0, 1)
    assert row0["is_weekend"] == int(first_ts.day_of_week >= 5)


def test_cyclical_encodings_are_bounded(hourly_series):
    feat = add_time_features(hourly_series, dropna=False)
    for col in ["hour_sin", "hour_cos", "month_sin", "month_cos"]:
        assert feat[col].min() >= -1.0 - 1e-9
        assert feat[col].max() <= 1.0 + 1e-9


def test_lag_features_shift_correctly(hourly_series):
    feat = add_time_features(hourly_series, lags=[1, 24], rolling_windows=[], dropna=False)
    target = hourly_series["PJME_MW"]
    # lag_1h at row i should equal the target at row i-1
    for i in [1, 2, 50, 100]:
        assert feat["lag_1h"].iloc[i] == target.iloc[i - 1]
    for i in [24, 25, 100]:
        assert feat["lag_24h"].iloc[i] == target.iloc[i - 24]


def test_rolling_features_do_not_leak_future_values(hourly_series):
    feat = add_time_features(hourly_series, lags=[], rolling_windows=[24], dropna=False)
    target = hourly_series["PJME_MW"]
    # roll_mean_24h at row i must equal mean of target[i-24:i] (strictly before i)
    i = 100
    expected_mean = target.iloc[i - 24 : i].mean()
    assert np.isclose(feat["roll_mean_24h"].iloc[i], expected_mean)
    # Changing a future value (after i) must not change the rolling feature at i
    mutated = hourly_series.copy()
    mutated.iloc[i + 1 :] = mutated.iloc[i + 1 :] + 10_000
    feat_mutated = add_time_features(mutated, lags=[], rolling_windows=[24], dropna=False)
    assert np.isclose(feat_mutated["roll_mean_24h"].iloc[i], feat["roll_mean_24h"].iloc[i])


def test_add_time_features_dropna_removes_warmup_rows(hourly_series):
    feat = add_time_features(hourly_series, lags=[1, 168], rolling_windows=[24])
    assert feat.isna().sum().sum() == 0
    assert len(feat) == len(hourly_series) - 168


def test_add_time_features_on_tiny_series_can_drop_everything(tiny_hourly_series):
    # Series shorter than the largest lag (168h) -> every row has at least
    # one NaN lag feature, so dropna leaves an empty (but valid) frame.
    feat = add_time_features(tiny_hourly_series)
    assert len(feat) == 0
    assert list(feat.columns)  # columns still defined


def test_time_based_split_is_chronological_and_unshuffled(hourly_series):
    feat = add_time_features(hourly_series)
    X_train, y_train, X_test, y_test = time_based_split(feat, test_frac=0.2)
    assert len(X_train) + len(X_test) == len(feat)
    assert X_train.index.max() < X_test.index.min()
    assert list(y_train.index) == list(X_train.index)


def test_time_based_split_on_empty_frame():
    import pandas as pd

    empty = pd.DataFrame(columns=["PJME_MW", "hour"])
    X_train, y_train, X_test, y_test = time_based_split(empty)
    assert len(X_train) == 0
    assert len(X_test) == 0
    assert len(y_train) == 0
    assert len(y_test) == 0
