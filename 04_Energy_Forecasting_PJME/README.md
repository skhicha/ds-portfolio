# PJME Energy Consumption Forecasting

A working hourly energy-demand forecasting pipeline on 16+ years of real PJM
Interconnection load data: ADF stationarity testing, leakage-safe calendar /
lag / rolling-window feature engineering, an XGBoost vs. SARIMA model
comparison with confidence intervals and residual diagnostics, a polished
interactive Streamlit dashboard, and a real pytest suite — all built on real
computations against the real `PJME_hourly.csv` dataset, not hardcoded
numbers.

**Achieved on the full dataset (145,224 hours after feature engineering,
chronological 80/20 split): XGBoost MAPE ≈ 0.86%, RMSE ≈ 362 MW on its
hourly held-out test set. SARIMA(1,1,1)(1,1,1,7), fit on daily means for
Jan–Nov 2017 and tested on Dec 2017, achieves MAPE ≈ 8.17%, RMSE ≈ 3,912 MW
(daily resolution).** These two numbers are evaluated on different test
periods and resolutions (hourly vs. daily, full-history split vs. a one-
month holdout), consistent with how the original notebook scoped SARIMA —
see "Notes on the two evaluations" below for why they aren't directly
comparable, and why XGBoost's number is unsurprisingly very low (hour-to-hour
load is close to a random walk, so `lag_1h` alone is extremely predictive).
See "Reproducing the full-history numbers reported above" below to
reproduce exactly.

## What it does

1. **Loads and cleans `data/PJME_hourly.csv`** (`src/data.py::load_raw`):
   parses timestamps, sorts chronologically, drops duplicate timestamps,
   resamples onto a strict hourly grid, and time-interpolates the small
   number of missing DST clock-change hours.
2. **Tests stationarity** (`src/data.py::adf_test`): Augmented Dickey-Fuller
   test on the raw series and on the first difference, with an explicit
   stationary/non-stationary verdict at p < 0.05. On the full real series,
   the raw hourly load is already found to be stationary (ADF stat ≈ -20.0,
   p ≈ 0.0) — the strong daily/weekly seasonality keeps the series
   mean-reverting at hourly resolution, so first-differencing (ADF stat ≈
   -53.9, p ≈ 0.0) sharpens but doesn't flip the verdict. This is reported
   honestly rather than forced to match the "classic non-stationary level
   series" textbook story.
3. **Engineers features** (`src/data.py::add_time_features`), leakage-safe
   throughout:
   - Calendar features: hour, day-of-week, month, quarter, day-of-year,
     week-of-year, is_weekend, plus sin/cos cyclical encodings of hour and
     month.
   - Lag features at 1, 2, 3, 24, 48, and 168 hours.
   - Rolling mean/std over 24h and 168h windows, computed on the target
     series *shifted by one hour* so no rolling statistic ever includes the
     value it's being used to help predict.
4. **Trains and compares two models** (`src/models.py`):
   - **XGBoost** (500 trees, depth 6, lr 0.05, early stopping) on the
     hourly feature frame, chronologically split 80/20 (train on the past,
     test on the future — never shuffled).
   - **SARIMA(1,1,1)(1,1,1,7)** on a daily-resampled subset (fitting SARIMA
     on 145k+ hourly observations is impractically slow for an interactive
     tool; the original notebook's scoping — train on ~11 months of daily
     means, test on the following month — is kept and made configurable).
5. **Evaluates** both models via MAPE (zero-actual-safe — rows where the
   actual value is exactly 0 are excluded from the average instead of
   dividing by zero) and RMSE (`src/evaluate.py`), and produces residual
   diagnostics (residual series, ±2σ bands) and a 95%-CI forecast band
   (±1.96×residual-σ around each prediction).
6. **Renders a polished interactive Streamlit dashboard** (`app.py`): a
   configurable training window, stationarity KPIs, a model-comparison
   table, an XGBoost tab (forecast + CI, residual diagnostics, feature
   importance), and a SARIMA tab (daily forecast + CI) — all computed live
   from whatever window the user selects, with `st.cache_data` /
   `st.cache_resource` so moving the sliders doesn't retrain from scratch
   on every widget interaction.

## Architecture

```
04_Energy_Forecasting_PJME/
├── app.py                      # Streamlit dashboard (entry point)
├── data/
│   └── PJME_hourly.csv         # raw hourly PJM load data (2002-2018)
├── src/
│   ├── data.py                 # load/clean, ADF test, feature engineering, time-based split
│   ├── models.py                # SARIMA + XGBoost training/forecasting (shared ForecastResult shape)
│   └── evaluate.py              # MAPE (zero-safe), RMSE, residual diagnostics / CI
├── tests/
│   ├── conftest.py             # synthetic hourly/daily fixtures, stationary & non-stationary series
│   ├── test_data.py            # load/clean, calendar/cyclical/lag/rolling correctness, split
│   ├── test_adf.py             # ADF test on known-stationary vs known-non-stationary series
│   ├── test_models.py          # XGBoost/SARIMA shapes, edge cases, one real-data integration test
│   └── test_evaluate.py        # MAPE/RMSE/residual-diagnostics correctness + edge cases
├── requirements.txt
├── pytest.ini
├── .streamlit/config.toml
├── .gitignore
└── README.md
```

**Data flow:** `data/PJME_hourly.csv` → `src.data.load_raw` →
`src.data.add_time_features` → `src.data.time_based_split` →
`src.models.train_xgboost` / `train_sarima` → `ForecastResult` →
`src.evaluate.mape` / `rmse` / `residual_diagnostics`. `app.py` calls the
exact same `src/` functions as the tests, so the dashboard's numbers are
always the real pipeline's numbers.

## Setup

```bash
cd 04_Energy_Forecasting_PJME

# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the interactive dashboard
streamlit run app.py
# open http://localhost:8501

# 4. Run the test suite
pytest -v
```

### Reproducing the full-history numbers reported above

```python
from src.data import load_raw, add_time_features, time_based_split, TARGET_COL
from src.models import train_xgboost, forecast_xgboost
from src.evaluate import mape, rmse

df = load_raw("data/PJME_hourly.csv")
feat = add_time_features(df, TARGET_COL)
X_train, y_train, X_test, y_test = time_based_split(feat, TARGET_COL, test_frac=0.2)
model = train_xgboost(X_train, y_train, X_test, y_test)
result = forecast_xgboost(model, X_test, y_test)
print(mape(result.y_true, result.y_pred), rmse(result.y_true, result.y_pred))
```

The Streamlit dashboard defaults to a 13-month rolling window (not the full
16-year history) purely for interactivity — widen the "Months of recent
history" slider to approach the full-history numbers, at the cost of a
slower retrain.

## Notes on the two evaluations

- XGBoost is evaluated hourly, on a chronological 80/20 split of whatever
  window is in scope (the full dataset for the headline number above).
- SARIMA is evaluated daily, trained on ~11 months of daily means and
  tested on the following month (Dec 2017 for the headline number), per
  the original notebook's scoping — full hourly SARIMA over 16 years is
  impractically slow for an interactive tool.
- Because the two models are scored on different resolutions and different
  test windows, their MAPE/RMSE numbers are not a strict apples-to-apples
  comparison; they're reported side by side for context, not as a single
  "winner" claim. XGBoost's very low MAPE is largely explained by `lag_1h`
  dominating feature importance — hour-to-hour load barely moves, so a
  near-random-walk forecast already performs very well at hourly
  resolution.

## Tech stack

Python, pandas, numpy, statsmodels (ADF test, SARIMAX), xgboost,
scikit-learn, Streamlit (dashboard), matplotlib (dashboard charts), pytest
(tests).

## Testing

```bash
pytest -v
```

29 tests, all passing, covering:
- Data loading/cleaning: deduplication, hourly resampling, time
  interpolation of missing hours.
- Feature engineering: calendar features match known dates, cyclical
  encodings bounded in [-1, 1], lag features shift correctly, rolling
  statistics provably do not leak future values (mutating future rows
  doesn't change a past row's rolling feature), warm-up row dropping, and a
  tiny-series edge case where every row has at least one NaN feature.
- Chronological train/test split (never shuffled) and its empty-input edge
  case.
- ADF test: correctly flags synthetic white noise as stationary and a
  synthetic random walk as non-stationary, confirms first-differencing a
  random walk makes it stationary, and handles too-short/all-NaN series
  without crashing.
- Model training/forecasting: output shapes match the test set with no
  NaNs for both XGBoost and SARIMA, confidence-interval ordering, an
  empty-test-set edge case for both models, and an automatic early-stopping
  fallback when the training set is smaller than the configured
  `early_stopping_rounds`. One slower test (`@pytest.mark.slow`, still fast
  in practice) runs the full XGBoost pipeline end-to-end on a real slice of
  `PJME_hourly.csv`.
- MAPE (a known hand-computed value, all-zero actuals returning NaN instead
  of crashing, mixed zero/nonzero actuals, empty input, perfect
  predictions) and RMSE (known value, empty input), plus residual
  diagnostics shape/band-direction checks and a zero-error edge case.

The Streamlit app was additionally smoke-tested with `streamlit.testing.v1.AppTest`
(full in-process script execution with zero exceptions at the default
13-month window and at both the smallest (3-month) and largest (36-month)
window settings) and via a live `streamlit run --server.headless true`
process confirmed serving HTTP 200 / a healthy `/_stcore/health` endpoint
with no errors in the server log.

## Disclaimer

`PJME_hourly.csv` is a real public PJM Interconnection dataset used here for
demonstration; forecasts produced by this tool are not operational grid
forecasts.
