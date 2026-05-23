# Baseline Regression Model

A simple OLS regression that predicts six market targets from eight macro-and-market features. Intentionally minimal — the point is to have a working end-to-end pipeline that future, better models can be measured against.

## What it does

For each target, fit:

```
target_return_t = β₀ + β₁·feature₁_t + β₂·feature₂_t + ... + ε_t
```

All features and targets are measured contemporaneously (same month). This makes the model *descriptive*, not predictive — see Limitations.

## How to run

From `backend/` with the venv active:

```bash
source venv/bin/activate
python -m regression_baseline.pipeline
```

First run takes ~30–60s (fetches ~10y from yfinance + FRED). Subsequent runs within 24h read the parquet cache and finish in ~1s. To force a re-fetch: `rm -rf regression_baseline/cache/`.

## Features (inputs)

| Feature | Source | Transform | Notes |
|---|---|---|---|
| `fed_funds_change` | FRED `FEDFUNDS` | `.diff()` (pp) | |
| `cpi_surprise` | FRED `CPIAUCSL` | `.pct_change(12)` | YoY headline inflation, **proxy for surprise** |
| `oil_change` | yfinance `CL=F` | `.pct_change()` | |
| `gdp_surprise` | FRED `GDPC1` | `.pct_change()` | Quarterly ffilled, **proxy for surprise** |
| `unrate_change` | FRED `UNRATE` | `.diff()` (pp) | |
| `pmi` | FRED `CFNAI` | level | **PMI proxy.** Used as level since CFNAI is already a deviation from trend (mean-zero by construction). |
| `dxy_change` | yfinance `DX-Y.NYB` | `.pct_change()` | |
| `vix_change` | yfinance `^VIX` | `.pct_change()` | Excluded for `sp500_return` and `nasdaq_return` (see below) |

## Targets (outputs)

| Target | Source | Transform |
|---|---|---|
| `sp500_return` | yfinance `^GSPC` | `.pct_change()` |
| `nasdaq_return` | yfinance `^IXIC` | `.pct_change()` |
| `dgs10_change` | FRED `DGS10` | `.diff()` (pp; multiply by 100 for bps) |
| `gold_return` | yfinance `GC=F` | `.pct_change()` |
| `oil_return` | yfinance `CL=F` | `.pct_change()` |
| `dxy_return` | yfinance `DX-Y.NYB` | `.pct_change()` |

## Feature exclusions

Some features must be dropped when fitting specific targets:

| Target | Excluded | Why |
|---|---|---|
| `sp500_return` | `vix_change` | VIX is implied volatility from SP500 options — mechanically derived from SP500. Including it leaks the target into the features. |
| `nasdaq_return` | `vix_change` | NASDAQ tracks SP500 closely enough that the same near-tautology applies. |
| `oil_return` | `oil_change` | Algebraic identity: same series, same transform — would force R²=1. |
| `dxy_return` | `dxy_change` | Algebraic identity. |

These are configured in `EXCLUDED_FEATURES` in `pipeline.py`. Adding new exclusion rules is a one-line edit there.

## Outputs

Written to `results/`:

- **`metrics.json`** — train and test R² / RMSE / MAE per target, plus `n_train`, `n_test`, `n_features`, `excluded_features`.
- **`coefficients.csv`** — one row per target, one column per feature, plus intercept. Empty cells = feature excluded for that target.
- **`predictions.csv`** — long format: date, target, actual, predicted, set (train/test).
- **`feature_correlations.csv`** — 8×8 Pearson correlation matrix of the feature set.
- **`feature_correlations.png`** — heatmap of the same, for visual inspection.

## Current results

Snapshot run dated **2026-05-23**. Monthly panel May 2017 → May 2026 (109 months, 87 train / 22 test, chronological 80/20 split). `results/metrics.json` is the source of truth — these numbers will drift as new months of data accrue.

| Target | Train R² | Test R² | Test RMSE | k | Read |
|---|---|---|---|---|---|
| `gold_return` | +0.42 | **+0.17** | 0.047 | 8 | best out-of-sample fit, still weak |
| `dgs10_change` | +0.35 | **+0.12** | 0.19 | 8 | modest signal, ~19bp monthly RMSE |
| `dxy_return` | +0.26 | **+0.09** | 0.020 | 7 | modest signal |
| `nasdaq_return` | +0.35 | **−0.38** | 0.059 | 7 | overfits; without VIX, no signal |
| `oil_return` | +0.31 | **−0.28** | 0.143 | 7 | overfits hard |
| `sp500_return` | +0.41 | **−0.52** | 0.044 | 7 | overfits; without VIX, no signal |

Where `k` is the number of features used after exclusions.

**Headline reading:** with `vix_change` excluded, the equity targets have *no* generalizable signal in this sample. Their train R²s are still 0.35–0.41, but that's pure in-sample overfit — the test R²s collapse to large negative values. A prior version of this baseline (with VIX kept as a feature for equities) reported test R²s of +0.30 (SP500) and +0.21 (NASDAQ); that fit was largely an artifact of VIX being mechanically derived from SP500. The honest baseline is what's reported above.

Directionally, several coefficients still pass a smell test (see `coefficients.csv`):
- `dxy_change` is negative for `sp500_return`, `nasdaq_return`, `gold_return` (stronger USD → equities and gold weaker ✓)
- `dxy_change` is strongly positive for `dgs10_change` (yields up ↔ USD stronger ✓)
- `oil_change` is positive for `sp500_return` (oil and equities both pro-cyclical ✓)

Magnitudes are not stable and should not be read literally — see Limitations #9 (multicollinearity) and `results/feature_correlations.png`.

## Limitations

These are not edge cases to fix later. They are properties of this baseline that bound what conclusions we can draw from it.

1. **"Surprise" is misnamed.** A real surprise = actual minus consensus economist forecast (e.g., Bloomberg ECO, Citi Surprise Index). We don't have access to historical consensus data on FRED, so `cpi_surprise` and `gdp_surprise` are change-as-proxy — equivalent to assuming the consensus forecast was always "no change from last period." YoY for CPI (12-month change) is at least the canonical reference; the assumption that economists expected zero YoY change is still wrong, but the units match the headline number. Real surprise data would change these features materially.

2. **PMI → CFNAI substitution.** Real PMI (ISM Manufacturing) is paywalled. CFNAI is a free FRED series that captures broad US economic activity from 85 indicators. It correlates with PMI but isn't identical: CFNAI is broader, monthly with revisions, and centered around 0 (PMI centered around 50). It is used as a *level* (no differencing) because the series is already constructed as a deviation from trend.

3. **GDP is quarterly, forward-filled to monthly.** Three of every four "monthly" GDP observations are stale carry-forwards. The regression sees zero change for two months, then a jump on quarter-end. This concentrates the GDP signal into 4 obs/year and biases the coefficient.

4. **Contemporaneous, not predictive.** Features and targets are both measured at month t. The model explains *what moved together*, not *what predicts what comes next*. To make this a forecast model, lag features by one period. As written, you cannot use this to trade.

5. **YoY CPI loses the first 12 months.** Because `cpi_surprise` is a 12-month percentage change, the first 12 rows of the feature matrix are NaN and get dropped. Modeling window starts 12 months later than the raw panel.

6. **Small sample.** 87 train obs / 7–8 features = ~10–12 obs per parameter. Wide train→test R² gaps and large negative test R²s on equity and oil targets are the canonical overfit signature.

7. **No regularization, no cross-validation.** Plain OLS, one train/test split. Ridge or lasso with rolling-window CV would give a more honest read of generalization. Skipped intentionally for a baseline.

8. **No regime awareness.** The 2017–2026 sample includes: low-vol bull market, COVID crash, ZIRP, inflation shock, rate hikes, sticky-yield regime. The model assumes one stable relationship across all of these.

9. **Severe multicollinearity in the macro block.** From `feature_correlations.csv`: `unrate_change` ↔ `pmi` = **−0.95**, `gdp_surprise` ↔ `pmi` = +0.67, `gdp_surprise` ↔ `unrate_change` = −0.68. The macro features are essentially measuring the same business-cycle signal three different ways. OLS will arbitrarily distribute weight across them; individual coefficients are not interpretable. Ridge regression with standardized features is the standard fix.

10. **Linear functional form.** Macro-to-market relationships are nonlinear (a 25bp hike during a banking crisis ≠ a 25bp hike at full employment). Linear regression captures none of this.

11. **Continuous-contract artifacts in futures.** `GC=F` and `CL=F` are front-month continuous contracts. They have small jumps at roll dates that aren't real returns to a position holder.

12. **Vintage data.** FRED serves the *latest revised* series, not what was known in real time. CPI gets revised after release. Real-time vs. revised data isn't distinguished here.

13. **Two of the targets are also features.** `oil` and `dxy` appear on both sides. Self-feature exclusion handles the algebraic identity; near-tautologies (VIX→SP500/NASDAQ) are handled separately. But other features that correlate with the same-month target move can still bias coefficients via channels we haven't excluded.

## How to extend

- **New feature/target definition:** edit `FEATURE_DEFS` / `TARGET_DEFS` in `pipeline.py`. Add a new transform branch in `_transform()` if needed (current branches: `pct`, `pct_yoy`, `diff`, `level`).
- **New exclusion rule:** add an entry to `EXCLUDED_FEATURES` in `pipeline.py`.
- **New raw series (e.g., add Bitcoin as a target):** edit `YFINANCE_TICKERS` or `FRED_SERIES` in `data.py`, then `rm -rf cache/` to force re-fetch, then add to `TARGET_DEFS` in `pipeline.py`.
- **Make it predictive:** in `pipeline.py`, replace the target build with `y.shift(-1)` so features at t predict returns at t+1. Drop the last row to handle the introduced NaN.
- **Add regularization:** swap `LinearRegression()` for `Ridge(alpha=...)` or `Lasso(alpha=...)` from `sklearn.linear_model`. Standardize features first with `StandardScaler` — penalty terms depend on coefficient magnitude, which depends on feature scale.
- **Cross-validation:** use `sklearn.model_selection.TimeSeriesSplit` instead of the single chronological split.

## File map

```
regression_baseline/
├── __init__.py                       # package marker
├── .gitignore                        # ignores cache/
├── README.md                         # this file
├── data.py                           # external fetching + monthly panel build
├── pipeline.py                       # features → targets → train → evaluate → save
├── cache/                            # gitignored parquet cache
│   ├── market_monthly.parquet
│   └── macro_monthly.parquet
└── results/                          # committed outputs
    ├── metrics.json
    ├── coefficients.csv
    ├── predictions.csv
    ├── feature_correlations.csv
    └── feature_correlations.png
```
