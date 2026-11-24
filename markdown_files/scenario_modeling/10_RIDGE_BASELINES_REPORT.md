# Scenario Playground Step 10 Ridge Baselines Report

## Status and scope

Step 10 is complete.

This step was necessary because it creates the first inspectable learned benchmark before a more flexible XGBoost model is attempted. It trained one Ridge regression per approved target, produced predictions only for the validation period, and compared those predictions with three naive baselines.

The approved sampling, features, targets, matrix, and chronological split were not reopened. No Step 11, XGBoost, walk-forward validation, final production artifact export, Supabase, backend, frontend, or server work was performed.

## Frozen matrix and split

The implementation used the approved raw, unscaled matrix:

```text
data/scenario_playground/processed/model_matrix/scenario_model_matrix_raw.csv
```

| Item | Result |
|---|---:|
| Matrix rows declared | 584 |
| Features used by every Ridge model | 58 |
| Targets modeled | 12 |
| Development rows fitted | 354 |
| Validation rows predicted | 110 |
| Boundary-purge rows parsed but excluded | 6 |
| Locked final-test rows declared | 114 |
| Locked final-test target rows parsed | 0 |

The retained periods are:

```text
development_train: 2013-08-05 through 2021-04-05
validation:        2021-05-10 through 2023-11-06
final_test_locked: begins 2023-12-11 and remains unavailable
```

The matrix reader deliberately stops after the first 470 chronological pre-final-test rows. It therefore obtains the development, validation, and six purge rows without parsing any of the remaining 114 locked rows. The split manifest may be read because it contains split metadata rather than target values.

The existing target-end-date boundary purge remains in force. The four development-boundary and two validation-boundary purge rows are not used for fitting or evaluation.

## Naive baselines

Each target was evaluated against:

1. `naive_zero`: always predict zero in the target's output unit.
2. `naive_training_mean`: always predict that target's mean calculated from the 354 development rows.
3. `naive_recent_trailing_21_observation`: predict the same instrument's movement over its latest 21 nonmissing source observations ending on the row's `target_baseline_date`.

The recent-trailing baseline is target-consistent:

- price, index, and ETF targets use the same selected source and percent-return formula as their target;
- the 10-year-yield target uses `DGS10` and a basis-point change;
- ETF history uses distribution-adjusted prices;
- index, currency-index, and continuous-futures history uses unadjusted closes;
- readers are bounded at `2023-11-02`, the latest validation target-baseline date;
- all 1,320 target-row trailing calculations pass their point-in-time check.

It uses 21 source observations, not 21 event-aware model rows.

## Ridge pipeline

Every target uses an independent scikit-learn pipeline:

```text
SimpleImputer(strategy="median", keep_empty_features=True)
StandardScaler()
Ridge(
    fit_intercept=True,
    solver="auto",
    tol=0.0001,
    alpha=selected per target
)
```

The imputer, scaler, and Ridge estimator are all fitted on `development_train` only. Validation rows are transformed by those fitted objects but never contribute to imputation medians, scaling means or scales, or model coefficients.

The approved matrix currently has no missing values. The imputer remains inside the pipeline as a safe, reproducible contract for later inference rather than as a whole-dataset preprocessing step.

The deterministic alpha grid is:

```text
0.01, 0.1, 1, 10, 100, 1,000, 10,000, 100,000, 1,000,000
```

For each target, validation MAE selects the alpha. Validation RMSE and then the smaller alpha are deterministic tie-breakers. This is permitted model selection on the validation set, but it means these metrics are validation-selection results rather than final unbiased performance estimates.

## Validation metrics for all targets

MAE and RMSE are in percent for the 11 return targets and basis points for `ten_year_yield_bps`.

| Target | Alpha | Ridge MAE | Ridge RMSE | OOS R² | Pred/actual corr. | Directional accuracy |
|---|---:|---:|---:|---:|---:|---:|
| S&P 500 | 100,000 | 3.7377 | 4.6694 | -0.0060 | 0.0462 | 59.09% |
| Nasdaq | 10,000 | 4.9829 | 6.2537 | -0.0369 | -0.0179 | 58.18% |
| 10-year yield (bps) | 1,000,000 | 26.2446 | 32.9241 | -0.1199 | 0.0831 | 42.73% |
| DXY | 1,000 | 1.8372 | 2.2659 | 0.0368 | 0.2509 | 61.82% |
| Gold | 10 | 2.6607 | 3.4170 | 0.2516 | 0.5153 | 71.82% |
| Oil | 1,000,000 | 8.9681 | 10.8048 | -0.0039 | -0.1301 | 49.09% |
| Technology | 1,000,000 | 5.2057 | 6.5095 | -0.0016 | -0.0146 | 61.82% |
| Energy | 10,000 | 6.9518 | 8.7284 | -0.0254 | 0.1266 | 63.64% |
| Financials | 100,000 | 4.2335 | 5.4672 | -0.0145 | -0.0332 | 52.73% |
| Utilities | 1,000 | 4.2339 | 5.6337 | -0.0264 | 0.2153 | 52.73% |
| Healthcare | 100,000 | 3.2963 | 3.9211 | -0.0098 | 0.2228 | 49.09% |
| Consumer Discretionary | 10,000 | 5.6084 | 7.0764 | -0.0335 | -0.0039 | 50.91% |

Prediction distributions, including minimum, 5th percentile, quartiles, median, 95th percentile, maximum, standard deviation, and sign counts, are persisted for every target and strategy. Metrics are also calculated separately for validation calendar years 2021, 2022, and 2023, containing 30, 43, and 37 validation rows respectively.

## Ridge versus naive baselines

Each cell below is `MAE / RMSE`.

| Target | Zero | Training mean | Recent trailing | Ridge |
|---|---:|---:|---:|---:|
| S&P 500 | 3.8205 / 4.7007 | 3.7388 / 4.6704 | 5.7350 / 7.1678 | **3.7377 / 4.6694** |
| Nasdaq | 5.0777 / **6.1783** | 4.9935 / 6.2027 | 7.5313 / 9.4160 | **4.9829** / 6.2537 |
| 10-year yield (bps) | **25.8818 / 32.3058** | 26.2420 / 32.9211 | 33.6909 / 43.3325 | 26.2446 / 32.9241 |
| DXY | 1.9118 / 2.3662 | 1.8847 / 2.3379 | 2.5217 / 3.1584 | **1.8372 / 2.2659** |
| Gold | 3.1722 / 3.9622 | 3.1573 / 3.9505 | 5.1093 / 6.2643 | **2.6607 / 3.4170** |
| Oil | **8.8916 / 10.8005** | 8.9679 / 10.8034 | 11.9663 / 15.0143 | 8.9681 / 10.8048 |
| Technology | 5.4914 / 6.7039 | **5.2057 / 6.5093** | 7.8895 / 9.5364 | 5.2057 / 6.5095 |
| Energy | 7.0508 / 8.8555 | 7.0570 / 8.8648 | 9.9892 / 12.6319 | **6.9518 / 8.7284** |
| Financials | 4.2349 / **5.4409** | 4.2350 / 5.4660 | 6.9297 / 8.3652 | **4.2335** / 5.4672 |
| Utilities | 4.4929 / 5.5927 | 4.4505 / **5.5697** | 6.6595 / 8.3135 | **4.2339** / 5.6337 |
| Healthcare | **3.2048** / 3.9534 | 3.2968 / 3.9300 | 4.7855 / 6.1088 | 3.2963 / **3.9211** |
| Consumer Discretionary | **5.5388 / 6.9721** | 5.6358 / 7.0246 | 8.5746 / 10.6532 | 5.6084 / 7.0764 |

Summary:

```text
Ridge beats the best naive MAE:   7 of 12 targets
Ridge beats the best naive RMSE:  5 of 12 targets
Positive Ridge OOS R-squared:     2 of 12 targets (DXY and Gold)
```

The strongest result is Gold: Ridge improves MAE by 15.73% and RMSE by 13.51% relative to the best naive baseline. DXY also has positive OOS R-squared and improves both error metrics. Several other apparent wins are small, and most targets have negative OOS R-squared.

Very large selected alphas for the 10-year yield, Oil, and Technology shrink their coefficients almost to zero. That is useful diagnostic evidence: the current linear model does not find stable validation signal for those outputs and behaves close to a training-mean model. It is not evidence that those targets should be removed.

## Files and outputs

### Code, configuration, tests, and documentation

```text
scenario_modeling/src/evaluation.py
scenario_modeling/src/ridge_baselines.py
scenario_modeling/tests/test_ridge_baselines.py
scenario_modeling/notebooks/08_ridge_baselines.ipynb
scenario_modeling/requirements.txt
markdown_files/scenario_modeling/10_RIDGE_BASELINES_REPORT.md
markdown_files/scenario_modeling/README.md
```

The modeling requirements record the exact Step 10 runtime:
`numpy==2.4.3`, `scikit-learn==1.9.0`, `scipy==1.18.0`,
`joblib==1.5.3`, `narwhals==2.24.0`, and
`threadpoolctl==3.6.0`. No dependency was added to the application
backend requirements.

### Generated local validation artifacts

All generated artifacts are under:

```text
data/scenario_playground/interim/models/ridge/
```

| File | Shape or role |
|---|---|
| `ridge_validation_predictions.csv` | 1,320 rows × 14 columns; 110 validation predictions for each of 12 targets |
| `ridge_validation_metrics.csv` | 48 rows × 24 columns; 12 targets × 4 strategies |
| `ridge_validation_metrics_by_year.csv` | 144 rows × 23 columns; 12 targets × 4 strategies × 3 years |
| `ridge_alpha_search.csv` | 108 rows × 25 columns; 12 targets × 9 alphas |
| `ridge_selected_configurations.csv` | 12 rows × 22 columns |
| `ridge_standardized_coefficients.csv` | 696 rows × 7 columns; 12 targets × 58 features |
| `ridge_preprocessing_fit_audit.csv` | 696 rows × 16 columns; train-only medians and scaling statistics |
| `ridge_trailing_baseline_audit.csv` | 1,320 rows × 18 columns; source dates, values, formulas, and point-in-time checks |
| `ridge_naive_comparison.csv` | 12 rows × 16 columns |
| `ridge_baseline_summary.json` | configuration, provenance, hashes, metrics, validation checks, and training gate |

No serialized model object or final production artifact was exported in Step 10.

## Validation

The implementation validates that:

- all 12 targets and all 58 approved features are present;
- exactly 354 development rows fit every pipeline;
- exactly 110 validation rows receive predictions;
- every imputer median and scaler mean matches development data;
- validation and locked-test rows are excluded from preprocessing fit;
- all 108 alpha candidates are evaluated;
- all three naive baselines and Ridge are evaluated for every target;
- all required overall and calendar-year metrics are present;
- all validation predictions are finite and reproducible;
- every trailing baseline uses only source values known by its baseline date;
- six boundary-purge rows remain excluded;
- the target-end-date purge remains in force;
- zero locked final-test target rows are parsed;
- no model objects or out-of-scope artifacts are exported.

The dedicated Step 10 test module contains 13 passing tests, including a sentinel test proving that the bounded matrix reader does not parse a row beyond its supplied limit.

## Interpretation and remaining limitations

The pipeline is reproducible and its predictions are numerically sensible, but predictive performance is modest:

- validation is a single historical period and also selected alpha;
- event-aware 21-session labels overlap within each split;
- walk-forward evaluation has not yet tested stability across repeated chronological folds;
- most Ridge OOS R-squared values are negative;
- the regional manufacturing PMI proxy and continuous `CL=F` futures limitations remain inherited from the approved matrix;
- the high-alpha solutions show that several targets are effectively intercept-dominant under a linear specification.

These are model findings, not Step 10 implementation blockers. The locked final test must remain untouched until the later plan explicitly authorizes its one-time evaluation.

## Completion gate

```text
STEP_10_RIDGE_COMPLETE_READY_FOR_STEP_11_XGBOOST_PROTOTYPE
```

Step 10 produced the required first out-of-sample validation predictions and establishes a transparent baseline for judging the shallow XGBoost prototype in Step 11.
