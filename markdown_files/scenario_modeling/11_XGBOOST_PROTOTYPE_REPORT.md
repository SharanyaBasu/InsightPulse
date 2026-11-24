# Scenario Playground Step 11 XGBoost Prototype Report

## Status and scope

Step 11 is complete.

The step was necessary because it tests whether a deliberately small boosted-tree model adds honest validation value beyond the transparent Step 10 baselines before the approach is expanded to all 12 targets.

Only the S&P 500 21-trading-session return was trained. Event-aware sampling remained final. The approved 584-row, 58-feature matrix, chronological development/validation split, and target-end-date purge were preserved. No weekly comparison, Step 12 walk-forward validation, other-target training, final-test access, production artifact export, Supabase, backend, frontend, or server work was performed.

## Frozen data boundary

| Item | Result |
|---|---:|
| Approved matrix rows | 584 |
| Pre-final-test matrix rows parsed | 470 |
| Development rows available | 354 |
| Validation rows | 110 |
| Boundary-purge rows excluded | 6 |
| Locked final-test rows declared | 114 |
| Locked final-test target rows parsed | 0 |
| Features | 58 |
| Trained targets | 1: `sp500` |

The selected recent-history model fits only 234 development rows:

```text
2016-04-15 through 2021-04-05
```

The full-development candidates fit all 354 retained development rows:

```text
2013-08-05 through 2021-04-05
```

All candidates predict the same frozen 110-row validation period:

```text
2021-05-10 through 2023-11-06
```

The loader parses exactly the first 470 chronological matrix rows and does not parse any of the remaining 114 locked rows.

## Candidate design

This was a bounded six-candidate experiment rather than a broad hyperparameter search.

Every candidate uses:

```text
objective:             reg:squarederror
early-stopping metric: validation MAE
tree method:           CPU histogram
maximum depth:         1 or 2
low learning rate:     0.02 to 0.03
row subsampling:       0.70 to 0.85
column subsampling:    0.65 to 0.75
L1 regularization:     0.10 to 0.75
L2 regularization:     10 to 30
deterministic seed:    20260725
threads:               1
```

Three conservative architectures were each tested on full development history and the most recent five years.

| Rank | Candidate | Train rows | Depth | Learning rate | Trees | Train MAE | Validation MAE | Validation RMSE | OOS R² |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `recent_5y__regularized_stumps` | 234 | 1 | 0.030 | 82 | 2.8013 | **3.5837** | **4.5239** | **0.0558** |
| 2 | `recent_5y__regularized_depth2` | 234 | 2 | 0.025 | 88 | 2.4861 | 3.6110 | 4.5484 | 0.0455 |
| 3 | `recent_5y__strongly_regularized_depth2` | 234 | 2 | 0.020 | 127 | 2.5280 | 3.6486 | 4.5802 | 0.0321 |
| 4 | `full_development__regularized_depth2` | 354 | 2 | 0.025 | 38 | 2.6298 | 3.6842 | 4.6315 | 0.0103 |
| 5 | `full_development__regularized_stumps` | 354 | 1 | 0.030 | 80 | 2.6134 | 3.6866 | 4.6443 | 0.0048 |
| 6 | `full_development__strongly_regularized_depth2` | 354 | 2 | 0.020 | 76 | 2.5357 | 3.7078 | 4.6697 | -0.0061 |

All six candidates stopped early.

## Selected configuration

```text
candidate:             recent_5y__regularized_stumps
training rows:         234
training dates:        2016-04-15 through 2021-04-05
max depth:             1
learning rate:         0.03
min child weight:      8
subsample:             0.85
column sample:         0.75
L1 alpha:              0.10
L2 lambda:             10
gamma:                 0
best iteration:        81, zero-based
trees used:            82
rounds evaluated:      122
early stopping rounds: 40
```

Selection used validation MAE, followed by validation RMSE, lower complexity, and candidate ID as deterministic tie-breakers.

The same candidate was trained twice from the same development data. Its maximum validation-prediction difference was exactly zero.

## Validation comparison

| Strategy | MAE | RMSE | OOS R² | Correlation | Directional accuracy | Prediction range |
|---|---:|---:|---:|---:|---:|---|
| Zero | 3.8205 | 4.7007 | -0.0195 | undefined | 0.00% | 0 to 0 |
| Training mean | 3.7388 | 4.6704 | -0.0064 | undefined | 59.09% | 1.0225 to 1.0225 |
| Recent trailing return | 5.7350 | 7.1678 | -1.3705 | -0.2314 | 45.45% | -10.8556 to 9.2993 |
| Ridge | 3.7377 | 4.6694 | -0.0060 | 0.0462 | 59.09% | 0.9290 to 1.1651 |
| Selected XGBoost | **3.5837** | **4.5239** | **0.0558** | **0.2617** | **64.55%** | -1.3545 to 2.1584 |

Relative to Ridge, the prototype improves:

```text
MAE:  4.12%
RMSE: 3.12%
```

It also beats the best naive baseline on both MAE and RMSE. Unlike Ridge, it produces both positive and negative predictions and has positive validation OOS R-squared.

These are configuration-selection results from one validation period, not final unbiased model performance.

## Train-versus-validation gap

For the selected candidate:

```text
development MAE:                 2.8013
validation MAE:                  3.5837
absolute generalization gap:     0.7823
validation/development ratio:    1.2793
development RMSE:                4.0934
validation RMSE:                 4.5239
```

The predefined warning requires both a ratio above 1.5 and an MAE gap above one percentage point. The selected stump model does not trigger it. The depth-2 candidates have larger gaps, supporting the shallower selection.

This does not prove the selected candidate is stable; Step 12 must repeat the comparison through purged walk-forward folds.

## Performance over time

| Validation year | Rows | XGBoost MAE | Ridge MAE | XGBoost RMSE | Ridge RMSE | XGBoost directional |
|---:|---:|---:|---:|---:|---:|---:|
| 2021 | 30 | 1.9607 | 2.0568 | 2.5435 | 2.6461 | 80.00% |
| 2022 | 43 | 5.0323 | 5.2482 | 5.8907 | 6.1342 | 51.16% |
| 2023 | 37 | 3.2161 | 3.3452 | 3.9078 | 3.9258 | 67.57% |

XGBoost improves MAE in each validation year, but 2022 remains substantially harder than 2021 or 2023.

The largest residuals cluster in volatile 2022 forward windows. The largest absolute error is the September 14, 2022 row: the realized 21-session return was approximately -12.69%, while the prototype predicted approximately +0.05%.

## VIX-regime diagnostics

| VIX regime | Rows | MAE | RMSE | OOS R² | Correlation | Directional |
|---|---:|---:|---:|---:|---:|---:|
| Below 20 | 54 | 2.6596 | 3.4061 | 0.0546 | 0.2435 | 64.81% |
| 20 to below 30 | 45 | 4.8426 | 5.6688 | 0.0460 | 0.2808 | 62.22% |
| 30 or above | 11 | 2.9698 | 4.0299 | -0.0916 | 0.5712 | 72.73% |

The medium-VIX regime has the largest errors. The high-VIX subset is small, so its correlation and directional result are not stable evidence.

## Stress-period diagnostics

| Period | Role | Rows | MAE | RMSE | OOS R² | Directional |
|---|---|---:|---:|---:|---:|---:|
| COVID market shock | Development in-sample | 18 | 6.7282 | 9.4960 | 0.2653 | 83.33% |
| 2022 inflation/tightening | Validation out-of-sample | 43 | 5.0323 | 5.8907 | 0.0030 | 51.16% |
| 2023 regional-bank stress | Validation out-of-sample | 8 | 2.0936 | 2.6197 | -0.9740 | 100.00% |

COVID cannot be treated as out-of-sample evidence because it belongs to the selected training window. The bank-stress subset contains only eight rows and its negative R² shows that directional accuracy alone can be misleading.

## Feature importance

The top five features by total gain are:

| Rank | Feature | Normalized total gain |
|---:|---|---:|
| 1 | `nasdaq_zscore_252d` | 20.07% |
| 2 | `cpi_yoy_pct` | 17.78% |
| 3 | `vix_change_points` | 15.93% |
| 4 | `sp500_zscore_252d` | 11.57% |
| 5 | `oil_drawdown_252d_pct` | 9.69% |

Feature importance is descriptive of this fitted sample and does not establish causality.

## Scenario-input sensitivity

The median-state one-feature-at-a-time sweep finds:

- `vix_change_points` changes the prototype prediction over an approximately 1.15-percentage-point span;
- the other seven scenario-shock inputs do not change the selected stump model's prediction over their selected-training ranges;
- VIX Change is the only scenario input used in any selected-model split;
- all sensitivity predictions are finite and within the training ranges;
- no clipping or post-model adjustment was applied.

This is the most important limitation of the prototype for the Scenario Playground. Its aggregate validation improvement does not mean it has learned useful sensitivity to all eight user controls. Step 12 must consider both predictive stability and scenario responsiveness. Ridge and XGBoost should both remain candidates.

## Files created or modified

### Code, tests, notebook, requirements, and documentation

```text
scenario_modeling/src/xgboost_prototype.py
scenario_modeling/tests/test_xgboost_prototype.py
scenario_modeling/notebooks/09_xgboost_prototype.ipynb
scenario_modeling/requirements.txt
markdown_files/scenario_modeling/11_XGBOOST_PROTOTYPE_REPORT.md
markdown_files/scenario_modeling/README.md
```

`xgboost==3.3.0` was installed into the registered
`backend/.venv` modeling environment and pinned in the modeling requirements.
Its NumPy and SciPy dependencies were already present and pinned.

### Generated local outputs

All generated outputs are under:

```text
data/scenario_playground/interim/models/xgboost_prototype/
```

| File | Shape or role |
|---|---|
| `sp500_candidate_summary.csv` | 6 candidates × 36 columns |
| `sp500_candidate_metrics.csv` | 12 candidate/evaluation-role rows × 26 columns |
| `sp500_learning_curves.csv` | 751 boosting-round records × 7 columns |
| `sp500_validation_predictions.csv` | 110 validation rows × 15 columns |
| `sp500_comparison_metrics.csv` | 5 strategies × 24 metric columns |
| `sp500_comparison_metrics_by_year.csv` | 15 strategy/year rows × 23 columns |
| `sp500_feature_importance.csv` | 58 features × 9 columns |
| `sp500_residuals_over_time.csv` | 110 rows × 12 columns |
| `sp500_vix_regime_metrics.csv` | 3 regimes × 24 columns |
| `sp500_crisis_period_metrics.csv` | 3 periods × 25 columns |
| `sp500_scenario_sensitivity.csv` | 80 sensitivity points × 11 columns |
| `sp500_scenario_sensitivity_summary.csv` | 8 inputs × 12 columns |
| `sp500_xgboost_prototype_summary.json` | Configuration, provenance, hashes, findings, validations, and gate |

No Booster, pipeline, or other production model artifact was exported.

## Limitations and Step 12 decision

- Validation performs both early stopping and candidate selection.
- Event-aware labels still overlap within each partition.
- Walk-forward stability has not been measured.
- Only S&P 500 was trained.
- The selected model uses only one of the eight scenario-shock features.
- COVID diagnostics are in-sample.
- Crisis and high-VIX subsets are small.
- The inherited PMI proxy and continuous `CL=F` limitations remain.

The result is promising enough to retain XGBoost, but not strong enough to promote it:

```text
Carry both Ridge and the selected shallow XGBoost candidate into
Step 12 purged walk-forward validation.
```

## Completion gate

```text
STEP_11_XGBOOST_PROTOTYPE_COMPLETE_READY_FOR_STEP_12_WALK_FORWARD_VALIDATION
```
