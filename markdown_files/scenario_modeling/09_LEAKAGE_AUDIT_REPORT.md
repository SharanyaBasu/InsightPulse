# Scenario Playground Step 9 Leakage and Data-Quality Audit Report

## Status and scope

Step 9 is complete.

The initial 59-feature matrix was audited, one structural feature-quality problem was corrected, the 58-feature matrix was regenerated, and all leakage and data-quality checks were rerun successfully.

No preprocessing was fit. No Ridge or XGBoost model was trained. No prediction was produced. The final-test target values were not used for distributions, correlations, outlier review, range decisions, or any other model decision.

## Audit sequence

```text
Initial post-CPI-policy matrix:
584 rows x 72 columns
1 key + 59 features + 12 targets

Issue found:
oil_volatility_20d_annualized_pct is structurally incoherent
through the April 2020 negative CL=F settlement

Correction:
exclude that one feature
retain all rows
leave all source data unchanged
perform no clipping or winsorization

Final re-audited matrix:
584 rows x 71 columns
1 key + 58 features + 12 targets
```

The corrected matrix is value-for-value equal to the initial matrix after dropping only the rejected oil-volatility column.

## Missingness result

The one genuine missing CPI Surprise row is:

```text
sample_date: 2013-08-16
```

It was excluded before the final audit because no compatible expectation exists strictly before the release. It was not assigned zero, future-filled, or imputed.

Final result:

```text
Rows:                         584
Missing feature values:        0
Missing target values:         0
Non-finite feature values:     0
Non-finite target values:      0
Rows imputed:                  0
```

## Extreme oil-volatility decision

The cached Yahoo `CL=F` series contains exactly one nonpositive close:

```text
Date:  2020-04-20
Close: approximately -37.63
```

Five selected state rows have 20-return windows that cross this settlement:

| Sample | State cutoff | Annualized oil volatility |
|---|---|---:|
| 2020-04-24 | 2020-04-22 | 1,173.05% |
| 2020-04-29 | 2020-04-28 | 1,180.69% |
| 2020-04-30 | 2020-04-28 | 1,180.69% |
| 2020-05-11 | 2020-05-07 | 1,193.81% |
| 2020-05-13 | 2020-05-11 | 1,196.13% |

The oil-market event is real. However, an arithmetic percentage return is not a stable quantity when a futures settlement crosses zero, so its rolling percentage volatility is not comparable with ordinary periods.

Final decision:

```text
Exclude oil_volatility_20d_annualized_pct from the model matrix.
Retain all five historical rows.
Retain the other oil state features and oil target.
Do not clip or winsorize the complete dataset.
```

The source Step 7 state table remains available for audit and is not rewritten.

## Audit scopes and final-test protection

| Scope | Rows | Allowed use in Step 9 |
|---|---:|---|
| Pre-final-test feature audit | 470 | Feature distributions, regimes, ranges, feature correlations |
| Purged pre-final-test target audit | 464 | Target distributions and feature/target correlations |
| Locked final test | 114 | Schema, timing, lock status, and missingness only |

The final-test period is:

```text
2023-12-07 through 2026-07-16
```

Its available matrix samples run from `2023-12-11` through `2026-06-11`. Its target magnitudes are excluded from all value-based audit tables and decisions.

## Scenario-shock ranges

The following ranges use pre-final-test features only. They include structural zeros. Release flags distinguish no event from an event with a measured zero shock.

| Contract input | Historical feature | Active rows/windows | Nonzero | Minimum | Median | Maximum | Unit |
|---|---|---:|---:|---:|---:|---:|---|
| `fed_funds_rate_change_bps` | same | 84 | 25 | -100 | 0 | 75 | bps |
| `cpi_surprise_pct` | same | 121 | 121 | -0.3712 | 0 | 0.6840 | percentage points |
| `oil_price_change_pct` | same | 470 | 470 | -32.3529 | 0.2545 | 55.0806 | percent |
| `gdp_growth_surprise_pct` | same | 41 | 41 | -3.8882 | 0 | 1.8210 | annualized percentage points |
| `unemployment_change_pct` | same | 124 | 93 | -2.2 | 0 | 10.3 | percentage points |
| `pmi_change_points` | `kc_fed_manufacturing_pmi_proxy_change_points` | 124 | 115 | -12 | 0 | 10 | PMI-style points |
| `dxy_change_pct` | same | 470 | 468 | -3.8535 | 0.0228 | 2.3048 | percent |
| `vix_change_points` | same | 470 | 465 | -6.76 | -0.375 | 28.17 | VIX points |

The persisted range report also contains the 1st, 5th, 25th, 75th, 95th, and 99th percentiles plus active-event-only minima, medians, and maxima.

Every input has at least 20 active and 20 nonzero pre-final-test observations. Variation is sufficient for a regularized Ridge baseline. Sparse macro features remain a reason to keep later XGBoost models shallow.

## Distribution and outlier review

Persisted distribution tables cover:

```text
58 final features
12 targets
```

Each table records:

```text
count
missing and non-finite count
zero and unique count
minimum and maximum
1st, 5th, 25th, 50th, 75th, 95th, and 99th percentiles
mean and standard deviation
IQR review bounds and review count
```

The IQR fields are diagnostics only. They do not delete, clip, or transform values. Sparse shock columns with a zero IQR are interpreted with their event flags and dedicated shock-range table.

Large retained observations include pandemic GDP growth, unemployment, VIX, oil returns, equity drawdowns, Treasury-yield changes, and sector returns. These represent documented market regimes rather than automatic data errors.

## Regime coverage

Pre-final-test feature rows cover:

| Calendar regime | Rows |
|---|---:|
| Pre-pandemic | 295 |
| 2020 pandemic shock | 47 |
| 2021–2022 recovery and inflation | 89 |
| 2023 pre-test period | 39 |

VIX regimes:

| VIX state | Rows |
|---|---:|
| Below 15 | 196 |
| 15 to below 25 | 208 |
| 25 or above | 66 |

Policy-rate regimes:

| Fed target midpoint | Rows |
|---|---:|
| Below 0.5% | 247 |
| 0.5% to below 3% | 174 |
| 3% or above | 49 |

The development data therefore includes calm, stressed, near-zero-rate, tightening, inflation, and pandemic conditions.

## Correlation review

Persisted correlation coverage:

```text
Feature pairs:         1,653
Target pairs:             66
Feature-target pairs:    696
```

Findings:

- 38 feature pairs have absolute Pearson correlation of at least 0.80.
- 8 target pairs have absolute Pearson correlation of at least 0.80.
- 1 feature-target pair exceeds 0.50 in absolute value: `oil_price_change_pct` with the `oil` target, approximately `0.594`.
- The strongest feature correlation is the Fed target midpoint with EFFR, approximately `0.999`.
- Nasdaq and Technology are the strongest target pair, approximately `0.961`.
- Broad equity return, drawdown, trend, and volatility features are naturally correlated.

These correlations are not target leakage. Each feature comes from Step 6 shocks or Step 7 pre-shock state data. They support the planned use of Ridge regularization and caution against high-capacity trees.

## Point-in-time checks

All checks pass:

```text
14 state availability/source-date families are on or before state_cutoff_date
615 event records pass their point-in-time checks
every available CPI expectation is strictly before release
1,755 oil/DXY/VIX source endpoints are on or before their boundaries
state_cutoff_date is before sample_date
shock_window_end_date is on or before sample_date
target_baseline_date equals state_cutoff_date
every target_end_date is exactly 21 canonical sessions after baseline
```

Market shock features ending at `sample_date` are known when the historical row is available. The target starts from the stored pre-shock baseline by contract; this is intentional scenario semantics, not future information.

## Target-overlap and split audit

Frozen raw periods:

| Period | Raw rows |
|---|---:|
| Development | 358 |
| Validation | 112 |
| Final test | 114 |

Target-end-date purge result:

| Role | Rows |
|---|---:|
| Development training | 354 |
| Development boundary purge | 4 |
| Validation evaluation | 110 |
| Validation boundary purge | 2 |
| Final test locked | 114 |

The purge does not remove 21 event-aware rows. It compares canonical-session positions and each row’s stored `target_end_date` with the next period’s first target baseline.

Passing conditions:

```text
every retained development label ends before 2021-05-06
every retained validation label ends before 2023-12-07
last retained baseline gap at each boundary = 24 canonical sessions
final-test rows are marked LOCKED_UNTIL_FINAL_EVALUATION
```

Overlapping labels still exist inside each period. Every later walk-forward fold must repeat the same target-end-date purge; random cross-validation remains prohibited.

## Leakage checklist

All 16 checks pass:

```text
no target-derived features
no future-filled state values
no future event or expectation values
no future market-shock endpoints
no scaler or learned preprocessing fit on the complete data
exact feature/target/provenance/audit separation
missing CPI row resolved without imputation
no missing or non-finite final values
oil-volatility structural problem corrected
audit correction changed only one feature
all targets use exact 21-session horizons
development labels do not cross validation
validation labels do not cross final test
purge uses canonical target-end dates, not 21 event rows
final test is locked and excluded from value-based audit
all eight shock representations have training variation
```

## Persisted Step 9 outputs

```text
data/scenario_playground/interim/audits/leakage/
  feature_distributions.csv
  target_distributions.csv
  missingness_by_year.csv
  regime_coverage.csv
  feature_correlations.csv
  target_correlations.csv
  feature_target_correlations.csv
  scenario_shock_ranges.csv
  extreme_observations.csv
  oil_volatility_review.csv
  leakage_checklist.csv
  leakage_audit_summary.json
```

Reusable implementation:

```text
scenario_modeling/src/leakage_audit.py
scenario_modeling/notebooks/07_leakage_audit.ipynb
scenario_modeling/tests/test_leakage_audit.py
```

## Remaining limitations

- The PMI shock is an explicit Kansas City Fed regional proxy, not national ISM PMI.
- Retained `CL=F` features and the oil target still inherit continuous-futures roll and contract-structure limitations.
- Event-aware rows overlap heavily inside each period.
- There are 354 development-fit and 110 validation-evaluation rows after purging, so model capacity must remain conservative.
- High feature and target correlations are expected; they make Ridge the appropriate first model.
- The final test is locked and must not be used for Step 10 model selection or preprocessing decisions.

## Training approval

```text
APPROVED_FOR_STEP_10_RIDGE_TRAINING
```

Approval means:

```text
fit Ridge and every learned preprocessing operation on
development_train rows only

use validation rows for initial out-of-sample predictions
and model decisions

do not use final_test_locked target values until the final
evaluation stage
```

## Completion decision

Step 9 is complete. The corrected matrix passes the full leakage checklist, has sufficient historical shock variation, and is ready for Step 10 Ridge training under the frozen split and train-only preprocessing rules.

No Step 10 work was performed.
