# Scenario Playground Step 8 Point-in-Time Matrix Report

## Status and scope

Step 8 is complete.

The completed Step 5 targets, Step 6 historical shocks, and Step 7 state features are assembled into one deterministic, raw point-in-time matrix. The model values, timing/event provenance, source-row audit fields, row/feature exclusions, and chronological split controls are stored separately.

No imputation, scaling, clipping, winsorization, learned preprocessing, model fitting, prediction, Supabase access, or application-code change occurred.

## Why the Step 8–9 batch is appropriate

Step 8 is the first stage where shocks, state, and targets coexist. Running the leakage audit immediately after assembly is safe and efficient because it can catch cross-family timing, missingness, role-separation, and outlier problems before training. Step 9 found one concrete feature-quality problem, so the matrix was regenerated and re-audited before being accepted.

The two steps remain modular:

- Step 8 owns assembly, schema, provenance, missing-row policy, periods, and purge metadata.
- Step 9 owns distribution review, leakage checks, the oil-volatility decision, and the training gate.

## Authoritative inputs

```text
data/scenario_playground/interim/sampling/selected_sampling_index.csv
data/scenario_playground/interim/features/shocks/scenario_shocks.csv
data/scenario_playground/interim/features/shocks/scenario_shock_summary.json
data/scenario_playground/interim/features/state/state_features.csv
data/scenario_playground/interim/features/state/state_feature_summary.json
data/scenario_playground/interim/targets/market_targets.csv
data/scenario_playground/interim/targets/market_target_summary.json
data/scenario_playground/raw/yahoo/GSPC.csv
```

All four row-indexed CSV inputs contain the same 585 unique, chronologically ordered `EVENT_AWARE` sample dates. Shared state cutoffs, shock boundaries, target boundaries, event identities, and model-availability dates are checked for exact equality.

Steps 6 and 7 already performed source-level as-of alignment. Step 8 therefore uses strict one-to-one sample-row joins. Re-running a looser as-of join between already materialized row tables would risk changing completed timing decisions.

## Missing CPI Surprise decision

The only missing model value in the 585 input rows is:

```text
sample_date: 2013-08-16
column:      cpi_surprise_pct
event:       CPI for observation month 2013-07
```

The initial CPI actual exists, but the Cleveland Fed archive does not contain a compatible expectation strictly before that release.

Final decision:

```text
Exclude this one row from the model matrix.
Do not replace it with zero.
Do not future-fill it.
Do not fit an imputer on the complete dataset.
```

Zero would incorrectly mean “the CPI release exactly matched expectations.” A later forecast would leak future information. The exclusion is recorded in:

```text
scenario_model_matrix_row_exclusions.csv
```

## Initial and final feature counts

The first assembled matrix had:

```text
Rows:     584 after the CPI row decision
Features: 59
Targets:  12
```

Step 9 rejected one state feature:

```text
oil_volatility_20d_annualized_pct
```

The final corrected feature families are:

| Feature family | Count |
|---|---:|
| Historical scenario shocks | 8 |
| Shock release flags | 5 |
| Macro state | 10 |
| Current market/VIX level | 1 |
| Market risk and trend | 34 |
| **Total features** | **58** |

The original Step 7 state table remains unchanged. The rejected oil-volatility feature is omitted only from the final model matrix, and all rows are retained.

## Final matrix

```text
File:        data/scenario_playground/processed/model_matrix/scenario_model_matrix_raw.csv
Rows:        584
Columns:     71
Date range:  2013-08-05 through 2026-06-11
```

Exact column roles:

| Role | Count |
|---|---:|
| Row key: `sample_date` | 1 |
| Raw feature columns | 58 |
| Raw target columns | 12 |
| **Matrix columns** | **71** |

The matrix is deliberately raw and unscaled. Every numeric value is equal to its Step 5, 6, or 7 source value after the documented row and feature exclusions.

## Feature, target, provenance, and audit separation

### Model-value file

```text
scenario_model_matrix_raw.csv
```

Contains only:

```text
sample_date
58 features
12 targets
```

### Timing and event provenance

```text
scenario_model_matrix_provenance.csv
```

Shape:

```text
584 rows x 21 columns
```

It retains the sampling design, model availability, state cutoff, shock-window boundaries, target boundaries, 21-session horizon, event IDs/types/observation periods, original release metadata, source metadata, and conservative timing-shift flags.

None of these provenance fields is a model feature.

### Source-row audit

```text
scenario_model_matrix_source_row_audit.csv
```

Shape:

```text
584 rows x 9 columns
```

It contains eligibility, exact-target-availability, shared-coverage, collision, and duplicate-removal checks. These audit fields are not model inputs.

### Column catalog

```text
scenario_model_matrix_column_catalog.csv
```

Shape:

```text
92 rows x 15 columns
```

Every model-matrix and provenance column has:

```text
role
family
source
source artifact
definition or formula
unit
point-in-time rule
proxy status
revision-risk classification
matrix inclusion status
usable matrix dates
notes
```

### Separate exclusions and split controls

```text
scenario_model_matrix_row_exclusions.csv
scenario_model_matrix_feature_exclusions.csv
scenario_model_matrix_split_manifest.csv
```

No endpoint source values from `market_target_audit.csv`, release-construction values from shock audits, availability dates, split flags, or audit results enter the model feature set.

## Proxy and revision-risk features

Two selected features are explicitly marked as proxies:

```text
kc_fed_manufacturing_pmi_proxy_change_points
real_rate_proxy_pct_points
```

The PMI proxy is not relabeled as national ISM PMI. The catalog also records point-in-time/release limitations for CPI, unemployment, and GDP and continuous-futures structure risk for retained oil features.

## Target availability

All 12 targets are finite for every final matrix row.

| Target | Final source | Field | Usable matrix rows | Matrix dates |
|---|---|---|---:|---|
| `sp500` | Yahoo `^GSPC` | `Close` | 584 | 2013-08-05 to 2026-06-11 |
| `nasdaq` | Yahoo `^IXIC` | `Close` | 584 | 2013-08-05 to 2026-06-11 |
| `ten_year_yield_bps` | FRED `DGS10` | `value` | 584 | 2013-08-05 to 2026-06-11 |
| `dxy` | Yahoo `DX-Y.NYB` | `Close` | 584 | 2013-08-05 to 2026-06-11 |
| `gold` | Yahoo `GLD` | `Adj Close` | 584 | 2013-08-05 to 2026-06-11 |
| `oil` | Yahoo `CL=F` | `Close` | 584 | 2013-08-05 to 2026-06-11 |
| `technology` | Yahoo `XLK` | `Adj Close` | 584 | 2013-08-05 to 2026-06-11 |
| `energy` | Yahoo `XLE` | `Adj Close` | 584 | 2013-08-05 to 2026-06-11 |
| `financials` | Yahoo `XLF` | `Adj Close` | 584 | 2013-08-05 to 2026-06-11 |
| `utilities` | Yahoo `XLU` | `Adj Close` | 584 | 2013-08-05 to 2026-06-11 |
| `healthcare` | Yahoo `XLV` | `Adj Close` | 584 | 2013-08-05 to 2026-06-11 |
| `consumer_discretionary` | Yahoo `XLY` | `Adj Close` | 584 | 2013-08-05 to 2026-06-11 |

Every label still ends on the exact stored 21st canonical `^GSPC` session. No endpoint is shifted or filled.

## Frozen chronological periods

The Step 4 canonical-calendar 60/20/20 planning dates are now frozen as the initial chronological boundaries:

| Period | Calendar boundary | Raw matrix rows |
|---|---|---:|
| Development | 2013-08-01 through 2021-05-05 | 358 |
| Validation | 2021-05-06 through 2023-12-06 | 112 |
| Final test | 2023-12-07 through 2026-07-16 | 114 |

The final test begins at the first test row’s target baseline, `2023-12-07`, and is marked:

```text
LOCKED_UNTIL_FINAL_EVALUATION
```

Step 9 does not use its target values for decisions.

## Target-overlap-aware purge

The purge uses:

```text
each row's target_end_date
the canonical ^GSPC session calendar
the next period's first target_baseline_date
```

Retention rule:

```text
retain earlier-period row only if
target_end_date < next period's first target_baseline_date
```

It does not remove 21 event-aware rows.

| Modeling role | Rows |
|---|---:|
| Development training | 354 |
| Development boundary purge | 4 |
| Validation evaluation | 110 |
| Validation boundary purge | 2 |
| Locked final test | 114 |

Purged development samples:

```text
2021-04-14
2021-04-23
2021-04-28
2021-04-30
```

Purged validation samples:

```text
2023-11-15
2023-11-17
```

The last retained development and validation baselines are each 24 canonical sessions before the next period’s first baseline, exceeding the 21-session target horizon.

## Persisted Step 8 outputs

```text
data/scenario_playground/processed/model_matrix/
  scenario_model_matrix_raw.csv
  scenario_model_matrix_provenance.csv
  scenario_model_matrix_source_row_audit.csv
  scenario_model_matrix_column_catalog.csv
  scenario_model_matrix_row_exclusions.csv
  scenario_model_matrix_feature_exclusions.csv
  scenario_model_matrix_split_manifest.csv
  scenario_model_matrix_summary.json
```

Reusable implementation:

```text
scenario_modeling/src/model_matrix.py
scenario_modeling/src/splitting.py
scenario_modeling/notebooks/06_point_in_time_matrix.ipynb
scenario_modeling/tests/test_model_matrix.py
```

## Validation result

All Step 8 checks pass:

```text
all Step 5-7 inputs contain 585 rows
all input sample indexes match one-to-one
all shared timing and event keys match
only the genuine missing CPI row is removed
no imputation, scaling, clipping, or winsorization occurs
all final model values are complete and finite
all final model values remain raw
state cutoff precedes sample date
shock window ends on or before sample date
target baseline equals state cutoff
all targets span exactly 21 canonical sessions
matrix roles are exact
provenance and audit columns remain separate
every matrix column has a catalog definition and source
split rows align one-to-one with matrix rows
target-end-date purge checks pass
all upstream point-in-time checks remain passing
```

## Completion decision

Step 8 is complete. The corrected processed matrix exists, contains 584 usable event-aware rows, 58 raw features, and all 12 targets, and has complete separate provenance and split controls.

The Step 9 report records the audit that approved this corrected version for Ridge training.
