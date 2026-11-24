# Scenario Playground Step 5 Market Targets Report

## Necessity and scope decision

Step 5 is necessary and correctly placed.

Step 4 selected the `EVENT_AWARE` sampling design and stored each row's:

```text
target_baseline_date
target_end_date
```

It intentionally did not calculate labels. Step 5 converts those fixed boundaries into the 12 model-contract outputs before scenario shocks or state features are added. This isolates label correctness from later feature work and catches source, unit, calendar, and adjustment errors early.

There is no conflict with the completed model contract or sampling design. The crucial implementation decision is:

```text
Never calculate a target from sample_date.
Use the Step 4 pre-shock target_baseline_date and exact target_end_date.
```

This report covers target construction only. No Step 6 shock construction, state features, model matrix, split, training, database, backend, or frontend work was performed.

## Input rows

Input:

```text
data/scenario_playground/interim/sampling/selected_sampling_index.csv
```

The input contains 585 label-eligible event-aware rows.

For each row:

```text
target_baseline_date = completed pre-shock market close
target_end_date = 21st observed ^GSPC session after baseline
```

Step 5 preserved all 585 rows. It did not forward-fill, backfill, shift an endpoint, shorten a horizon, clip a target, or drop a row.

Coverage:

```text
First sample date:          2013-08-05
Last sample date:           2026-06-11
First target baseline:      2013-08-01
Last target endpoint:       2026-07-10
Target horizon:             exactly 21 trading sessions
```

## Target formulas and units

Eleven price or index targets use:

```text
target_percent =
    100 * (future_value / baseline_value - 1)
```

The 10-year yield target uses:

```text
ten_year_yield_bps =
    100 * (future_yield_percent - baseline_yield_percent)
```

The yield result is a basis-point change, not a price return.

The compact output columns use the final contract names:

```text
sp500
nasdaq
ten_year_yield_bps
dxy
gold
oil
technology
energy
financials
utilities
healthcare
consumer_discretionary
```

## Source and value-field decisions

Step 5 reused the final Step 3 decisions without reopening source research.

| Target | Source | Field | Formula type | Unit |
|---|---|---|---|---|
| `sp500` | Yahoo `^GSPC` | `Close` | Index return | Percent |
| `nasdaq` | Yahoo `^IXIC` | `Close` | Index return | Percent |
| `ten_year_yield_bps` | FRED `DGS10` | `value` | Yield difference × 100 | Basis points |
| `dxy` | Yahoo `DX-Y.NYB` | `Close` | Index return | Percent |
| `gold` | Yahoo `GLD` | `Adj Close` | Adjusted ETF return | Percent |
| `oil` | Yahoo `CL=F` | `Close` | Continuous-futures return | Percent |
| `technology` | Yahoo `XLK` | `Adj Close` | Adjusted ETF return | Percent |
| `energy` | Yahoo `XLE` | `Adj Close` | Adjusted ETF return | Percent |
| `financials` | Yahoo `XLF` | `Adj Close` | Adjusted ETF return | Percent |
| `utilities` | Yahoo `XLU` | `Adj Close` | Adjusted ETF return | Percent |
| `healthcare` | Yahoo `XLV` | `Adj Close` | Adjusted ETF return | Percent |
| `consumer_discretionary` | Yahoo `XLY` | `Adj Close` | Adjusted ETF return | Percent |

Adjustment policy:

- `^GSPC` and `^IXIC` are direct price-index closes; they do not include dividends.
- DXY is an index close.
- `CL=F` is a futures close, so no dividend adjustment is appropriate.
- GLD and the six sector ETFs use adjusted close so distributions and split adjustments are handled consistently.
- `DGS10` is a percentage-point yield level and receives no price-return transformation.

No source field is mixed within a target's history.

## Implementation sequence

The reusable implementation is:

```text
scenario_modeling/src/targets.py
```

It performs the acceptance sequence required by the Step 5 instruction:

1. Load and validate the 585 selected event-aware rows.
2. Load the 12 accepted raw series.
3. Verify every stored baseline/end pair is exactly 21 observed sessions apart.
4. Construct the S&P 500 target first.
5. Independently recalculate representative S&P rows from raw CSV strings with `Decimal`.
6. Expand to the remaining 11 targets only after the S&P first pass succeeds.
7. Independently verify representative `DGS10` basis-point calculations.
8. Produce distributions and explicit extreme-value review rows.
9. Persist compact targets separately from baseline/future audit values.

The module has no downloader, shock builder, state-feature builder, splitter, model, API, or database behavior.

## S&P 500 manual validation

Six dates were recalculated independently from raw CSV strings. All agree with the vectorized target to considerably less than `1e-10` percentage points.

| Sample | Current date | Current `^GSPC` | Future date | Future `^GSPC` | Hand-calculated target |
|---|---|---:|---|---:|---:|
| 2013-08-05 | 2013-08-01 | 1,706.869995 | 2013-08-30 | 1,632.969971 | -4.329564% |
| 2016-11-02 | 2016-11-01 | 2,111.719971 | 2016-12-01 | 2,191.080078 | +3.758079% |
| 2019-12-09 | 2019-12-05 | 3,117.429932 | 2020-01-07 | 3,237.179932 | +3.841305% |
| 2020-02-21 | 2020-02-19 | 3,386.149902 | 2020-03-19 | 2,409.389893 | -28.845740% |
| 2023-02-24 | 2023-02-22 | 3,991.050049 | 2023-03-23 | 3,948.719971 | -1.060625% |
| 2026-06-11 | 2026-06-09 | 7,386.649902 | 2026-07-10 | 7,575.390137 | +2.555153% |

The pandemic decline is large but mathematically and historically consistent with its stored dates. It remains in the dataset.

## Manual yield conversion

Four `DGS10` rows were independently checked.

| Sample | Current date/value | Future date/value | Calculation | Result |
|---|---|---|---|---:|
| 2013-08-05 | 2013-08-01, 2.74% | 2013-08-30, 2.78% | `(2.78 - 2.74) × 100` | +4 bps |
| 2020-02-10 | 2020-02-06, 1.65% | 2020-03-09, 0.54% | `(0.54 - 1.65) × 100` | -111 bps |
| 2022-09-23 | 2022-09-21, 3.51% | 2022-10-20, 4.24% | `(4.24 - 3.51) × 100` | +73 bps |
| 2026-06-11 | 2026-06-09, 4.53% | 2026-07-10, 4.56% | `(4.56 - 4.53) × 100` | +3 bps |

This validates both sign and unit conversion.

## Target distributions

Every distribution contains 585 finite values and zero missing values.

| Target | Unit | Mean | Standard deviation | Minimum | P01 | Median | P99 | Maximum |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| S&P 500 | % | 1.056 | 4.290 | -28.846 | -11.756 | 1.540 | 9.545 | 14.590 |
| Nasdaq | % | 1.443 | 5.294 | -27.163 | -14.149 | 2.033 | 13.748 | 16.931 |
| 10-year yield | bps | 0.850 | 23.479 | -111.000 | -56.000 | 0.000 | 58.000 | 73.000 |
| DXY | % | 0.178 | 1.838 | -6.765 | -4.368 | 0.144 | 4.208 | 5.523 |
| Gold | % | 0.800 | 4.417 | -15.373 | -8.801 | 0.473 | 11.300 | 14.076 |
| Oil | % | 1.092 | 16.858 | -56.193 | -31.756 | 0.370 | 48.932 | 173.177 |
| Technology | % | 1.924 | 5.754 | -27.182 | -13.996 | 2.275 | 15.602 | 23.545 |
| Energy | % | 0.662 | 8.429 | -53.309 | -22.679 | 0.773 | 23.100 | 37.403 |
| Financials | % | 1.006 | 5.624 | -37.006 | -13.122 | 1.514 | 15.433 | 22.548 |
| Utilities | % | 1.005 | 4.610 | -26.022 | -13.835 | 1.426 | 10.986 | 14.286 |
| Healthcare | % | 0.930 | 3.998 | -20.944 | -9.037 | 1.209 | 8.606 | 23.146 |
| Consumer Discretionary | % | 1.048 | 5.538 | -31.579 | -16.613 | 1.418 | 14.092 | 19.989 |

The distributions are directionally plausible:

- DXY is the narrowest return distribution.
- Oil and Energy are the most volatile commodity-related outputs.
- Nasdaq and Technology are more volatile than the broad S&P 500.
- The yield is centered close to zero but contains large tightening/easing moves.
- The largest broad-equity and sector declines occur during the 2020 shock.

No distribution was standardized, clipped, winsorized, or filtered in Step 5.

## Extreme-value review

Review thresholds are diagnostics:

```text
Most price/index targets: absolute return over 50%
Oil futures:             absolute return over 150%
10-year yield:           absolute move over 100 bps
```

Four rows cross those thresholds:

| Target | Sample | Baseline/end | Baseline | Future | Target |
|---|---|---|---:|---:|---:|
| 10-year yield | 2020-02-10 | 2020-02-06 to 2020-03-09 | 1.65% | 0.54% | -111 bps |
| Oil | 2020-04-29 | 2020-04-28 to 2020-05-28 | 12.34 | 33.71 | +173.177% |
| Oil | 2020-04-30 | 2020-04-28 to 2020-05-28 | 12.34 | 33.71 | +173.177% |
| Energy | 2020-02-21 | 2020-02-19 to 2020-03-19 | 20.9955 | 9.8030 | -53.309% |

The two oil rows have different event-aware sample dates but the same pre-shock close and 21-session endpoint. Step 4 already documented that event-aware rows can have completely overlapping target windows. Their shocks and event identities differ, so the rows remain distinct.

These observations are retained because:

- their raw baseline and future values are present;
- their formulas are finite;
- their horizons are exactly 21 sessions;
- their dates correspond to the pandemic market shock or oil recovery; and
- deleting them during target construction would hide real source behavior.

The oil target still carries the accepted `CL=F` continuous-futures and roll-convention limitation. Later model evaluation must test robustness, but Step 5 does not alter the label.

## Missing trading-day and endpoint validation

For every row and target:

1. The baseline date exists in the target's selected raw series.
2. The endpoint date exists in the target's selected raw series.
3. Both dates are observed `^GSPC` sessions.
4. Their canonical-session position difference is exactly 21.
5. The lookup is exact; no nearest-date or as-of join is used.

Results:

```text
Rows checked:                         585
Minimum session gap:                   21
Maximum session gap:                   21
Missing baseline values:                0
Missing future values:                  0
Missing target values:                  0
Non-finite target values:               0
Rows dropped during Step 5:             0
```

The six candidate rows Step 4 found ineligible are not silently reintroduced. Step 5 starts from the accepted 585-row selected index.

## Persisted outputs

Generated files are under:

```text
data/scenario_playground/interim/targets/
```

### Compact target table

```text
market_targets.csv
```

Shape:

```text
585 rows × 20 columns
```

Schema:

```text
sample_date
model_availability_date
state_cutoff_date
target_baseline_date
target_end_date
target_horizon_sessions
event_types_json
event_ids_json
sp500
nasdaq
ten_year_yield_bps
dxy
gold
oil
technology
energy
financials
utilities
healthcare
consumer_discretionary
```

The compact file is the later model-matrix target input.

### Endpoint audit table

```text
market_target_audit.csv
```

Shape:

```text
585 rows × 43 columns
```

For each target it retains:

```text
<target>_baseline_value
<target>_future_value
<target>_calculated_target
```

This audit data is separate so future baseline/future source values cannot accidentally become model features.

### Manual checks

```text
manual_target_checks.csv
```

Contains six S&P and four yield checks, including raw values, independent calculation, vectorized result, difference, and pass flag.

### Extreme-value review

```text
market_target_extremes.csv
```

Contains the four reviewed threshold exceedances. No row is removed.

### Metadata and distributions

```text
market_target_summary.json
```

Contains:

- source paths and SHA-256 hashes;
- output schemas and hashes;
- formulas, fields, units, and adjustment policies;
- all manual-check records;
- complete distribution summaries;
- extreme-review counts;
- horizon validation; and
- scope/acceptance checks.

CSV is used because Parquet support is not installed in the existing environment. Output order and numeric formatting are deterministic.

## Notebook and tests

Notebook:

```text
scenario_modeling/notebooks/03_market_targets.ipynb
```

It:

1. loads local caches through `scenario_modeling.src.targets`;
2. displays source and formula decisions;
3. performs the S&P first-pass review;
4. shows all 12 target columns;
5. displays every distribution;
6. inspects all four extremes;
7. verifies yield conversion; and
8. runs the Step 5 acceptance checks.

Focused tests:

```text
scenario_modeling/tests/test_targets.py
```

Coverage includes:

- percentage-return formula;
- 3.90% to 4.05% equals +15 bps;
- zero-baseline rejection;
- exact-date lookup without forward fill;
- adjusted/unadjusted source-field policies;
- all 585 rows and 12 target columns;
- exact 21-session horizons;
- hard-checked S&P and yield values;
- finite/missing distribution checks;
- retained extreme observations; and
- deterministic ordering and serialization.

## Validation result

All Step 5 validations pass:

```text
selected_sampling_design_is_event_aware = true
input_rows_preserved_without_dropping = true
all_target_baselines_equal_state_cutoffs = true
all_target_windows_are_exactly_21_sessions = true
all_12_contract_targets_present = true
all_exact_endpoint_lookups_succeeded = true
no_target_values_missing = true
all_target_values_finite = true
all_manual_checks_passed = true
yield_basis_point_conversion_checked = true
no_forward_fill_backfill_or_endpoint_shift = true
source_field_policies_match_step_3 = true
```

## Remaining limitations

- `^GSPC` and `^IXIC` are price indexes rather than total-return investments.
- `CL=F` follows Yahoo's continuous-futures history and roll conventions.
- Event-aware rows can share part or all of the same 21-session target window.
- Extreme pandemic observations reduce effective stability but are valid and retained.
- The compact table is labels only; no feature matrix exists yet.
- Later chronological splitting must purge or embargo overlapping labels.

These are documented modeling limitations, not Step 5 blockers.

## Completion decision

Step 5 is complete. All 12 labels exist for all 585 selected rows, the S&P target was independently validated before expansion, the yield conversion was manually verified, distributions and extremes were reviewed, and deterministic offline outputs are available.

Step 6 was not started.
