# Scenario Playground Step 7 State-Feature Report

## Status and scope

Step 7 is complete.

It constructs a separate point-in-time state-feature table for all 585 selected `EVENT_AWARE` rows. Every state value represents information available on or before the row's completed pre-shock `state_cutoff_date`.

It does not load or join Step 6 shocks, load Step 5 targets, assemble a model matrix, create splits, train models, access Supabase, or modify application code.

## Why this step is necessary

The same shock can produce a different market response in different environments. The state features let later models distinguish, for example:

```text
low versus high volatility
near-zero versus restrictive policy rates
steep versus inverted yield curves
calm versus drawdown equity markets
positive versus negative recent market trends
```

The feature family is intentionally limited to 46 columns. This is within the required 25-60 range and avoids an undisciplined technical-indicator expansion before the core model is evaluated.

## Input and timing rule

Input:

```text
data/scenario_playground/interim/sampling/selected_sampling_index.csv
```

Coverage:

```text
Rows:                    585
First sample:            2013-08-05
Last sample:             2026-06-11
First state cutoff:      2013-08-01
Last state cutoff:       2026-06-09
```

Every state cutoff strictly precedes its operational sample date.

Point-in-time rules:

- Market and yield series use the latest observation at or before the cutoff.
- EFFR becomes available on the following observed equity session.
- CPI, unemployment, and GDP advance values use conservative next-session availability because exact cached historical release times are absent.
- Rolling calculations end on the selected source observation at or before the cutoff.
- No future-fill, backward-fill, full-sample scaling, clipping, or imputation occurs.

## Feature counts

```text
Macro state features:           10
Current VIX level:               1
Market risk and trend features: 35
Total state features:           46
```

### Macro state

```text
fed_funds_target_midpoint_pct
effective_fed_funds_rate_pct
cpi_mom_pct
cpi_yoy_pct
unemployment_rate_pct
real_gdp_qoq_annualized_pct
ten_year_yield_pct
two_year_yield_pct
yield_curve_slope_10y_2y_pct_points
real_rate_proxy_pct_points
```

Definitions:

| Feature | Source and formula | Unit |
|---|---|---|
| Fed target midpoint | Latest `(DFEDTARL + DFEDTARU) / 2` | percent |
| Effective Fed Funds | Latest EFFR published by the next session | percent |
| CPI MoM | Initial-vintage `100 * (current/prior - 1)` | percent |
| CPI YoY | Initial-vintage `100 * (current/12-month-prior - 1)` | percent |
| Unemployment | Latest initially released UNRATE | percent |
| Real GDP growth | Latest BEA advance real-GDP QoQ SAAR value from the Atlanta TrackRecord | annualized percent |
| 10-year yield | Latest DGS10 | percent |
| 2-year yield | Latest DGS2 | percent |
| Yield-curve slope | `DGS10 - DGS2` | percentage points |
| Real-rate proxy | Fed target midpoint minus CPI YoY | percentage points |

The real-rate column remains explicitly identified as a proxy.

### Current volatility state

```text
vix_level_points
```

Source: FRED `VIXCLS`, using the latest value at or before the cutoff.

### Market risk and trend

The following seven features are calculated independently for:

```text
sp500: Yahoo ^GSPC Close
nasdaq: Yahoo ^IXIC Close
gold: Yahoo GLD adjusted Close
oil: Yahoo CL=F Close
dxy: Yahoo DX-Y.NYB Close
```

Features per asset:

```text
return_5d_pct
return_21d_pct
volatility_20d_annualized_pct
drawdown_252d_pct
ma_distance_50d_pct
ma_distance_200d_pct
zscore_252d
```

Formulas:

```text
return_Nd_pct =
    100 * (current / value N source observations earlier - 1)

volatility_20d_annualized_pct =
    100 * sqrt(252)
    * sample standard deviation of the latest 20 arithmetic daily returns

drawdown_252d_pct =
    100 * (current / trailing_252_observation_max - 1)

ma_distance_Nd_pct =
    100 * (current / trailing_N_observation_mean - 1)

zscore_252d =
    (current - trailing_252_mean)
    / trailing_252_sample_standard_deviation
```

All 200- and 252-observation features have complete trailing histories before the first selected row.

## Missingness and finiteness

Results:

```text
Rows:                      585
State features:             46
Missing state values:        0
Non-finite state values:     0
Rows removed:                0
```

No imputation was required or performed.

## Availability and staleness

Fourteen independently retained source-date or availability-date families were checked. Every date is on or before its row's `state_cutoff_date`.

Macro age reflects the normal time between monthly or quarterly releases:

| State | Maximum age | Median age |
|---|---:|---:|
| Fed target midpoint | 0 days | 0 days |
| Effective Fed Funds | 1 day | 0 days |
| CPI MoM | 77 days | 19 days |
| CPI YoY | 51 days | 19 days |
| Unemployment | 72 days | 16 days |
| Real GDP advance growth | 144 days | 46 days |
| 10-year yield | 0 days | 0 days |
| 2-year yield | 0 days | 0 days |

The longer CPI MoM maximum reflects the source gap around the 2025 shutdown. The implementation carries only the last value that had actually become available; it does not fill a future release backward.

All five Yahoo market families and VIX have a maximum state-cutoff age of zero days on the selected rows.

## Known market limitation

The cached Yahoo `CL=F` history contains one nonpositive close during the April 2020 oil disruption.

Consequences visible in the untrimmed state features include:

```text
maximum oil 20-day annualized arithmetic-return volatility:
approximately 1,196.13%
```

This is retained and documented. Step 7 does not clip or winsorize it. Step 9 must review its influence as part of the required outlier and leakage audit.

## Point-in-time validation

All validations pass:

```text
All 585 selected rows preserved
All 46 declared features present
Feature count is between 25 and 60
All definitions are present
No state values are missing
All state values are finite
Every source availability date is on or before the state cutoff
Every rolling feature ends on or before the state cutoff
Macro releases are never backfilled before availability
All long rolling lookbacks are complete
Every state cutoff strictly precedes the sample date
```

## Persisted outputs

```text
data/scenario_playground/interim/features/state/
  state_features.csv
  state_feature_availability_audit.csv
  state_feature_summary.json
```

Shapes:

```text
state_features.csv:                   585 rows x 48 columns
state_feature_availability_audit.csv: 585 rows x 46 columns
```

The main file contains two key columns and 46 numeric state features. The availability audit is separate so release dates, source observation dates, and staleness metadata cannot accidentally become model features.

## Reusable implementation

```text
scenario_modeling/src/feature_sources.py
scenario_modeling/src/state_features.py
scenario_modeling/notebooks/05_state_features.ipynb
scenario_modeling/tests/test_state_features.py
```

The module can be run from the repository root with:

```powershell
backend\.venv\Scripts\python.exe -m scenario_modeling.src.state_features
```

Use `--dry-run` to construct and validate without rewriting the interim outputs.

## Completion decision

Step 7 passes its acceptance criteria. The state table is complete, finite, date-aligned, point-in-time checked, and remains independent from shocks and targets.

The oil-volatility extreme is a documented Step 9 review item, not a Step 7 construction failure.
