# Scenario Playground Step 16 Confidence Diagnostics Report

## Status and scope

Step 16 is complete:

```text
STEP_16_CONFIDENCE_DIAGNOSTICS_COMPLETE
```

The implementation uses accepted Step 15 explanation and sensitivity evidence
and preserves the corrected all-target Ridge inference workflow unchanged.

It does not train or test all-target XGBoost, access final-test targets, export
fitted models, modify backend or frontend code, create an API, access
Supabase, or start servers.

## Confidence methodology

Confidence is:

```text
RULE_BASED_NON_COMPENSATORY
```

This means unrelated strengths cannot average away a serious limitation.

Examples:

- an in-range scenario cannot upgrade a weak Step 13 model;
- a familiar state cannot erase poor walk-forward performance;
- missing XGBoost agreement cannot receive a favorable default;
- a good low-VIX error does not erase crisis-period residual evidence; and
- none of the 12 targets can receive medium or high confidence.

The output keeps seven concepts separate:

```text
input-range warnings
state-extrapolation warnings
model-performance warnings
scenario-responsiveness warnings
regime warnings
state-freshness warnings
overall research-confidence result
```

## Historical range and percentile diagnostics

The implementation diagnoses:

```text
8 scenario shocks
+
45 current-state features
=
53 continuous inputs
```

For every input it records:

- historical minimum and maximum;
- 1st, 5th, 25th, 50th, 75th, 95th, and 99th percentiles;
- empirical percentile of the current value;
- standardized distance from the training mean;
- outside-range and extreme-within-range status; and
- confirmation that no clipping or winsorization occurred.

For the executed example:

- all eight scenario shocks are inside their training ranges;
- `gold_return_5d_pct` is extreme but still inside its range;
- `gold_return_21d_pct` is below its training minimum; and
- `gold_ma_distance_50d_pct` is below its training minimum.

These current-state extrapolations are reported for every target.

## Distance from familiar historical combinations

The complete 58-feature row is transformed with the same fitted median imputer
and standard scaler used by Ridge. Euclidean distance to the nearest authorized
training row is compared with the leave-one-out nearest-neighbor distance
distribution of all 464 authorized rows.

| Row | Nearest distance | Historical neighbor percentile | Status | Nearest historical sample |
|---|---:|---:|---|---|
| Baseline | 7.0994 | 93.32 | `WITHIN_OBSERVED_COMBINATION_SUPPORT` | 2023-10-09 |
| Scenario | 7.7260 | 95.04 | `UNUSUAL_COMBINATION` | 2018-08-13 |

The scenario is unusual but is not beyond the maximum observed historical
nearest-neighbor distance.

## State freshness

The reproducible diagnostic date is:

```text
2026-07-26
```

The default state is:

```text
sample date: 2026-06-11
state cutoff: 2026-06-09
calendar age: 45 days
status: STALE
```

The state remains usable for research, but all 12 targets receive an explicit
state-freshness warning.

## Walk-forward, regime, and residual evidence

For each target, Step 16 reads the persisted Step 13:

- five-fold Ridge errors;
- pooled walk-forward MAE, RMSE, and out-of-sample R-squared;
- low-, medium-, and high-VIX regime errors;
- residual distribution;
- worst observed sample date and maximum absolute error;
- naive-benchmark comparison; and
- target-specific Step 13 classification.

The current state is in the low-VIX regime. No target's low-VIX MAE is more
than 25% above its pooled Ridge MAE, so the executed example has zero elevated
current-regime warnings. High-VIX and extreme-period evidence remains in each
target payload.

## Per-target results

| Target | Frozen Step 13 classification | Overall result | Walk-forward MAE | OOS R² | Negligible inputs | Unstable inputs |
|---|---|---|---:|---:|---:|---:|
| S&P 500 | `APPROVED_WITH_LOW_CONFIDENCE` | `LOW_RESEARCH_CONFIDENCE_WITH_CONTEXT_WARNINGS` | 3.575 | 0.014 | 1 | 2 |
| Nasdaq | `NAIVE_FALLBACK_REQUIRED` | `VERY_LOW_RESEARCH_CONFIDENCE` | 5.299 | -0.383 | 2 | 3 |
| 10-year yield | `NAIVE_FALLBACK_REQUIRED` | `VERY_LOW_RESEARCH_CONFIDENCE` | 19.765 bps | -0.084 | 8 | 3 |
| DXY | `NAIVE_FALLBACK_REQUIRED` | `VERY_LOW_RESEARCH_CONFIDENCE` | 1.446 | -0.042 | 0 | 0 |
| Gold | `NAIVE_FALLBACK_REQUIRED` | `VERY_LOW_RESEARCH_CONFIDENCE` | 2.989 | -0.029 | 0 | 2 |
| Oil | `NAIVE_FALLBACK_REQUIRED` | `VERY_LOW_RESEARCH_CONFIDENCE` | 14.941 | -0.409 | 7 | 3 |
| Technology | `NAIVE_FALLBACK_REQUIRED` | `VERY_LOW_RESEARCH_CONFIDENCE` | 5.648 | -0.490 | 8 | 2 |
| Energy | `APPROVED_WITH_LOW_CONFIDENCE` | `LOW_RESEARCH_CONFIDENCE_WITH_CONTEXT_WARNINGS` | 7.211 | -0.028 | 0 | 5 |
| Financials | `NAIVE_FALLBACK_REQUIRED` | `VERY_LOW_RESEARCH_CONFIDENCE` | 5.104 | -0.119 | 4 | 2 |
| Utilities | `NAIVE_FALLBACK_REQUIRED` | `VERY_LOW_RESEARCH_CONFIDENCE` | 4.256 | -0.188 | 0 | 6 |
| Healthcare | `NAIVE_FALLBACK_REQUIRED` | `VERY_LOW_RESEARCH_CONFIDENCE` | 3.328 | -0.008 | 8 | 0 |
| Consumer Discretionary | `APPROVED_WITH_LOW_CONFIDENCE` | `LOW_RESEARCH_CONFIDENCE_WITH_CONTEXT_WARNINGS` | 4.800 | -0.018 | 1 | 4 |

The nine weak-evidence targets remain very-low-confidence research models.
They still return the numeric Ridge predictions required by corrected Step 14.
No naive benchmark replaces those predictions.

The three previously promoted targets remain only low confidence, and the
unusual scenario combination, extrapolated gold state features, and stale
state are exposed as context warnings.

## Warning results

The executed output contains 88 structured warning records.

| Category | Targets warned | Main reason |
|---|---:|---|
| Input range | 0 | All eight example shocks are in range |
| State extrapolation | 12 | Two state features outside range and scenario combination unusual |
| Model performance | 12 | Low/weak Step 13 evidence, naive comparison, and/or nonpositive OOS R² |
| Scenario responsiveness | 11 | Negligible responses and/or unstable historical directions |
| Current VIX regime | 0 | Current low-VIX MAE is not elevated by the defined rule |
| State freshness | 12 | Default state is 45 days old |

DXY has no Step 15 responsiveness warning because all eight final-fit slopes
clear the response threshold and its historical meaningful directions are
consistent. Its confidence remains very low because its predictive metrics
are weak.

## Postponed XGBoost handling

For all 12 targets:

```text
status = UNAVAILABLE_INTENTIONALLY_POSTPONED
available = false
included_in_confidence = false
score = null
penalty_or_bonus = NONE
```

The missing comparator does not raise or lower confidence. The metadata states
that Ridge-versus-XGBoost agreement can be added in a later model version.

No XGBoost model was trained or called.

## Files created

### Reusable Python

- `scenario_modeling/src/confidence_diagnostics.py`

### Focused tests

- `scenario_modeling/tests/test_confidence_diagnostics.py`

### Executed notebook

- `scenario_modeling/notebooks/14_confidence_diagnostics.ipynb`

The notebook contains seven executed code cells, uses the
`insightpulse-backend` kernel, and has zero error outputs.

### Generated local evidence

Under `data/scenario_playground/interim/confidence/`:

- `input_diagnostics.csv`
- `target_confidence_diagnostics.csv`
- `warning_records.csv`
- `joint_feature_distance.json`
- `state_freshness.json`
- `target_confidence.json`
- `confidence_metadata_schema.json`
- `confidence_diagnostics_summary.json`

No fitted model binary was exported.

## Frozen Step 17 confidence schema

`confidence_metadata_schema.json` freezes:

- required per-target diagnostic fields;
- the six warning categories;
- allowed overall research-confidence statuses;
- the non-compensatory method;
- the unavailable-XGBoost contract;
- prohibition on strong confidence;
- research-only and noncausal flags; and
- benchmark-only naive behavior.

All generated output hashes are recorded in
`confidence_diagnostics_summary.json`.

## Validation results

### Focused Steps 15 and 16 tests

```powershell
backend\.venv\Scripts\python.exe -m unittest scenario_modeling.tests.test_scenario_explanations scenario_modeling.tests.test_confidence_diagnostics -v
```

```text
18/18 passed
```

### Complete Scenario Modeling suite

```powershell
backend\.venv\Scripts\python.exe -m unittest discover -s scenario_modeling\tests -p "test_*.py" -v
```

```text
132/132 passed
```

### Notebooks

| Notebook | Kernel | Code cells | Errors |
|---|---|---:|---:|
| `13_ridge_explanations_and_sensitivity.ipynb` | `insightpulse-backend` | 7/7 | 0 |
| `14_confidence_diagnostics.ipynb` | `insightpulse-backend` | 7/7 | 0 |

The sandbox prevented IPython from writing its optional user-level history
database. That did not affect notebook execution, model calculations, cached
outputs, or saved notebook results.

### Other acceptance checks

The following pass:

- both Python modules compile;
- all 12 targets have confidence diagnostics;
- all 53 continuous inputs have finite range diagnostics;
- joint-distance values are finite;
- Step 13 walk-forward metrics match exactly;
- all warning categories are structured;
- XGBoost agreement is unavailable, unscored, and excluded;
- no target is promoted beyond Step 13 evidence;
- neutral scenario effects remain zero;
- feature order, state isolation, and release-flag isolation remain exact;
- repeated diagnostics are deterministic;
- schemas validate;
- persisted hashes validate;
- no naive benchmark substitution occurs;
- all outputs remain research only and noncausal;
- locked final-test target rows parsed remain zero; and
- final-test access remains zero.

No dependency was installed. `scenario_modeling/requirements.txt` was updated
only to identify the existing pinned environment as supporting work through
Step 16.

## Remaining limitations

- Nine targets have weak predictive evidence.
- Eleven targets have at least one scenario-responsiveness warning.
- Thirty-nine target/input pairs are negligible under the documented
  thresholds.
- Thirty-two target/input pairs have unstable historical directions.
- The scenario is an unusual historical feature combination.
- Two default-state gold features are outside the fitting range.
- The default state is stale as of the diagnostic date.
- Final-test targets remain unopened, so no untouched final-test estimate is
  available.
- Ridge contributions and sensitivity slopes are predictive associations, not
  causal effects.
- No uncertainty interval, production bundle, or application API exists.
- All-target XGBoost agreement remains unavailable by design.

## Step 17 readiness

Steps 15 and 16 now provide accepted, frozen metadata schemas for:

- exact Ridge explanations;
- scenario sensitivity evidence;
- input and state ranges;
- feature-combination distance;
- walk-forward, regime, and residual evidence;
- target confidence results;
- structured warnings; and
- explicit unavailable-XGBoost handling.

Step 17 can safely export:

```text
12 fitted Ridge pipelines
their fitted preprocessing
the exact 58-feature schema
input and target definitions
the latest default state snapshot and provenance
explanation metadata
sensitivity evidence
confidence metadata
warnings
a reloadable inference interface
```

The Step 17 loader must not retrain or require the training matrix. That work
was not begun here.
