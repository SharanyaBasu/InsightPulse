# Scenario Playground Step 15 Ridge Explanations and Sensitivity Report

## Status and scope

Step 15 is complete:

```text
STEP_15_EXPLANATIONS_AND_SENSITIVITY_COMPLETE
```

The implementation explains the corrected Step 14 all-target Ridge workflow
without changing its sampling, features, targets, split boundaries, selected
alphas, inference semantics, or confidence evidence.

It preserves:

- 12 independent target-specific Ridge pipelines;
- the frozen 584-row, 58-feature event-aware matrix;
- the 464 authorized pre-final-test fitting rows;
- all 114 final-test target rows as locked and unparsed;
- numeric baseline, scenario, and incremental predictions for all targets;
- benchmark-only naive models;
- the Kansas City Fed manufacturing PMI proxy disclosure; and
- research-only predictive, rather than causal, interpretation.

No Step 17 artifact export, all-target XGBoost work, backend or frontend
change, API, Supabase access, or server work was performed.

## Why Steps 15 and 16 were safe to combine

Step 15 produces exact model-contribution and sensitivity evidence. Step 16
consumes those accepted outputs but does not alter the Ridge models.

The internal order remained:

1. Build and validate Step 15 explanations and sensitivity evidence.
2. Use only accepted Step 15 outputs in Step 16.

The code, tests, notebooks, reports, generated outputs, and acceptance gates
remain separate so either step can be reviewed independently.

## Exact Ridge explanation method

For every target and for both the baseline and scenario row, the implementation
uses the actual fitted pipeline:

```text
training-fitted median imputer
->
training-fitted standard scaler
->
target-specific Ridge coefficients and intercept
```

The exact decomposition is:

```text
prediction =
Ridge intercept
+
sum(
    standardized feature value
    *
    fitted Ridge coefficient
)
```

The implementation records, for every feature:

```text
raw value
imputed value
training scaler mean
training scaler scale
standardized value
Ridge coefficient
feature contribution
absolute contribution
feature role
feature and target units
```

This distinction matters because a neutral raw shock value of zero does not
necessarily become a standardized value of zero. There are 240
baseline/scenario contribution rows where the raw value is zero but its
standardized value is nonzero.

## Contribution roles and isolation

Every explanation separates:

```text
model intercept
current-state contribution
scenario-shock contribution
release-flag contribution
model prediction
```

For the scenario-minus-baseline explanation:

- current-state contribution change is zero;
- release-flag contribution change is zero;
- intercept change is zero; and
- scenario-shock contribution changes reconcile to the incremental prediction.

The implementation does not allow state or release-flag changes to be
mislabelled as user-shock effects.

## Reconciliation results

| Item | Result |
|---|---:|
| Targets explained | 12 |
| Feature contributions | 1,392 |
| Scenario-minus-baseline feature deltas | 696 |
| Reconciliation records | 36 |
| Reconciliations passed | 36 |
| Maximum absolute reconciliation error | 2.22044604925031e-16 |
| Required tolerance | 1e-9 |

All baseline predictions, scenario predictions, and incremental differences
reconcile within tolerance.

## Human-readable drivers

The output produces deterministic top positive and negative drivers for:

```text
baseline row
scenario row
scenario-minus-baseline contribution changes
```

There are 192 top-driver records. Every explanation includes:

- feature name and role;
- raw value and unit;
- standardized Ridge contribution;
- target output unit; and
- a warning that the contribution is a predictive association, not a causal
  effect.

Existing Step 13 low-confidence and weak-evidence warnings remain attached to
all target explanation payloads.

## Unusually large contributions

Each example-row contribution is compared with the 99th percentile of that
feature's absolute contribution across the 464 authorized fitting rows for the
same target.

The executed example produced:

```text
0 unusually large contribution rows
```

This only means the individual example contributions stayed within that
historical contribution diagnostic. It does not make the model accurate or
production ready.

## Sensitivity method

For each of the eight scenario inputs:

1. Select representative historical rows nearest the 10th, 50th, and 90th
   percentiles of `vix_level_points`.
2. Keep all 45 state features fixed.
3. Keep all five release flags fixed throughout each sweep.
4. Hold the other seven shocks at neutral zero.
5. Vary the selected shock across observed quantiles, seven evenly spaced
   in-range values, and neutral zero.
6. Call every one of the 12 target-specific Ridge pipelines.
7. Record the scenario prediction and incremental effect.

For event-driven shocks, the relevant event flag remains one for the entire
sweep, including the neutral point. This prevents the curve from mixing shock
magnitude with an event-context change.

The generated evidence contains:

| Output | Rows |
|---|---:|
| Sensitivity curve points | 3,060 |
| Target/input/state summaries | 288 |
| Target/input summaries | 96 |
| SVG target figures | 12 |

Because the final models are linear Ridge pipelines, all final-fit sensitivity
curves are monotonic and continuous. This is expected model structure, not
evidence that the response is economically correct.

## Findings by scenario input

The meaningful-response threshold is:

```text
0.05 percentage points for price-return targets
1 basis point for the 10-year-yield target
```

| Scenario input | Responsive targets | Targets with negligible final-fit response | Targets with unstable historical fold direction |
|---|---:|---|---|
| `fed_funds_rate_change_bps` | 5/12 | Consumer Discretionary, Financials, Healthcare, Nasdaq, Oil, Technology, 10-year yield | Consumer Discretionary, Energy, Financials, Gold, S&P 500, Utilities |
| `cpi_surprise_pct` | 5/12 | Financials, Healthcare, Nasdaq, Oil, S&P 500, Technology, 10-year yield | Consumer Discretionary, Energy, Nasdaq, Oil, 10-year yield, Utilities |
| `oil_price_change_pct` | 9/12 | Healthcare, Technology, 10-year yield | None |
| `gdp_growth_surprise_pct` | 8/12 | Healthcare, Oil, Technology, 10-year yield | Consumer Discretionary, Energy, Financials, Gold, Nasdaq, S&P 500, Technology, 10-year yield |
| `unemployment_change_pct` | 8/12 | Healthcare, Oil, Technology, 10-year yield | Consumer Discretionary, Energy, Nasdaq, Oil, Technology, Utilities |
| `pmi_change_points` | 7/12 | Financials, Healthcare, Oil, Technology, 10-year yield | Energy, Utilities |
| `dxy_change_pct` | 7/12 | Financials, Healthcare, Oil, Technology, 10-year yield | Utilities |
| `vix_change_points` | 8/12 | Healthcare, Oil, Technology, 10-year yield | Oil, 10-year yield, Utilities |

The PMI input continues to mean:

```text
pmi_change_points
->
kc_fed_manufacturing_pmi_proxy_change_points
```

It is not described as national ISM history.

## Findings by target

| Target | Responsive inputs | Negligible inputs | Inputs with unstable historical direction |
|---|---:|---|---|
| S&P 500 | 7/8 | CPI | Fed Funds, GDP |
| Nasdaq | 6/8 | CPI, Fed Funds | CPI, GDP, Unemployment |
| 10-year yield | 0/8 | All eight | CPI, GDP, VIX |
| DXY | 8/8 | None | None |
| Gold | 8/8 | None | Fed Funds, GDP |
| Oil | 1/8 | All except Oil Price | CPI, Unemployment, VIX |
| Technology | 0/8 | All eight | GDP, Unemployment |
| Energy | 8/8 | None | CPI, Fed Funds, GDP, PMI, Unemployment |
| Financials | 4/8 | CPI, DXY, Fed Funds, PMI | Fed Funds, GDP |
| Utilities | 8/8 | None | CPI, DXY, Fed Funds, PMI, Unemployment, VIX |
| Healthcare | 0/8 | All eight | None |
| Consumer Discretionary | 7/8 | Fed Funds | CPI, Fed Funds, GDP, Unemployment |

Across all 96 target/input pairs:

```text
39 negligible final-fit responses
32 unstable historical walk-forward directions
0 limited conservative direction-reference mismatches
0 unusually large sensitivity spans
```

The zero reference-mismatch count applies only to the deliberately small set
of conservative plausibility references. It is not a claim that every
relationship is economically intuitive.

## Files created

### Reusable Python

- `scenario_modeling/src/scenario_explanations.py`

### Focused tests

- `scenario_modeling/tests/test_scenario_explanations.py`

### Executed notebook

- `scenario_modeling/notebooks/13_ridge_explanations_and_sensitivity.ipynb`

The notebook contains seven executed code cells, uses the
`insightpulse-backend` kernel, and has zero error outputs.

### Generated local evidence

Under `data/scenario_playground/interim/explanations/`:

- `feature_contributions.csv`
- `contribution_deltas.csv`
- `prediction_reconciliation.csv`
- `top_drivers.csv`
- `sensitivity_curves.csv`
- `sensitivity_state_summary.csv`
- `sensitivity_summary.csv`
- `target_explanations.json`
- `explanation_metadata_schema.json`
- `scenario_explanations_summary.json`
- `figures/<target>_sensitivity.svg` for all 12 targets

No fitted model binary was exported.

## Frozen Step 17 explanation schema

`explanation_metadata_schema.json` freezes:

- required target explanation fields;
- required contribution-row fields;
- the 12-target, eight-input, three-state sensitivity dimensions;
- reconciliation tolerance;
- confidence-preservation boundary; and
- the noncausal interpretation requirement.

All generated file and SVG hashes are recorded in
`scenario_explanations_summary.json`.

## Validation

The following passed:

- exact 12-target and 58-feature order;
- exact contribution and delta counts;
- all numeric contribution values finite;
- all 36 prediction reconciliations;
- only shock contributions change incrementally;
- all eight inputs tested for all 12 targets and three VIX states;
- all final Ridge curves finite, monotonic, and continuous;
- fixed state and release flags;
- neutral shocks produce zero incremental effect;
- repeated inference is deterministic;
- naive models are not substituted;
- explanation metadata schema validates;
- output and SVG hashes validate;
- zero locked final-test target rows parsed;
- zero final-test accesses; and
- no model binary export.

## Remaining limitations

- Ridge explanations describe the fitted linear associations, not causal
  market effects.
- Thirty-nine target/input pairs have negligible final-fit responses.
- Thirty-two target/input pairs have unstable directions across historical
  folds.
- Strong shrinkage makes the 10-year yield, Technology, and Healthcare
  insensitive to all eight shocks under the documented thresholds.
- A monotonic linear curve can still have the wrong sign or weak predictive
  usefulness.
- No uncertainty interval or production artifact exists.
- All-target XGBoost was intentionally not evaluated.

## Completion and handoff

Step 15 is complete and its accepted evidence is suitable for Step 16 and for
later inclusion in the Step 17 Ridge bundle.
