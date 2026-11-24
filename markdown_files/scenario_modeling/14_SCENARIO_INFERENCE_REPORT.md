# Scenario Playground Step 14 Corrected All-Target Ridge Inference Report

## Status and scope

Corrected Step 14 is complete:

```text
STEP_14_ALL_TARGET_RIDGE_INFERENCE_COMPLETE
```

The correction was necessary because the original Step 14 implementation used
the Step 13 evaluation classifications as hard serving gates. It fitted Ridge
for only S&P 500, Energy, and Consumer Discretionary, substituted naive
baselines for the other nine targets, and returned `null` scenario and
incremental fields for those nine.

The corrected implementation now:

- builds 12 independent target-specific Ridge pipelines;
- returns finite numeric Ridge baseline, scenario, and incremental predictions
  for all 12 targets;
- preserves every Step 13 metric and classification as confidence evidence;
- labels nine targets explicitly as weak-evidence experimental models;
- retains zero, training-mean, and trailing strategies as evaluation
  benchmarks only;
- never substitutes a naive value for a Ridge inference output;
- keeps all 114 final-test target rows locked;
- remains research-only, noncausal, and not production-approved.

No all-target XGBoost evaluation, Steps 15–17, model artifact export, API,
Supabase, backend, frontend, or server work was performed.

## Callable architecture

Reusable implementation:

```text
scenario_modeling/src/scenario_inference.py
```

Primary builder:

```python
from scenario_modeling.src.scenario_inference import (
    build_research_inference_engine,
)

engine = build_research_inference_engine()
```

Reusable prediction call:

```python
result = engine.predict(scenario_shocks)
```

Convenience call:

```python
from scenario_modeling.src.scenario_inference import predict_scenario

result = predict_scenario(scenario_shocks)
```

Building one engine and reusing `engine.predict(...)` avoids refitting on each
request.

The engine:

1. Uses the bounded reader to parse only the 464 authorized pre-final-test
   rows.
2. Verifies the frozen 58-feature order and all 12 target names.
3. Reads the 12 frozen Step 13 configurations and assessments.
4. Verifies each target's alpha, unit, model family, history strategy,
   training-row count, classification, and final-test lock.
5. Creates 12 separate median-imputer, standard-scaler, and Ridge pipelines.
6. Fits every preprocessing step and Ridge model on the same 464 authorized
   rows only.
7. Loads the latest complete target-free Step 7 state.
8. Creates a neutral baseline row and a scenario row with identical state and
   release flags.
9. Calls the correct target-specific pipeline for all 12 targets.
10. Returns model predictions, Step 13 evidence, naive comparisons, and
    warnings in one structured result.

No shared multi-output estimator is used. No fitted pipeline is serialized.

## Training and locked-data boundary

| Item | Verified result |
|---|---|
| Approved matrix | 584 rows, 58 features |
| Authorized pre-final-test rows fitted | 464 |
| Training samples | 2013-08-05 through 2023-11-06 |
| Last training target end | 2023-12-04 |
| Locked final-test rows declared | 114 |
| Locked final-test target rows parsed | 0 |
| Locked final-test access count | 0 |
| Latest cached state sample | 2026-06-11 |
| Latest state cutoff | 2026-06-09 |
| Latest state source | Step 7 `state_features.csv` |
| Latest state source contains targets | No |

The latest state is later than the model-training period. That is an explicit
research extrapolation, not hidden model support.

## Exact 58-feature mapping

Each inference row contains:

```text
8 scenario shocks
+ 5 release flags
+ 45 current-state features
= 58 ordered model features
```

### Eight user inputs

| Contract input | Frozen model feature | Unit | Neutral |
|---|---|---|---:|
| `fed_funds_rate_change_bps` | `fed_funds_rate_change_bps` | basis points | 0 |
| `cpi_surprise_pct` | `cpi_surprise_pct` | percentage points | 0 |
| `oil_price_change_pct` | `oil_price_change_pct` | percent | 0 |
| `gdp_growth_surprise_pct` | `gdp_growth_surprise_pct` | annualized percentage points | 0 |
| `unemployment_change_pct` | `unemployment_change_pct` | percentage points | 0 |
| `pmi_change_points` | `kc_fed_manufacturing_pmi_proxy_change_points` | PMI-style index points | 0 |
| `dxy_change_pct` | `dxy_change_pct` | percent | 0 |
| `vix_change_points` | `vix_change_points` | VIX points | 0 |

The PMI input remains explicitly mapped to the Kansas City Fed manufacturing
proxy. It is not presented as historical national ISM PMI.

### Five fixed release flags

```text
fomc_release_flag
cpi_release_flag
unemployment_release_flag
gdp_release_flag
pmi_release_flag
```

When flags are omitted, a nonzero event shock activates its matching flag.
They are derived once and copied unchanged into the baseline and scenario
rows. The caller may provide all five flags explicitly, including a
zero-surprise release context.

Oil, DXY, and VIX are market-window shocks and do not have dedicated release
flags.

### Forty-five state features

The 45 state features include the frozen macro, rate, volatility, return,
trend, drawdown, moving-average-distance, and z-score features approved in
Steps 7–9. They are identical between the baseline and scenario rows.

`oil_volatility_20d_annualized_pct` remains excluded under the Step 9
nonpositive-`CL=F` decision.

The exact ordered mapping and provenance remain in:

```text
data/scenario_playground/interim/inference/feature_mapping.csv
```

## Baseline, scenario, and incremental semantics

For every target:

```text
baseline prediction =
fixed current state
+ eight neutral shock magnitudes
+ fixed release flags

scenario prediction =
the same fixed current state
+ eight user-entered shock magnitudes
+ the same fixed release flags

incremental scenario effect =
scenario prediction - baseline prediction
```

Only the eight shock-magnitude columns may change. The implementation verifies:

- exact 58-feature ordering;
- state equality;
- release-flag equality;
- changed columns are a subset of the eight shock columns;
- no clipping or winsorization;
- a neutral eight-zero scenario gives exactly zero incremental output for all
  12 targets.

## Frozen target configurations and Step 13 evidence

MAE and RMSE are percentage points for price-return targets and basis points
for the 10-year-yield target. Direction is directional accuracy. The
classification column preserves the original Step 13 result.

| Target | Alpha | Unit | MAE | RMSE | OOS R² | Direction | Best naive | Ridge beat MAE/RMSE | Classification |
|---|---:|---|---:|---:|---:|---:|---|---|---|
| S&P 500 | 10,000 | percent | 3.575 | 5.027 | 0.014 | 66.7% | training mean | Yes / Yes | `APPROVED_WITH_LOW_CONFIDENCE` |
| Nasdaq | 10,000 | percent | 5.299 | 7.096 | -0.383 | 66.7% | training mean | No / No | `NAIVE_FALLBACK_REQUIRED` |
| 10-year yield | 1,000,000 | basis points | 19.765 | 26.599 | -0.084 | 44.7% | zero | No / No | `NAIVE_FALLBACK_REQUIRED` |
| DXY | 100 | percent | 1.446 | 1.911 | -0.042 | 54.3% | zero | No / No | `NAIVE_FALLBACK_REQUIRED` |
| Gold | 10 | percent | 2.989 | 3.775 | -0.029 | 54.3% | training mean | No / No | `NAIVE_FALLBACK_REQUIRED` |
| Oil | 1,000,000 | percent | 14.941 | 24.647 | -0.409 | 46.7% | zero | No / No | `NAIVE_FALLBACK_REQUIRED` |
| Technology | 1,000,000 | percent | 5.648 | 7.706 | -0.490 | 68.7% | training mean | No / No | `NAIVE_FALLBACK_REQUIRED` |
| Energy | 10,000 | percent | 7.211 | 10.590 | -0.028 | 60.7% | zero | Yes / No | `APPROVED_WITH_LOW_CONFIDENCE` |
| Financials | 100,000 | percent | 5.104 | 7.062 | -0.119 | 54.0% | training mean | No / No | `NAIVE_FALLBACK_REQUIRED` |
| Utilities | 1,000 | percent | 4.256 | 5.763 | -0.188 | 59.7% | training mean | No / No | `NAIVE_FALLBACK_REQUIRED` |
| Healthcare | 100,000 | percent | 3.328 | 4.361 | -0.008 | 58.7% | training mean | No / No | `NAIVE_FALLBACK_REQUIRED` |
| Consumer Discretionary | 10,000 | percent | 4.800 | 6.561 | -0.018 | 61.7% | training mean | Yes / No | `APPROVED_WITH_LOW_CONFIDENCE` |

`NAIVE_FALLBACK_REQUIRED` is retained as historical evaluation evidence. It no
longer means that Step 14 substitutes the benchmark for the Ridge prediction.

## Target metadata and warning behavior

Every target output includes:

```text
target
model family
independent-pipeline structure
selected alpha
target unit
Step 13 evaluation classification
confidence level
validation MAE
validation RMSE
out-of-sample R-squared
directional accuracy
best naive MAE and RMSE strategies
whether Ridge beat each benchmark
scenario-responsiveness counts and limitations
known source and model limitations
applicable warning codes
production-ready status
causal-effect status
```

The three prior promotion-gate targets return:

```text
confidence_level = LOW_CONFIDENCE_EXPERIMENTAL
LOW_CONFIDENCE_EXPERIMENTAL_RIDGE
```

The other nine return:

```text
confidence_level = WEAK_EVIDENCE_EXPERIMENTAL
WEAK_EVIDENCE_EXPERIMENTAL_RIDGE
RIDGE_DID_NOT_RELIABLY_OUTPERFORM_BEST_NAIVE_BENCHMARK
STEP13_NAIVE_FALLBACK_REQUIRED_RETAINED_AS_EVIDENCE_ONLY
```

All 12 include:

```text
RESEARCH_ONLY_NOT_PRODUCTION_APPROVED
PREDICTIVE_ASSOCIATION_NOT_CAUSAL_EFFECT
PMI_INPUT_USES_KANSAS_CITY_FED_MANUFACTURING_PROXY
```

## Naive benchmark policy

These strategies remain available in Step 13 research and evaluation evidence:

```text
naive_zero
naive_training_mean
naive_recent_trailing_21_observation
```

Each target result identifies its best naive strategy and comparison metrics.
Every result also records:

```text
serving_role = EVALUATION_BENCHMARK_ONLY_NOT_SUBSTITUTED
substituted_for_ridge_output = false
```

The corrected example has:

```text
naive benchmark substitutions: 0
null Ridge outputs: 0
```

## Executed example

Input:

```json
{
  "fed_funds_rate_change_bps": 0.0,
  "cpi_surprise_pct": 0.2,
  "oil_price_change_pct": 5.0,
  "gdp_growth_surprise_pct": 0.0,
  "unemployment_change_pct": 0.0,
  "pmi_change_points": 0.0,
  "dxy_change_pct": 1.0,
  "vix_change_points": 5.0
}
```

Derived and fixed release context:

```json
{
  "fomc_release_flag": 0,
  "cpi_release_flag": 1,
  "unemployment_release_flag": 0,
  "gdp_release_flag": 0,
  "pmi_release_flag": 0
}
```

Results using the cached 2026-06-11 state:

| Target | Confidence | Baseline | Scenario | Incremental | Unit |
|---|---|---:|---:|---:|---|
| S&P 500 | Low confidence | 0.622545 | 0.559086 | -0.063459 | percent |
| Nasdaq | Weak evidence | 0.872261 | 0.797250 | -0.075011 | percent |
| 10-year yield | Weak evidence | 0.476885 | 0.480230 | 0.003345 | basis points |
| DXY | Weak evidence | -0.195015 | 0.406944 | 0.601959 | percent |
| Gold | Weak evidence | 2.265318 | 2.153837 | -0.111480 | percent |
| Oil | Weak evidence | 1.072739 | 1.075137 | 0.002399 | percent |
| Technology | Weak evidence | 1.813204 | 1.812471 | -0.000732 | percent |
| Energy | Low confidence | 0.093058 | 0.026582 | -0.066476 | percent |
| Financials | Weak evidence | 0.828518 | 0.822248 | -0.006271 | percent |
| Utilities | Weak evidence | 1.249663 | 0.997988 | -0.251675 | percent |
| Healthcare | Weak evidence | 0.981351 | 0.978166 | -0.003185 | percent |
| Consumer Discretionary | Low confidence | 0.674845 | 0.587243 | -0.087602 | percent |

These are predictive research associations, not causal effects or production
forecasts.

The example also reports:

- two current-state range warnings for Gold state features;
- one DXY baseline warning because the current-state prediction is below the
  Step 13 out-of-sample prediction range;
- no clipping or winsorization.

## Input validation

The interface rejects:

- missing or extra shock fields;
- booleans, nonnumeric values, NaN, or infinity;
- incomplete or nonfinite current-state mappings;
- incomplete release-flag mappings;
- flag values other than exactly zero or one;
- unknown range policies;
- feature-set or feature-order mismatches.

Out-of-training-range shocks are preserved and warned by default. Strict mode
raises `ScenarioValidationError`. Neither mode silently clips values.

## Files modified

### Python and tests

- `scenario_modeling/src/scenario_inference.py`
  - fits all 12 independent Ridge pipelines;
  - removes fallback-serving branches;
  - returns numeric all-target inference;
  - exposes Step 13 evidence and confidence warnings;
  - preserves benchmark-only naive metadata.
- `scenario_modeling/tests/test_scenario_inference.py`
  - 18 corrected all-target acceptance tests.

### Notebook

- `scenario_modeling/notebooks/12_scenario_inference.ipynb`
  - demonstrates the callable interface;
  - shows all 12 numeric outputs;
  - validates isolation, neutral behavior, metadata, and lock status;
  - was executed top-to-bottom using `insightpulse-backend`.

### Documentation

- `markdown_files/scenario_modeling/13_ALL_TARGETS_REPORT.md`
- `markdown_files/scenario_modeling/14_SCENARIO_INFERENCE_REPORT.md`
- `markdown_files/scenario_modeling/README.md`

No dependency was installed, so `scenario_modeling/requirements.txt` required
no dependency change.

## Generated local research evidence

All generated evidence remains ignored by Git under:

```text
data/scenario_playground/interim/inference/
```

| File | Rows/objects | Purpose |
|---|---:|---|
| `feature_mapping.csv` | 58 | Exact ordered inference mapping |
| `input_ranges.csv` | 53 | Eight shock and 45 state ranges |
| `model_gate_audit.csv` | 12 | All-target Ridge, confidence, alpha, unit, and benchmark-substitution audit |
| `example_result.json` | 1 | Complete executed 12-target scenario result |
| `neutral_result.json` | 1 | All-target neutral behavior |
| `scenario_inference_summary.json` | 1 | Validation, configuration, lock, and completion evidence |

No model binary or production artifact was generated.

## Validation results

### Python and inference

```powershell
backend\.venv\Scripts\python.exe -m py_compile scenario_modeling\src\scenario_inference.py
backend\.venv\Scripts\python.exe -m scenario_modeling.src.scenario_inference --dry-run
backend\.venv\Scripts\python.exe -m scenario_modeling.src.scenario_inference
```

Results:

```text
12 independent Ridge pipelines
12 finite numeric baselines
12 finite numeric scenario predictions
12 finite numeric incremental effects
12 neutral incremental effects exactly zero
0 naive benchmark substitutions
0 locked final-test target rows parsed
0 locked final-test accesses
27/27 internal acceptance validations passed
```

### Tests

```powershell
backend\.venv\Scripts\python.exe -m unittest scenario_modeling.tests.test_scenario_inference -v
backend\.venv\Scripts\python.exe -m unittest discover -s scenario_modeling\tests -p "test_*.py" -v
```

Results:

```text
Focused corrected Step 14 tests: 18/18 passed
Complete Scenario Modeling suite: 114/114 passed
```

### Notebook

```text
Notebook: scenario_modeling/notebooks/12_scenario_inference.ipynb
Kernel: insightpulse-backend
Python: backend/.venv/Scripts/python.exe
Code cells executed: 7/7
Notebook errors: 0
```

The notebook was executed with the installed Jupyter kernel client. No new
notebook dependency was installed.

### Safety and repository checks

```text
Generated inference files ignored by Git: PASS
Obsolete fallback/null serving-pattern scan: PASS
Secret-pattern scan: PASS
Model binaries exported: 0
Final-test target access: 0
```

## Remaining limitations

- No target earned the strongest Step 13 experimental-inference
  classification.
- Only S&P 500, Energy, and Consumer Discretionary passed the previous
  low-confidence promotion gate.
- Nine models did not reliably outperform their best naive benchmarks.
- Most pooled out-of-sample R-squared values are weak or negative.
- Scenario response magnitude and sign are unstable for several targets and
  often weaken in the latest fold.
- Making all 12 models callable does not improve their historical model
  quality.
- The latest cached state is dated 2026-06-11 rather than live.
- The current state extrapolates beyond the 2013–2023 fitting period.
- The PMI input uses the disclosed Kansas City Fed regional manufacturing
  proxy.
- Oil retains continuous-`CL=F` futures and roll-effect limitations.
- No uncertainty interval, explanation payload, production bundle, or API has
  been implemented.
- The locked final test remains unopened, so no untouched final-test estimate
  exists.

## Completion and next gate

Corrected Step 14 is complete:

```text
STEP_14_ALL_TARGET_RIDGE_INFERENCE_COMPLETE
```

The project is ready for a separate all-target XGBoost evaluation if the team
chooses to perform one. That evaluation must remain separate, preserve the
same matrix, target-end-date purging, final-test lock, and target-specific
confidence reporting, and must not reinterpret the Ridge results as causal.

This step did not perform any all-target XGBoost work, begin Steps 15–17,
export production artifacts, access Supabase, modify backend or frontend code,
or start a server.
