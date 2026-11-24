# Scenario Playground Step 17 Artifact Export Report

## Status and scope

Step 17 is complete:

```text
STEP_17_ARTIFACT_EXPORT_COMPLETE
```

Step 17 remained necessary and correctly scoped because corrected Step 14
could call all 12 Ridge models only after fitting them in memory. The exported
bundle creates the boundary between notebook research and future application
integration:

```text
fit the frozen models once during export
-> save fitted pipelines and frozen metadata
-> load and predict later without retraining or model data
```

No all-target XGBoost work, backend/frontend modification, API endpoint,
Supabase access, external API call, server start, or locked-final-test access
was performed.

## Exported bundle

Bundle version:

```text
scenario-playground-ridge-v1.0.0
```

Local export directory:

```text
scenario_modeling/artifacts/scenario_playground_ridge/v1.0.0/
```

Format:

```text
joblib
```

The directory contains 14 files. The checksum index covers the other 13 files
and excludes only itself to avoid a recursive self-hash.

| File | Purpose |
|---|---|
| `scenario_model_bundle.joblib` | 12 fitted pipelines plus lean runtime metadata and feature-only diagnostic references |
| `model_manifest.json` | machine-readable model, data, schema, runtime, security, and lock contract |
| `artifact_checksums.json` | SHA-256 index for every other bundle file |
| `feature_columns.json` | exact ordered 58-feature schema |
| `feature_definitions.json` | feature roles, units, sources, formulas, and point-in-time rules |
| `feature_ranges.json` | historical ranges and quantiles for 53 continuous shock/state inputs |
| `target_definitions.json` | 12 targets, units, Ridge roster, alphas, and confidence classes |
| `evaluation_summary.json` | validation metrics, naive comparisons, responsiveness, and limitations |
| `inference_configuration.json` | input/release mappings, default state, dates, and isolation policy |
| `explanation_schema.json` | frozen `scenario_explanation_metadata_v1` contract |
| `confidence_schema.json` | frozen `scenario_confidence_metadata_v1` contract |
| `environment.json` | Python, dependency versions, requirements hash, and joblib security warning |
| `requirements_snapshot.txt` | exact modeling requirements at export |
| `reference_signature.json` | accepted Step 14-16 reference fields for tolerance-aware reload verification |

Generated Step 17 evidence is stored under:

```text
data/scenario_playground/interim/artifact_export/
```

It contains:

```text
artifact_export_summary.json
example_runtime_result.json
neutral_runtime_result.json
fresh_process_verification.json
```

Both the fitted bundle and generated evidence are ignored by Git.

## Lean runtime architecture

Callable loader:

```python
from scenario_modeling.src.scenario_runtime import (
    load_scenario_model_bundle,
)

engine = load_scenario_model_bundle(bundle_path)
result = engine.predict(scenario_shocks)
```

The loaded `ScenarioModelRuntime`:

- has no training method;
- loads 12 already-fitted pipelines;
- does not import or call the matrix/training builders;
- does not read `data/scenario_playground`;
- does not read target labels;
- does not call external APIs or Supabase;
- validates bundle hashes before deserialization by default; and
- validates every pipeline, alpha, unit, fitted feature order, and frozen
  schema after loading.

The payload does not contain the training-oriented `ResearchInferenceEngine`,
the complete training DataFrame, or target labels.

It does include a compact feature-only diagnostic reference:

```text
464 standardized 58-feature rows
464 sample dates
53 continuous-input value distributions
nearest-neighbor distances
per-target contribution thresholds
```

This target-free reference is required to reproduce the accepted Step 15
contribution checks and Step 16 empirical percentile and historical-neighbor
diagnostics without reading the training matrix at runtime.

## Model registry

All 12 entries are independent:

```text
SimpleImputer(strategy="median")
-> StandardScaler()
-> Ridge(frozen target-specific alpha)
```

| Target | Alpha | Output unit | Frozen Step 13 evidence class |
|---|---:|---|---|
| `sp500` | 10,000 | percent | `APPROVED_WITH_LOW_CONFIDENCE` |
| `nasdaq` | 10,000 | percent | `NAIVE_FALLBACK_REQUIRED` |
| `ten_year_yield_bps` | 1,000,000 | basis points | `NAIVE_FALLBACK_REQUIRED` |
| `dxy` | 100 | percent | `NAIVE_FALLBACK_REQUIRED` |
| `gold` | 10 | percent | `NAIVE_FALLBACK_REQUIRED` |
| `oil` | 1,000,000 | percent | `NAIVE_FALLBACK_REQUIRED` |
| `technology` | 1,000,000 | percent | `NAIVE_FALLBACK_REQUIRED` |
| `energy` | 10,000 | percent | `APPROVED_WITH_LOW_CONFIDENCE` |
| `financials` | 100,000 | percent | `NAIVE_FALLBACK_REQUIRED` |
| `utilities` | 1,000 | percent | `NAIVE_FALLBACK_REQUIRED` |
| `healthcare` | 100,000 | percent | `NAIVE_FALLBACK_REQUIRED` |
| `consumer_discretionary` | 10,000 | percent | `APPROVED_WITH_LOW_CONFIDENCE` |

The nine `NAIVE_FALLBACK_REQUIRED` classifications remain weak-evidence
warnings. They do not suppress or replace their Ridge outputs in the corrected
all-Ridge research interface. Naive strategies remain evaluation benchmarks
only.

## Frozen inference contract

Eight required public inputs:

```text
fed_funds_rate_change_bps
cpi_surprise_pct
oil_price_change_pct
gdp_growth_surprise_pct
unemployment_change_pct
pmi_change_points
dxy_change_pct
vix_change_points
```

The PMI contract preserves the disclosed mapping:

```text
pmi_change_points
->
kc_fed_manufacturing_pmi_proxy_change_points
```

The exact 58-feature order is:

```text
8 scenario-shock features
+ 5 release flags
+ 45 current-state features
```

Definitions:

```text
baseline =
fixed state + neutral shocks + fixed release flags

scenario =
the same fixed state + user shocks + the same release flags

incremental =
scenario - baseline
```

Only the eight shock magnitudes may differ between the pair.

The bundled default state is versioned and truthfully dated:

```text
sample date: 2026-06-11
state cutoff date: 2026-06-09
state features: 45
```

An explicit complete 45-feature state is also supported. The caller may supply
its sample and cutoff dates. If a caller-provided state has no sample date,
freshness is explicitly `STATE_DATE_UNAVAILABLE`.

## Complete accepted example

Scenario:

```text
Fed Funds: 0 bps
CPI surprise: +0.2 percentage points
Oil: +5%
GDP growth surprise: 0 annualized percentage points
Unemployment: 0 percentage points
PMI: 0 points
DXY: +1%
VIX: +5 points
```

| Target | Unit | Baseline | Scenario | Incremental |
|---|---|---:|---:|---:|
| S&P 500 | percent | 0.622545 | 0.559086 | -0.063459 |
| Nasdaq | percent | 0.872261 | 0.797250 | -0.075011 |
| 10-year yield | basis points | 0.476885 | 0.480230 | +0.003345 |
| DXY | percent | -0.195015 | 0.406944 | +0.601959 |
| Gold | percent | 2.265318 | 2.153837 | -0.111480 |
| Oil | percent | 1.072739 | 1.075137 | +0.002399 |
| Technology | percent | 1.813204 | 1.812471 | -0.000732 |
| Energy | percent | 0.093058 | 0.026582 | -0.066476 |
| Financials | percent | 0.828518 | 0.822248 | -0.006271 |
| Utilities | percent | 1.249663 | 0.997988 | -0.251675 |
| Healthcare | percent | 0.981351 | 0.978166 | -0.003185 |
| Consumer Discretionary | percent | 0.674845 | 0.587243 | -0.087602 |

All 36 values are finite Python numbers.

## Explanations and confidence

The reloaded runtime returns:

- exact standardized Ridge contribution decomposition;
- baseline, scenario, and incremental top drivers;
- contribution-group reconciliation;
- historical shock-sensitivity evidence;
- input range and percentile diagnostics;
- 58-feature historical-neighbor distance;
- regime-specific walk-forward evidence;
- residual and naive-benchmark evidence;
- state freshness;
- structured warnings; and
- non-compensatory overall research confidence.

The current example preserves:

```text
explanation schema: scenario_explanation_metadata_v1
confidence schema: scenario_confidence_metadata_v1
warning records: 88
XGBoost status: UNAVAILABLE_INTENTIONALLY_POSTPONED
```

All explanations are predictive decompositions. They do not claim causality.

## Fresh-process verification

The verification started a separate Python process and installed guards before
loading:

```text
Pipeline.fit: blocked
SimpleImputer.fit: blocked
StandardScaler.fit: blocked
Ridge.fit: blocked
all data/scenario_playground reads: blocked
```

Result:

| Check | Result |
|---|---:|
| Pipelines loaded | 12 |
| Fit attempts | 0 |
| Training-data reads | 0 |
| Target-label reads | 0 |
| External API calls | 0 |
| Supabase calls | 0 |
| Finite outputs | pass |
| Neutral increments exactly zero | pass |
| Explanation fields match | pass |
| Confidence fields match | pass |
| Warning fields match | pass |
| Repeated-load determinism | pass |
| Locked-final-test access | 0 |

Comparison tolerance:

```text
absolute tolerance: 1e-10
maximum observed numeric difference: 4.8197001945027296e-12
mismatched fields: 0
```

## Integrity, security, and environment

Validation confirmed:

- all SHA-256 entries match;
- tampering with a metadata file is detected before load;
- the manifest and joblib payload carry the same bundle version;
- the payload has no target-label or complete-training-frame fields;
- a scan of all 13 text bundle files found zero credential-shaped values;
- no new dependency was installed for Step 17;
- existing exact dependencies remain in
  `scenario_modeling/requirements.txt`; and
- joblib's trusted-source-only security requirement is documented in the
  manifest, environment metadata, loader docstring, and handoff guide.

## Validation results

```text
Python compile:
PASS

Focused Step 17 suite:
16 tests passed

Complete Scenario Modeling suite:
148 tests passed in 114.029 seconds

Step 17 notebook:
5/5 code cells executed
0 error outputs
kernel: insightpulse-backend

Bundle hash validation:
13/13 covered files matched

Credential-shaped text scan:
0 findings

Git ignore validation:
bundle ignored by scenario_modeling/artifacts/
generated evidence ignored by data/scenario_playground/interim/
```

## Team handoff

The operational guide is:

```text
scenario_modeling/MODEL_BUNDLE_HANDOFF.md
```

Offline smoke command:

```powershell
backend\.venv\Scripts\python.exe -m scenario_modeling.run_bundle_smoke_test `
  --bundle scenario_modeling\artifacts\scenario_playground_ridge\v1.0.0 `
  --forbid-training-data
```

Because fitted model binaries are ignored by Git, the complete version
directory must be transferred through a trusted team channel with its manifest
and checksum file intact.

## Remaining limitations

- All outputs are research-only predictive associations.
- Three targets retain low confidence and nine retain very low/weak evidence.
- Nine Ridge models did not reliably beat their best naive evaluation
  benchmark.
- The bundled state is dated 2026-06-11 and becomes increasingly stale until a
  complete newer state is supplied.
- PMI remains the disclosed Kansas City Fed manufacturing proxy.
- The bundle is tied to pinned Python/scikit-learn/joblib versions.
- Joblib files are unsafe when obtained from an untrusted source.
- All-target XGBoost remains intentionally postponed and is not required for
  this valid all-Ridge bundle.

Any future model-family, data, schema, default-state, or XGBoost change must
produce a newly versioned bundle rather than silently replacing `v1.0.0`.

## Scope confirmation

Step 17 performed:

```text
artifact export
lean runtime creation
fresh-process reload
offline predictions
explanation/confidence preservation
integrity and handoff validation
```

Step 17 did not perform:

```text
all-target XGBoost
backend changes
frontend changes
FastAPI endpoint creation
Supabase access
external API access
server startup
locked-final-test access
unrelated work
```
