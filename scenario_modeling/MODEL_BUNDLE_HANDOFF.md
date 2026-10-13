# Scenario Playground Ridge Bundle Handoff

## What this bundle is

The Step 17 bundle is a research-only, offline inference package containing:

- 12 independent fitted `SimpleImputer -> StandardScaler -> Ridge` pipelines;
- the exact ordered 58-feature schema;
- a versioned 45-feature default state snapshot;
- the eight public scenario-input mappings and five release-flag mappings;
- target units, selected alphas, validation evidence, and warnings;
- Step 15 contribution-explanation metadata;
- Step 16 confidence and extrapolation diagnostics; and
- a manifest and SHA-256 checksums.

It does not contain target labels or a complete training DataFrame. Loading it
does not retrain models or read `data/scenario_playground`.

Current version:

```text
scenario-playground-ridge-v1.0.0
```

## Current model behavior

Ridge v1 contains 12 independent fitted target-specific Ridge pipelines. Each
pipeline consumes the same ordered 58-feature row:

```text
8 user scenario shocks
+ 5 release flags
+ 45 market and macro state features
= 58 total features
```

For every target, the runtime returns a 21-trading-day baseline prediction,
scenario prediction, and incremental scenario effect. The result also includes
Ridge contribution explanations, confidence diagnostics, input and prediction
range checks, and structured warnings.

Default local location:

```text
scenario_modeling/artifacts/scenario_playground_ridge/v1.0.0/
```

The artifact directory is intentionally ignored by Git.

The complete `v1.0.0` directory is required. The `.joblib` file alone is not a
supported handoff because the runtime also uses the manifest, SHA-256
checksums, feature and target schemas, bundled state configuration,
explanation and confidence schemas, evaluation metadata, and reference
signature.

The bundle will be distributed separately from Git source control:

```text
TODO: insert the trusted Ridge v1 artifact download location after it is
published for the team.
```

## Security boundary

Joblib uses Python pickle semantics and can execute code while loading.

Only load a bundle received through a trusted team channel. Keep the complete
version directory together and validate `artifact_checksums.json` before use.
The supplied loader validates hashes by default, but a valid hash does not make
an untrusted source safe.

## Environment

The authoritative project environment and notebook kernel are:

```text
backend/.venv
Python (InsightPulse backend .venv)
kernel name: insightpulse-backend
```

Install the pinned modeling requirements:

```powershell
backend\.venv\Scripts\python.exe -m pip install -r scenario_modeling\requirements.txt
```

The exact inference-only dependency versions recorded by the exported bundle
are:

```text
joblib==1.5.3
numpy==2.4.3
pandas==3.0.1
scikit-learn==1.9.0
scipy==1.18.0
threadpoolctl==3.6.0
```

`scenario_modeling/requirements.txt` contains these packages but is broader
than runtime inference: it also includes research downloaders, XGBoost, and
the VS Code/Jupyter kernel stack.

## Load and predict

Run from the repository root:

```python
from scenario_modeling.src.scenario_runtime import (
    load_scenario_model_bundle,
)

bundle_path = (
    "scenario_modeling/artifacts/"
    "scenario_playground_ridge/v1.0.0"
)
engine = load_scenario_model_bundle(bundle_path)

scenario_shocks = {
    "fed_funds_rate_change_bps": 0.0,
    "cpi_surprise_pct": 0.2,
    "oil_price_change_pct": 5.0,
    "gdp_growth_surprise_pct": 0.0,
    "unemployment_change_pct": 0.0,
    "pmi_change_points": 0.0,
    "dxy_change_pct": 1.0,
    "vix_change_points": 5.0,
}

result = engine.predict(scenario_shocks)
```

The eight inputs are required and must be finite numbers:

| Input | Unit |
|---|---|
| `fed_funds_rate_change_bps` | basis points |
| `cpi_surprise_pct` | percentage points |
| `oil_price_change_pct` | percent |
| `gdp_growth_surprise_pct` | annualized percentage points |
| `unemployment_change_pct` | percentage points |
| `pmi_change_points` | PMI-style index points |
| `dxy_change_pct` | percent |
| `vix_change_points` | VIX index points |

`pmi_change_points` preserves the frozen Kansas City Fed manufacturing proxy
mapping. It is not relabeled as an official ISM PMI history.

## Returned targets

Each target returns numeric `baseline_prediction`, `scenario_prediction`, and
`incremental_scenario_effect` over 21 trading days.

| Target | Unit |
|---|---|
| `sp500` | percent |
| `nasdaq` | percent |
| `ten_year_yield_bps` | basis points |
| `dxy` | percent |
| `gold` | percent |
| `oil` | percent |
| `technology` | percent |
| `energy` | percent |
| `financials` | percent |
| `utilities` | percent |
| `healthcare` | percent |
| `consumer_discretionary` | percent |

Definitions:

```text
baseline =
fixed current state + neutral shocks + fixed release flags

scenario =
the same state + user shocks + the same release flags

incremental =
scenario - baseline
```

The models estimate predictive associations, not causal effects. Naive models
remain evaluation benchmarks only and never replace a Ridge output.

All 12 Ridge targets remain callable and return numeric outputs. Historical
weak-evidence classifications and confidence warnings are preserved in the
result; they do not suppress a Ridge target or substitute a naive prediction.

## State and release flags

If no state is supplied, the runtime uses the bundled snapshot:

```text
sample date: 2026-06-11
state cutoff date: 2026-06-09
state feature count: 45
```

`engine.predict(scenario_shocks)` uses this snapshot automatically when no
explicit state is supplied. This is a fixed, reproducible research/demo state,
not current or live market data. The state values and metadata are included in
the exported bundle, so runtime does not read local historical data to
reconstruct them. Dynamic or live state refresh is future integration work and
is not required for the Ridge v1 handoff.

The runtime reports the real snapshot date and evaluates freshness relative to
the call date. It does not pretend that the bundled state is live. As the
bundle ages, stale-state warnings are expected.

An integration can supply a complete explicit state:

```python
current_state = {
    name: value
    for name, value in zip(
        engine.state_feature_names,
        your_45_values,
        strict=True,
    )
}

result = engine.predict(
    scenario_shocks,
    current_state=current_state,
    state_sample_date="2026-07-20",
    state_cutoff_date="2026-07-19",
)
```

All 45 fields are required for an override. If dates are omitted, the runtime
returns an explicit `STATE_DATE_UNAVAILABLE` freshness warning.

Release flags are derived once from nonzero event shocks and held fixed between
baseline and scenario. A caller may instead supply all five flags:

```python
release_flags = {
    "fomc_release_flag": 0,
    "cpi_release_flag": 1,
    "unemployment_release_flag": 0,
    "gdp_release_flag": 0,
    "pmi_release_flag": 0,
}

result = engine.predict(
    scenario_shocks,
    release_flags=release_flags,
)
```

## Runtime independence

Loading and calling Ridge v1 does not require:

- retraining or fitting;
- the training matrix or target labels;
- raw, interim, or processed research datasets;
- Supabase or another database;
- external APIs or network access;
- environment credentials or API keys; or
- notebooks or research-only training modules.

## Backend integration boundary

The current source imports successfully when Python is launched from the
repository root. The documented backend launch convention runs from inside
`backend/`, which does not automatically place the repository-root
`scenario_modeling` package on the backend import path.

Backend integration must make the modeling package available through the
team's chosen repository/package structure and include the inference runtime
dependencies listed above. This handoff intentionally does not choose or
implement that integration. Do not use machine-specific `sys.path` edits as a
substitute for a repository-level package decision.

## Offline smoke test

This command blocks fit calls and rejects any attempt to read
`data/scenario_playground`:

```powershell
backend\.venv\Scripts\python.exe -m scenario_modeling.run_bundle_smoke_test `
  --bundle scenario_modeling\artifacts\scenario_playground_ridge\v1.0.0 `
  --forbid-training-data
```

This smoke test is the supported runtime handoff verification for teammates
who receive the minimal committed source and the complete separately
distributed `v1.0.0` bundle. It relies only on files included in that handoff;
no research test modules are required.

## Distribution

Because model binaries are ignored by Git:

1. Obtain the complete `v1.0.0` directory from the trusted team location that
   will replace the TODO above.
2. Transfer the directory through a trusted team channel.
3. Do not rename, omit, or separately edit files inside it.
4. Keep `model_manifest.json` and `artifact_checksums.json` beside the joblib
   payload.
5. Run the offline smoke test after transfer.
6. Create a new version directory for any future model, schema, state, or
   XGBoost roster change; never silently replace `v1.0.0`.

The current all-Ridge bundle is valid without all-target XGBoost. XGBoost can
be evaluated later and incorporated into a newly versioned bundle.
