# Scenario Playground Step 6 Scenario-Shock Report

## Status and scope

Step 6 is complete with one documented source-boundary missing value.

The implementation constructs the historical equivalents of the eight Scenario Playground inputs on the 585 selected `EVENT_AWARE` rows. It uses only Step 3 caches and Step 4 timing boundaries.

It does not load Step 5 targets, construct state features, assemble a model matrix, create splits, train models, access Supabase, or modify application code.

## Why this step is necessary

The model cannot learn from user-entered shocks unless historical features use the same economic meanings and units. In particular:

- a CPI or GDP surprise must remain actual minus a pre-release expectation;
- VIX change must remain an index-point change;
- Fed Funds change must remain basis points;
- an absent event must be distinguishable from an event whose measured shock was exactly zero; and
- the regional manufacturing proxy must not be silently described as national ISM PMI.

## Inputs

Primary row index:

```text
data/scenario_playground/interim/sampling/selected_sampling_index.csv
```

Normalized events:

```text
data/scenario_playground/interim/sampling/normalized_event_records.csv
```

Selected cached sources:

```text
Federal Reserve FOMC decision history
ALFRED CPIAUCSL and UNRATE output_type=2 chunks
Cleveland Fed monthly inflation-nowcast JSON
Atlanta Fed GDPNow TrackRecord workbook
Kansas City Fed manufacturing-proxy event records
FRED DCOILWTICO
Yahoo DX-Y.NYB Close
FRED VIXCLS
```

No source was refreshed and no external request was made.

## Final historical shock representations

| Contract input | Historical model feature | Formula | Unit | Event/window behavior |
|---|---|---|---|---|
| `fed_funds_rate_change_bps` | Same | Official target midpoint after minus before, represented in the cached decision record as basis points | bps | Nonzero or zero only on attached FOMC events; holds remain explicit zero-bps events |
| `cpi_surprise_pct` | Same | Initial-release headline seasonally adjusted CPI month-over-month inflation minus latest Cleveland Fed headline CPI nowcast strictly before release | percentage points | Nonzero or zero only on attached CPI events |
| `oil_price_change_pct` | Same | `100 * (DCOILWTICO_end / DCOILWTICO_start - 1)` using the latest observation at or before each Step 4 boundary | percent | Calculated on every row |
| `gdp_growth_surprise_pct` | Same | BEA advance real-GDP quarter-over-quarter SAAR growth minus the GDPNow forecast identified by the TrackRecord as immediately before the advance estimate | annualized percentage points | Nonzero or zero only on attached GDP events |
| `unemployment_change_pct` | Same | Current initial UNRATE release minus the prior available initial UNRATE release | percentage points | Nonzero or zero only on attached unemployment events |
| `pmi_change_points` | `kc_fed_manufacturing_pmi_proxy_change_points` | Current `50 + 0.5 * KC composite` minus its prior monthly level | PMI-style points | Nonzero or zero only on attached PMI events |
| `dxy_change_pct` | Same | `100 * (DX-Y.NYB_end / DX-Y.NYB_start - 1)` | percent | Calculated on every row |
| `vix_change_points` | Same | `VIXCLS_end - VIXCLS_start` | VIX points | Calculated on every row |

The PMI mapping is intentional:

```text
user contract: pmi_change_points
historical model column: kc_fed_manufacturing_pmi_proxy_change_points
```

It preserves the eight-input contract without claiming that the regional Kansas City series is national ISM data.

## Output counts

Coverage:

```text
Rows:              585
First sample:      2013-08-05
Last sample:       2026-06-11
Shock features:    8
Release flags:     5
Numeric features: 13
```

Attached event counts:

| Event type | Events |
|---|---:|
| FOMC | 104 |
| CPI | 152 |
| Unemployment | 154 |
| GDP advance | 51 |
| PMI proxy | 154 |
| **Total** | **615** |

The 615 events appear on 585 rows because Step 4 retained 30 legitimate same-day collisions.

Release flags:

```text
fomc_release_flag
cpi_release_flag
unemployment_release_flag
gdp_release_flag
pmi_release_flag
```

These flags distinguish an absent macro event from a present event with a zero measured shock.

## Missingness

All shock and release-flag values are complete except:

```text
cpi_surprise_pct: 1 missing value
```

The missing row is:

```text
Sample date:       2013-08-16
Observation month: 2013-07
Release date:      2013-08-15
```

The ALFRED initial CPI actual is present, but the Cleveland archive begins its usable headline forecasts after this release. A post-release forecast was not used and the value was not replaced with zero.

The November 2025 CPI event remains usable. Its immediately preceding October `CPIAUCSL` level is absent from the release vintage, so the implementation uses the same-measure `Actual CPI Inflation` value embedded on the release date in the Cleveland archive. Its expectation remains the latest forecast strictly before release.

The later matrix step must apply its documented missing-row policy to the single July 2013 CPI gap. Step 6 performs no imputation or row removal.

## Observed shock ranges

Ranges include structural zeros on rows without the corresponding macro event.

| Feature | Missing | Nonzero | Minimum | Maximum |
|---|---:|---:|---:|---:|
| Fed Funds change | 0 | 31 | -100 bps | +75 bps |
| CPI surprise | 1 | 151 | -0.3712 pp | +0.6840 pp |
| Oil price change | 0 | 585 | -32.3529% | +55.0806% |
| GDP growth surprise | 0 | 51 | -3.8882 pp | +2.4557 pp |
| Unemployment change | 0 | 116 | -2.2 pp | +10.3 pp |
| KC Fed PMI-style proxy change | 0 | 142 | -12 points | +10 points |
| DXY change | 0 | 583 | -3.8535% | +2.3048% |
| VIX change | 0 | 579 | -9.26 points | +28.17 points |

These are construction diagnostics, not clipping or winsorization decisions.

## Point-in-time validation

All implemented checks pass:

```text
All 585 selected rows preserved
All eight historical shock features present
All five release flags present
All 615 attached events audited
Release-flag counts equal attached-event counts
Macro shocks are zero when their event type is absent
Every available CPI expectation is strictly before release
All event availability dates are on or before the sample date
Every state cutoff precedes the attached event availability
All market source dates are on or before their window boundaries
All non-missing values are finite
The regional PMI proxy name remains explicit
```

Maximum market-source staleness:

| Feature | Start maximum | End maximum |
|---|---:|---:|
| Oil | 0 calendar days | 3 calendar days |
| DXY | 0 calendar days | 3 calendar days |
| VIX | 0 calendar days | 0 calendar days |

No market window uses a source value after its boundary.

## Persisted outputs

```text
data/scenario_playground/interim/features/shocks/
  scenario_shocks.csv
  scenario_shock_event_audit.csv
  scenario_market_shock_audit.csv
  scenario_shock_summary.json
```

Shapes:

```text
scenario_shocks.csv:             585 rows x 21 columns
scenario_shock_event_audit.csv:  615 rows x 23 columns
scenario_market_shock_audit.csv: 1,755 rows x 16 columns
```

The main shock file contains eight key/provenance columns and thirteen numeric feature columns. The two audit files remain separate so source endpoint values and event construction details cannot accidentally become model features.

## Reusable implementation

```text
scenario_modeling/src/feature_sources.py
scenario_modeling/src/shock_features.py
scenario_modeling/notebooks/04_scenario_shocks.ipynb
scenario_modeling/tests/test_shock_features.py
```

The module can be run from the repository root with:

```powershell
backend\.venv\Scripts\python.exe -m scenario_modeling.src.shock_features
```

Use `--dry-run` to construct and validate without rewriting the interim outputs.

## Completion decision

Step 6 passes its implementation and point-in-time acceptance checks. Its eight historical feature definitions are reproducible, source and unit decisions are explicit, and the one unavailable source-boundary CPI value is preserved honestly.

Step 8 must decide whether to exclude that one row or handle it through a training-only missing-data policy. Step 6 does not make that matrix-level decision.
