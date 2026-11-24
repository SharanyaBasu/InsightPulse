# Scenario Playground Demo Guide

Use this as a short speaking guide for demos. The most important qualification
is:

> The Scenario Playground is a research model that estimates historical
> predictive associations over the next 21 trading days. It does not prove
> that a scenario causes a market outcome, and it is not production-approved.

## The 30-second explanation

The user describes a hypothetical economic scenario with eight shock values,
such as a CPI surprise, an oil-price move, or a VIX increase.

The system combines those values with:

- the latest available market and macroeconomic state;
- five flags describing which kind of economic release is being considered;
  and
- 12 independently trained Ridge regression models.

It immediately returns three 21-trading-day estimates for each of 12 market
targets:

```text
baseline    = expected change with neutral, zero-sized shocks
scenario    = expected change with the user's shocks
incremental = scenario - baseline
```

The current state and release flags are identical in the baseline and scenario
rows. Only the eight user-entered shock magnitudes change. This isolates the
model's scenario adjustment from changes in the surrounding market state.

## What we wanted to build

The goal was to turn the Scenario Playground's eight UI inputs into a
historically grounded answer to:

> Given the same market conditions, how would the model's 21-trading-day
> projection change when these shock assumptions change?

The model predicts percentage changes, not future price levels. The exception
is the 10-year Treasury yield, for which it predicts a change in basis points.
Twenty-one trading days is approximately one market month.

The prediction is produced immediately. We do not wait 21 days to show it;
21 trading days describes the future outcome window used to train and
interpret the prediction.

## The eight scenario inputs

Every neutral value is zero.

| Input | Meaning | Unit | Historical representation used for training |
|---|---|---|---|
| `fed_funds_rate_change_bps` | Change in the Fed's target-rate midpoint | Basis points | Official FOMC decision records and FRED target rates |
| `cpi_surprise_pct` | Initially reported monthly CPI inflation minus the latest compatible forecast available before release | Percentage points | Point-in-time ALFRED CPI actual minus Cleveland Fed headline CPI nowcast |
| `oil_price_change_pct` | Change in oil price over the historical shock window | Percent | FRED WTI spot oil (`DCOILWTICO`) |
| `gdp_growth_surprise_pct` | BEA advance real-GDP growth minus the latest pre-release GDPNow forecast | Annualized percentage points | BEA advance value and Atlanta Fed GDPNow TrackRecord |
| `unemployment_change_pct` | Current initially reported unemployment rate minus the prior initially reported rate | Percentage points | Point-in-time ALFRED `UNRATE` releases |
| `pmi_change_points` | Change in manufacturing activity | PMI-style index points | Kansas City Fed manufacturing proxy, mapped to `50 + 0.5 × composite` |
| `dxy_change_pct` | Change in the U.S. Dollar Index | Percent | Yahoo `DX-Y.NYB` |
| `vix_change_points` | Change in the VIX level | VIX index points | FRED `VIXCLS` |

Important input details:

- CPI and GDP are genuine surprise concepts: actual minus an expectation that
  was available before the release.
- A `+0.2` CPI surprise means inflation was 0.2 percentage points above the
  forecast, not that CPI increased by 20%.
- A `+25` Fed Funds value means a 25-basis-point increase.
- A `+5` VIX value means five index points, not 5%.
- The historical PMI source is a disclosed regional Kansas City Fed proxy,
  not official national ISM PMI history.

## What the model actually receives

Each model input row has exactly 58 ordered features:

```text
8 scenario-shock values
+ 5 economic-release flags
+ 45 current-state features
= 58 model features
```

### Five release flags

The flags are:

```text
fomc_release_flag
cpi_release_flag
unemployment_release_flag
gdp_release_flag
pmi_release_flag
```

They distinguish “no event occurred” from “an event occurred but its measured
shock was zero.” If flags are not supplied, nonzero macro shocks activate
their matching flags. The same resolved flags are then used for both baseline
and scenario prediction.

Oil, DXY, and VIX are market-window shocks and do not have release flags.

### Forty-five current-state features

The state describes the conditions surrounding the scenario:

- 10 macro and rate features: Fed target and effective rates, CPI monthly and
  yearly inflation, unemployment, advance GDP growth, 10-year and 2-year
  yields, the 10y–2y slope, and a real-rate proxy;
- the current VIX level; and
- 34 market risk and trend features across the S&P 500, Nasdaq, gold, oil, and
  DXY: 5-day and 21-day returns, volatility, drawdowns, moving-average
  distances, and long-window z-scores.

The oil 20-day percentage-volatility feature was deliberately excluded. The
negative April 2020 `CL=F` futures price made ordinary percentage volatility
mathematically unstable.

## How the historical data was collected

The download system cached raw sources separately under:

```text
data/scenario_playground/raw/
```

The main raw forms are:

- Yahoo daily market CSV files with date, OHLC, adjusted close, and volume;
- FRED and ALFRED observation JSON files with dates, values, and vintages;
- structured FOMC, CPI, unemployment, GDP, and PMI event records;
- official forecast/release workbooks and source manifests; and
- metadata sidecars recording source, retrieval, coverage, and validation
  information.

The source inventory downloaded 17 Yahoo candidates and retained 17 usable
FRED candidate histories, then selected the final source for each input and
target. Downloading a candidate did not automatically make it a model source.

Revision-sensitive macro data was handled point in time. For example, CPI and
unemployment use initially available values rather than silently replacing
history with today's revised values, and forecasts must predate their release.

The shared selected-source period is approximately August 2013 through
mid-July 2026. The final model rows run from August 5, 2013 through June 11,
2026; the earlier ending is necessary because a row needs 21 later trading
sessions to calculate its training target.

## Sampling and what one row looks like

Rows are event-aware rather than arbitrary weekly snapshots. A row is created
when an FOMC, CPI, unemployment, GDP, or PMI event becomes usable. If multiple
events become usable on the same trading date, they share one row with
multiple event identities.

The features are the market values including the 8 shocks (x), and the label (y) is the change in the 12 targets from time t to t + 21 days

The selected index originally contained 585 usable dates. One early CPI row
had an actual release but no compatible pre-release forecast. It was removed
instead of being filled with zero or future information, leaving:

```text
584 rows
58 model features
12 future targets
71 columns in the raw model matrix, including sample_date
```

Provenance, source dates, target dates, exclusions, and split roles are kept
in separate audit files rather than being passed to the models.

A shortened real row looks like this:

```text
sample_date:                    2013-08-05
unemployment_release_flag:      1
unemployment_change_pct:       -0.2 percentage points
oil_price_change_pct:          -1.223%
dxy_change_pct:                -0.559%
vix_change_points:             -1.1 points
current vix_level_points:      12.94
observed next-21-day S&P 500:  -4.330%
```

During training, the 58 known-at-the-time values are the inputs, while the
observed market change over the next 21 sessions is the label. Future target
values are never used as input features.

## The 12 outputs

Eleven targets are cumulative returns over the next 21 trading sessions:

```text
100 × (future value / baseline value - 1)
```

The 10-year-yield output is:

```text
100 × (future yield in percent - baseline yield in percent)
```

| Output | Historical proxy and field | Unit | Final Ridge alpha |
|---|---|---|---:|
| S&P 500 | Yahoo `^GSPC` Close | Percent | 10,000 |
| Nasdaq | Yahoo `^IXIC` Close | Percent | 10,000 |
| 10-year Treasury yield | FRED `DGS10` value | Basis points | 1,000,000 |
| DXY | Yahoo `DX-Y.NYB` Close | Percent | 100 |
| Gold | Yahoo `GLD` Adjusted Close | Percent | 10 |
| Oil | Yahoo `CL=F` Close | Percent | 1,000,000 |
| Technology | Yahoo `XLK` Adjusted Close | Percent | 1,000,000 |
| Energy | Yahoo `XLE` Adjusted Close | Percent | 10,000 |
| Financials | Yahoo `XLF` Adjusted Close | Percent | 100,000 |
| Utilities | Yahoo `XLU` Adjusted Close | Percent | 1,000 |
| Healthcare | Yahoo `XLV` Adjusted Close | Percent | 100,000 |
| Consumer Discretionary | Yahoo `XLY` Adjusted Close | Percent | 10,000 |

Indexes and futures use ordinary close. ETFs use adjusted close so splits and
distributions are treated consistently. The yield is a yield-level change,
not a bond-price return.

## How the models were trained

There is one independent model per target:

```text
median imputer
-> standard scaler
-> Ridge regression with a target-specific alpha
```

Ridge is regularized linear regression. Regularization shrinks unstable
coefficients, which is valuable because the dataset is small relative to its
58 features. A very large alpha means the evidence supported heavy shrinkage
and therefore only a small scenario response.

Training was chronological:

- preprocessing was fitted only inside each training period;
- five expanding walk-forward folds tested repeated future periods;
- rows whose 21-day labels crossed a validation boundary were purged;
- Ridge was compared with zero, training-mean, and trailing naive baselines;
  and
- a shallow S&P 500 XGBoost prototype was not selected because its modest
  error improvement did not justify ignoring most scenario inputs.

After the model choices were frozen, each exported Ridge pipeline was fitted
on 464 authorized pre-final-test rows. Six split-boundary rows remained
purged, and 114 locked final-test target rows were never accessed.

The exported bundle loads already-fitted models. It does not retrain or need
the training matrix during inference.

## How to read the three predictions

Suppose S&P 500 output is:

```text
baseline:     +0.62%
scenario:     +0.56%
incremental:  -0.06 percentage points
```

This means:

- under the fixed state and neutral shocks, the model projects a 0.62%
  cumulative S&P 500 change over 21 trading days;
- under the same state and the user's shocks, it projects 0.56%; and
- the scenario shifts the model projection down by 0.06 percentage points.

Do not add the scenario to the baseline. The relationship is:

```text
scenario = baseline + incremental
```

For the 10-year yield, the same three fields are basis-point changes. A
scenario value of `+12` means the yield is projected to be 12 basis points
higher after 21 trading sessions.

## Accepted example

This example uses CPI `+0.2` percentage points, oil `+5%`, DXY `+1%`, VIX
`+5` points, and zero for the other four shocks.

| Target | Baseline | Scenario | Incremental | Unit |
|---|---:|---:|---:|---|
| S&P 500 | 0.622545 | 0.559086 | -0.063459 | Percent |
| Nasdaq | 0.872261 | 0.797250 | -0.075011 | Percent |
| 10-year yield | 0.476885 | 0.480230 | +0.003345 | Basis points |
| DXY | -0.195015 | 0.406944 | +0.601959 | Percent |
| Gold | 2.265318 | 2.153837 | -0.111480 | Percent |
| Oil | 1.072739 | 1.075137 | +0.002399 | Percent |
| Technology | 1.813204 | 1.812471 | -0.000732 | Percent |
| Energy | 0.093058 | 0.026582 | -0.066476 | Percent |
| Financials | 0.828518 | 0.822248 | -0.006271 | Percent |
| Utilities | 1.249663 | 0.997988 | -0.251675 | Percent |
| Healthcare | 0.981351 | 0.978166 | -0.003185 | Percent |
| Consumer Discretionary | 0.674845 | 0.587243 | -0.087602 | Percent |

These values demonstrate the output contract; they are not a claim that the
scenario will produce those market moves.

## Explanations, diagnostics, and confidence

The result also explains and qualifies the numbers:

- exact Ridge feature contributions that reconcile to each prediction;
- top positive and negative baseline, scenario, and incremental drivers;
- whether shocks or state values are outside their historical ranges;
- how unusual the full 58-feature combination is relative to training rows;
- walk-forward errors, regime performance, and naive comparisons;
- historical shock-response stability;
- state-snapshot freshness; and
- structured research warnings.

A feature contribution is a predictive model decomposition, not a causal
economic explanation.

Current evidence is deliberately labeled conservatively:

- S&P 500, Energy, and Consumer Discretionary are
  `LOW_RESEARCH_CONFIDENCE_WITH_CONTEXT_WARNINGS`;
- the other nine are `VERY_LOW_RESEARCH_CONFIDENCE`;
- all 12 still return numeric Ridge outputs; and
- the historical `NAIVE_FALLBACK_REQUIRED` classification is warning evidence
  only. A naive forecast is never substituted for a Ridge prediction.

The bundled default state has sample date `2026-06-11` and information cutoff
`2026-06-09`. It is a reproducible snapshot, not live market data, so stale
state warnings are expected until an integration supplies all 45 features
from a newer state.

## Quick demo questions

**Are these price forecasts?**\
No. Eleven outputs are percentage changes over the next 21 trading days. The
10-year yield output is a basis-point change.

**Does “incremental” mean the shock caused that move?**\
No. It is the difference between two predictions from a historical
association model while state and flags are held fixed.

**What happens if every shock is zero?**\
Scenario equals baseline and every incremental output is exactly zero.

**Why are some incremental effects very small?**\
Strong Ridge regularization shrank relationships that were not stable in the
historical data. Small output can represent weak evidence, not economic
irrelevance.

**Why use event-aware rows?**\
They align training examples with actual macro releases and FOMC decisions
instead of arbitrary weekly dates.

**Why 12 separate models?**\
Each target has different behavior, unit, and required regularization.

**Does the model use Supabase at inference?**\
No. The research bundle is local and offline. Backend, Supabase, and frontend
integration are separate work.

**Is the final test result available?**\
No. Its target values remained locked, so the current evidence is
walk-forward research validation rather than a final untouched test score.

## Minimal callable example

Run from the repository root in `backend/.venv`:

```python
from scenario_modeling.src.scenario_runtime import load_scenario_model_bundle

engine = load_scenario_model_bundle(
    "scenario_modeling/artifacts/scenario_playground_ridge/v1.0.0"
)

result = engine.predict({
    "fed_funds_rate_change_bps": 0.0,
    "cpi_surprise_pct": 0.2,
    "oil_price_change_pct": 5.0,
    "gdp_growth_surprise_pct": 0.0,
    "unemployment_change_pct": 0.0,
    "pmi_change_points": 0.0,
    "dxy_change_pct": 1.0,
    "vix_change_points": 5.0,
})

print(result["targets"]["sp500"])
```

The loader verifies the trusted bundle, loads 12 fitted pipelines, and makes
predictions without retraining, reading the training matrix, calling external
APIs, or accessing Supabase.

## Best one-sentence demo conclusion

> InsightPulse holds the supplied market context fixed, changes only eight
> user-defined economic shocks, and uses 12 transparent Ridge models to show
> the baseline, total scenario projection, and incremental model response over
> the next 21 trading days—with explicit warnings wherever the historical
> evidence is weak.
