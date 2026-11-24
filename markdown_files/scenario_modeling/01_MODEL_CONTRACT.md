# InsightPulse Scenario Playground Model Contract

## Purpose

This document freezes the meaning of one Scenario Playground model row before historical data collection begins. The frontend user enters eight shock magnitudes only; no shock date, event timestamp, release date, prediction date, or shock-window selector is user-facing.

---

## Training-row semantics

One training row represents the market and macroeconomic information available at a pre-shock state snapshot at time `t`, paired with the historical shock for that event or window, used to predict movement over the following 21 trading days.

Each row contains current market-state features, current macro-state features, eight historical scenario-shock features, and forward 21-trading-day target values.

### State-before-shock invariant

```text
State features must represent conditions before the shock being modeled.
The shock is applied hypothetically after that state snapshot.
The target represents movement over the following 21 trading days.
```

For event-based shocks such as Fed, CPI, GDP, unemployment, or PMI releases, state features must not contain the market reaction to that event. If CPI is released before the market opens, use the previous completed market close as the pre-shock state rather than the close after the release.

Release timestamps and effective dates may be retained internally during dataset construction. Exact alignment rules are deferred to sampling design, shock feature construction, and point-in-time dataset construction.

---

## Timestamp and information timing

The conceptual historical row timestamp is the completed market close immediately before the modeled shock. For ordinary non-event rows, it is the latest completed close used for the state snapshot; for event rows, it is the selected pre-event close.

For current inference:

```text
state snapshot =
the latest completed market close and latest macro information
available at that point
```

Market information is known only after the relevant close is complete. Macro information is known only after its release and availability time.

Historical rows must not use a macro value before release, a post-event close as the pre-event state, a later revised value as if it were known at `t`, future-filled data, or future observations in rolling features.

---

## Prediction horizon

The horizon is exactly 21 trading days, using the 21st future trading observation rather than 21 calendar days. Rows without a valid 21st future observation are excluded from training and evaluation.

During training, future observations are used only to calculate the label. During live inference, the model predicts the next 21 trading days immediately; the user does not wait 21 days for the prediction.

---

## Scenario input contract

| Input name | Meaning | Unit | Historical feature correspondence | Neutral |
|---|---|---|---|---:|
| `fed_funds_rate_change_bps` | Policy/effective Fed Funds rate change for a defined decision/release window | basis points | New released rate minus previous applicable rate | `0` |
| `cpi_surprise_pct` | Actual CPI inflation minus the pre-release expectation | percentage points | Actual CPI inflation minus pre-release consensus/nowcast | `0` |
| `oil_price_change_pct` | Oil movement over a fixed modeling shock window | percent | `100 * (oil_end / oil_start - 1)` | `0` |
| `gdp_growth_surprise_pct` | Actual GDP growth minus the pre-release expectation | percentage points | Actual GDP growth minus pre-release consensus/nowcast | `0` |
| `unemployment_change_pct` | Newly released unemployment rate minus the previous release | percentage points | New unemployment rate minus previous released rate | `0` |
| `pmi_change_points` | Released PMI level change | index points | New PMI release minus previous PMI release | `0` |
| `dxy_change_pct` | DXY movement over a fixed modeling shock window | percent | `100 * (DXY_end / DXY_start - 1)` | `0` |
| `vix_change_points` | VIX index movement over a fixed modeling shock window | VIX points | `VIX_end - VIX_start` | `0` |

Example inputs are `fed_funds_rate_change_bps = 25`, `cpi_surprise_pct = 0.3`, and `vix_change_points = 5`.

The canonical CPI and GDP names mean true surprises. If historical expectations are unavailable, use distinct names such as `cpi_surprise_proxy` or `gdp_surprise_proxy`; never silently use a raw change under a true-surprise name. Every proxy must record its method, coverage, quality, and limitation.

The historical windows for `oil_price_change_pct`, `dxy_change_pct`, and `vix_change_points` are not finalized. Later data-source and sampling work must select a fixed one-day, event-window, weekly, or other definition. The frontend will not select the window; the same definition must be used in training and inference.

---

## Target output contract

For all price or index targets:

```text
target_pct =
    100 * (value_at_t_plus_21_trading_days / value_at_t - 1)
```

The output unit is `%` and represents cumulative percentage movement over the horizon.

For the 10-year yield:

```text
ten_year_yield_bps =
    100 * (yield_at_t_plus_21_trading_days - yield_at_t)
```

Yield values are percentage points, so 3.90% to 4.05% equals +15 bps. This is a yield change, not a price return.

| Target | Intended proxy | Current value type | Formula | Unit | Neutral interpretation | Pending source decision |
|---|---|---|---|---|---|---|
| `sp500` | Preferably `SPY` or broad S&P 500 proxy; backend has `^GSPC` | Price/index level | `100 * (future/current - 1)` | `%` | No S&P 500 movement | ETF versus index and adjustment convention |
| `nasdaq` | Preferably `QQQ` or broad Nasdaq proxy; backend has `^IXIC` | Price/index level | `100 * (future/current - 1)` | `%` | No Nasdaq movement | ETF versus index and adjustment convention |
| `ten_year_yield_bps` | FRED `DGS10` or Yahoo `^TNX` | Yield percentage-point level | `100 * (future - current)` | `bps` | No yield change | Select authoritative series and units |
| `dxy` | `DX-Y.NYB` or approved DXY proxy | Index level | `100 * (future/current - 1)` | `%` | No DXY movement | Confirm source and calendar |
| `gold` | Preferably `GLD` or consistently handled gold series; backend has `GC=F` | Price level | `100 * (future/current - 1)` | `%` | No gold movement | Select ETF, futures, or spot and adjustment policy |
| `oil` | WTI spot, rolled futures, or approved ETF; backend has `CL=F` | Price level | `100 * (future/current - 1)` | `%` | No oil movement | Select source and roll/adjustment policy |
| `technology` | `XLK` | Sector ETF price level | `100 * (future/current - 1)` | `%` | No Technology movement | Confirm direct `XLK` history |
| `energy` | `XLE` | Sector ETF price level | `100 * (future/current - 1)` | `%` | No Energy movement | Confirm direct `XLE` history |
| `financials` | `XLF` | Sector ETF price level | `100 * (future/current - 1)` | `%` | No Financials movement | Confirm direct `XLF` history |
| `utilities` | Preferably `XLU` | Sector ETF price level | `100 * (future/current - 1)` | `%` | No Utilities movement | Direct `XLU` support is currently absent |
| `healthcare` | `XLV` | Sector ETF price level | `100 * (future/current - 1)` | `%` | No Healthcare movement | Confirm direct `XLV` history |
| `consumer_discretionary` | `XLY` | Sector ETF price level | `100 * (future/current - 1)` | `%` | No Consumer Discretionary movement | Confirm direct `XLY` history |

The target names are final; concrete source instruments are not. The current backend’s placeholder sector mappings do not define the model targets, and Utilities currently lacks direct `XLU` support.

For the overlapping input/target pairs:

```text
oil_price_change_pct -> oil
dxy_change_pct -> dxy
```

The input is the hypothetical initial shock applied after the state snapshot. The output is the cumulative return from the pre-shock starting value through the 21st future trading day. It may include both the imposed initial shock and additional movement during the remaining horizon.

---

## Baseline, scenario, and incremental predictions

For current Scenario Playground inference:

```text
state snapshot =
the latest completed market close and latest macro information
available at that point

baseline row =
the state snapshot plus eight neutral shock values of zero

scenario row =
the exact same state snapshot plus the eight user-entered shocks

baseline prediction =
the model prediction from the baseline row

scenario prediction =
the model prediction from the scenario row

incremental scenario effect =
scenario prediction minus baseline prediction
```

Only the eight shock fields change between the rows. State features are not overwritten by hypothetical inputs. A neutral scenario can still produce a nonzero baseline prediction.

The notebook must preserve, separately for all twelve targets:

```text
baseline_projection
scenario_projection
incremental_scenario_effect
```

---

## Missing-data and leakage rules

If the 21st future trading-day target is unavailable, exclude the row; do not shorten or relabel the target.

If a feature is unavailable at `t`, do not backfill it from the future, use a later revised value as historical knowledge, fabricate a true surprise, or use a post-shock close as the pre-shock state.

Any imputation must be fit on training data only and documented. If the latest inference state is incomplete, do not fabricate or silently substitute an unrelated proxy; mark the affected prediction unavailable or return a data-quality warning.

---

## Finalized Step 1 decisions

```text
One row is a pre-shock state snapshot paired with a historical shock
and a 21-trading-day future target.
The state-before-shock invariant is mandatory.
The user enters shock magnitudes only, with no user-facing time input.
The horizon is exactly 21 trading days.
The eight canonical input names and twelve target names are fixed.
Neutral shock values are zero.
Price/index outputs are cumulative percentage returns.
The 10-year yield output is a cumulative basis-point change.
True surprises remain distinct from proxy features.
Baseline, scenario, and incremental outputs are preserved separately.
Future information is never used in historical rows.
```

---

## Decisions deferred to later steps

```text
Final target instruments and source selection
^GSPC versus SPY
^IXIC versus QQQ
DGS10 versus ^TNX
GC=F versus GLD or spot gold
CL=F versus WTI spot, rolled futures, or an ETF
Adjusted versus unadjusted prices
Dividend and futures-roll treatment
Historical PMI source
Historical consensus/nowcast sources
Historical release timestamps and effective dates
Point-in-time vintage availability
Exact oil, DXY, and VIX shock windows
Fed Funds Rate definition
Direct Utilities/XLU history
Common calendars and missing-period handling
```

The exact historical alignment rules will be implemented later during sampling design, shock feature construction, and point-in-time dataset construction. No user-facing event timestamp or shock-window selector will be added.
