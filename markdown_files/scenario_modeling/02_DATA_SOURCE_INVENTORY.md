# InsightPulse Scenario Playground
## Step 2 — Data-Source and Coverage Inventory

Status: completed as a source and coverage inventory only.

No datasets were downloaded, no external APIs were called during this step, no live Supabase/Postgres database was queried, and no backend or frontend code was changed.

This document follows the authoritative model contract in `markdown_files/scenario_modeling/01_MODEL_CONTRACT.md`.

---

## 1. Executive summary

The notebook-modeling pipeline must use a separate historical dataset. Supabase/Postgres is the application database, not the notebook training-data store for this phase.

The future data root is:

```text
data/scenario_playground/
  raw/
  interim/
  processed/
```

The current repository provides prototype access to:

- Yahoo Finance through `yfinance` for daily market data.
- FRED through `fredapi` for macroeconomic and Treasury data.
- Finnhub for current equity quotes only.
- CoinGecko for current crypto quotes only.

The repository does not currently provide historical CPI or GDP expectation snapshots, historical economic-release timestamps, point-in-time macro vintages for modeling rows, PMI ingestion, direct `XLU` Utilities support, historical Scenario Playground shocks, or forward 21-trading-day labels.

The main expected coverage limiter is historical, point-in-time surprise data rather than raw market-price history.

### Status vocabulary

```text
Accepted source decision:
  Provisional source convention accepted for Step 3 collection.

Provisional candidate:
  Plausible source, but coverage, alignment, or semantics remain open.

Blocked pending verification:
  Cannot be used until access, coverage, licensing, or reproducibility is verified.

Deferred:
  Requires later sampling or modeling tests and is intentionally not fixed here.
```

---

## 2. Confirmed repository support

### 2.1 Yahoo market provider

`backend/providers/yahoo_provider.py` requests:

```text
^GSPC, ^IXIC,
VGK, EWJ, EEM, MTUM, VTV, IWF,
XLY, XLP, XLE, XLF, XLV, XLK, XLI, XLB, XLRE, XLC,
^IRX, ^FVX, ^TNX, HYG, LQD, TLT, ^VIX,
CL=F, NG=F, GC=F, SI=F, HG=F,
DX-Y.NYB, EURUSD=X, GBPUSD=X, AUDUSD=X,
USDJPY=X, USDCHF=X, CEW,
BTC-USD, ETH-USD
```

Confirmed behavior:

- Requests daily data and only the latest 365 calendar days.
- Uses `Close`, drops missing values, concatenates all successful series, and applies a complete-row `dropna()`.
- Does not explicitly choose adjusted or unadjusted prices.
- Does not explicitly define a market calendar.
- Does not include `XLU`.

The current yfinance documentation lists `auto_adjust=True` as the download default. Because the repository does not set this explicitly, Step 3 must freeze the adjustment convention. [yfinance download documentation](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html)

The yfinance project is an unofficial wrapper and states that downloaded data remain subject to Yahoo terms. Commercial or redistributable use must be verified. [yfinance project notice](https://github.com/ranaroussi/yfinance)

### 2.2 FRED provider

`backend/providers/fred_provider.py` currently requests:

| Backend name | FRED identifier |
|---|---|
| `CPI` | `CPIAUCSL` |
| `Unemployment` | `UNRATE` |
| `Fed_Funds_Rate` | `FEDFUNDS` |
| `GDP` | `GDPC1` |
| `DGS2` | `DGS2` |
| `DGS10` | `DGS10` |

Confirmed behavior:

- Requests full available series because no start date is passed.
- Combines monthly, quarterly, and daily observations.
- Forward-fills values and drops rows that remain incomplete.
- Does not retain release timestamps, ALFRED vintages, or original-versus-revised values.

FRED supports real-time periods, vintage dates, and initial-release-only output, but that capability is not currently used. [FRED observations API](https://fred.stlouisfed.org/docs/api/fred/series_observations.html)

The notebook pipeline must not copy the backend's forward-fill behavior into point-in-time event features without release dates.

### 2.3 Other providers and calculated metrics

Finnhub provides current quotes for `AAPL`, `MSFT`, `NVDA`, `AMZN`, `GOOGL`, `META`, `TSLA`, `SPY`, `QQQ`, and `DIA`; it is not a historical modeling source. CoinGecko provides current crypto quotes; crypto is not required by the model contract.

The existing Scenario route is a placeholder: it computes one-period deltas from two backend rows and uses incorrect placeholder sector mappings. It is not a training-data source.

`backend/services/calculation_service.py` currently calculates 5-day returns, 20-day annualized log-return volatility, drawdown, 252-day or available-history z-scores, 50-day and 200-day moving-average distances, yield-curve slope, a Fed Funds minus CPI real-rate proxy, one-day DXY return, and BTC/ETH ratio. These are state-feature candidates, not historical shocks or forward labels.

---

## 3. Accepted source decisions for Step 3

1. Use a separate notebook historical dataset, independent of Supabase/Postgres.
2. Use FRED/ALFRED for official macro actuals, Treasury yields, and vintage metadata where available.
3. Use FRED target-rate series for the primary historical policy-rate shock.
4. Keep `EFFR` as policy state or validation, not the primary user-facing shock.
5. Preserve Manufacturing PMI Change as the contract meaning of `pmi_change_points`, but use the explicitly named Kansas City Fed regional PMI-style proxy for free MVP historical training after direct ISM history proved unsuitable.
6. Keep ISM Manufacturing and Services data outside the primary path unless separately licensed; never combine Manufacturing and Services PMI into one feature.
7. Use `VIXCLS` as the preferred VIX candidate and Yahoo `^VIX` as the backup comparison.
8. Keep true DXY separate from the FRED broad trade-weighted dollar index.
9. Keep true surprises separate from change-based proxy features.
10. Store raw source data by provider and series before creating comparisons or processed tables.

These are source-collection decisions, not final feature-engineering or sampling decisions.

---

## 4. Federal Funds policy shock

Use the official machine-readable FRED series:

```text
DFEDTARL
DFEDTARU
DFEDTAR
EFFR
```

For the target-range period:

```text
target_midpoint_pct = (DFEDTARL + DFEDTARU) / 2

fed_funds_rate_change_bps =
    100 * (target_midpoint_pct - previous_target_midpoint_pct)
```

For the historical single-target period, use `DFEDTAR`.

`EFFR` remains a state or validation series. It is not the primary user-facing policy shock. FOMC release timestamps must be retained separately for state-before-shock alignment. Do not build the primary historical shock by scraping FOMC statements.

| Variable | Role | Source | Frequency | Unit | Transformation | Status |
|---|---|---|---|---|---|---|
| `fed_funds_target_lower_pct` | State/shock source | FRED `DFEDTARL` | Event/effective-date | Percent | Preserve value and vintage | Accepted candidate |
| `fed_funds_target_upper_pct` | State/shock source | FRED `DFEDTARU` | Event/effective-date | Percent | Preserve value and vintage | Accepted candidate |
| `fed_funds_target_single_pct` | Historical state/shock source | FRED `DFEDTAR` | Event/effective-date | Percent | Use in single-target period | Accepted candidate |
| `fed_funds_target_midpoint_pct` | State | Derived target midpoint | Event/effective-date | Percent | Midpoint of lower and upper target | Accepted derived state |
| `fed_funds_rate_change_bps` | Scenario input | Derived target change | Event-based | Basis points | Difference from previous applicable target times 100 | Accepted provisional definition |
| `effective_fed_funds_rate_pct` | State/validation | FRED `EFFR` | Daily | Percent | Preserve daily value | Accepted validation candidate |

The exact treatment of effective dates, same-day releases, and unchanged FOMC decisions remains a sampling decision.

---

## 5. CPI candidates and matching definitions

`CPIAUCSL` alone is not a complete CPI-surprise source. It is one possible actual series.

| Definition | Candidate identifier | Frequency | Unit | Status |
|---|---|---|---|---|
| Headline CPI, seasonally adjusted | `CPIAUCSL` | Monthly | CPI index | Accepted candidate |
| Headline CPI, not seasonally adjusted | `CPIAUCNS` | Monthly | CPI index | Accepted candidate |
| Core CPI, seasonally adjusted | `CPILFESL` | Monthly | CPI index | Accepted candidate |
| Core CPI, not seasonally adjusted | Candidate `CUSR0000SA0L1E`; verify official metadata | Monthly | CPI index | Blocked pending identifier verification |

The actual CPI measure and expectation must match on:

```text
Headline versus core
Month-over-month versus year-over-year
Seasonally adjusted versus not seasonally adjusted
Reference month
Release vintage
```

No forecast or nowcast may be subtracted from an actual with a different definition.

Provisional definition:

```text
cpi_surprise_pct =
    actual CPI inflation released by BLS
    - expectation available before the release
```

The exact CPI measure remains unresolved until historical expectation coverage is measured. The Cleveland Fed provides pre-release CPI nowcasts, but a public description is not proof that a reproducible historical downloader and complete archive have been verified. [Cleveland Fed Inflation Nowcasting](https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting)

Possible expectation candidates are Cleveland Fed nowcasts, Philadelphia Fed SPF, or a licensed consensus source. Raw CPI changes must not be named `cpi_surprise_pct`.

---

## 6. Expectation snapshot rule

For CPI and GDP:

```text
expectation =
the latest forecast or nowcast published strictly before the
corresponding official release timestamp
```

Store the forecast timestamp, retrieval timestamp, forecast vintage, and official release timestamp separately.

For GDP, the provisional comparison is:

```text
final pre-release GDPNow estimate
versus
the BEA advance estimate of real GDP growth
```

Both values must use the same convention, provisionally seasonally adjusted annualized quarter-over-quarter real GDP growth. Do not use an arbitrary earlier GDPNow update or a later revised GDP actual.

The Atlanta Fed describes GDPNow as a running nowcast rather than an official forecast. Its visible archive does not prove that a stable historical downloader and complete pre-release coverage have been verified. [Atlanta Fed GDPNow](https://www.atlantafed.org/cqer/research/gdpnow/)

The Philadelphia Fed provides real-time datasets and SPF forecast files, but SPF is not automatically equivalent to a release-specific GDPNow snapshot. [Philadelphia Fed real-time data](https://www.philadelphiafed.org/surveys-and-data/real-time-data-research)

Historical forecast availability and exact release matching remain Step 3 verification items.

---

## 7. PMI status

The contract meaning remains Manufacturing PMI Change. Step 3 did not silently call a regional series the national ISM series. The selected historical representation is:

```text
historical proxy feature name =
    kc_fed_manufacturing_pmi_proxy_change_points

kc_pmi_style_level =
    50 + 0.5 * Kansas City composite diffusion points

later proxy change =
    current kc_pmi_style_level
    - previous kc_pmi_style_level
```

The Federal Reserve Bank of Kansas City Tenth District Manufacturing Survey is the final free student-MVP source. Its composite uses the same five equally weighted categories as the national ISM Manufacturing PMI—production, new orders, employment, supplier delivery time, and raw-material inventories—and is a manufacturing diffusion index. Its raw neutral value is 0; the deterministic PMI-style mapping has neutral 50.

The important limitation is geographic: this is a seven-state Tenth District measure, not the national ISM Manufacturing PMI. The user-facing contract label remains `PMI Change`, but every modeling artifact must disclose the internal proxy name and regional scope. Exact national ISM semantics would require separately licensed long-history data.

The official source workbook contains 301 monthly values from July 2001 through July 2026. The point-in-time-usable cache contains 163 consecutive observations from January 2013 through July 2026 with official release dates from January 24, 2013 through July 23, 2026. Exact historical times are unavailable, so each release becomes model-usable on the next US equity trading session. The workbook is revised history because seasonal factors are recalculated; it is not an initial-release value-vintage archive.

Across the 64 trustworthy DBnomics/ISM observations from May 2020 through August 2025, the regional PMI-style level has 0.936 Pearson level correlation with national ISM, a 2.097-point mean absolute level difference, and the same side of 50 in 57 months. This is validation only; the rejected DBnomics cache is not a model input.

The Kansas City Fed permits reproduction without quantity limit when not for private gain. Local attributed student research caching and training are accepted for this MVP; raw data remains Git-ignored, redistribution is not approved by this decision, and production/private-gain use requires a new review.

The preserved DBnomics `ISM/pmi/pm` file is `REJECTED_PRIMARY_CANDIDATE`: it has only 64 trustworthy observations through August 2025; September-December 2025 contain `11.1`, `10.0`, `10.0`, and `10.3`; structured historical release dates are absent; and the mirror does not resolve underlying ISM bulk-use restrictions.

---

## 8. VIX source correction

The VIX inventory is independent of PMI licensing.

| Variable | Preferred | Backup | Frequency | Unit | Transformation | Status |
|---|---|---|---|---|---|---|
| `vix_level_points` | FRED `VIXCLS` | Yahoo `^VIX` | Daily close | VIX points | Preserve close and provenance | Accepted comparison candidates |
| `vix_change_points` | Selected VIX series | Alternate candidate | Fixed internal window, not selected | VIX points | `VIX_end - VIX_start` | Step 4 window decision |

FRED identifies `VIXCLS` as the CBOE Volatility Index, daily close, and index points. CBOE/FRED citation and licensing metadata must be retained in provenance. [FRED VIXCLS](https://fred.stlouisfed.org/series/VIXCLS)

The frontend's current percentage display is inconsistent with the canonical `vix_change_points` unit and must not determine the historical source definition.

---

## 9. Complete required-input inventory

| Input | Meaning and unit | Preferred source | Backup/source comparison | Frequency and timing | Transformation | Coverage, revision, and PIT status |
|---|---|---|---|---|---|---|
| `fed_funds_rate_change_bps` | Policy target change, basis points | FRED `DFEDTARL`, `DFEDTARU`, `DFEDTAR` | FRED `EFFR` for state/validation | Event/effective-date | Target midpoint or single-target difference times 100 | Long official history likely; target effective date and FOMC release timestamp must be aligned |
| `cpi_surprise_pct` | Actual CPI inflation minus pre-release expectation, percentage points | BLS/FRED CPI candidate matched to expectation | Cleveland Fed nowcast, SPF, licensed consensus | Monthly; normally 8:30 a.m. ET release | Actual minus latest strictly pre-release expectation | Actual history likely long; expectation archive and vintage coverage unverified |
| `oil_price_change_pct` | Oil movement over a fixed window, percent | FRED `DCOILWTICO` | Yahoo `CL=F`, `USO` | Daily/market window | `100 * (end/start - 1)` | Spot has no futures roll but publication/calendar gaps; exact window deferred |
| `gdp_growth_surprise_pct` | GDP growth actual minus pre-release expectation, percentage points | BEA advance real GDP growth/FRED `GDPC1` | Final pre-release Atlanta GDPNow; SPF comparison | Quarterly; normally 8:30 a.m. ET | Same SA annualized QoQ convention for both values | Revisions are material; exact GDPNow history and release matching unverified |
| `unemployment_change_pct` | Released unemployment-rate change, percentage points | BLS Employment Situation/FRED `UNRATE` | BLS source series directly | Monthly; normally 8:30 a.m. ET | New release minus prior release | Actual history likely long; release vintage and timestamp must be stored |
| `pmi_change_points` | Manufacturing PMI-style change, index points | Kansas City Fed Tenth District manufacturing composite mapped to a 50-neutral PMI-style level | Exact national ISM history only if separately licensed; DBnomics candidate rejected | Monthly; official release date plus next trading session because historical time is unavailable | Later difference of `50 + 0.5 × KC composite`; internal proxy name required | Accepted for the free student MVP with regional-scope and revised-history limitations; 2013-01 through 2026-07 point-in-time-usable |
| `dxy_change_pct` | True DXY movement over a fixed window, percent | Yahoo `DX-Y.NYB` or approved DXY source | FRED `DTWEXBGS`, explicitly named broad dollar index | Daily/market window | `100 * (end/start - 1)` | FRED backup is not DXY; source continuity and terms unverified |
| `vix_change_points` | VIX movement over a fixed window, points | FRED `VIXCLS` | Yahoo `^VIX` | Daily close/market window | `end - start` | Official daily close candidate; exact window and cross-source alignment deferred |

---

## 10. Target-source inventory

| Target | Provisional proxy | Current value type | Prediction formula and unit | Neutral interpretation | Source decision/status |
|---|---|---|---|---|---|
| `sp500` | `SPY` or `^GSPC` | ETF or index level | `100 * (future/current - 1)`, % | No S&P 500 movement | Provisional comparison; index-versus-ETF and price-adjustment choice remain open |
| `nasdaq` | `^IXIC`; QQQ comparison | Nasdaq Composite index or Nasdaq-100 ETF level | `100 * (future/current - 1)`, % | No Nasdaq movement | Prefer `^IXIC` for semantic Nasdaq Composite fidelity; QQQ is Nasdaq-100 |
| `ten_year_yield_bps` | FRED `DGS10` | Yield percentage-point level | `100 * (future - current)`, bps | No yield change | Prefer `DGS10`; compare Yahoo `^TNX` for timing and units |
| `dxy` | True DXY source, currently Yahoo `DX-Y.NYB` | Dollar-index level | `100 * (future/current - 1)`, % | No DXY movement | Do not substitute `DTWEXBGS` without renaming the target |
| `gold` | `GLD`, `GC=F`, or approved spot gold | ETF, futures, or spot price | `100 * (future/current - 1)`, % | No gold movement | Provisional comparison; ETF adjustment and futures-roll policy remain open |
| `oil` | FRED `DCOILWTICO` | WTI spot price | `100 * (future/current - 1)`, % | No oil movement | Provisional preference for spot; compare `CL=F` and `USO` |
| `technology` | `XLK` | Sector ETF price level | `100 * (future/current - 1)`, % | No Technology movement | Direct backend support exists; adjustment and common-history verification pending |
| `energy` | `XLE` | Sector ETF price level | `100 * (future/current - 1)`, % | No Energy movement | Direct backend support exists; verification pending |
| `financials` | `XLF` | Sector ETF price level | `100 * (future/current - 1)`, % | No Financials movement | Direct backend support exists; verification pending |
| `utilities` | `XLU` | Sector ETF price level | `100 * (future/current - 1)`, % | No Utilities movement | Direct `XLU` support is missing and must be verified/added |
| `healthcare` | `XLV` | Sector ETF price level | `100 * (future/current - 1)`, % | No Healthcare movement | Direct backend support exists under Health Care naming; verification pending |
| `consumer_discretionary` | `XLY` | Sector ETF price level | `100 * (future/current - 1)`, % | No Consumer Discretionary movement | Direct backend support exists; verification pending |

### Target comparison notes

- `SPY` is an investable ETF that seeks to track the S&P 500 price and yield performance; `^GSPC` is the direct index candidate. [SPY official page](https://www.ssga.com/us/en/individual/etfs/state-street-spdr-sp-500-etf-trust-spy)
- The Nasdaq Composite covers the broad Nasdaq-listed universe, while QQQ tracks the Nasdaq-100. They are not interchangeable. [Nasdaq Composite](https://indexes.nasdaq.com/Index/Overview/COMP) · [QQQ official page](https://www.invesco.com/qqq-etf/en/about.html)
- FRED `DGS10` is a daily percentage Treasury-yield series. `^TNX` is the existing Yahoo comparison candidate. [DGS10](https://fred.stlouisfed.org/series/DGS10/)
- FRED `DCOILWTICO` is daily WTI spot from EIA and avoids futures-roll construction, although release timing and missing dates must be handled. [WTI spot](https://fred.stlouisfed.org/series/DCOILWTICO)
- FRED `DTWEXBGS` is a broad trade-weighted dollar index, not DXY. [Broad dollar index](https://fred.stlouisfed.org/series/DTWEXBGS)

### Raw versus adjusted prices

Step 3 must explicitly collect and compare raw and adjusted variants where available.

Raw close means price movement and index-like interpretation. Adjusted close means split/dividend-adjusted or total-return-like interpretation.

The model must not mix adjusted ETF prices with raw index or futures prices without documenting the target meaning. Futures also require an explicit continuous-contract and roll policy.

---

## 11. Candidate market-state feature inventory

These variables describe the latest valid pre-shock market and macro state. They are not user-entered shocks.

| Feature family | Canonical variables | Source/unit | Transformation | Repository support and limitation | Status |
|---|---|---|---|---|---|
| Current broad-market levels | `sp500_level`, `nasdaq_level` | Selected S&P/Nasdaq proxy; index or price level | Latest valid completed close | Yahoo `^GSPC` and `^IXIC` exist; ETF alternatives do not exist in the historical provider | Provisional |
| Commodity levels | `gold_level`, `oil_level` | Selected gold and WTI source | Latest valid source value | Yahoo `GC=F` and `CL=F` exist; spot alternatives are not in backend | Provisional |
| Dollar and volatility levels | `dxy_level`, `vix_level_points` | DXY index and VIX points | Latest valid close | Yahoo DXY and `^VIX` exist; FRED validation candidates are not in backend | Provisional |
| Sector levels | `technology_level`, `energy_level`, `financials_level`, `utilities_level`, `healthcare_level`, `consumer_discretionary_level` | `XLK`, `XLE`, `XLF`, `XLU`, `XLV`, `XLY` | Latest valid close | All except `XLU` are present in the backend provider/schema | `XLU` blocked pending source addition |
| Policy state | `fed_funds_target_midpoint_pct`, `effective_fed_funds_rate_pct` | FRED target series and `EFFR` | Latest value available at `t` | Backend stores monthly `FEDFUNDS`, not target range or EFFR | Accepted candidate |
| CPI state | `cpi_level`, `cpi_mom_pct`, `cpi_yoy_pct` | Matched CPI raw series | Level and lagged percentage changes | Backend stores `CPIAUCSL` level and forward-fills | Vintage/release handling blocked |
| Labor state | `unemployment_rate_pct` | BLS/FRED `UNRATE` | Latest released rate | Backend stores level without release metadata | Vintage/release handling blocked |
| GDP state | `real_gdp_level`, `real_gdp_qoq_annualized_pct` | BEA/FRED `GDPC1` | Lagged growth using released vintage | Backend stores level without release metadata | Vintage/release handling blocked |
| Yield state | `two_year_yield_pct`, `ten_year_yield_pct` | FRED `DGS2`, `DGS10` | Latest valid daily yield | Backend supports both through FRED and `^TNX` through Yahoo | Accepted comparison |
| Yield curve | `yield_curve_slope_10y_2y_pct_points` | `DGS10 - DGS2`; validation `T10Y2Y` | Percentage-point difference | Backend calculates this metric | Accepted candidate |
| Real-rate proxy | `real_rate_proxy_pct_points` | Policy rate minus CPI YoY | Explicitly named proxy | Backend calculates Fed Funds minus CPI YoY without vintage timing | Provisional proxy |
| Recent returns | `*_1d_return_pct`, `*_5d_return_pct`, `*_21d_return_pct` | Market source for S&P, Nasdaq, gold, oil, DXY, sectors | Close-to-close return | Backend calculates only selected 5-day returns | Candidate; exact windows to verify |
| Rolling volatility | `*_20d_volatility_pct` | Market sources | Annualized standard deviation of daily log returns | Backend calculates for selected assets | Candidate |
| Drawdown | `*_drawdown_from_peak_pct` | Market sources | Current value relative to trailing peak | Backend uses available history and labels the window `1y` | Exact lookback required |
| Moving-average distance | `*_ma_distance_50d`, `*_ma_distance_200d` | Market sources | `(current - MA) / MA` | Backend calculates for selected assets | Candidate |
| Rolling z-score | `*_price_zscore_252d` | Market sources | Trailing mean and standard deviation | Backend falls back to available history after 60 observations | Must not silently label short history as 252-day |
| Release flag | `cpi_release_flag`, `gdp_release_flag`, `unemployment_release_flag`, `pmi_release_flag`, `fed_decision_flag` | Official release calendars | Boolean | Not present in repository | Required candidate |
| Release recency | `days_since_cpi_release`, `days_since_gdp_release`, `days_since_unemployment_release`, `days_since_pmi_release`, `days_since_fed_decision` | Release timestamps | Calendar/trading-day distance | Not present in repository | Required candidate |
| Source quality | `*_source_quality`, `*_vintage_quality`, `*_proxy_quality` | Dataset provenance | Categorical quality/version metadata | Not present in repository | Required metadata |

---

## 12. Point-in-time and release-timing rules

The current backend is not point-in-time safe for Scenario Playground training because it stores observation dates but not publication timestamps or historical vintage values.

Required rules:

- Use only the value released by time `t`.
- Store observation date, release timestamp, vintage date, and source version separately.
- Never forward-fill a macro value into dates before its release.
- Never use a revised value as if it were known on the historical row date.
- Never use a post-release market close as the pre-shock state when the event occurred earlier that day.
- Never calculate rolling features with observations after `t`.
- Preserve original-release and revised values where available.

FRED supports vintage-date and initial-release-only queries, but the repository does not currently use them. [FRED vintage documentation](https://fred.stlouisfed.org/docs/api/fred/series_observations.html)

Expected normal release timing:

| Source | Typical timing |
|---|---|
| BLS CPI | 8:30 a.m. ET |
| BLS Employment Situation | 8:30 a.m. ET |
| BEA GDP | 8:30 a.m. ET |
| FOMC statement | 2:00 p.m. ET |
| Selected Kansas City Fed manufacturing proxy | Official archive release date; exact historical time unavailable, so next trading session |
| ISM Manufacturing/Services (not selected) | Candidate/reference timing only; no bulk history is used |

Release calendars can change, so the actual release timestamp must be stored with each event.

---

## 13. Shock-window comparison

The exact windows for oil, DXY, and VIX remain deferred to Step 4.

| Window | Definition | Benefit | Limitation |
|---|---|---|---|
| One trading day | `100 * (P_t / P_t-1 - 1)`; VIX uses point difference | Dense and simple | May not correspond to a specific event |
| Event window | Pre-event close to same-day or next-day close | Closest to release interpretation | Requires accurate timestamps and event matching |
| Five trading days | `100 * (P_t / P_t-5 - 1)`; VIX point difference | Less sensitive to intraday timing | Blurs immediate response and overlaps more |
| Weekly | Fixed weekly observations | Reduces calendar noise | Smaller sample and less intuitive user interpretation |

The frontend must provide magnitude only. The selected internal window must be fixed for both historical training and live scenario inference.

---

## 14. Coverage and common-history assessment

No actual downloaded row counts, missing-data rates, or exact common date ranges were measured in Step 2.

Confirmed repository behavior:

- Yahoo requests only the latest 365 calendar days.
- FRED requests full available series but does not preserve historical vintages.
- Yahoo's complete-row `dropna()` can reduce usable history to the intersection of every requested ticker.
- SQLite row counts and date ranges are not evidence of current Supabase production coverage.

Reasonable estimates pending Step 3 measurement:

| Dataset choice | Likely limiter | Expected implication |
|---|---|---|
| Broad indexes plus FRED macro/yields | Release and vintage alignment | Potentially several decades |
| All required sector ETFs | ETF inception and common trading history | Likely late 1990s onward |
| GLD included | GLD inception in 2004 | Likely 2004 onward |
| USO included | USO inception in 2006 | Likely 2006 onward |
| True CPI/GDP surprises | Expectation and nowcast archives | Shorter than official actual history |
| Kansas City Fed manufacturing proxy | Official workbook and release-date archive | 2013-01 through 2026-07 usable; does not limit the August 2013 shared start |
| DXY through Yahoo | Vendor coverage and terms | Must be measured |
| All targets plus true surprises and timestamps | Intersection of all source families | Likely strictest practical period |

The final training period must not be selected until Step 3 measures coverage and Step 4 tests sampling alternatives.

---

## 15. Raw and interim storage design

Raw files must remain separated by source and series:

data/scenario_playground/
├── raw/
│   ├── yahoo/
│   ├── fred/
│   ├── alfred/
│   ├── bls/
│   ├── bea/
│   ├── federal_reserve/
│   ├── ism/
│   ├── cleveland_fed/
│   └── atlanta_fed/
├── interim/
│   └── source_comparisons/
└── processed/

Files such as the following belong under `interim/source_comparisons/`, not under `raw/`:

data/scenario_playground/interim/source_comparisons/
  sp500_candidates.parquet
  yield_candidates.parquet
  gold_candidates.parquet
  oil_candidates.parquet

The raw layer must preserve, where practical:

- Original values.
- Original source schema.
- Source-specific dates.
- Retrieval metadata.
- Series identifiers.
- Request parameters.
- Vintage parameters.
- Release metadata.

Recommended formats:

- Parquet for raw time series and interim comparisons.
- JSON for provenance, coverage, request parameters, vintages, and source quality.
- CSV only for small source tables or manually reviewed event data.
- Markdown for source and modeling documentation.

No data directories are created by this Step 2 amendment.

---

## 16. Step 3 download inventory

The provisional FRED/ALFRED list is:

data sources:
  DFEDTAR
  DFEDTARL
  DFEDTARU
  EFFR
  CPIAUCSL
  CPIAUCNS
  CPILFESL
  UNRATE
  GDPC1
  DGS2
  DGS10
  VIXCLS
  DCOILWTICO
  DTWEXBGS
  T10Y2Y

These are not all final model features. Some are source-comparison or validation candidates.

Step 3 must verify the official identifier for a not-seasonally-adjusted core CPI series before adding it to the download set.

Step 3 must also collect or verify:

- FOMC release timestamps and target-rate effective dates.
- CPI release dates, definitions, and historical expectation snapshots.
- BEA advance GDP release timestamps and final pre-release GDPNow snapshots.
- Employment Situation release timestamps.
- Selected Kansas City Fed proxy values, official release dates, next-session timing, and noncommercial-use terms.
- Candidate market target histories and adjustment conventions.

---

## 17. Sources blocked or still provisional

### Blocked pending access verification

- Exact national ISM Manufacturing PMI bulk history remains unavailable without a separate licensing decision; it is not required by the accepted free MVP proxy path.
- Historical Cleveland Fed CPI nowcast snapshots and reproducible download method.
- Historical Atlanta Fed GDPNow snapshots and exact pre-release matching.
- Historical consensus expectations if a licensed provider is required.
- Not-seasonally-adjusted core CPI identifier and coverage.
- Direct `XLU` history through the selected market provider.
- Commercial or redistributable use of Yahoo/yfinance data.

### Provisional candidates

- `SPY` versus `^GSPC`.
- `^IXIC` versus `QQQ`.
- FRED `DGS10` versus Yahoo `^TNX`.
- `GLD` versus `GC=F` versus spot gold.
- WTI spot versus `CL=F` versus `USO`.
- Yahoo DXY versus another true DXY source.
- Raw versus adjusted ETF prices.
- Target-range midpoint versus alternative policy-rate representations for state features.

### Definitions deferred

- Exact CPI headline/core and month-over-month/year-over-year convention.
- Exact GDP growth convention after the provisional SA annualized QoQ test.
- Any future replacement of the accepted manufacturing-only regional proxy with separately licensed national ISM data.
- Exact oil, DXY, and VIX shock windows.
- Event-row sampling and non-event-row sampling.
- Final common training period.

---

## 18. Step 3 measurements required

Step 3 must measure rather than assume:

- Exact start and end dates for every candidate source.
- Row counts and missing-data rates.
- Duplicate dates and non-trading dates.
- ETF inception dates and continuity.
- `XLU` availability.
- Unit and timestamp alignment between `DGS10` and `^TNX`.
- DXY continuity and source definition.
- Gold and oil futures-roll behavior.
- Raw versus adjusted price differences.
- Historical coverage of CPI nowcasts and GDPNow.
- Whether expectation timestamps are strictly before official releases.
- Historical release timestamps and time zones.
- ALFRED initial-release and vintage availability.
- Kansas City proxy geographic/revision limitations and continued compliance with its noncommercial-use terms.
- Earliest common date across all required inputs and targets.

---

## 19. Decisions deferred to Step 4 sampling tests

Step 4 must test, rather than assume:

- One-day versus event-window versus multi-day oil shocks.
- One-day versus event-window versus multi-day DXY shocks.
- One-day versus event-window versus multi-day VIX shocks.
- Event-only rows versus regular daily rows.
- Shock-window overlap and dependence.
- State-before-release close selection.
- The effect of missing or delayed macro releases.
- Whether recent-history or recency-weighted training is preferable.

---

## 20. Final handoff

The finalized Step 2 documentation file is:

markdown_files/scenario_modeling/02_DATA_SOURCE_INVENTORY.md

Accepted source decisions:

- Separate notebook dataset outside Supabase.
- FRED/ALFRED for official macro and yield candidates.
- `DFEDTARL`, `DFEDTARU`, and `DFEDTAR` for the provisional historical policy target.
- `EFFR` for policy state or validation.
- Kansas City Fed Tenth District manufacturing composite as the explicitly named, 50-neutral PMI-style historical proxy; national ISM remains the conceptual reference only.
- FRED `VIXCLS` as the preferred VIX candidate, with `^VIX` as backup.
- Source-separated raw storage and comparison-only interim storage.

Sources still blocked or provisional:

- Historical expectations and nowcast archives.
- Exact national ISM bulk history only if a future product requires national rather than accepted regional-proxy semantics.
- Not-seasonally-adjusted core CPI identifier.
- ETF/index/futures/spot target choices.
- DXY source continuity.
- Raw versus adjusted price policy.
- `XLU` direct history.

Step 3 must verify coverage, access, licensing, release timing, vintages, identifiers, calendars, and missing periods. Step 4 must test shock-window and sampling alternatives.

No data directories, notebooks, Python modules, backend/frontend files, database tables, or unrelated files were changed by this amendment.
