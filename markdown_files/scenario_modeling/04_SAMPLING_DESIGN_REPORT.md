# Scenario Playground Step 4 Sampling Design Report

## Status and scope

Step 4 constructs and compares two row indexes from the accepted Step 3 caches:

- `WEEKLY`: one row at the final observed equity session of each ISO week.
- `EVENT_AWARE`: one row for each unique equity session when one or more selected economic events become usable.

The selected MVP design is:

```text
EVENT_AWARE
```

This step defines dates, event identities, target-horizon eligibility, and point-in-time boundaries. It does **not** calculate shock magnitudes, state features, target values, final splits, model matrices, predictions, or models.

## Why sampling is needed

The source data mixes daily markets, monthly releases, quarterly GDP releases, and irregular FOMC decisions. Sampling determines what one training row means and prevents calendar dates, release dates, and revised observation periods from being treated as interchangeable.

A valid row needs:

1. a completed pre-shock market close for state;
2. one or more shocks that become known after that state;
3. a deterministic date when those shocks are usable;
4. the exact 21st later trading session for all 12 targets; and
5. no future or revised information in the state.

## Cached evidence used

No source was refreshed. Step 4 reads only these accepted local paths:

- Canonical calendar and selected Yahoo targets: `data/scenario_playground/raw/yahoo/`
- Selected yield target: `data/scenario_playground/raw/fred/DGS10.observations.json`
- FOMC events: `data/scenario_playground/raw/federal_reserve/fomc_decisions.json`
- CPI first appearances: `data/scenario_playground/raw/alfred/CPIAUCSL.vintage_*.json`
- Unemployment first appearances: `data/scenario_playground/raw/alfred/UNRATE.vintage_*.json`
- GDP advance dates: `data/scenario_playground/raw/atlanta_fed/GDPTrackingModelDataAndForecasts.xlsx`, worksheet `TrackRecord`
- Manufacturing-proxy releases: `data/scenario_playground/raw/kansas_city_fed/manufacturing_pmi_proxy_events.json`

The shared Step 3 period used for candidate construction is:

```text
2013-08-01 through 2026-07-16
```

The start is the accepted Cleveland Fed CPI-nowcast coverage boundary. The end is the last usable selected `DGS10` target observation.

## Canonical market calendar

The canonical calendar is the 3,258 non-missing `Close` dates in the selected Yahoo `^GSPC` cache from 2013-08-01 through 2026-07-16.

`^GSPC` is appropriate because:

- it is the selected S&P 500 target;
- it directly records observed US equity sessions;
- it includes actual holiday and special-closure behavior;
- it avoids approximating trading days with weekdays; and
- it supports exact positional counting to the 21st future session.

Calendar rules:

- Saturdays and Sundays are never model sessions.
- A weekly anchor is the last observed `^GSPC` session in the ISO week.
- A release with unknown time becomes usable on the next observed session.
- A known pre-close FOMC release can become usable that session.
- A known at/after-close or non-session FOMC release becomes usable on the next observed session.
- A target endpoint is never shifted to another date. If the selected target value is absent at the exact endpoint, the row is ineligible.
- A row is also ineligible when its exact 21st future session falls after the shared selected-target coverage.

The existing deterministic holiday utility and observed `^GSPC` dates disagree once in this period: `2021-12-31` is present in `^GSPC` but the utility classifies it as closed. The observed selected-market series is authoritative, so the real session is retained. No large calendar dependency was added.

## Exact meaning of one model row

### Conceptual timestamp `t`

The conceptual `t` is the completed canonical market close immediately before the modeled shock or shock interval.

```text
state_cutoff_date = conceptual pre-shock t
target_baseline_date = state_cutoff_date
target_end_date = the 21st canonical session after t
```

The target starts at the **pre-shock close**, not a post-event close. This preserves the Step 1 contract: the future cumulative movement can include the user-imposed initial shock and the subsequent movement through trading session 21.

### Stored `sample_date`

`sample_date` is an operational row key:

- Weekly: the interval's final observed session.
- Event-aware: the event `model_availability_date`.

It is the date by which all shocks attached to the row are usable. It is not a replacement for the conceptual pre-shock `t`; the separate `state_cutoff_date` and `target_baseline_date` retain that meaning.

### Release-time rules

| Release condition | State cutoff | Model availability | Market-close input availability |
|---|---|---|---|
| Pre-market, exact time | Prior completed close | Release session | Release-session close is usable only after it completes |
| During market, exact time | Prior completed close | Release session | Release-session close is usable only after it completes |
| At/after market close, exact time | Release-day completed close when it is a session | Next observed session | Next-session close is usable only after it completes |
| Exact time unavailable | Prior completed close | Strictly next observed session | No close after `sample_date` may be used |
| Weekend or market holiday | Prior observed close | Next observed session | The next observed close is the first reaction close |

Only FOMC records currently have exact cached historical times. CPI, unemployment, GDP, and the Kansas City manufacturing proxy use conservative next-session availability.

## Selected target fields and coverage

Step 4 did not change the Step 3 target decisions.

| Target | Selected source | Field checked | Usable observations | First | Last |
|---|---|---|---:|---|---|
| S&P 500 | Yahoo `^GSPC` | `Close` | 19,256 | 1950-01-03 | 2026-07-17 |
| Nasdaq | Yahoo `^IXIC` | `Close` | 13,978 | 1971-02-05 | 2026-07-17 |
| 10-year yield | FRED `DGS10` | `value` | 16,119 | 1962-01-02 | 2026-07-16 |
| DXY | Yahoo `DX-Y.NYB` | `Close` | 14,102 | 1971-01-04 | 2026-07-17 |
| Gold | Yahoo `GLD` | `Adj Close` | 5,448 | 2004-11-18 | 2026-07-17 |
| Oil | Yahoo `CL=F` | `Close` | 6,502 | 2000-08-23 | 2026-07-17 |
| Technology | Yahoo `XLK` | `Adj Close` | 6,933 | 1998-12-22 | 2026-07-17 |
| Energy | Yahoo `XLE` | `Adj Close` | 6,933 | 1998-12-22 | 2026-07-17 |
| Financials | Yahoo `XLF` | `Adj Close` | 6,933 | 1998-12-22 | 2026-07-17 |
| Utilities | Yahoo `XLU` | `Adj Close` | 6,933 | 1998-12-22 | 2026-07-17 |
| Healthcare | Yahoo `XLV` | `Adj Close` | 6,933 | 1998-12-22 | 2026-07-17 |
| Consumer Discretionary | Yahoo `XLY` | `Adj Close` | 6,933 | 1998-12-22 | 2026-07-17 |

Direct indexes, DXY, futures, and the yield use their native close/value. ETFs use adjusted close so later return targets account for distributions. Step 4 checks endpoint presence only; it calculates no target.

## Weekly candidate

### Definition

Each ISO week contributes its final observed `^GSPC` session, normally Friday.

For each weekly row:

```text
sample_date = final observed session of the week
state_cutoff_date = prior weekly anchor
shock interval = (prior anchor, current anchor]
target_baseline_date = prior weekly anchor
target_end_date = 21st canonical session after the prior anchor
```

The first row uses 2013-08-01 as a short initial state cutoff because there is no earlier anchor inside the accepted shared period. It is marked `initial_partial_interval`.

The index stores all event IDs, types, observation periods, original release dates, availability dates, and sources that became usable in the interval. It calculates no event shock.

### Holiday anchors

- Raw weekly anchors: **677**
- Anchors not on Friday: **25**
- Friday market-holiday shifts: **24**
- Final partial-week anchor at the shared end: **1**

### Exact counts

- Raw anchors: **677**
- Label-eligible anchors: **654**
- Removed by the 21-session/shared-coverage boundary: **4**
- Removed because a selected target is absent at the exact baseline or endpoint: **19**
- Weeks containing at least one event: **449**
- Weeks containing multiple events: **156**
- First sample: **2013-08-02**
- Last raw sample: **2026-07-16**
- Last label-eligible sample: **2026-06-18**

Most endpoint-missing removals occur because `DGS10` is absent on a federal holiday when the equity market is open. Two weekly endpoints also lack DXY and oil values. These rows are excluded rather than filled or moved.

### Multiple same-type events

All identities remain in the weekly row. If this rejected design were used later:

- Fed decisions would be summed as signed basis-point changes.
- CPI and GDP surprises would be summed as signed surprises within the interval.
- Unemployment and PMI would use the newest release minus the last release before the interval, which is the net interval change.
- Oil, DXY, and VIX would use one endpoint change across the complete weekly shock window.

These extra aggregation rules are one reason not to select weekly sampling.

## Event-aware candidate

### Event-date construction

| Event type | Structured timing source | Date meaning |
|---|---|---|
| FOMC | `fomc_decisions.json` | Official announcement date/time and cached model-availability rule |
| CPI | Earliest non-missing `CPIAUCSL` ALFRED vintage column | Initial point-in-time appearance date, not observation month |
| Unemployment | Earliest non-missing `UNRATE` ALFRED vintage column | Initial point-in-time appearance date, not observation month |
| GDP | Atlanta Fed `TrackRecord` release-date column | BEA advance-estimate release date |
| PMI proxy | Kansas City Fed structured release record | Official archive/release-page date |

### Definition

Each unique observed `model_availability_date` contributes one row.

```text
sample_date = unique model-availability session
state_cutoff_date = pre-event completed close
target_baseline_date = state_cutoff_date
target_end_date = 21st canonical session after state cutoff
```

When several events share `sample_date`, the row uses the earliest pre-event cutoff among them. This is conservative: state is guaranteed to precede every attached event.

### Exact event counts

| Type | Events |
|---|---:|
| FOMC | 105 |
| CPI | 155 |
| Unemployment | 155 |
| GDP advance | 51 |
| Manufacturing proxy | 155 |
| **Total before same-day deduplication** | **621** |

Raw structured events by model-availability year:

| Year | FOMC | CPI | Unemployment | GDP | PMI proxy | Total |
|---|---:|---:|---:|---:|---:|---:|
| 2013 | 3 | 5 | 5 | 1 | 5 | 19 |
| 2014 | 8 | 12 | 12 | 4 | 12 | 48 |
| 2015 | 8 | 12 | 12 | 4 | 12 | 48 |
| 2016 | 8 | 12 | 12 | 4 | 12 | 48 |
| 2017 | 8 | 12 | 12 | 4 | 12 | 48 |
| 2018 | 8 | 12 | 12 | 4 | 12 | 48 |
| 2019 | 9 | 12 | 12 | 4 | 12 | 49 |
| 2020 | 9 | 12 | 12 | 4 | 12 | 49 |
| 2021 | 8 | 12 | 12 | 4 | 12 | 48 |
| 2022 | 8 | 12 | 12 | 4 | 12 | 48 |
| 2023 | 8 | 12 | 12 | 4 | 12 | 48 |
| 2024 | 8 | 12 | 12 | 4 | 12 | 48 |
| 2025 | 8 | 11 | 11 | 4 | 12 | 46 |
| 2026 | 4 | 7 | 7 | 2 | 6 | 26 |

### Same-day collisions

- Unique event-aware dates: **591**
- Dates containing multiple events: **30**
- Events on those dates: **60**
- Duplicate candidate rows removed: **30**
- Event identities lost: **0**

Collision combinations:

| Event-type combination | Dates |
|---|---:|
| `cpi+pmi` | 2 |
| `fomc+cpi` | 8 |
| `fomc+gdp` | 3 |
| `gdp+pmi` | 16 |
| `unemployment+pmi` | 1 |

Event order is deterministic: FOMC, CPI, unemployment, GDP, then PMI; ties use release date and stable event ID. Every ID is retained in the row's JSON-list columns.

### Exact candidate counts

- Raw unique event-aware rows: **591**
- Label-eligible rows: **585**
- Removed by the 21-session/shared-coverage boundary: **4**
- Removed because a selected target is absent at the exact baseline or endpoint: **2**
- First sample: **2013-08-05**
- Last raw sample: **2026-07-15**
- Last label-eligible sample: **2026-06-11**

The two endpoint-presence removals have pre-shock baselines on 2020-10-12 and 2020-11-11, when the equity market traded but `DGS10` was not published. No value is filled from another date.

## Event representation

Both candidate indexes represent every event inside the shared period.

| Type | Available | Weekly represented | Weekly combined with another event | Event-aware represented | Event-aware combined with another event | Omitted | Shifted later |
|---|---:|---:|---:|---:|---:|---:|---:|
| FOMC | 105 | 105 | 73 | 105 | 11 | 0 | 20 |
| CPI | 155 | 155 | 99 | 155 | 10 | 0 | 155 |
| Unemployment | 155 | 155 | 73 | 155 | 1 | 0 | 155 |
| GDP | 51 | 51 | 43 | 51 | 19 | 0 | 51 |
| PMI proxy | 155 | 155 | 40 | 155 | 19 | 0 | 155 |

Of the 20 FOMC later-session shifts, 19 are conservative shifts caused by unavailable exact historical announcement times. All CPI, unemployment, GDP, and PMI records are conservatively shifted because exact times are absent from the accepted structured caches.

## Market-based shock-window boundary

Oil, DXY, and VIX do not have scheduled macro release events. Step 4 stores a deterministic future feature boundary without calculating the feature:

```text
shock_window_start_date = state_cutoff_date
shock_window_end_date = sample_date
```

Later feature construction must calculate:

- `oil_price_change_pct`: percent change in FRED `DCOILWTICO`
- `dxy_change_pct`: percent change in Yahoo `DX-Y.NYB` `Close`
- `vix_change_points`: point change in FRED `VIXCLS`

Only completed observations at or before `sample_date` may be used. No future close is permitted. Unknown-time releases can produce a conservative multi-session reaction window because their availability moves to the next session. The window remains compatible with user inputs because its outputs retain percent, percent, and VIX-point units respectively.

Step 4 stores only these boundaries; it does not calculate the three changes.

## Counts by year

| Year | Weekly raw | Weekly eligible | Event-aware raw | Event-aware eligible |
|---|---:|---:|---:|---:|
| 2013 | 22 | 20 | 18 | 18 |
| 2014 | 52 | 51 | 44 | 44 |
| 2015 | 53 | 52 | 46 | 46 |
| 2016 | 52 | 50 | 46 | 46 |
| 2017 | 52 | 51 | 48 | 48 |
| 2018 | 52 | 50 | 47 | 47 |
| 2019 | 52 | 50 | 47 | 47 |
| 2020 | 53 | 52 | 49 | 47 |
| 2021 | 52 | 51 | 46 | 46 |
| 2022 | 52 | 50 | 43 | 43 |
| 2023 | 52 | 51 | 42 | 42 |
| 2024 | 52 | 50 | 45 | 45 |
| 2025 | 52 | 51 | 44 | 44 |
| 2026 | 29 | 25 | 26 | 22 |

The period includes pre-pandemic expansion, the 2020 shock and emergency decisions, the inflation/tightening cycle, and later years. Event-aware rows are not concentrated in only one year; eligible annual counts remain between 42 and 48 for every full calendar year.

## Planning-only chronological counts

For a common comparison, Step 4 divides canonical calendar time at 60% and 80%. These are **not final splits** and have no purge or embargo.

| Planning period | Calendar dates | Weekly eligible rows | Event-aware eligible rows | Raw events before deduplication |
|---|---|---:|---:|---:|
| Train estimate | 2013-08-01 to 2021-05-05 | 393 | 359 | 374 |
| Validation estimate | 2021-05-06 to 2023-12-06 | 131 | 112 | 123 |
| Test estimate | 2023-12-07 to 2026-07-16 | 130 | 114 | 124 |

Raw event composition in those planning periods:

| Planning period | FOMC | CPI | Unemployment | GDP | PMI proxy | Total |
|---|---:|---:|---:|---:|---:|---:|
| Train estimate | 64 | 93 | 93 | 31 | 93 | 374 |
| Validation estimate | 20 | 31 | 31 | 10 | 31 | 123 |
| Test estimate | 21 | 31 | 31 | 10 | 31 | 124 |

Both candidates leave enough rows for chronological Ridge evaluation and a shallow, strongly regularized XGBoost benchmark. The counts do not imply that overlapping rows are independent.

## Row spacing

Metrics below use label-eligible rows.

| Metric | Weekly | Event-aware |
|---|---:|---:|
| Minimum gap, trading sessions | 4 | 1 |
| Median gap, trading sessions | 5 | 5 |
| Maximum gap, trading sessions | 10 | 20 |
| Minimum gap, calendar days | 6 | 1 |
| Median gap, calendar days | 7 | 8 |
| Maximum gap, calendar days | 14 | 28 |
| Consecutive-session row pairs | 0 | 39 |
| Gaps over 21 sessions | 0 | 0 |

Event-aware spacing distribution:

| Gap | Row pairs |
|---|---:|
| 1 session | 39 |
| 2-3 sessions | 156 |
| 4-5 sessions | 115 |
| 6-10 sessions | 230 |
| 11-20 sessions | 44 |
| Over 20 sessions | 0 |

The largest eligible event-aware gap is 20 sessions, between 2025-09-26 and 2025-10-24. There is no unsupported multi-month event drought.

## Target-window overlap

Two consecutive row windows are counted as overlapping when their pre-shock baselines are fewer than 21 canonical sessions apart.

| Metric | Weekly | Event-aware |
|---|---:|---:|
| Consecutive eligible row pairs | 653 | 584 |
| Pairs with overlapping windows | 653 | 584 |
| Approximate overlap proportion | 100% | 100% |
| Mean shared future sessions | 16.046 | 15.466 |
| Maximum shared future sessions | 20 | 21 |

Event-aware rows have slightly less average overlap but are still highly dependent. Later chronological splitting must:

- keep time order;
- prevent a training label from crossing into validation or test;
- apply a purge/embargo of at least 21 canonical sessions at split boundaries; and
- avoid interpreting the nominal row count as the number of independent observations.

No final split is created in Step 4.

## Point-in-time validation

All deterministic checks pass:

- every event model-availability date is an observed session;
- no event is used before its release date;
- every state cutoff precedes the modeled event, including same-day after-close handling;
- every next-session shift follows the observed calendar;
- no observation month is used as a release date;
- CPI and unemployment retain first-vintage ALFRED paths;
- GDP retains the advance-release TrackRecord path;
- every nonblank target endpoint is exactly 21 sessions after its baseline;
- no market close after `sample_date` is included in a shock window;
- event-aware dates are unique;
- selected dates are unique; and
- every selected row is label-eligible.

The raw normalized event table preserves release date, time, timezone, date kind, source, observation period, source availability, normalized availability, conservative-shift flags, and target-boundary checks so later steps can audit every row.

## Candidate comparison

### Ridge

Weekly:

- regular spacing and 654 eligible rows;
- simpler matrix alignment;
- but one coefficient can mix shocks from different dates inside the same interval;
- additional aggregation rules weaken direct scenario interpretation.

Event-aware:

- 585 eligible rows, including 359 in the planning-only train period;
- macro shocks retain direct release alignment;
- signed coefficients are easier to relate to user-entered shock fields;
- overlap and correlated shocks still require regularization and honest time splits.

Both are large enough for Ridge. Event-aware has the stronger contract fit.

### Shallow XGBoost

Weekly:

- enough rows for a small regularized benchmark;
- interval combinations create heterogeneous rows and can encourage unstable interaction splits.

Event-aware:

- enough rows for shallow, strongly regularized trees;
- event types are sparse, so depth, learning rate, and feature count must remain conservative;
- 30 collision rows preserve legitimate same-day interactions without duplicate labels.

Neither design supports a high-capacity model or random cross-validation.

### Speed and maintenance

Weekly is simpler to describe as a calendar frequency, but it requires later same-type and cross-type shock aggregation. Event-aware requires event normalization and collision handling, which Step 4 now implements once in a reusable module. With that code present, the ongoing maintenance difference is small.

## Final decision

```text
Selected: EVENT_AWARE
Rejected: WEEKLY
```

Reasons:

1. Event-aware loses only 69 currently eligible rows relative to weekly: 585 versus 654.
2. It directly matches the contract's pre-shock state plus event-shock semantics.
3. All 621 event identities are retained; 30 duplicate dates are removed with zero identity loss.
4. It avoids weekly aggregation of unrelated macro releases.
5. It supports all five event-based inputs and gives the three market-based inputs an explicit, no-future-close window.
6. It remains large enough for Ridge and a shallow, strongly regularized XGBoost benchmark.
7. The extra implementation complexity is now isolated in one module and stable normalized schemas.

Accepted limitations:

- row spacing is irregular;
- all consecutive target windows overlap;
- exact historical times are unavailable for four event families;
- the conservative next-session policy can widen the market-shock window;
- a regional Kansas City manufacturing proxy remains the accepted PMI representation;
- same-day events can carry multiple nonzero macro shocks; and
- later splits require a 21-session purge/embargo.

Consequences for later steps:

- State features must be joined as of `state_cutoff_date`.
- Macro shocks must be constructed only from the attached event IDs and point-in-time paths.
- Oil, DXY, and VIX changes must use the stored shock-window boundaries.
- All 12 targets must use `target_baseline_date` and the exact stored `target_end_date`.
- Chronological splitting must account for target overlap.
- Ridge should be the stable primary benchmark; XGBoost should remain shallow and strongly regularized.

The sampling choice does not need to be reopened unless a cached-data inconsistency is discovered.

## Reusable implementation

The focused implementation is:

```text
scenario_modeling/src/sampling.py
```

Responsibilities:

- load accepted cached target fields and event records;
- build the observed `^GSPC` calendar;
- derive CPI/unemployment first-appearance dates;
- parse GDP TrackRecord advance dates;
- normalize release availability;
- construct weekly and event-aware indexes;
- deduplicate same-day events without losing identity;
- check exact 21-session and selected-target eligibility;
- calculate comparison metadata; and
- persist deterministic CSV/JSON artifacts.

It contains no downloader, feature builder, target calculator, splitter, trainer, evaluator, API behavior, or database access.

## Persisted files

Generated interim data is stored under:

```text
data/scenario_playground/interim/sampling/
```

Files:

```text
normalized_event_records.csv
weekly_candidate_index.csv
event_aware_candidate_index.csv
selected_sampling_index.csv
sampling_comparison_summary.json
```

CSV is used because neither `pyarrow` nor `fastparquet` is installed in the existing environment. Dates use ISO `YYYY-MM-DD`, rows and JSON-list values have deterministic ordering, and the summary stores source/cache hashes plus output hashes.

### Normalized event schema

```text
event_id
event_type
event_source
event_source_file
event_observation_period
original_release_date
original_release_time
original_release_timezone
release_date_kind
source_model_availability_date
model_availability_date
availability_rule
next_session_shift_applied
conservative_next_session_shift
calendar_normalization_applied
state_cutoff_date
shock_window_start_date
shock_window_end_date
target_baseline_date
target_end_date
target_horizon_sessions
has_21_future_sessions
full_horizon_within_shared_coverage
all_target_values_available
missing_target_fields_json
label_eligible
ineligibility_reason
```

### Weekly schema

```text
sampling_design, sample_date, anchor_date, iso_year, iso_week,
state_cutoff_date, interval_start_exclusive, interval_end_inclusive,
shock_window_start_date, shock_window_end_date, target_baseline_date,
target_end_date, target_horizon_sessions, has_21_future_sessions,
full_horizon_within_shared_coverage, all_target_values_available,
missing_target_fields_json, label_eligible, ineligibility_reason,
anchor_is_shifted, anchor_shift_reason, initial_partial_interval,
event_count, event_types_json, event_ids_json,
event_observation_periods_json, event_release_dates_json,
event_availability_dates_json, event_sources_json,
multiple_same_type_event_types_json
```

### Event-aware and selected schema

```text
sampling_design, sample_date, model_availability_date, state_cutoff_date,
shock_window_start_date, shock_window_end_date, target_baseline_date,
target_end_date, target_horizon_sessions, has_21_future_sessions,
full_horizon_within_shared_coverage, all_target_values_available,
missing_target_fields_json, label_eligible, ineligibility_reason,
event_count, collision_count, duplicate_candidates_removed,
event_types_json, event_ids_json, event_observation_periods_json,
event_release_dates_json, event_release_times_json,
event_timezones_json, event_sources_json, event_release_date_kinds_json,
next_session_shift_applied, conservative_next_session_shift,
calendar_normalization_applied
```

`selected_sampling_index.csv` contains only the 585 label-eligible event-aware rows.

## Notebook and tests

Notebook:

```text
scenario_modeling/notebooks/02_sampling_design.ipynb
```

It builds both candidates through `scenario_modeling.src.sampling`, displays exact counts, event representation, collisions, spacing, overlap, validation checks, and the final decision. It uses only local caches.

Focused tests:

```text
scenario_modeling/tests/test_sampling.py
```

They cover weekly anchors, Good Friday handling, strict next-session shifts, unknown and after-close event timing, same-day deduplication, event preservation, deterministic ordering, exact 21-session eligibility, final-boundary removal, observation-month separation, selected-date uniqueness, stable schemas, offline cache construction, and repeatable serialization.

## Readiness

Step 4 is ready for the next numbered step because one final design is selected, its eligible index is persisted, timing is explicit, all event identities are retained, and later code can consume stable row boundaries without repeating the sampling decision.

Step 4 did not start feature construction, target calculation, model-matrix construction, final splitting, model training, Supabase work, backend work, frontend work, or any server.
