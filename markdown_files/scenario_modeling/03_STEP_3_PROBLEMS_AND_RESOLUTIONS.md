# Step 3 Problems and Resolutions

This document is a high-level record of the main problems encountered while completing the Scenario Playground raw-data download step, why they occurred, and how they were resolved.

The detailed source decisions, coverage, row counts, and metadata remain in `03_RAW_DATA_DOWNLOAD_REPORT.md`.

## Final outcome

Step 3 now has a practical raw-data path for all eight scenario inputs and all twelve output targets.

The two most significant blockers were:

1. Finding trustworthy Manufacturing PMI history with usable release dates.
2. Building a complete FOMC decision history that included unchanged-rate meetings.

Both were resolved for the student MVP. No features, targets, model matrix, or models were created during this work.

## Problems encountered

| Problem | Why it happened | Resolution | Remaining limitation |
|---|---|---|---|
| The original DBnomics/ISM PMI file was not trustworthy enough for training | It contained only 68 observations, only 64 were trustworthy, and September-December 2025 contained implausible values. It also had observation months rather than historical release dates. | The file was preserved unchanged as raw evidence but marked `REJECTED_PRIMARY_CANDIDATE`, so it cannot be selected by the primary modeling path. | It remains useful only as a limited comparison source through August 2025. |
| Direct official ISM access did not provide a practical free historical dataset | Direct requests to the ISM calendar and terms pages returned CAPTCHA pages. Complete bulk history, structured release dates, and acceptable reuse rights could not be established. | The direct requests are recorded as blocked probes rather than successful datasets. No required model input depends on them. | Exact national ISM history would require a separate access and licensing decision. |
| Replacing ISM risked silently changing the meaning of "PMI Change" | Many free manufacturing surveys are regional business indexes rather than national PMI data. Treating one as ISM would be misleading. | The Kansas City Fed Tenth District Manufacturing Survey was accepted as an explicitly named regional PMI-style proxy. It uses the same five equally weighted manufacturing components, and its raw diffusion value is mapped with `50 + 0.5 * composite`. | It is regional, not national. Modeling code must use `kc_fed_manufacturing_pmi_proxy_change_points` and disclose the proxy. |
| Kansas City Fed release dates were not in the historical workbook | The workbook contains observation values but does not contain the publication date for every observation. Observation month cannot safely be treated as release date. | Official yearly archive pages from 2013 onward were downloaded and deterministically parsed. Each observation now has a separate observation month, release date, and model-availability date. | Exact historical release times are unavailable, so every value uses the next trading session. |
| Some Kansas City archive pages initially appeared incomplete | Concurrent requests sometimes returned soft or incomplete pages, and the 2024 archive omitted the June release. | Incomplete years are retried sequentially. The June 2024 record is filled from its official dated release page, which is retained with the other source files. | Values come from the latest revised workbook rather than a full archive of initial-release value vintages. |
| Reading the official PMI workbook could have introduced another dependency | The source is XLSX, while the existing modeling requirements did not need a new spreadsheet package. | A deterministic standard-library ZIP/XML parser was added for the required worksheet content. No new package was added to `requirements.txt`. | The parser is intentionally source-specific and validates the XLSX signature and expected layout. |
| FRED policy-rate series did not provide a complete FOMC event calendar | `DFEDTAR`, `DFEDTARL`, and `DFEDTARU` show policy levels and changes but do not identify meetings where rates were held unchanged. | Official Federal Reserve historical meeting pages, the current calendar, and linked statements were downloaded and parsed independently of rate changes. FRED is then used only for numeric before/after policy levels. | Public decision coverage starts in 1994, when reliable public announcement history begins. |
| Historical FOMC pages contain different date formats and event types | Meetings can span two months, statement dates can differ from heading dates, some meetings were canceled, and some conference calls did not produce a public policy decision. | The parser handles cross-month headings, lets an official statement date override the heading when appropriate, excludes canceled meetings, and includes unscheduled events only when an official decision statement exists. | Some older decisions do not have a published announcement time. |
| The first current-calendar parser included one event that was not a rate decision | The initial link pattern was broad enough to classify the August 22, 2025 monetary-policy strategy notation vote as a scheduled rate decision. | The parser now accepts actual meeting-statement HTML links rather than every monetary-policy press release. The strategy vote was removed from the structured decisions and source manifest. | Future parser validation must continue distinguishing rate decisions from other FOMC announcements. |
| FOMC release times were not consistently available | Modern statements commonly publish a time, while many older official pages do not. Applying the modern 2:00 p.m. convention historically would introduce unsupported assumptions. | Exact official times are stored only when present. Missing times, after-close releases, and non-trading-day releases use a conservative next-trading-session availability rule. | 188 records currently require conservative next-session treatment. |
| A restricted-network refresh failed during validation | The local execution sandbox initially blocked an outbound Federal Reserve request. This was an environment restriction, not evidence that the Federal Reserve source was unavailable. | The failed attempt wrote a diagnostic record and preserved the valid cache. The authorized retry completed successfully. | Network availability is still required for a future explicit refresh, but not for normal cached runs. |
| A failed refresh could otherwise destroy working data | Download or parser failures can occur after a valid cache already exists. Overwriting first would leave the workspace without usable data. | Downloads and parsing complete before replacing a valid cache. Failures are written separately, and tests confirmed the prior file hashes remain unchanged. | Failure records are append-only, so old failures remain visible after a later successful refresh. |
| Some provider webpages were accessible but did not provide structured historical data | Receiving HTML or HTTP 200 only proves that a page responded. It does not prove that a reproducible historical dataset was downloaded. BLS also returned HTTP 403 for some schedule pages. | Structured APIs, JSON feeds, Excel workbooks, FRED/ALFRED vintages, or practical alternate sources were selected. HTML-only probes are not treated as completed data paths. | Certain blocked webpages remain documented as candidate-access evidence only. |
| Ideal survey expectations were not freely available for every surprise feature | Historical consensus forecasts are often licensed, paid, or difficult to reconstruct point in time. | Cleveland Fed inflation nowcasts were selected for CPI expectations, and Atlanta Fed GDPNow history was selected for GDP expectations. FRED/ALFRED supplies point-in-time actual values. | Step 4 must perform the exact pre-release matching; Step 3 only downloaded the required raw inputs. |
| The original short PMI history made the shared modeling period too small | Using only May 2020-August 2025 produced roughly 275 weekly observations and too few chronological training rows for a stable boosted-model benchmark. | The accepted Kansas City release records extend before the August 2013 CPI-expectation start, so PMI no longer limits the shared period. | XGBoost should still be shallow and strongly regularized because 21-day labels overlap. |
| Download metadata could accidentally expose credentials | FRED/ALFRED uses the existing project key, and failure URLs can include query parameters. | Keys are loaded by environment-variable name, request parameters are redacted before persistence, and cached files/reports were scanned for secret values and authenticated URLs. | Secrets must continue to stay in the environment and must never be added to notebook cells or raw metadata. |

## Important final numbers

### Manufacturing proxy

- Official workbook: 301 monthly values from July 2001 through July 2026.
- Structured point-in-time records: 163 monthly observations from January 2013 through July 2026.
- Official release-date coverage: January 24, 2013 through July 23, 2026.
- Missing months or duplicate observation months: none.
- Trustworthy overlap with DBnomics/ISM: 64 months.
- Level correlation over that overlap: approximately 0.936.

### FOMC decisions

- Total policy decisions: 274.
- Scheduled decisions: 260.
- Applicable unscheduled decisions: 14.
- Rate increases: 51.
- Rate decreases: 40.
- Holds: 183.
- Coverage: February 4, 1994 through June 17, 2026.
- Official statement/FRED target cross-check mismatches: none.

### Shared period

- Approximate shared raw period: August 1, 2013 through July 16, 2026.
- Approximate label-eligible weekly observations: 672.
- Approximate label-eligible event-aware observations: 603.
- Ridge has enough data for chronological training, validation, and testing.
- A shallow, strongly regularized XGBoost benchmark can be evaluated, but the data does not support a high-capacity production model.

## Context for the next step

Step 3 resolved how the required raw information can be obtained and cached. Step 4 may use these source decisions without reopening PMI or FOMC source research for the student MVP.

Step 4 still needs to perform the actual transformations, including:

- aligning releases to trading dates;
- matching forecasts to actual releases point in time;
- calculating the eight scenario inputs;
- constructing state features;
- constructing the 21-trading-day targets;
- comparing weekly and event-aware sampling;
- and building the model-ready dataset.

Those transformations were deliberately not performed in Step 3.
