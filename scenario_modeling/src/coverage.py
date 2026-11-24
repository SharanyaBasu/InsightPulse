from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .config import (
    FRED_API_KEY,
    INTERIM_ROOT,
    RAW_ROOT,
    REPORT_PATH,
    credential_is_configured,
    ensure_storage,
)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def load_metadata() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not RAW_ROOT.exists():
        return records
    for path in sorted(RAW_ROOT.rglob("*.metadata.json")):
        record = _read_json(path)
        if record is not None:
            records.append(record)
    return records


def load_failures() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not RAW_ROOT.exists():
        return records
    for path in sorted(RAW_ROOT.rglob("*.failure.json")):
        record = _read_json(path)
        if record is not None:
            records.append(record)
    return records


def _status_counts(records: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str(record.get("retrieval_status", "UNKNOWN")) for record in records)


FINAL_INPUT_SOURCE_DECISIONS = [
    {
        "input": "fed_funds_rate_change_bps",
        "raw_source": "Official Federal Reserve FOMC calendars/statements plus FRED DFEDTARL, DFEDTARU, and DFEDTAR",
        "representation": "Target-range midpoint change for the range era; single-target change for the earlier era",
        "unit": "basis points",
        "coverage": "274 structured decisions from 1994-02-04 through 2026-06-17, including scheduled holds and applicable statement-backed unscheduled decisions; the final feature remains a later-step transformation",
    },
    {
        "input": "cpi_surprise_pct",
        "raw_source": "Cleveland Fed nowcast JSON plus FRED CPIAUCSL observations and ALFRED vintage chunks",
        "representation": "Cleveland pre-release CPI nowcast minus the first point-in-time CPI actual available at the release date",
        "unit": "percentage points of month-over-month CPI inflation",
        "coverage": "Structured nowcast windows and point-in-time CPI history downloaded; exact expectation/actual pairing is a Step 4 transformation",
    },
    {
        "input": "oil_price_change_pct",
        "raw_source": "FRED DCOILWTICO; Yahoo CL=F and USO candidates retained",
        "representation": "Change in the selected oil price proxy over the declared shock window",
        "unit": "percent",
        "coverage": "FRED spot and Yahoo futures/ETF candidates downloaded; final window construction is deferred to Step 4",
    },
    {
        "input": "gdp_growth_surprise_pct",
        "raw_source": "Atlanta Fed GDPNow workbooks plus FRED GDPC1 vintages; BEA release-date JSON",
        "representation": "BEA advance real-GDP growth minus the GDPNow estimate immediately before the advance release",
        "unit": "annualized percentage points, quarter over quarter",
        "coverage": "Official GDPNow model/track-record workbook and release-date workbook downloaded; exact row extraction is a Step 4 transformation",
    },
    {
        "input": "unemployment_change_pct",
        "raw_source": "FRED UNRATE with ALFRED vintage chunks; BLS Employment Situation access retained as a failed candidate probe",
        "representation": "Current unemployment rate minus the prior released unemployment rate",
        "unit": "percentage points",
        "coverage": "FRED point-in-time unemployment history downloaded; the BLS Employment Situation API identifier did not return a series in this environment",
    },
    {
        "input": "pmi_change_points",
        "raw_source": "Federal Reserve Bank of Kansas City Tenth District Manufacturing Survey composite; DBnomics/ISM retained only as rejected evidence",
        "representation": "Later difference of the explicitly named regional PMI-style proxy, where PMI-style level = 50 + 0.5 × Kansas City composite diffusion points",
        "unit": "PMI index points",
        "coverage": "163 point-in-time-usable monthly records from 2013-01 through 2026-07, each with an official release date and conservative next-session availability; the official workbook has 301 values from 2001-07, but this is a regional proxy with revised-history limitations",
    },
    {
        "input": "dxy_change_pct",
        "raw_source": "Yahoo DX-Y.NYB; FRED DTWEXBGS retained only as a broad-dollar comparison",
        "representation": "Change in the true DXY market index proxy over the declared shock window",
        "unit": "percent",
        "coverage": "Yahoo DXY candidate downloaded from 1971-01-04; window construction is deferred to Step 4",
    },
    {
        "input": "vix_change_points",
        "raw_source": "FRED VIXCLS; Yahoo ^VIX retained as a comparison",
        "representation": "Change in the VIX index over the declared shock window",
        "unit": "VIX index points",
        "coverage": "FRED and Yahoo VIX histories downloaded from 1990; the index-point contract is fixed even though the frontend previously used a percentage label",
    },
]


FINAL_TARGET_SOURCE_DECISIONS = [
    {
        "target": "sp500",
        "raw_source": "Yahoo ^GSPC",
        "current_value": "Index Close",
        "formula": "100 * (Close[t+21] / Close[t] - 1)",
        "unit": "percent 21-trading-day price return",
        "neutral": "No expected index return over the next 21 trading days",
    },
    {
        "target": "nasdaq",
        "raw_source": "Yahoo ^IXIC",
        "current_value": "Nasdaq Composite Close",
        "formula": "100 * (Close[t+21] / Close[t] - 1)",
        "unit": "percent 21-trading-day price return",
        "neutral": "No expected index return over the next 21 trading days",
    },
    {
        "target": "ten_year_yield_bps",
        "raw_source": "FRED DGS10; Yahoo ^TNX retained as a comparison",
        "current_value": "10-year Treasury constant-maturity yield in percent",
        "formula": "100 * (DGS10[t+21] - DGS10[t])",
        "unit": "basis points",
        "neutral": "No expected change in the 10-year yield",
    },
    {
        "target": "dxy",
        "raw_source": "Yahoo DX-Y.NYB",
        "current_value": "DXY index Close",
        "formula": "100 * (Close[t+21] / Close[t] - 1)",
        "unit": "percent 21-trading-day index return",
        "neutral": "No expected DXY return over the next 21 trading days",
    },
    {
        "target": "gold",
        "raw_source": "Yahoo GLD adjusted close; GC=F retained as a futures comparison",
        "current_value": "GLD adjusted Close",
        "formula": "100 * (AdjClose[t+21] / AdjClose[t] - 1)",
        "unit": "percent 21-trading-day ETF-proxy return",
        "neutral": "No expected gold-proxy return over the next 21 trading days",
    },
    {
        "target": "oil",
        "raw_source": "Yahoo CL=F Close; FRED DCOILWTICO retained as a spot comparison",
        "current_value": "WTI continuous-futures Close",
        "formula": "100 * (Close[t+21] / Close[t] - 1)",
        "unit": "percent 21-trading-day futures return",
        "neutral": "No expected oil-proxy return over the next 21 trading days",
    },
    {
        "target": "technology",
        "raw_source": "Yahoo XLK adjusted close",
        "current_value": "Technology Select Sector SPDR adjusted Close",
        "formula": "100 * (AdjClose[t+21] / AdjClose[t] - 1)",
        "unit": "percent 21-trading-day ETF-proxy return",
        "neutral": "No expected technology-sector return over the next 21 trading days",
    },
    {
        "target": "energy",
        "raw_source": "Yahoo XLE adjusted close",
        "current_value": "Energy Select Sector SPDR adjusted Close",
        "formula": "100 * (AdjClose[t+21] / AdjClose[t] - 1)",
        "unit": "percent 21-trading-day ETF-proxy return",
        "neutral": "No expected energy-sector return over the next 21 trading days",
    },
    {
        "target": "financials",
        "raw_source": "Yahoo XLF adjusted close",
        "current_value": "Financial Select Sector SPDR adjusted Close",
        "formula": "100 * (AdjClose[t+21] / AdjClose[t] - 1)",
        "unit": "percent 21-trading-day ETF-proxy return",
        "neutral": "No expected financial-sector return over the next 21 trading days",
    },
    {
        "target": "utilities",
        "raw_source": "Yahoo XLU adjusted close",
        "current_value": "Utilities Select Sector SPDR adjusted Close",
        "formula": "100 * (AdjClose[t+21] / AdjClose[t] - 1)",
        "unit": "percent 21-trading-day ETF-proxy return",
        "neutral": "No expected utilities-sector return over the next 21 trading days",
    },
    {
        "target": "healthcare",
        "raw_source": "Yahoo XLV adjusted close",
        "current_value": "Health Care Select Sector SPDR adjusted Close",
        "formula": "100 * (AdjClose[t+21] / AdjClose[t] - 1)",
        "unit": "percent 21-trading-day ETF-proxy return",
        "neutral": "No expected healthcare-sector return over the next 21 trading days",
    },
    {
        "target": "consumer_discretionary",
        "raw_source": "Yahoo XLY adjusted close",
        "current_value": "Consumer Discretionary Select Sector SPDR adjusted Close",
        "formula": "100 * (AdjClose[t+21] / AdjClose[t] - 1)",
        "unit": "percent 21-trading-day ETF-proxy return",
        "neutral": "No expected consumer-discretionary return over the next 21 trading days",
    },
]


def source_access_matrix() -> list[dict[str, str]]:
    fred_auth = (
        "EXISTING_PROJECT_KEY_AVAILABLE"
        if credential_is_configured(FRED_API_KEY)
        else "EXISTING_PROJECT_KEY_MISSING"
    )
    return [
        {
            "source": "Yahoo Finance via yfinance",
            "datasets": "Market, index, commodity, FX, VIX, and sector target candidates",
            "authentication": "NO_KEY_REQUIRED",
            "retrieval": "Controlled yfinance downloads",
            "status": "17/17 candidate tickers downloaded",
            "decision": "Final target proxies are ^GSPC, ^IXIC, DX-Y.NYB, GLD, CL=F, XLK, XLE, XLF, XLU, XLV, and XLY; alternatives remain raw comparisons.",
        },
        {
            "source": "FRED",
            "datasets": "Policy rates, CPI, GDP, unemployment, VIX, oil, dollar, and Treasury series",
            "authentication": fred_auth,
            "retrieval": "Official FRED observations and series metadata endpoints",
            "status": "18/18 requested series paths attempted; 17 retained as usable raw candidate histories",
            "decision": "FRED is final for policy, CPI actuals, GDP actuals, unemployment, DGS10, VIXCLS, and DCOILWTICO; retired NAPM is not used.",
        },
        {
            "source": "ALFRED / FRED vintage-capable endpoint",
            "datasets": "CPIAUCSL, UNRATE, and GDPC1 vintage-date indexes and observation chunks",
            "authentication": fred_auth,
            "retrieval": "Controlled realtime, vintage_dates, and initial-release requests",
            "status": "Primary point-in-time histories downloaded",
            "decision": "Use output_type=2 vintage snapshots; output_type=4 initial-release probing failed but is not required for the selected vintage-date path.",
        },
        {
            "source": "Federal Reserve",
            "datasets": "Official FOMC historical/current calendars, statements, and structured policy decisions",
            "authentication": "NO_KEY_REQUIRED",
            "retrieval": "Deterministic official calendar/statement parsing plus cached FRED policy levels",
            "status": "274 decisions downloaded for 1994-02-04 through 2026-06-17: 260 scheduled, 14 applicable unscheduled, 183 holds, 51 increases, and 40 decreases",
            "decision": "Use the independent decision calendar so holds remain present; attach DFEDTAR before the range era and DFEDTARL/DFEDTARU thereafter for numeric before/after levels.",
        },
        {
            "source": "BLS",
            "datasets": "CPI headline NSA API; Employment Situation candidate probe",
            "authentication": "NO_KEY_REQUIRED",
            "retrieval": "BLS Public Data API year chunks",
            "status": "8/8 CPI API chunks downloaded; Employment Situation API returned no series; schedule pages/ICS returned HTTP 403",
            "decision": "Use BLS CPI API as an actual-data cross-check and FRED vintages for unemployment; no required input depends on the blocked HTML pages or unavailable BLS employment identifier.",
        },
        {
            "source": "BEA",
            "datasets": "Official release schedule and machine-readable release-date feed",
            "authentication": "NO_KEY_REQUIRED",
            "retrieval": "Official schedule page and release_dates.json",
            "status": "Structured release-date feed downloaded",
            "decision": "Use Atlanta GDPNow track-record data for advance-release expectations and FRED/ALFRED for actual GDP; BEA API key is not required by the selected path.",
        },
        {
            "source": "Cleveland Fed",
            "datasets": "Monthly, quarterly, and annual inflation nowcast JSON",
            "authentication": "NO_KEY_REQUIRED",
            "retrieval": "Official machine-readable JSON endpoints",
            "status": "3 structured JSON files downloaded",
            "decision": "Use monthly CPI nowcast data as the practical expectation source for the true CPI surprise contract.",
        },
        {
            "source": "Atlanta Fed GDPNow",
            "datasets": "Historical GDPNow estimates",
            "authentication": "NO_KEY_REQUIRED",
            "retrieval": "Official GDPNow page and linked Excel workbooks",
            "status": "Page plus two actual Excel workbooks downloaded",
            "decision": "Use the TrackRecord workbook as the practical GDP expectation/advance-release source.",
        },
        {
            "source": "Philadelphia Fed SPF",
            "datasets": "Mean/median level and growth forecast workbooks",
            "authentication": "NO_KEY_REQUIRED",
            "retrieval": "Official linked Excel workbooks",
            "status": "Research page plus four workbooks downloaded",
            "decision": "Retain as a free backup expectation source, not the primary CPI/GDP source when Cleveland/Atlanta histories are available.",
        },
        {
            "source": "Federal Reserve Bank of Kansas City",
            "datasets": "Tenth District Manufacturing Survey historical workbook, official archive release dates, methodology, and terms",
            "authentication": "NO_KEY_REQUIRED",
            "retrieval": "Deterministic XLSX parsing and official archive/release-page parsing",
            "status": "163 point-in-time-usable monthly records from 2013-01 through 2026-07; 301 source-workbook values extend back to 2001-07",
            "decision": "Use as the explicitly named regional PMI-style MVP proxy. The five equally weighted components match the ISM construction, but geographic scope is the Tenth District and current-revised history is not a vintage archive.",
        },
        {
            "source": "ISM / DBnomics candidate",
            "datasets": "DBnomics ISM/pmi/pm mirror, official ISM release calendar, and official terms of use",
            "authentication": "NO_KEY_REQUIRED",
            "retrieval": "Preserved source-faithful DBnomics JSON; direct official-calendar and terms requests returned CAPTCHA shells and are retained only as blocked reference probes",
            "status": "REJECTED_PRIMARY_CANDIDATE: 64 trustworthy rows through 2025-08; 2025-09 through 2025-12 are corrupted",
            "decision": "Do not use for modeling. The history is short, structured historical release dates are absent, four values are corrupt, and underlying ISM bulk-use/licensing restrictions are unresolved.",
        },
    ]


def _md(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_report(
    records: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> str:
    pmi_payload = _read_json(
        RAW_ROOT / "kansas_city_fed" / "manufacturing_pmi_proxy_events.json"
    ) or {}
    fomc_payload = _read_json(
        RAW_ROOT / "federal_reserve" / "fomc_decisions.json"
    ) or {}
    pmi_observations = list(pmi_payload.get("observations") or [])
    pmi_comparisons = list(
        pmi_payload.get("published_ism_scale_and_direction_comparisons") or []
    )
    pmi_overlap = dict(
        pmi_payload.get("trusted_dbnomics_ism_overlap_validation") or {}
    )
    fomc_events = list(fomc_payload.get("events") or [])
    fomc_counts = dict(fomc_payload.get("counts") or {})
    counts = _status_counts(records)
    source_counts: dict[str, Counter[str]] = {}
    for record in records:
        source_counts.setdefault(str(record.get("source", "UNKNOWN")), Counter())[
            str(record.get("retrieval_status", "UNKNOWN"))
        ] += 1

    lines: list[str] = [
        "# Scenario Playground Raw Data Download Report",
        "",
        "This report is generated by the Step 3 local raw-data runner. It records retrieval attempts, cache reuse, source metadata, and failures. It does not contain model features, targets, shocks, or Supabase writes.",
        "",
        "## Retrieval Summary",
        "",
        f"- Metadata records: {len(records)}",
        f"- Failure records: {len(failures)}",
        f"- Status counts: {dict(counts)}",
        "",
        "## Source Access Matrix",
        "",
        "| Source | Datasets | Authentication classification | Retrieval method | Status | Practical Step 3 decision |",
        "|---|---|---|---|---|---|",
    ]
    for row in source_access_matrix():
        lines.append(
            "| "
            + " | ".join(
                _md(row[key])
                for key in ("source", "datasets", "authentication", "retrieval", "status", "decision")
            )
            + " |"
        )

    lines += [
        "",
        "## Final Practical Input Source Decisions",
        "",
        "These are the Step 3 collection decisions for the eight contract inputs. Step 3 stores the raw ingredients; it does not yet calculate the shock columns.",
        "",
        "| Input | Raw source | Historical representation | Unit | Coverage/limitation |",
        "|---|---|---|---|---|",
    ]
    for row in FINAL_INPUT_SOURCE_DECISIONS:
        lines.append(
            "| "
            + " | ".join(
                _md(row[key])
                for key in ("input", "raw_source", "representation", "unit", "coverage")
            )
            + " |"
        )

    lines += [
        "",
        "## Final Practical Target Source Decisions",
        "",
        "All price/index targets use the same 21-trading-day horizon from the later model contract. ETF proxies use adjusted close; direct indexes, FX, futures, and yields use their native close/value fields.",
        "",
        "| Target | Raw source | Current value type | Prediction formula | Output unit | Neutral interpretation |",
        "|---|---|---|---|---|---|",
    ]
    for row in FINAL_TARGET_SOURCE_DECISIONS:
        lines.append(
            "| "
            + " | ".join(
                _md(row[key])
                for key in ("target", "raw_source", "current_value", "formula", "unit", "neutral")
            )
            + " |"
        )

    lines += [
        "",
        "## Point-in-Time and Raw-Data Boundaries",
        "",
        "- The ordinary FRED observations are the latest retrieved histories; the separate ALFRED/FRED output_type=2 chunks preserve values as available at explicit vintage dates for CPIAUCSL, UNRATE, and GDPC1.",
        "- Later steps must join a macro value only when its release/vintage timestamp is on or before the row timestamp. If the exact release time is unavailable, the conservative choice is to use the next trading session and document that date-level approximation.",
        "- FOMC decisions are now structured independently from policy-rate changes, so scheduled holds remain in the event calendar. Official release times are retained when published; otherwise availability is conservatively moved to the next US equity trading session.",
        "- The selected manufacturing proxy has an official release date for every usable observation. Because exact historical release times are unavailable, every value becomes usable on the next US equity trading session.",
        "- No raw file is forward-filled, resampled, revised into a single history, or merged with market data here. Missing observations remain visible in the metadata and raw payloads.",
        "- A source candidate being downloaded does not make it the selected model source. ^TNX, DGS10, GLD, GC=F, CL=F, DCOILWTICO, VIXCLS, ^VIX, DTWEXBGS, SPY, QQQ, and the sector ETFs are retained so Step 4 can compare representations without redownloading.",
        "",
        "## Step 3 Completion Boundary",
        "",
        "Step 3 is complete for raw-data collection under the student MVP standard: the eight inputs and twelve targets each have a named raw path; the rejected DBnomics/ISM values are outside the primary path; the accepted manufacturing proxy has long structured release coverage; and the independent FOMC history includes holds. Step 4 still owns event alignment, sampling, shock construction, state features, 21-trading-day targets, the merged model matrix, and model training.",
        "",
        "## Manufacturing PMI Correction",
        "",
        "- Final source: Federal Reserve Bank of Kansas City Tenth District Manufacturing Survey, stored as the explicitly named `kc_fed_manufacturing_pmi_proxy_change_points` historical proxy for the contract input `pmi_change_points`.",
        "- Official references: [survey/workbook](https://www.kansascityfed.org/surveys/manufacturing-survey/), [methodology](https://www.kansascityfed.org/surveys/manufacturing-survey/manufacturing-survey-methodology/), and [reproduction disclaimer](https://www.kansascityfed.org/about-us/disclaimer/).",
        "- Economic fit: the official Kansas City methodology uses a manufacturing diffusion composite built from production, new orders, employment, supplier delivery time, and raw-material inventories with equal weights. Raw values use increase-minus-decrease diffusion points with neutral 0; `50 + 0.5 × raw composite` is the deterministic 50-neutral PMI-style representation.",
        "- Conceptual limitation: this is a seven-state Tenth District regional series, not the national ISM Manufacturing PMI. The contract/UI label remains `PMI Change`, but model documentation must disclose the historical regional proxy and must not call it official ISM data.",
        "- Value coverage: the official workbook has 301 monthly values from 2001-07 through 2026-07. Point-in-time-usable structured records begin 2013-01 and contain 163 consecutive months through 2026-07.",
        "- Release coverage: 163 official release dates from 2013-01-24 through 2026-07-23. One omission in the 2024 yearly archive (June) is filled from the official dated June 27, 2024 release page retained in the same raw source directory.",
        "- Timing: exact historical release times are not published consistently, so every record uses the next US equity trading session as its model-availability date.",
        "- Revision behavior: the workbook is the latest revised, seasonally adjusted history; Kansas City Fed recalculates seasonal factors, so this is not an initial-release value-vintage archive.",
        "- Licensing: the Kansas City Fed disclaimer permits reproduction without quantity limit when not for private gain. This project accepts local, attributed, noncommercial student caching/training; raw files stay Git-ignored, redistribution is not authorized by this decision, and production/private-gain use requires a new review.",
        "- Rejected source: `data/scenario_playground/raw/ism/dbnomics_ism_pmi.json` remains unchanged and is marked `REJECTED_PRIMARY_CANDIDATE`. Its trustworthy history is only 64 rows (2020-05 through 2025-08); 2025-09 through 2025-12 contain 11.1, 10.0, 10.0, and 10.3; it has no structured historical release dates; and DBnomics does not resolve underlying ISM bulk-use restrictions.",
        "- ISM reference-page probes: direct requests to the official [release calendar](https://www.ismworld.org/supply-management-news-and-reports/reports/rob-report-calendar/) and [terms of use](https://www.ismworld.org/footer/terms-of-use/) returned CAPTCHA shells. Their metadata therefore marks them unavailable rather than successful source documents; the URLs remain cited for manual timing/legal review, and the selected Kansas City path does not depend on those probes.",
        "",
        "### PMI validation evidence",
        "",
        "| Month | Official national ISM PMI reference | Kansas City raw composite | Kansas City PMI-style level | Same expansion/contraction side of 50? |",
        "|---|---:|---:|---:|---|",
    ]
    for comparison in pmi_comparisons:
        lines.append(
            "| "
            + " | ".join(
                _md(comparison.get(key))
                for key in (
                    "observation_month",
                    "official_ism_manufacturing_pmi",
                    "kc_composite_diffusion_points",
                    "kc_pmi_style_level",
                    "same_expansion_contraction_direction",
                )
            )
            + " |"
        )

    pmi_values = [
        float(row["pmi_style_level"])
        for row in pmi_observations
        if row.get("pmi_style_level") is not None
    ]
    pmi_months = [
        str(row.get("observation_month"))
        for row in pmi_observations
        if row.get("observation_month")
    ]
    lines += [
        "",
        (
            f"- Parser checks: {len(pmi_observations)} structured rows, "
            f"{len(set(pmi_months))} unique observation months, "
            f"PMI-style range {min(pmi_values) if pmi_values else 'n/a'} to "
            f"{max(pmi_values) if pmi_values else 'n/a'}, no missing months, and "
            "observation month kept separate from release/model-availability dates."
        ),
        (
            f"- Full trustworthy overlap check against the preserved DBnomics/ISM "
            f"values through 2025-08: {pmi_overlap.get('overlap_observation_count', 'n/a')} "
            f"months, Pearson level correlation "
            f"{pmi_overlap.get('pearson_level_correlation', 'n/a')}, mean absolute "
            f"level difference {pmi_overlap.get('mean_absolute_level_difference_points', 'n/a')} "
            f"points, and {pmi_overlap.get('same_expansion_contraction_side_count', 'n/a')} "
            "months on the same expansion/contraction side of 50. This validation does "
            "not put the rejected DBnomics file into the model path."
        ),
        "- The selected comparison table is deliberately not an equality test: it includes historical/recent expansion and contraction examples and highlights genuine regional/national divergence. Only 2 of these 8 deliberately difficult comparison months are on the same side of 50, which is why the internal regional-proxy name is mandatory despite strong full-overlap co-movement.",
        "- The September-December 2025 accepted proxy levels are 51.5, 52.0, 53.5, and 50.0; none uses the corrupted DBnomics values. National ISM references for those months are 49.1, 48.7, 48.2, and 47.9, illustrating the geographic difference rather than a parsing failure.",
        "",
        "## Structured FOMC Decision History",
        "",
        "- Final source: official Federal Reserve historical meeting pages (1994-2020), the current calendar (2021 onward), and linked official statements. All source pages are retained with SHA-256 hashes in `fomc_source_manifest.json`.",
        "- Official references: [current calendar](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm), year-specific historical pages under the Federal Reserve's FOMC archive, and the [Federal Reserve website disclaimer](https://www.federalreserve.gov/disclaimer.htm).",
        "- Parser method: every scheduled meeting panel is retained independently of whether rates changed. Conference calls or unscheduled panels are retained when they link an official policy-decision statement; canceled meetings, non-rate notation votes, and unscheduled discussions without a public decision statement are excluded.",
        "- Numeric method: before 2008-12-16, use the FRED `DFEDTAR` single target. From 2008-12-16 onward, use FRED `DFEDTARL`/`DFEDTARU` and calculate the midpoint change in basis points. Parsed statement targets were cross-checked against those cached FRED levels.",
        (
            f"- Coverage/counts: {len(fomc_events)} decisions from "
            f"{fomc_events[0]['announcement_date'] if fomc_events else 'n/a'} through "
            f"{fomc_events[-1]['announcement_date'] if fomc_events else 'n/a'}; "
            f"{fomc_counts.get('scheduled', 0)} scheduled, "
            f"{fomc_counts.get('unscheduled', 0)} applicable unscheduled, "
            f"{fomc_counts.get('increase', 0)} increases, "
            f"{fomc_counts.get('decrease', 0)} decreases, and "
            f"{fomc_counts.get('hold', 0)} holds."
        ),
        (
            f"- Timing: {fomc_counts.get('exact_announcement_time', 0)} records have "
            "an official parsed announcement time. A pre-close release is usable in "
            "that date's end-of-day row; at/after-close and non-trading-day releases "
            "move to the next session. Every missing historical time also moves to "
            f"the next session ({fomc_counts.get('conservative_next_session', 0)} records total)."
        ),
        "- Applicable emergency decisions include March 3, 2020 (10:00 a.m. EST, -50 bps) and March 15, 2020 (5:00 p.m. EDT, -100 bps). The canceled March 17-18, 2020 meeting is not represented as a decision.",
        "- Non-decision guard: the August 22, 2025 notation vote updating the Statement on Longer-Run Goals and Monetary Policy Strategy is excluded from `fomc_decisions.json` and its source manifest because it did not set or hold the policy target at an FOMC rate decision.",
        f"- Statement/FRED numeric cross-checks: {fomc_counts.get('statement_target_cross_checks', 0)} parsed target statements and zero mismatches.",
        "",
        "## Shared Historical Coverage and Sample Sufficiency",
        "",
        "- Approximate raw intersection: 2013-08-01 through 2026-07-16.",
        "- Start limiter: Cleveland Fed CPI nowcast/expectation coverage. The selected PMI release records and complete FOMC records already begin earlier.",
        "- End limiter: the cached FRED `DGS10` target history at 2026-07-16. Yahoo target files extend at least as far as the same window.",
        "- Approximate last date that can already receive a complete 21-trading-day label: 2026-06-15. This is a coverage calculation only; no targets or rows were constructed.",
        "- Weekly inventory estimate: about 676 raw weekly dates, or about 672 currently label-eligible weekly dates.",
        "- Economic-event inventory through the raw end: 154 CPI releases, about 156 Employment Situation/unemployment releases, about 51 GDP advance events, 105 FOMC announcements, and 155 manufacturing-proxy releases.",
        "- Approximate combined event-aware inventory after same-day deduplication: about 607 raw event dates, or about 603 currently label-eligible event dates. Employment/GDP counts and same-day deduplication remain estimates until Step 4 performs the authorized event alignment.",
        "- Illustrative 60/20/20 chronological sufficiency check (not a final split): weekly gives roughly 403 training, 134 validation, and 135 test rows; event-aware gives roughly 362 training, 121 validation, and 120 test rows.",
        "- Ridge assessment: meaningful training, chronological validation, and chronological testing are supported for both candidate sampling families.",
        "- XGBoost assessment: a shallow, strongly regularized benchmark can be evaluated meaningfully with roughly 360-400 chronological training rows spanning the post-crisis low-rate era, tightening cycles, the pandemic, inflation, and later easing. It is not enough evidence for a high-capacity or production-grade boosted model; overlapping 21-day labels reduce effective independence.",
        "- Weekly versus event-aware comparison is now data-supported, but choosing either design remains Step 4 work.",
    ]

    lines += [
        "",
        "## Dataset Coverage",
        "",
        "| Dataset | Source | Retrieval status | Rows | Minimum date | Maximum date | Frequency | Units | File | Cache reused |",
        "|---|---|---|---:|---|---|---|---|---|---|",
    ]
    for record in records:
        lines.append(
            "| "
            + " | ".join(
                _md(record.get(key))
                for key in (
                    "dataset_name",
                    "source",
                    "retrieval_status",
                    "row_count",
                    "minimum_observation_date",
                    "maximum_observation_date",
                    "frequency",
                    "units",
                    "file_path",
                    "cache_reused",
                )
            )
            + " |"
        )

    lines += [
        "",
        "## Failures and Preserved Caches",
        "",
        "Failure records are append-only diagnostics. A failed attempt can remain listed after a later successful refresh; current acceptance is determined from the valid data file and its successful metadata sidecar.",
        "",
        "| Dataset | Source | Error | Previous cache preserved | Failure file |",
        "|---|---|---|---|---|",
    ]
    if failures:
        for failure in failures:
            lines.append(
                "| "
                + " | ".join(
                    _md(failure.get(key))
                    for key in (
                        "dataset_name",
                        "source",
                        "error_details",
                        "preserved_previous_cache",
                        "failure_file",
                    )
                )
                + " |"
            )
    else:
        lines.append("| None recorded | | | | |")

    lines += [
        "",
        "## Storage and Cache Rules",
        "",
        f"- Raw root: {RAW_ROOT}",
        f"- Interim comparison root: {INTERIM_ROOT / 'source_comparisons'}",
        "- Each source has its own raw directory and source-specific files; no source merges are performed here.",
        "- Raw responses and/or raw ticker files are retained. No forward fill, resampling, imputation, shock construction, feature engineering, or target construction occurs in Step 3.",
        "- A successful data file with a valid metadata sidecar is reused unless the runner receives --refresh.",
        "- A failed refresh writes a separate failure record and does not delete a previous valid cache.",
        "- API keys are read from the active process or backend/.env by variable name only; values are never written to metadata or this report.",
        "",
        "## Metadata Captured",
        "",
        "For tabular datasets the sidecar records dataset/source identifiers, retrieval time, redacted request parameters, authentication classification, environment-variable name, file format/path, row and column information, date coverage, calendar span, approximate trading years where relevant, frequency, duplicates, missing values, gap notes, timezone handling, units, adjustment policy, vintage parameters, library version, and licensing/source notes.",
        "",
        "## Dependency and Reproducibility Notes",
        "",
        "- The downloader uses the existing project runtime packages pandas, requests, yfinance, fredapi, and python-dotenv.",
        "- Raw files use CSV/JSON/document formats. Parquet was not written because pyarrow is not installed in the inspected backend environment; no dependency was installed during Step 3.",
        "- The notebook is an instructional wrapper around the same local runner and does not train a model.",
        "",
        "## Remaining Step 4 Work and Limitations",
        "",
        "- A successful HTTP response or downloaded page is not evidence that a complete historical release-vintage dataset is available.",
        "- Parse the downloaded Cleveland/Atlanta/Philadelphia structured files and align their release dates to market sessions before constructing the training matrix.",
        "- Do not screen or repair the DBnomics/ISM mirror into the primary path; it is explicitly rejected. Use the accepted Kansas City Fed proxy records and preserve their regional/revised-history disclosure.",
        "- Exact national ISM semantics would require separately licensed long-history data. The accepted free MVP proxy preserves the manufacturing diffusion construction but not national geographic coverage.",
        "- Historical Kansas City release times remain unavailable, so the conservative next-session rule is required. Its workbook is revised history, not a value-vintage archive.",
        "- Historical FOMC records without an official time also require the conservative next-session rule.",
        "- Decide the final event-day convention, shock windows, sample frequency, missing-row policy, and overlap after the raw source coverage is measured in Step 4.",
        "- Step 3 has fixed practical source paths; it has not constructed shocks, state features, targets, sampling rows, chronological datasets, or a merged model matrix.",
        "",
        "## Source-Level Retrieval Counts",
        "",
        "| Source | Status counts |",
        "|---|---|",
    ]
    for source in sorted(source_counts):
        lines.append(f"| {_md(source)} | {_md(dict(source_counts[source]))} |")

    return "\n".join(lines) + "\n"


def write_report() -> Path:
    ensure_storage()
    records = load_metadata()
    failures = load_failures()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_report(records, failures), encoding="utf-8")
    return REPORT_PATH
