from __future__ import annotations

import hashlib
import html
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, time, timedelta
from importlib.metadata import version
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests

from ..cache import (
    record_failure,
    valid_cache,
    write_bytes_atomic,
    write_json_atomic,
)
from ..config import NO_KEY_REQUIRED, PROJECT_ROOT, RAW_SOURCE_DIRS, safe_request_parameters
from ..metadata import persist_metadata, summarize_document, summarize_tabular
from ..trading_calendar import (
    is_us_equity_trading_session,
    next_us_equity_trading_session,
)


CURRENT_CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
HISTORICAL_URL_TEMPLATE = (
    "https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"
)
FIRST_RELIABLE_PUBLIC_DECISION_YEAR = 1994
PARSER_VERSION = "insightpulse-fomc-decisions-v2"
FED_BASE_URL = "https://www.federalreserve.gov"

_MONTH_NUMBERS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
_MONTH_ALIASES = {
    **_MONTH_NUMBERS,
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
_HISTORICAL_PANEL_SPLIT = re.compile(
    r"<div\s+class=[\"'][^\"']*panel panel-default[^\"']*[\"'][^>]*>",
    re.IGNORECASE,
)
_ANCHOR_PATTERN = re.compile(
    r"<a\b[^>]*\bhref=[\"'](?P<href>[^\"']+)[\"'][^>]*>"
    r"(?P<label>.*?)</a>",
    re.IGNORECASE | re.DOTALL,
)
_CURRENT_STATEMENT_PATTERN = re.compile(
    r"<a\b[^>]*\bhref=[\"'](?P<href>[^\"']*"
    r"/newsevents/pressreleases/monetary(?P<day>\d{8})a\.htm"
    r"(?:\?[^\"']*)?)[\"'][^>]*>\s*HTML\s*</a>",
    re.IGNORECASE | re.DOTALL,
)


def _library_version(package: str) -> str | None:
    try:
        return version(package)
    except Exception:
        return None


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _strip_html(value: str) -> str:
    value = re.sub(r"<script\b.*?</script>", " ", value, flags=re.IGNORECASE | re.DOTALL)
    value = re.sub(r"<style\b.*?</style>", " ", value, flags=re.IGNORECASE | re.DOTALL)
    value = re.sub(r"<[^>]+>", " ", value)
    return " ".join(html.unescape(value).split())


def _get(url: str, *, timeout: int = 60) -> tuple[bytes, dict[str, str], int]:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.get(
                url,
                timeout=timeout,
                headers={"User-Agent": "InsightPulse-ScenarioModeling/step3"},
            )
            response.raise_for_status()
            return response.content, dict(response.headers), response.status_code
        except Exception as exc:
            last_error = exc
            if attempt == 2:
                break
    raise RuntimeError(f"Unable to retrieve {url}: {last_error}")


def _fetch_many(urls: list[str], *, max_workers: int = 6) -> dict[str, tuple[bytes, dict[str, str], int]]:
    results: dict[str, tuple[bytes, dict[str, str], int]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_get, url): url for url in urls}
        for future in as_completed(futures):
            url = futures[future]
            results[url] = future.result()
    return results


def _parse_heading_date(heading: str, year: int) -> date:
    slash_match = re.search(
        r"\b([A-Za-z]+)/([A-Za-z]+)\s+(\d{1,2})\s*-\s*(\d{1,2})",
        heading,
    )
    if slash_match:
        second_month = _MONTH_ALIASES.get(slash_match.group(2).lower())
        if second_month is None:
            raise RuntimeError(
                f"Unknown month abbreviation in FOMC heading: {heading}"
            )
        return date(year, second_month, int(slash_match.group(4)))
    match = re.search(
        r"\b("
        + "|".join(_MONTH_NUMBERS)
        + r")\s+(\d{1,2})"
        + r"(?:\s*-\s*(?:("
        + "|".join(_MONTH_NUMBERS)
        + r")\s+)?(\d{1,2}))?",
        heading,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise RuntimeError(f"Unable to parse FOMC event date from heading: {heading}")
    first_month = _MONTH_NUMBERS[match.group(1).lower()]
    second_month = (
        _MONTH_NUMBERS[match.group(3).lower()]
        if match.group(3)
        else first_month
    )
    day = int(match.group(4) or match.group(2))
    return date(year, second_month, day)


def _statement_link_from_panel(panel: str) -> str | None:
    for anchor in _ANCHOR_PATTERN.finditer(panel):
        label = _strip_html(anchor.group("label")).strip().lower()
        if label == "statement":
            return urljoin(FED_BASE_URL, html.unescape(anchor.group("href")))
    return None


def _historical_events(page: bytes, year: int, source_file: str) -> list[dict[str, Any]]:
    text = page.decode("utf-8", errors="replace")
    events: list[dict[str, Any]] = []
    for panel in _HISTORICAL_PANEL_SPLIT.split(text)[1:]:
        heading_match = re.search(r"<h5\b[^>]*>(.*?)</h5>", panel, re.I | re.S)
        if heading_match is None:
            continue
        heading = _strip_html(heading_match.group(1))
        lower_heading = heading.lower()
        if "cancelled" in lower_heading or "canceled" in lower_heading:
            continue
        statement_url = _statement_link_from_panel(panel)
        scheduled = (
            "meeting" in lower_heading
            and "conference call" not in lower_heading
            and "unscheduled" not in lower_heading
        )
        applicable_unscheduled = (
            ("conference call" in lower_heading or "unscheduled" in lower_heading)
            and statement_url is not None
        )
        if not scheduled and not applicable_unscheduled:
            continue
        decision_date = _parse_heading_date(heading, year)
        if statement_url:
            url_date = re.search(r"(?<!\d)((?:19|20)\d{6})(?!\d)", statement_url)
            if url_date:
                decision_date = datetime.strptime(url_date.group(1), "%Y%m%d").date()
        events.append(
            {
                "announcement_date": decision_date.isoformat(),
                "event_type": "scheduled" if scheduled else "unscheduled",
                "official_event_heading": heading,
                "statement_url": statement_url,
                "calendar_source_url": HISTORICAL_URL_TEMPLATE.format(year=year),
                "calendar_source_file": source_file,
            }
        )
    return events


def _current_events(page: bytes, source_file: str) -> list[dict[str, Any]]:
    text = page.decode("utf-8", errors="replace")
    events: list[dict[str, Any]] = []
    for match in _CURRENT_STATEMENT_PATTERN.finditer(text):
        decision_date = datetime.strptime(match.group("day"), "%Y%m%d").date()
        if decision_date.year < 2021:
            continue
        events.append(
            {
                "announcement_date": decision_date.isoformat(),
                "event_type": "scheduled",
                "official_event_heading": (
                    f"{decision_date.isoformat()} FOMC meeting statement"
                ),
                "statement_url": urljoin(
                    CURRENT_CALENDAR_URL,
                    html.unescape(match.group("href")),
                ),
                "calendar_source_url": CURRENT_CALENDAR_URL,
                "calendar_source_file": source_file,
            }
        )
    return events


def _load_fred_observations(series_id: str) -> dict[date, float]:
    path = (
        PROJECT_ROOT
        / "data"
        / "scenario_playground"
        / "raw"
        / "fred"
        / f"{series_id}.observations.json"
    )
    if not path.exists():
        raise RuntimeError(
            f"Required cached FRED policy series is missing: {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    values: dict[date, float] = {}
    for row in payload.get("observations", []):
        raw_value = row.get("value")
        if raw_value in (None, "", "."):
            continue
        values[date.fromisoformat(row["date"])] = float(raw_value)
    if not values:
        raise RuntimeError(f"Required cached FRED series {series_id} has no values")
    return values


def _latest_on_or_before(values: dict[date, float], day: date) -> tuple[date, float]:
    eligible = [candidate for candidate in values if candidate <= day]
    if not eligible:
        raise RuntimeError(f"No policy observation exists on or before {day}")
    selected = max(eligible)
    return selected, values[selected]


def _policy_level(
    day: date,
    *,
    single: dict[date, float],
    lower: dict[date, float],
    upper: dict[date, float],
) -> dict[str, Any]:
    if day < date(2008, 12, 16):
        observation_date, target = _latest_on_or_before(single, day)
        return {
            "policy_expression": "single_target",
            "observation_date": observation_date,
            "lower": target,
            "upper": target,
            "midpoint": target,
        }
    lower_date, lower_value = _latest_on_or_before(lower, day)
    upper_date, upper_value = _latest_on_or_before(upper, day)
    return {
        "policy_expression": "target_range",
        "observation_date": max(lower_date, upper_date),
        "lower": lower_value,
        "upper": upper_value,
        "midpoint": (lower_value + upper_value) / 2.0,
    }


def _same_policy(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        abs(float(left["lower"]) - float(right["lower"])) < 1e-9
        and abs(float(left["upper"]) - float(right["upper"])) < 1e-9
    )


def _policy_before_after(
    event_date: date,
    *,
    single: dict[date, float],
    lower: dict[date, float],
    upper: dict[date, float],
) -> tuple[dict[str, Any], dict[str, Any], date | None]:
    before = _policy_level(
        event_date - timedelta(days=1),
        single=single,
        lower=lower,
        upper=upper,
    )
    after = before
    effective_date: date | None = None
    for offset in range(0, 3):
        candidate_day = event_date + timedelta(days=offset)
        candidate = _policy_level(
            candidate_day,
            single=single,
            lower=lower,
            upper=upper,
        )
        if not _same_policy(before, candidate):
            after = candidate
            effective_date = candidate["observation_date"]
            break
    return before, after, effective_date


def _release_time(text: str) -> tuple[str | None, str | None]:
    match = re.search(
        r"For release at\s+(\d{1,2})(?::(\d{2}))?\s*"
        r"([ap])\.?\s*m\.?\s*([A-Z]{2,5})",
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None, None
    hour = int(match.group(1))
    minute = int(match.group(2) or "00")
    meridiem = match.group(3).lower()
    if meridiem == "p" and hour != 12:
        hour += 12
    if meridiem == "a" and hour == 12:
        hour = 0
    return f"{hour:02d}:{minute:02d}:00", match.group(4).upper()


def _policy_sentence(text: str) -> str | None:
    lower = text.lower()
    phrases = (
        "target range for the federal funds rate",
        "federal funds rate target",
        "intended federal funds rate",
    )
    indexes = [lower.find(phrase) for phrase in phrases if lower.find(phrase) >= 0]
    if not indexes:
        return None
    index = min(indexes)
    start = max(text.rfind(".", 0, index) + 1, 0)
    end = text.find(".", index)
    if end < 0:
        end = min(len(text), index + 600)
    return text[start : end + 1].strip()


def _rate_number(value: str) -> float:
    normalized = (
        value.replace("–", "-")
        .replace("—", "-")
        .replace("¼", "-1/4")
        .replace("½", "-1/2")
        .replace("¾", "-3/4")
        .strip()
    )
    mixed = re.fullmatch(r"(\d+)-(\d+)/(\d+)", normalized)
    if mixed:
        return float(mixed.group(1)) + float(mixed.group(2)) / float(mixed.group(3))
    fraction = re.fullmatch(r"(\d+)/(\d+)", normalized)
    if fraction:
        return float(fraction.group(1)) / float(fraction.group(2))
    return float(normalized)


def _statement_target_range(sentence: str | None) -> tuple[float, float] | None:
    if not sentence:
        return None
    number = r"(\d+(?:\.\d+)?(?:[-–—]\d+/\d+)?|\d+/\d+|0)"
    range_match = re.search(
        r"target range for the federal funds rate(?:\s+\w+){0,8}?\s+"
        + number
        + r"\s+(?:to|through|-)\s+"
        + number
        + r"\s*percent",
        sentence,
        flags=re.IGNORECASE,
    )
    if range_match:
        return _rate_number(range_match.group(1)), _rate_number(range_match.group(2))
    single_match = re.search(
        r"(?:federal funds rate target|intended federal funds rate)"
        r"(?:\s+\w+){0,10}?\s+"
        + number
        + r"\s*percent",
        sentence,
        flags=re.IGNORECASE,
    )
    if single_match:
        target = _rate_number(single_match.group(1))
        return target, target
    return None


def _statement_file_path(event_date: date, statement_url: str, content_type: str | None) -> Path:
    suffix = Path(urlparse(statement_url).path).suffix.lower()
    if suffix not in {".htm", ".html", ".pdf"}:
        suffix = ".pdf" if content_type and "pdf" in content_type.lower() else ".html"
    basename = Path(urlparse(statement_url).path.rstrip("/")).stem or "statement"
    safe_basename = re.sub(r"[^A-Za-z0-9_.-]+", "_", basename)
    return (
        RAW_SOURCE_DIRS["federal_reserve"]
        / "statements"
        / f"{event_date.isoformat()}_{safe_basename}{suffix}"
    )


def _availability(
    event_date: date,
    release_time: str | None,
) -> tuple[date, str]:
    if release_time is None:
        return (
            next_us_equity_trading_session(event_date),
            "Exact historical announcement time unavailable; use next US equity trading session.",
        )
    parsed_time = time.fromisoformat(release_time)
    if is_us_equity_trading_session(event_date) and parsed_time < time(16, 0):
        return (
            event_date,
            "Official announcement time is before the 4:00 p.m. New York close; usable in an end-of-day row on the announcement date.",
        )
    return (
        next_us_equity_trading_session(event_date),
        "Official announcement was at/after the close or on a non-trading day; use next US equity trading session.",
    )


def _document_metadata(
    *,
    dataset_name: str,
    data_path: Path,
    source_url: str,
    content: bytes,
    content_type: str | None,
    note: str,
) -> dict[str, Any]:
    metadata = summarize_document(
        dataset_name=dataset_name,
        source="Board of Governors of the Federal Reserve System",
        data_path=data_path,
        request_parameters={"method": "GET", "url": source_url, "timeout_seconds": 60},
        authentication_classification=NO_KEY_REQUIRED,
        credential_environment_variable=None,
        source_library_version=_library_version("requests"),
        note=note,
        license_note=(
            "Federal Reserve Board website materials are treated as public domain "
            "unless a page identifies third-party copyright; source attribution is retained."
        ),
        dataset_status="SOURCE_FAITHFUL_RAW_DOCUMENT",
    )
    metadata["content_type"] = content_type
    metadata["sha256"] = _sha256(content)
    return metadata


def download_federal_reserve_documents(*, refresh: bool = False) -> list[dict[str, Any]]:
    source_dir = RAW_SOURCE_DIRS["federal_reserve"]
    data_path = source_dir / "fomc_decisions.json"
    request_parameters = {
        "current_calendar_url": CURRENT_CALENDAR_URL,
        "historical_url_template": HISTORICAL_URL_TEMPLATE,
        "first_year": FIRST_RELIABLE_PUBLIC_DECISION_YEAR,
        "latest_year": date.today().year,
        "required_cached_policy_series": ["DFEDTAR", "DFEDTARL", "DFEDTARU"],
    }
    cached, metadata = valid_cache(data_path)
    if cached and not refresh:
        return [{**(metadata or {}), "cache_reused": True}]

    try:
        historical_years = list(
            range(FIRST_RELIABLE_PUBLIC_DECISION_YEAR, 2021)
        )
        calendar_urls = [
            HISTORICAL_URL_TEMPLATE.format(year=year)
            for year in historical_years
        ] + [CURRENT_CALENDAR_URL]
        calendar_responses = _fetch_many(calendar_urls, max_workers=6)

        events: list[dict[str, Any]] = []
        for year in historical_years:
            url = HISTORICAL_URL_TEMPLATE.format(year=year)
            source_file = f"calendars/fomc_historical_{year}.html"
            events.extend(
                _historical_events(calendar_responses[url][0], year, source_file)
            )
        events.extend(
            _current_events(
                calendar_responses[CURRENT_CALENDAR_URL][0],
                "fomc_calendars.html",
            )
        )

        unique_events: dict[str, dict[str, Any]] = {}
        for event in sorted(events, key=lambda row: row["announcement_date"]):
            key = event["announcement_date"]
            existing = unique_events.get(key)
            if existing is None:
                unique_events[key] = event
            elif existing.get("statement_url") is None and event.get("statement_url"):
                unique_events[key] = event
            elif existing["event_type"] != event["event_type"]:
                raise RuntimeError(f"Conflicting FOMC event types on {key}")
        events = list(unique_events.values())

        statement_urls = sorted(
            {
                str(event["statement_url"])
                for event in events
                if event.get("statement_url")
            }
        )
        statement_responses = _fetch_many(statement_urls, max_workers=5)

        single = _load_fred_observations("DFEDTAR")
        lower = _load_fred_observations("DFEDTARL")
        upper = _load_fred_observations("DFEDTARU")

        structured_events: list[dict[str, Any]] = []
        source_statement_files: dict[str, tuple[Path, bytes, dict[str, str]]] = {}
        statement_cross_checks = 0
        statement_cross_check_mismatches: list[str] = []
        for event in events:
            event_date = date.fromisoformat(event["announcement_date"])
            statement_url = event.get("statement_url")
            statement_text = ""
            statement_file: Path | None = None
            if statement_url:
                content, headers, _ = statement_responses[str(statement_url)]
                statement_file = _statement_file_path(
                    event_date,
                    str(statement_url),
                    headers.get("content-type"),
                )
                source_statement_files[str(statement_url)] = (
                    statement_file,
                    content,
                    headers,
                )
                if "pdf" not in (headers.get("content-type") or "").lower():
                    statement_text = _strip_html(
                        content.decode("utf-8", errors="replace")
                    )

            release_time, release_timezone = _release_time(statement_text)
            sentence = _policy_sentence(statement_text)
            parsed_statement_target = _statement_target_range(sentence)
            before, after, effective_date = _policy_before_after(
                event_date,
                single=single,
                lower=lower,
                upper=upper,
            )
            change_bps = round(
                100.0 * (float(after["midpoint"]) - float(before["midpoint"])),
                6,
            )
            if change_bps > 0:
                action = "increase"
            elif change_bps < 0:
                action = "decrease"
            else:
                action = "hold"
            availability_date, availability_rule = _availability(
                event_date,
                release_time,
            )

            if parsed_statement_target is not None:
                statement_cross_checks += 1
                if (
                    abs(parsed_statement_target[0] - float(after["lower"])) > 0.011
                    or abs(parsed_statement_target[1] - float(after["upper"])) > 0.011
                ):
                    statement_cross_check_mismatches.append(event_date.isoformat())

            structured_events.append(
                {
                    **event,
                    "announcement_time": release_time,
                    "announcement_timezone": release_timezone,
                    "announcement_time_source": (
                        "official statement page"
                        if release_time
                        else "not published in the parsed official source"
                    ),
                    "model_availability_date": availability_date.isoformat(),
                    "model_availability_rule": availability_rule,
                    "policy_expression": after["policy_expression"],
                    "policy_expression_before": before["policy_expression"],
                    "policy_expression_after": after["policy_expression"],
                    "target_before_lower_pct": float(before["lower"]),
                    "target_before_upper_pct": float(before["upper"]),
                    "target_before_midpoint_pct": float(before["midpoint"]),
                    "target_after_lower_pct": float(after["lower"]),
                    "target_after_upper_pct": float(after["upper"]),
                    "target_after_midpoint_pct": float(after["midpoint"]),
                    "policy_change_bps": change_bps,
                    "decision": action,
                    "is_hold": action == "hold",
                    "fred_effective_observation_date": (
                        effective_date.isoformat() if effective_date else None
                    ),
                    "numeric_policy_source_before": (
                        "FRED DFEDTAR"
                        if before["policy_expression"] == "single_target"
                        else "FRED DFEDTARL and DFEDTARU"
                    ),
                    "numeric_policy_source_after": (
                        "FRED DFEDTAR"
                        if after["policy_expression"] == "single_target"
                        else "FRED DFEDTARL and DFEDTARU"
                    ),
                    "official_policy_sentence": sentence,
                    "statement_target_cross_check": (
                        {
                            "lower_pct": parsed_statement_target[0],
                            "upper_pct": parsed_statement_target[1],
                            "matches_fred": event_date.isoformat()
                            not in statement_cross_check_mismatches,
                        }
                        if parsed_statement_target is not None
                        else None
                    ),
                    "statement_source_file": (
                        str(statement_file.relative_to(source_dir)).replace("\\", "/")
                        if statement_file
                        else None
                    ),
                }
            )

        structured_events.sort(key=lambda row: row["announcement_date"])
        event_dates = [row["announcement_date"] for row in structured_events]
        if event_dates != sorted(event_dates) or len(event_dates) != len(set(event_dates)):
            raise RuntimeError("FOMC announcement dates are not ordered and unique")
        if not any(row["decision"] == "hold" for row in structured_events):
            raise RuntimeError("No unchanged-rate FOMC decisions were parsed")
        if not any(row["event_type"] == "unscheduled" for row in structured_events):
            raise RuntimeError("No applicable unscheduled FOMC decisions were parsed")
        if statement_cross_check_mismatches:
            raise RuntimeError(
                "Official statement targets disagree with cached FRED policy levels on: "
                + ", ".join(statement_cross_check_mismatches)
            )
        for year in range(2013, min(date.today().year, 2025) + 1):
            scheduled_count = sum(
                row["event_type"] == "scheduled"
                and row["announcement_date"].startswith(f"{year}-")
                for row in structured_events
            )
            minimum_expected = 7 if year == 2020 else 8
            if scheduled_count < minimum_expected:
                raise RuntimeError(
                    f"Only {scheduled_count} scheduled FOMC decisions were parsed for {year}"
                )

        counts = {
            "scheduled": sum(
                row["event_type"] == "scheduled" for row in structured_events
            ),
            "unscheduled": sum(
                row["event_type"] == "unscheduled" for row in structured_events
            ),
            "increase": sum(row["decision"] == "increase" for row in structured_events),
            "decrease": sum(row["decision"] == "decrease" for row in structured_events),
            "hold": sum(row["decision"] == "hold" for row in structured_events),
            "exact_announcement_time": sum(
                row["announcement_time"] is not None for row in structured_events
            ),
            "conservative_next_session": sum(
                "next US equity trading session" in row["model_availability_rule"]
                for row in structured_events
            ),
            "statement_target_cross_checks": statement_cross_checks,
        }
        payload = {
            "dataset_name": "fomc_policy_decisions",
            "parser_version": PARSER_VERSION,
            "source": "Board of Governors of the Federal Reserve System",
            "numeric_policy_sources": {
                "before_2008_12_16": "FRED DFEDTAR single target",
                "from_2008_12_16": (
                    "FRED DFEDTARL/DFEDTARU target range; midpoint used for basis-point change"
                ),
            },
            "coverage_policy": (
                "1994 is the first year of reliable public FOMC decision announcements. "
                "Every scheduled meeting on the official historical pages is retained, "
                "including no-change meetings. Conference calls/unscheduled panels are "
                "retained when an official statement is linked."
            ),
            "timing_policy": (
                "Use the official release time when available. For an end-of-day market "
                "row, a pre-close release is available that day; an at/after-close or "
                "non-trading-day release is available next session. When historical "
                "time is unavailable, always use the next US equity trading session."
            ),
            "counts": counts,
            "events": structured_events,
        }

        # Parsing and validation are complete. Only now replace source-faithful caches.
        source_dir.mkdir(parents=True, exist_ok=True)
        source_manifest: list[dict[str, Any]] = []
        for year in historical_years:
            url = HISTORICAL_URL_TEMPLATE.format(year=year)
            content, headers, _ = calendar_responses[url]
            path = source_dir / "calendars" / f"fomc_historical_{year}.html"
            write_bytes_atomic(path, content)
            source_manifest.append(
                {
                    "kind": "historical_calendar",
                    "year": year,
                    "source_url": url,
                    "file": str(path.relative_to(source_dir)).replace("\\", "/"),
                    "sha256": _sha256(content),
                    "content_type": headers.get("content-type"),
                }
            )

        current_content, current_headers, _ = calendar_responses[CURRENT_CALENDAR_URL]
        current_path = source_dir / "fomc_calendars.html"
        write_bytes_atomic(current_path, current_content)
        current_metadata = _document_metadata(
            dataset_name="fomc_current_calendar_source",
            data_path=current_path,
            source_url=CURRENT_CALENDAR_URL,
            content=current_content,
            content_type=current_headers.get("content-type"),
            note=(
                "Official current FOMC calendar retained as source-faithful input to "
                "the separate structured fomc_decisions.json parser output."
            ),
        )
        persist_metadata(current_path, current_metadata)
        source_manifest.append(
            {
                "kind": "current_calendar",
                "source_url": CURRENT_CALENDAR_URL,
                "file": str(current_path.relative_to(source_dir)).replace("\\", "/"),
                "sha256": _sha256(current_content),
                "content_type": current_headers.get("content-type"),
            }
        )

        for statement_url, (path, content, headers) in source_statement_files.items():
            write_bytes_atomic(path, content)
            source_manifest.append(
                {
                    "kind": "statement",
                    "source_url": statement_url,
                    "file": str(path.relative_to(source_dir)).replace("\\", "/"),
                    "sha256": _sha256(content),
                    "content_type": headers.get("content-type"),
                }
            )

        manifest_path = source_dir / "fomc_source_manifest.json"
        manifest_payload = {
            "dataset_name": "fomc_source_manifest",
            "parser_version": PARSER_VERSION,
            "records": sorted(source_manifest, key=lambda row: row["file"]),
        }
        write_json_atomic(manifest_path, manifest_payload)
        manifest_metadata = summarize_document(
            dataset_name="fomc_source_manifest",
            source="Board of Governors of the Federal Reserve System",
            data_path=manifest_path,
            request_parameters=request_parameters,
            authentication_classification=NO_KEY_REQUIRED,
            credential_environment_variable=None,
            source_library_version=_library_version("requests"),
            note=(
                "Hash manifest for every source-faithful calendar and statement file "
                "used by the deterministic FOMC decision parser."
            ),
            license_note=(
                "Federal Reserve Board website materials are treated as public domain "
                "unless a page identifies third-party copyright; attribution is retained."
            ),
            dataset_status="SOURCE_MANIFEST",
        )
        manifest_metadata["row_count"] = len(source_manifest)
        manifest_metadata["parser_version"] = PARSER_VERSION
        persist_metadata(manifest_path, manifest_metadata)

        write_json_atomic(data_path, payload)
        frame = pd.DataFrame(structured_events)
        frame["announcement_date"] = pd.to_datetime(frame["announcement_date"])
        metadata = summarize_tabular(
            frame,
            dataset_name="fomc_policy_decisions",
            source="Board of Governors of the Federal Reserve System",
            data_path=data_path,
            frequency="Scheduled and applicable unscheduled FOMC decision events",
            units="Federal funds target/range percent and midpoint change in basis points",
            request_parameters=request_parameters,
            authentication_classification=NO_KEY_REQUIRED,
            credential_environment_variable=None,
            source_library_version=_library_version("requests"),
            date_column="announcement_date",
            vintage_parameters={
                "calendar": "official historical/current FOMC pages",
                "numeric_policy_values": "latest cached FRED observations",
            },
            license_note=(
                "Federal Reserve Board website materials are treated as public domain "
                "unless a page identifies third-party copyright; attribution is retained."
            ),
            source_notes=(
                "Structured independently from policy-rate changes, so scheduled holds "
                "remain present. Exact historical times are used only when the official "
                "statement publishes them."
            ),
        )
        metadata.update(
            {
                "parser_version": PARSER_VERSION,
                "scheduled_decision_count": counts["scheduled"],
                "unscheduled_decision_count": counts["unscheduled"],
                "increase_count": counts["increase"],
                "decrease_count": counts["decrease"],
                "hold_count": counts["hold"],
                "exact_announcement_time_count": counts[
                    "exact_announcement_time"
                ],
                "conservative_next_session_count": counts[
                    "conservative_next_session"
                ],
                "statement_target_cross_check_count": statement_cross_checks,
                "statement_target_cross_check_mismatch_count": 0,
                "minimum_model_availability_date": min(
                    row["model_availability_date"] for row in structured_events
                ),
                "maximum_model_availability_date": max(
                    row["model_availability_date"] for row in structured_events
                ),
                "source_manifest_file": str(manifest_path),
                "source_manifest_record_count": len(source_manifest),
                "cache_reused": False,
            }
        )
        structured_metadata = persist_metadata(data_path, metadata)
        return [current_metadata, manifest_metadata, structured_metadata]
    except Exception as exc:
        failure_path = record_failure(
            data_path,
            dataset_name="fomc_policy_decisions",
            source="Board of Governors of the Federal Reserve System",
            authentication_classification=NO_KEY_REQUIRED,
            credential_environment_variable=None,
            request_parameters=safe_request_parameters(request_parameters),
            error=str(exc),
        )
        return [
            {
                "dataset_name": "fomc_policy_decisions",
                "source": "Board of Governors of the Federal Reserve System",
                "retrieval_status": "SOURCE_UNAVAILABLE",
                "error_details": str(exc),
                "failure_file": str(failure_path),
                "file_path": str(data_path),
            }
        ]
