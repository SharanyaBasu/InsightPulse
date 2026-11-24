from __future__ import annotations

import hashlib
import html
import io
import json
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from importlib.metadata import version
from pathlib import Path
from typing import Any
from urllib.parse import urljoin
from xml.etree import ElementTree

import pandas as pd
import requests

from ..cache import (
    record_failure,
    valid_cache,
    write_bytes_atomic,
    write_json_atomic,
)
from ..config import NO_KEY_REQUIRED, RAW_SOURCE_DIRS, safe_request_parameters
from ..metadata import persist_metadata, summarize_document, summarize_tabular
from ..trading_calendar import next_us_equity_trading_session


LANDING_URL = "https://www.kansascityfed.org/surveys/manufacturing-survey/"
METHODOLOGY_URL = (
    "https://www.kansascityfed.org/surveys/manufacturing-survey/"
    "manufacturing-survey-methodology/"
)
DISCLAIMER_URL = "https://www.kansascityfed.org/about-us/disclaimer/"
ARCHIVE_URL_TEMPLATE = (
    "https://www.kansascityfed.org/surveys/manufacturing-survey-archive/"
    "{year}-manufacturing-surveys/"
)
FIRST_OBSERVATION_YEAR = 2001
FIRST_STRUCTURED_RELEASE_YEAR = 2013
PARSER_VERSION = "insightpulse-kc-manufacturing-v1"
_KNOWN_OFFICIAL_ARCHIVE_OMISSIONS = {
    (2024, 6): {
        "release_date": "2024-06-27",
        "release_page_url": (
            "https://www.kansascityfed.org/surveys/manufacturing-survey/"
            "tenth-district-manufacturing-activity-fell-moderately-in-june/"
        ),
        "official_archive_entry_title": (
            "Tenth District Manufacturing Activity Fell Moderately in June"
        ),
        "release_source_kind": "official_release_page_missing_from_yearly_archive",
    }
}

_MONTH_NUMBERS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}
_DATE_PATTERN = re.compile(
    r"\b("
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
    r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|"
    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
    r")\.?\s+(\d{1,2})(?:,\s+(\d{4}))?\b",
    re.IGNORECASE,
)
_ANCHOR_PATTERN = re.compile(
    r"<a\b[^>]*\bhref=[\"'](?P<href>[^\"']+)[\"'][^>]*>"
    r"(?P<label>.*?)</a>",
    re.IGNORECASE | re.DOTALL,
)


def _library_version(package: str) -> str | None:
    try:
        return version(package)
    except Exception:
        return None


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


def _fetch_archives(years: list[int]) -> dict[int, tuple[bytes, dict[str, str], int]]:
    results: dict[int, tuple[bytes, dict[str, str], int]] = {}
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(_get, ARCHIVE_URL_TEMPLATE.format(year=year)): year
            for year in years
        }
        for future in as_completed(futures):
            year = futures[future]
            results[year] = future.result()
    return results


def _discover_workbook_url(landing_html: str) -> str:
    candidates: list[str] = []
    for match in re.finditer(r"\bhref=[\"']([^\"']+\.xlsx(?:\?[^\"']*)?)[\"']", landing_html, re.I):
        href = html.unescape(match.group(1))
        if "histor" in href.lower() or "manufact" in href.lower():
            candidates.append(urljoin(LANDING_URL, href))
    if not candidates:
        raise RuntimeError("No historical manufacturing workbook link was found on the official page")
    return candidates[0]


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    return [
        "".join(node.text or "" for node in item.findall(".//x:t", namespace))
        for item in root.findall("x:si", namespace)
    ]


def _first_sheet_path(archive: zipfile.ZipFile) -> str:
    namespace = {
        "x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "p": "http://schemas.openxmlformats.org/package/2006/relationships",
    }
    workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
    sheet = workbook.find("x:sheets/x:sheet", namespace)
    if sheet is None:
        raise RuntimeError("The official workbook contains no worksheet")
    relationship_id = sheet.attrib[
        "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
    ]
    relationships = ElementTree.fromstring(
        archive.read("xl/_rels/workbook.xml.rels")
    )
    for relationship in relationships.findall("p:Relationship", namespace):
        if relationship.attrib.get("Id") == relationship_id:
            target = relationship.attrib["Target"].lstrip("/")
            if target.startswith("xl/"):
                return target
            return f"xl/{target}"
    raise RuntimeError("The first worksheet relationship could not be resolved")


def _cell_column(reference: str) -> str:
    match = re.match(r"([A-Z]+)", reference)
    return match.group(1) if match else ""


def _parse_workbook(content: bytes) -> list[dict[str, Any]]:
    if not content.startswith(b"PK"):
        raise RuntimeError("Kansas City Fed historical workbook is not a valid XLSX/ZIP file")
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        strings = _shared_strings(archive)
        sheet_path = _first_sheet_path(archive)
        root = ElementTree.fromstring(archive.read(sheet_path))

    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rows: dict[int, dict[str, Any]] = {}
    for row in root.findall(".//x:sheetData/x:row", namespace):
        row_number = int(row.attrib["r"])
        values: dict[str, Any] = {}
        for cell in row.findall("x:c", namespace):
            column = _cell_column(cell.attrib.get("r", ""))
            value_node = cell.find("x:v", namespace)
            if value_node is None:
                inline = cell.find(".//x:t", namespace)
                value: Any = inline.text if inline is not None else None
            else:
                value = value_node.text
                if cell.attrib.get("t") == "s" and value is not None:
                    value = strings[int(value)]
            values[column] = value
        rows[row_number] = values

    composite_row_number = None
    for row_number, values in rows.items():
        if str(values.get("A", "")).strip().lower() == "composite index":
            composite_row_number = row_number
            break
    if composite_row_number is None:
        raise RuntimeError("Composite Index row was not found in the official workbook")

    date_row_number = None
    for row_number in range(composite_row_number - 1, 0, -1):
        if len(rows.get(row_number, {})) >= 10:
            date_row_number = row_number
            break
    if date_row_number is None:
        raise RuntimeError("Monthly date row was not found in the official workbook")

    observations: list[dict[str, Any]] = []
    for column, raw_value in rows[composite_row_number].items():
        if column == "A" or raw_value in (None, ""):
            continue
        raw_date = rows[date_row_number].get(column)
        if raw_date in (None, ""):
            continue
        try:
            serial = float(raw_date)
            observation_month = date(1899, 12, 30) + timedelta(days=int(serial))
        except (TypeError, ValueError):
            parsed = pd.to_datetime(raw_date, errors="coerce")
            if pd.isna(parsed):
                continue
            observation_month = parsed.date()
        try:
            composite = float(raw_value)
        except (TypeError, ValueError):
            continue
        observations.append(
            {
                "observation_month": observation_month.replace(day=1),
                "kc_composite_diffusion_points": composite,
            }
        )

    observations.sort(key=lambda row: row["observation_month"])
    if not observations:
        raise RuntimeError("No composite observations were parsed from the official workbook")
    return observations


def _parse_archive_release_dates(content: bytes, year: int) -> dict[tuple[int, int], dict[str, str]]:
    text = content.decode("utf-8", errors="replace")
    records: dict[tuple[int, int], dict[str, str]] = {}
    for anchor in _ANCHOR_PATTERN.finditer(text):
        label = _strip_html(anchor.group("label"))
        if "manufactur" not in label.lower():
            continue
        following = _strip_html(text[anchor.end() : anchor.end() + 700])
        date_match = _DATE_PATTERN.search(following)
        if date_match is None:
            continue
        release_year = int(date_match.group(3) or year)
        if release_year != year:
            continue
        month_key = date_match.group(1).lower().rstrip(".")
        month = _MONTH_NUMBERS[month_key]
        release_day = int(date_match.group(2))
        release_date = date(release_year, month, release_day)
        key = (release_year, month)
        candidate = {
            "release_date": release_date.isoformat(),
            "release_page_url": urljoin(
                ARCHIVE_URL_TEMPLATE.format(year=year),
                html.unescape(anchor.group("href")),
            ),
            "official_archive_entry_title": label,
            "release_source_kind": "official_yearly_archive_entry",
        }
        existing = records.get(key)
        if existing and existing["release_date"] != candidate["release_date"]:
            raise RuntimeError(
                f"Multiple release dates were parsed for {year}-{month:02d}: "
                f"{existing['release_date']} and {candidate['release_date']}"
            )
        records[key] = candidate
    return records


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _document_metadata(
    *,
    dataset_name: str,
    data_path: Path,
    source_url: str,
    note: str,
    license_note: str,
    content: bytes,
    content_type: str | None,
) -> dict[str, Any]:
    metadata = summarize_document(
        dataset_name=dataset_name,
        source="Federal Reserve Bank of Kansas City",
        data_path=data_path,
        request_parameters={"method": "GET", "url": source_url, "timeout_seconds": 60},
        authentication_classification=NO_KEY_REQUIRED,
        credential_environment_variable=None,
        source_library_version=_library_version("requests"),
        note=note,
        license_note=license_note,
        dataset_status="SOURCE_FAITHFUL_RAW_DOCUMENT",
    )
    metadata["content_type"] = content_type
    metadata["sha256"] = _sha256(content)
    return metadata


def _published_ism_comparisons(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = {
        row["observation_month"].isoformat(): row
        for row in observations
    }
    references = [
        (
            "2011-04-01",
            60.4,
            "https://www.prnewswire.com/news-releases/institute-for-supply-management-makes-annual-adjustments-to-seasonal-factors-for-ism-manufacturing-pmi-and-diffusion-indexes-and-ism-non-manufacturing-nmi-and-diffusion-indexes-242576861.html",
        ),
        (
            "2019-03-01",
            55.3,
            "https://www.ismworld.org/supply-management-news-and-reports/news-publications/inside-supply-management-magazine/blog/2019-04/rob-roundup-march-pmi/",
        ),
        (
            "2022-10-01",
            50.2,
            "https://www.ismworld.org/supply-management-news-and-reports/news-publications/inside-supply-management-magazine/blog/2022/2022-11/rob-roundup-october-2022-manufacturing-pmi/",
        ),
        (
            "2025-08-01",
            48.7,
            "https://www.ismworld.org/globalassets/pub/research-and-surveys/rob/pmi/haay202508pmi.pdf",
        ),
        (
            "2025-09-01",
            49.1,
            "https://www.ismworld.org/globalassets/pub/research-and-surveys/rob/pmi/de4d202509pmi.pdf",
        ),
        (
            "2025-10-01",
            48.7,
            "https://www.ismworld.org/globalassets/pub/research-and-surveys/rob/pmi/boo202510pmi.pdf",
        ),
        (
            "2025-11-01",
            48.2,
            "https://www.ismworld.org/supply-management-news-and-reports/news-publications/inside-supply-management-magazine/blog/2025/2025-12/ism-pmi-reports-roundup-november-2025-manufacturing/",
        ),
        (
            "2025-12-01",
            47.9,
            "https://www.ismworld.org/globalassets/pub/research-and-surveys/rob/pmi/humpt202512pmi.pdf",
        ),
    ]
    comparisons: list[dict[str, Any]] = []
    for observation_month, ism_value, source_url in references:
        selected_row = selected.get(observation_month)
        if selected_row is None:
            continue
        kc_raw = float(selected_row["kc_composite_diffusion_points"])
        kc_pmi_style = 50.0 + 0.5 * kc_raw
        comparisons.append(
            {
                "observation_month": observation_month,
                "official_ism_manufacturing_pmi": ism_value,
                "official_ism_source_url": source_url,
                "kc_composite_diffusion_points": kc_raw,
                "kc_pmi_style_level": kc_pmi_style,
                "same_expansion_contraction_direction": (
                    (ism_value >= 50.0) == (kc_pmi_style >= 50.0)
                ),
                "interpretation": (
                    "Scale/direction comparison only. The Kansas City series is a "
                    "regional proxy and is not expected to equal the national ISM series."
                ),
            }
        )
    return comparisons


def _trusted_dbnomics_overlap_validation(
    structured_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    path = RAW_SOURCE_DIRS["ism"] / "dbnomics_ism_pmi.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    docs = ((payload.get("series") or {}).get("docs") or [])
    if not docs:
        return {}
    document = docs[0]
    dates = document.get("period_start_day") or document.get("period") or []
    values = document.get("value") or []
    trusted_ism = pd.DataFrame(
        {
            "observation_month": pd.to_datetime(dates, errors="coerce"),
            "official_ism_pmi": pd.to_numeric(values, errors="coerce"),
        }
    )
    trusted_ism = trusted_ism[
        trusted_ism["observation_month"] <= pd.Timestamp("2025-08-01")
    ].dropna()
    proxy = pd.DataFrame(structured_rows)
    proxy["observation_month"] = pd.to_datetime(proxy["observation_month"])
    overlap = trusted_ism.merge(
        proxy[["observation_month", "pmi_style_level"]],
        on="observation_month",
        how="inner",
    )
    if overlap.empty:
        return {}
    correlation = float(
        overlap["official_ism_pmi"].corr(overlap["pmi_style_level"])
    )
    mean_absolute_difference = float(
        (overlap["official_ism_pmi"] - overlap["pmi_style_level"]).abs().mean()
    )
    same_side = int(
        (
            (overlap["official_ism_pmi"] >= 50.0)
            == (overlap["pmi_style_level"] >= 50.0)
        ).sum()
    )
    return {
        "purpose": (
            "Validation only; the rejected DBnomics cache is not used as a model input."
        ),
        "trusted_overlap_rule": (
            "Use DBnomics/ISM observations only through 2025-08, before the four "
            "confirmed corrupted values."
        ),
        "overlap_observation_count": int(len(overlap)),
        "minimum_overlap_month": overlap["observation_month"].min().date().isoformat(),
        "maximum_overlap_month": overlap["observation_month"].max().date().isoformat(),
        "pearson_level_correlation": round(correlation, 6),
        "mean_absolute_level_difference_points": round(
            mean_absolute_difference, 6
        ),
        "same_expansion_contraction_side_count": same_side,
        "same_side_share": round(same_side / len(overlap), 6),
    }


def download_kansas_city_fed_manufacturing(
    *,
    refresh: bool = False,
) -> list[dict[str, Any]]:
    source_dir = RAW_SOURCE_DIRS["kansas_city_fed"]
    data_path = source_dir / "manufacturing_pmi_proxy_events.json"
    request_parameters = {
        "landing_url": LANDING_URL,
        "methodology_url": METHODOLOGY_URL,
        "archive_url_template": ARCHIVE_URL_TEMPLATE,
        "first_year": FIRST_OBSERVATION_YEAR,
        "first_structured_release_year": FIRST_STRUCTURED_RELEASE_YEAR,
        "latest_year": date.today().year,
    }
    cached, metadata = valid_cache(data_path)
    if cached and not refresh:
        return [{**(metadata or {}), "cache_reused": True}]

    license_note = (
        "The Kansas City Fed disclaimer permits reproduction without quantity limit "
        "when not for private gain. Local, attributed student research caching and "
        "model training are accepted for this MVP. Raw files remain Git-ignored and "
        "must not be redistributed; production or private-gain use requires a fresh "
        "terms/licensing review."
    )
    try:
        landing, landing_headers, _ = _get(LANDING_URL)
        methodology, methodology_headers, _ = _get(METHODOLOGY_URL)
        disclaimer, disclaimer_headers, _ = _get(DISCLAIMER_URL)
        workbook_url = _discover_workbook_url(landing.decode("utf-8", errors="replace"))
        workbook, workbook_headers, _ = _get(workbook_url)
        if not workbook.startswith(b"PK"):
            raise RuntimeError("Official manufacturing history did not have an XLSX signature")

        years = list(range(FIRST_STRUCTURED_RELEASE_YEAR, date.today().year + 1))
        archives = _fetch_archives(years)
        observations = _parse_workbook(workbook)

        release_records: dict[tuple[int, int], dict[str, str]] = {}
        for year in years:
            parsed = _parse_archive_release_dates(archives[year][0], year)
            required_months = {
                row["observation_month"].month
                for row in observations
                if row["observation_month"].year == year
            }
            if required_months - {month for _, month in parsed}:
                # A small number of concurrent archive requests can return a soft,
                # incomplete page. Retry that year sequentially before rejecting it.
                archives[year] = _get(ARCHIVE_URL_TEMPLATE.format(year=year))
                parsed = _parse_archive_release_dates(archives[year][0], year)
            release_records.update(parsed)
        release_records.update(_KNOWN_OFFICIAL_ARCHIVE_OMISSIONS)

        supplemental_release_pages = {
            key: _get(record["release_page_url"])
            for key, record in _KNOWN_OFFICIAL_ARCHIVE_OMISSIONS.items()
        }
        for (year, month), (content, _, _) in supplemental_release_pages.items():
            text = _strip_html(content.decode("utf-8", errors="replace"))
            expected_date = date.fromisoformat(
                _KNOWN_OFFICIAL_ARCHIVE_OMISSIONS[(year, month)]["release_date"]
            )
            if (
                expected_date.strftime("%B") not in text
                or str(expected_date.day) not in text
                or str(expected_date.year) not in text
            ):
                raise RuntimeError(
                    f"Supplemental official release page did not validate for {year}-{month:02d}"
                )

        observation_keys = {
            (row["observation_month"].year, row["observation_month"].month)
            for row in observations
            if row["observation_month"].year >= FIRST_STRUCTURED_RELEASE_YEAR
        }
        missing_release_dates = sorted(observation_keys - set(release_records))
        if missing_release_dates:
            raise RuntimeError(
                "Official release dates were not parsed for: "
                + ", ".join(f"{year}-{month:02d}" for year, month in missing_release_dates)
            )

        structured_rows: list[dict[str, Any]] = []
        for row in observations:
            observation_month = row["observation_month"]
            if observation_month.year < FIRST_STRUCTURED_RELEASE_YEAR:
                continue
            release = release_records[(observation_month.year, observation_month.month)]
            release_date = date.fromisoformat(release["release_date"])
            if release_date < observation_month:
                raise RuntimeError(
                    f"Release date precedes observation month for {observation_month}"
                )
            raw_composite = float(row["kc_composite_diffusion_points"])
            if not -100.0 <= raw_composite <= 100.0:
                raise RuntimeError(
                    f"Implausible Kansas City composite value {raw_composite} "
                    f"for {observation_month}"
                )
            structured_rows.append(
                {
                    "observation_month": observation_month.isoformat(),
                    "release_date": release_date.isoformat(),
                    "release_time": None,
                    "release_timezone": None,
                    "model_availability_date": next_us_equity_trading_session(
                        release_date
                    ).isoformat(),
                    "model_availability_rule": (
                        "Exact historical release time is unavailable; use beginning "
                        "with the next US equity trading session."
                    ),
                    "kc_composite_diffusion_points": raw_composite,
                    "kc_composite_neutral_value": 0.0,
                    "pmi_style_level": 50.0 + 0.5 * raw_composite,
                    "pmi_style_neutral_value": 50.0,
                    "pmi_style_mapping": (
                        "50 + 0.5 * Kansas City composite diffusion points. This maps "
                        "the increase-minus-decrease convention to a 50-centered "
                        "PMI-style scale; it does not make the regional proxy the "
                        "national ISM series."
                    ),
                    "release_page_url": release["release_page_url"],
                    "official_archive_entry_title": release[
                        "official_archive_entry_title"
                    ],
                    "release_source_kind": release["release_source_kind"],
                }
            )

        dates = [row["observation_month"] for row in structured_rows]
        if dates != sorted(dates) or len(dates) != len(set(dates)):
            raise RuntimeError("Observation months are not strictly ordered and unique")
        expected_months = pd.period_range(dates[0], dates[-1], freq="M")
        if len(expected_months) != len(structured_rows):
            raise RuntimeError("Unexpected missing months exist in the official workbook")
        if dates[0] > "2013-01-01" or dates[-1] < "2026-01-01":
            raise RuntimeError(
                f"Historical coverage is unexpectedly short: {dates[0]} through {dates[-1]}"
            )

        comparisons = _published_ism_comparisons(observations)
        if len(comparisons) < 6:
            raise RuntimeError("Too few published ISM comparison observations were available")
        overlap_validation = _trusted_dbnomics_overlap_validation(structured_rows)
        if (
            overlap_validation
            and overlap_validation["overlap_observation_count"] < 60
        ):
            raise RuntimeError(
                "The trustworthy DBnomics/ISM validation overlap is unexpectedly short"
            )

        payload = {
            "dataset_name": "kansas_city_fed_manufacturing_pmi_proxy_events",
            "parser_version": PARSER_VERSION,
            "source": "Federal Reserve Bank of Kansas City",
            "selection_status": "PRIMARY_MVP_PROXY_READY_WITH_LIMITATIONS",
            "contract_input": "pmi_change_points",
            "historical_proxy_feature_name": (
                "kc_fed_manufacturing_pmi_proxy_change_points"
            ),
            "economic_concept": (
                "Tenth District manufacturing composite diffusion index, constructed "
                "from the same five equally weighted components used by the national "
                "ISM Manufacturing PMI: production, new orders, employment, supplier "
                "delivery time, and raw-material inventories."
            ),
            "geographic_scope": (
                "Regional: the seven-state Tenth Federal Reserve District. It is not "
                "the national ISM Manufacturing PMI."
            ),
            "frequency": "Monthly",
            "source_workbook_value_coverage": {
                "minimum_observation_month": observations[0][
                    "observation_month"
                ].isoformat(),
                "maximum_observation_month": observations[-1][
                    "observation_month"
                ].isoformat(),
                "observation_count": len(observations),
            },
            "point_in_time_usable_coverage": {
                "minimum_observation_month": structured_rows[0][
                    "observation_month"
                ],
                "maximum_observation_month": structured_rows[-1][
                    "observation_month"
                ],
                "observation_count": len(structured_rows),
            },
            "raw_units": (
                "Kansas City composite diffusion points: percent reporting increase "
                "minus percent reporting decrease; neutral 0"
            ),
            "pmi_style_units": "PMI-style index points; neutral 50",
            "revision_policy": (
                "The official historical workbook contains the latest revised, "
                "seasonally adjusted history. Kansas City Fed recomputes seasonal "
                "factors annually, so values can differ modestly from first releases. "
                "Release dates are point-in-time, but this cache is not a vintage-value archive."
            ),
            "availability_policy": (
                "Official archive publication date plus a conservative next-US-equity-"
                "trading-session rule because exact historical release times are unavailable."
            ),
            "licensing_policy": license_note,
            "source_urls": {
                "landing": LANDING_URL,
                "methodology": METHODOLOGY_URL,
                "disclaimer": DISCLAIMER_URL,
                "workbook": workbook_url,
                "archive_template": ARCHIVE_URL_TEMPLATE,
            },
            "published_ism_scale_and_direction_comparisons": comparisons,
            "trusted_dbnomics_ism_overlap_validation": overlap_validation,
            "observations": structured_rows,
        }

        # All downloads and parsing have succeeded. Only now replace the valid cache.
        source_dir.mkdir(parents=True, exist_ok=True)
        source_files = [
            (
                source_dir / "manufacturing_survey.html",
                landing,
                LANDING_URL,
                landing_headers,
                "Official survey landing page and workbook link.",
            ),
            (
                source_dir / "manufacturing_survey_methodology.html",
                methodology,
                METHODOLOGY_URL,
                methodology_headers,
                "Official construction and seasonal-adjustment methodology.",
            ),
            (
                source_dir / "kansas_city_fed_disclaimer.html",
                disclaimer,
                DISCLAIMER_URL,
                disclaimer_headers,
                "Official reproduction and website-use terms.",
            ),
            (
                source_dir / "manufacturing_survey_historical.xlsx",
                workbook,
                workbook_url,
                workbook_headers,
                "Official source-faithful historical manufacturing survey workbook.",
            ),
        ]
        for path, content, url, headers, note in source_files:
            write_bytes_atomic(path, content)
            document_metadata = _document_metadata(
                dataset_name=f"kc_fed_{path.stem}",
                data_path=path,
                source_url=url,
                note=note,
                license_note=license_note,
                content=content,
                content_type=headers.get("content-type"),
            )
            persist_metadata(path, document_metadata)

        archive_hashes: dict[str, str] = {}
        for year in years:
            content, headers, _ = archives[year]
            archive_path = source_dir / "archives" / f"manufacturing_surveys_{year}.html"
            write_bytes_atomic(archive_path, content)
            archive_hashes[str(year)] = _sha256(content)
        payload["archive_source_sha256"] = archive_hashes

        for (year, month), (content, headers, _) in supplemental_release_pages.items():
            record = _KNOWN_OFFICIAL_ARCHIVE_OMISSIONS[(year, month)]
            supplemental_path = (
                source_dir
                / "supplemental_release_pages"
                / f"manufacturing_release_{year}_{month:02d}.html"
            )
            write_bytes_atomic(supplemental_path, content)
            persist_metadata(
                supplemental_path,
                _document_metadata(
                    dataset_name=f"kc_fed_manufacturing_release_{year}_{month:02d}",
                    data_path=supplemental_path,
                    source_url=record["release_page_url"],
                    note=(
                        "Official release page retained because the corresponding "
                        "entry is omitted from the official yearly archive page."
                    ),
                    license_note=license_note,
                    content=content,
                    content_type=headers.get("content-type"),
                ),
            )

        write_json_atomic(data_path, payload)
        frame = pd.DataFrame(structured_rows)
        frame["observation_month"] = pd.to_datetime(frame["observation_month"])
        metadata = summarize_tabular(
            frame,
            dataset_name="kansas_city_fed_manufacturing_pmi_proxy_events",
            source="Federal Reserve Bank of Kansas City",
            data_path=data_path,
            frequency="Monthly",
            units=(
                "Raw regional composite diffusion points (neutral 0) and deterministic "
                "PMI-style level (neutral 50)"
            ),
            request_parameters={
                **request_parameters,
                "workbook_url": workbook_url,
            },
            authentication_classification=NO_KEY_REQUIRED,
            credential_environment_variable=None,
            source_library_version=_library_version("requests"),
            date_column="observation_month",
            vintage_parameters={
                "value_vintage": "latest revised official workbook",
                "release_dates": "official archive publication dates",
            },
            license_note=license_note,
            source_notes=(
                "Accepted MVP proxy with limitations. Regional Tenth District scope "
                "replaces unavailable/restricted national ISM history without being "
                "silently represented as the national ISM series."
            ),
        )
        metadata.update(
            {
                "selection_status": "PRIMARY_MVP_PROXY_READY_WITH_LIMITATIONS",
                "contract_input": "pmi_change_points",
                "historical_proxy_feature_name": (
                    "kc_fed_manufacturing_pmi_proxy_change_points"
                ),
                "release_date_count": len(structured_rows),
                "minimum_release_date": min(
                    row["release_date"] for row in structured_rows
                ),
                "maximum_release_date": max(
                    row["release_date"] for row in structured_rows
                ),
                "minimum_model_availability_date": min(
                    row["model_availability_date"] for row in structured_rows
                ),
                "maximum_model_availability_date": max(
                    row["model_availability_date"] for row in structured_rows
                ),
                "exact_release_time_count": 0,
                "next_trading_session_rule_count": len(structured_rows),
                "missing_month_count": 0,
                "published_ism_comparison_count": len(comparisons),
                "same_direction_comparison_count": sum(
                    bool(row["same_expansion_contraction_direction"])
                    for row in comparisons
                ),
                "trusted_dbnomics_overlap_validation": overlap_validation,
                "parser_version": PARSER_VERSION,
                "source_file_sha256": {
                    "workbook": _sha256(workbook),
                    "landing": _sha256(landing),
                    "methodology": _sha256(methodology),
                    "disclaimer": _sha256(disclaimer),
                },
                "cache_reused": False,
            }
        )
        return [persist_metadata(data_path, metadata)]
    except Exception as exc:
        failure_path = record_failure(
            data_path,
            dataset_name="kansas_city_fed_manufacturing_pmi_proxy_events",
            source="Federal Reserve Bank of Kansas City",
            authentication_classification=NO_KEY_REQUIRED,
            credential_environment_variable=None,
            request_parameters=safe_request_parameters(request_parameters),
            error=str(exc),
        )
        return [
            {
                "dataset_name": "kansas_city_fed_manufacturing_pmi_proxy_events",
                "source": "Federal Reserve Bank of Kansas City",
                "retrieval_status": "SOURCE_UNAVAILABLE",
                "error_details": str(exc),
                "failure_file": str(failure_path),
                "file_path": str(data_path),
            }
        ]
