from __future__ import annotations

from datetime import date
import json
from importlib.metadata import version
from typing import Any

import pandas as pd
import requests

from ..cache import (
    record_failure,
    safe_name,
    valid_cache,
    write_json_atomic,
)
from ..config import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    NO_KEY_REQUIRED,
    RAW_SOURCE_DIRS,
    safe_request_parameters,
)
from ..metadata import persist_metadata, summarize_tabular


BLS_API_BASE = "https://api.bls.gov/publicAPI/v2/timeseries/data"
BLS_SERIES = [
    ("cpi_headline_nsa", "CUUR0000SA0", "CPI index, not seasonally adjusted"),
]


def _library_version(package: str) -> str | None:
    try:
        return version(package)
    except Exception:
        return None


def _year_chunks() -> list[tuple[int, int]]:
    start_year = int(DEFAULT_START_DATE[:4])
    end_year = int(DEFAULT_END_DATE[:4])
    chunks: list[tuple[int, int]] = []
    # The public API may silently truncate requests longer than ten years.
    # Use explicit ten-year windows so the raw cache has complete coverage.
    for chunk_start in range(start_year, end_year + 1, 10):
        chunks.append((chunk_start, min(chunk_start + 9, end_year)))
    return chunks


def _bls_frame(payload: dict[str, Any]) -> pd.DataFrame:
    series = (payload.get("Results") or {}).get("series") or []
    if not series:
        raise RuntimeError("BLS returned no series")
    observations = series[0].get("data") or []
    frame = pd.DataFrame(observations)
    if frame.empty:
        raise RuntimeError("BLS returned no observations")
    if {"year", "period"}.issubset(frame.columns):
        month = frame["period"].astype(str).str.extract(r"M(\d{2})")[0]
        frame["date"] = pd.to_datetime(
            frame["year"].astype(str) + "-" + month.fillna("01") + "-01",
            errors="coerce",
        )
    if "value" in frame.columns:
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame


def download_bls_series(
    label: str,
    series_id: str,
    units: str,
    *,
    refresh: bool = False,
) -> list[dict[str, Any]]:
    source_dir = RAW_SOURCE_DIRS["bls"] / "api"
    results: list[dict[str, Any]] = []
    for chunk_start, chunk_end in _year_chunks():
        data_path = source_dir / f"{safe_name(series_id)}_{chunk_start}_{chunk_end}.json"
        request_parameters = {
            "series_id": series_id,
            "startyear": str(chunk_start),
            "endyear": str(chunk_end),
            "format": "json",
        }
        cached, metadata = valid_cache(data_path)
        if cached and not refresh:
            results.append({**(metadata or {}), "cache_reused": True})
            continue
        try:
            response = requests.get(
                f"{BLS_API_BASE}/{series_id}",
                params={
                    "startyear": chunk_start,
                    "endyear": chunk_end,
                },
                timeout=30,
                headers={"User-Agent": "InsightPulse-ScenarioModeling/step3"},
            )
            response.raise_for_status()
            payload = response.json()
            frame = _bls_frame(payload)
            write_json_atomic(data_path, payload)
            missing_count = int(frame["value"].isna().sum()) if "value" in frame else 0
            coverage_frame = frame.drop(columns=["latest"], errors="ignore")
            status = "DOWNLOADED_WITH_GAPS" if missing_count else "SUCCESSFULLY_DOWNLOADED"
            metadata = summarize_tabular(
                coverage_frame,
                dataset_name=f"bls_{label}_{chunk_start}_{chunk_end}",
                source="BLS Public Data API",
                data_path=data_path,
                frequency="Monthly",
                units=units,
                request_parameters=request_parameters,
                authentication_classification=NO_KEY_REQUIRED,
                credential_environment_variable=None,
                source_library_version=_library_version("requests"),
                date_column="date",
                license_note="BLS public data citation and source terms must be retained.",
                source_notes=f"BLS series {series_id}; raw API response preserved.",
                retrieval_status=status,
            )
            metadata["bls_series_id"] = series_id
            metadata["cache_reused"] = False
            results.append(persist_metadata(data_path, metadata))
        except Exception as exc:
            failure_path = record_failure(
                data_path,
                dataset_name=f"bls_{label}_{chunk_start}_{chunk_end}",
                source="BLS Public Data API",
                authentication_classification=NO_KEY_REQUIRED,
                credential_environment_variable=None,
                request_parameters=safe_request_parameters(request_parameters),
                error=str(exc),
            )
            results.append(
                {
                    "dataset_name": f"bls_{label}_{chunk_start}_{chunk_end}",
                    "source": "BLS Public Data API",
                    "retrieval_status": "SOURCE_UNAVAILABLE",
                    "error_details": str(exc),
                    "failure_file": str(failure_path),
                    "file_path": str(data_path),
                }
            )
    return results


def download_bls_documents(*, refresh: bool = False) -> list[dict[str, Any]]:
    # The BLS schedule pages and ICS feed were attempted during the source
    # inventory and returned HTTP 403 in this environment.  The reproducible
    # selected path is the public API below; the preserved failure records
    # remain in raw/bls/failures for auditability.
    results: list[dict[str, Any]] = []
    for label, series_id, units in BLS_SERIES:
        results.extend(
            download_bls_series(
                label,
                series_id,
                units,
                refresh=refresh,
            )
        )
    return results


def rebuild_cached_bls_metadata() -> list[dict[str, Any]]:
    """Recompute coverage sidecars from preserved raw BLS API payloads."""
    source_dir = RAW_SOURCE_DIRS["bls"] / "api"
    units_by_series = {series_id: units for _, series_id, units in BLS_SERIES}
    rebuilt: list[dict[str, Any]] = []
    for data_path in sorted(source_dir.glob("*.json")):
        if data_path.name.endswith(".metadata.json"):
            continue
        parts = data_path.stem.split("_")
        if len(parts) != 3 or parts[0] not in units_by_series:
            continue
        series_id, chunk_start, chunk_end = parts
        try:
            payload = json.loads(data_path.read_text(encoding="utf-8"))
            frame = _bls_frame(payload)
            coverage_frame = frame.drop(columns=["latest"], errors="ignore")
            missing_count = int(frame["value"].isna().sum())
            status = "DOWNLOADED_WITH_GAPS" if missing_count else "SUCCESSFULLY_DOWNLOADED"
            metadata = summarize_tabular(
                coverage_frame,
                dataset_name=f"bls_cpi_headline_nsa_{chunk_start}_{chunk_end}",
                source="BLS Public Data API",
                data_path=data_path,
                frequency="Monthly",
                units=units_by_series[series_id],
                request_parameters={
                    "series_id": series_id,
                    "startyear": chunk_start,
                    "endyear": chunk_end,
                    "format": "json",
                },
                authentication_classification=NO_KEY_REQUIRED,
                credential_environment_variable=None,
                source_library_version=_library_version("requests"),
                date_column="date",
                license_note="BLS public data citation and source terms must be retained.",
                source_notes=f"BLS series {series_id}; raw API response preserved.",
                retrieval_status=status,
            )
            metadata["bls_series_id"] = series_id
            metadata["cache_reused"] = False
            rebuilt.append(persist_metadata(data_path, metadata))
        except Exception:
            continue
    return rebuilt
