from __future__ import annotations

import json
import re
from importlib.metadata import version
from pathlib import Path
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
    EXISTING_PROJECT_KEY_AVAILABLE,
    EXISTING_PROJECT_KEY_MISSING,
    FRED_API_KEY,
    RAW_SOURCE_DIRS,
    get_credential,
    safe_request_parameters,
)
from ..metadata import persist_metadata, summarize_tabular


FRED_API_BASE = "https://api.stlouisfed.org/fred"
FRED_SERIES_URL = f"{FRED_API_BASE}/series"
FRED_OBSERVATIONS_URL = f"{FRED_API_BASE}/series/observations"

FRED_CANDIDATES = [
    "DFEDTAR",
    "DFEDTARL",
    "DFEDTARU",
    "EFFR",
    "FEDFUNDS",
    "CPIAUCSL",
    "CPIAUCNS",
    "CPILFESL",
    "CUUR0000SA0L2",
    "UNRATE",
    "GDPC1",
    "DGS2",
    "DGS10",
    "VIXCLS",
    "DCOILWTICO",
    "DTWEXBGS",
    "T10Y2Y",
]


def _library_version(package: str) -> str | None:
    try:
        return version(package)
    except Exception:
        return None


def _safe_error(exc: Exception, credential: str | None = None) -> str:
    value = str(exc)
    if credential:
        value = value.replace(credential, "<redacted>")
    value = re.sub(r"([?&](?:api_key|token|key)=)[^&\s]+", r"\1<redacted>", value)
    return value


def _auth_classification() -> str:
    return (
        EXISTING_PROJECT_KEY_AVAILABLE
        if get_credential(FRED_API_KEY)
        else EXISTING_PROJECT_KEY_MISSING
    )


def request_fred_json(
    endpoint: str,
    parameters: dict[str, Any],
    *,
    timeout: int = 30,
) -> dict[str, Any]:
    credential = get_credential(FRED_API_KEY)
    if not credential:
        raise RuntimeError("FRED_API_KEY is not configured in the active process or backend/.env")
    params = dict(parameters)
    params["api_key"] = credential
    params["file_type"] = "json"
    response = requests.get(
        endpoint,
        params=params,
        timeout=timeout,
        headers={"User-Agent": "InsightPulse-ScenarioModeling/step3"},
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("FRED returned a non-object JSON response")
    return payload


def _series_info(series_id: str) -> dict[str, Any]:
    payload = request_fred_json(
        FRED_SERIES_URL,
        {"series_id": series_id},
    )
    series = payload.get("seriess") or []
    return series[0] if series else {}


def _observations_frame(payload: dict[str, Any]) -> pd.DataFrame:
    observations = payload.get("observations") or []
    frame = pd.DataFrame(observations)
    if frame.empty:
        raise RuntimeError("FRED returned no observations")
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if "value" in frame.columns:
        numeric = pd.to_numeric(frame["value"], errors="coerce")
        frame["value_numeric_for_coverage_only"] = numeric
    return frame


def download_series(series_id: str, *, refresh: bool = False) -> dict[str, Any]:
    source_dir = RAW_SOURCE_DIRS["fred"]
    data_path = source_dir / f"{safe_name(series_id)}.observations.json"
    request_parameters = {
        "series_id": series_id,
        "observation_start": DEFAULT_START_DATE,
        "observation_end": DEFAULT_END_DATE,
        "output_type": 1,
        "sort_order": "asc",
    }
    auth_classification = _auth_classification()

    cached, metadata = valid_cache(data_path)
    if cached and not refresh:
        return {**(metadata or {}), "cache_reused": True}

    try:
        payload = request_fred_json(
            FRED_OBSERVATIONS_URL,
            request_parameters,
        )
        frame = _observations_frame(payload)
        info = _series_info(series_id)
        series_info_path = source_dir / f"{safe_name(series_id)}.series.json"
        write_json_atomic(series_info_path, {"seriess": [info]})
        write_json_atomic(data_path, payload)
        frequency = info.get("frequency") or info.get("frequency_short")
        units = info.get("units") or info.get("units_short")
        missing_count = int(frame["value_numeric_for_coverage_only"].isna().sum())
        status = "DOWNLOADED_WITH_GAPS" if missing_count else "SUCCESSFULLY_DOWNLOADED"
        metadata = summarize_tabular(
            frame,
            dataset_name=f"fred_{series_id}",
            source="FRED",
            data_path=data_path,
            frequency=frequency,
            units=units,
            request_parameters=request_parameters,
            authentication_classification=auth_classification,
            credential_environment_variable=FRED_API_KEY,
            source_library_version=_library_version("fredapi"),
            date_column="date",
            vintage_parameters={
                "realtime_start": payload.get("realtime_start"),
                "realtime_end": payload.get("realtime_end"),
                "output_type": payload.get("output_type"),
            },
            license_note="FRED source citation and series-specific usage notes must be retained.",
            source_notes=info.get("notes"),
            retrieval_status=status,
        )
        metadata["source_series_metadata_file"] = str(series_info_path)
        metadata["missing_value_count_total"] = missing_count
        metadata["cache_reused"] = False
        return persist_metadata(data_path, metadata)
    except Exception as exc:
        error = _safe_error(exc, get_credential(FRED_API_KEY))
        failure_path = record_failure(
            data_path,
            dataset_name=f"fred_{series_id}",
            source="FRED",
            authentication_classification=auth_classification,
            credential_environment_variable=FRED_API_KEY,
            request_parameters=request_parameters,
            error=error,
        )
        return {
            "dataset_name": f"fred_{series_id}",
            "source": "FRED",
            "retrieval_status": "SOURCE_UNAVAILABLE",
            "error_details": error,
            "failure_file": str(failure_path),
            "file_path": str(data_path),
        }


def download_all(*, refresh: bool = False) -> list[dict[str, Any]]:
    return [download_series(series_id, refresh=refresh) for series_id in FRED_CANDIDATES]
