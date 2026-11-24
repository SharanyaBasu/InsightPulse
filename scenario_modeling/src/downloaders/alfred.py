from __future__ import annotations

from importlib.metadata import version
import json
from typing import Any

import pandas as pd

from ..cache import (
    record_failure,
    safe_name,
    valid_cache,
    write_json_atomic,
)
from ..config import (
    EXISTING_PROJECT_KEY_AVAILABLE,
    EXISTING_PROJECT_KEY_MISSING,
    DEFAULT_END_DATE,
    FRED_API_KEY,
    RAW_SOURCE_DIRS,
    get_credential,
    safe_request_parameters,
)
from ..metadata import persist_metadata, summarize_tabular
from .fred import FRED_OBSERVATIONS_URL, _safe_error, request_fred_json


FRED_VINTAGEDATES_URL = "https://api.stlouisfed.org/fred/series/vintagedates"

# These are the historical macro series used to construct surprise/change
# inputs.  Their vintage histories are cached separately from the ordinary
# latest-observation downloads so later steps can enforce point-in-time joins.
VINTAGE_SERIES = ("CPIAUCSL", "UNRATE", "GDPC1")
VINTAGE_DATE_CHUNK_SIZE = 50


CAPABILITY_TESTS = {
    "realtime_period": {
        "series_id": "CPIAUCSL",
        "realtime_start": "2020-01-01",
        "realtime_end": "2020-01-31",
        "observation_start": "2019-01-01",
        "observation_end": "2020-01-31",
        "output_type": 1,
    },
    "vintage_date": {
        "series_id": "CPIAUCSL",
        "vintage_dates": "2020-01-15",
        "observation_start": "2019-01-01",
        "observation_end": "2020-01-31",
        "output_type": 2,
    },
    "initial_release": {
        "series_id": "CPIAUCSL",
        "observation_start": "2019-01-01",
        "observation_end": "2020-01-31",
        "output_type": 4,
    },
}


def _library_version(package: str) -> str | None:
    try:
        return version(package)
    except Exception:
        return None


def run_capability_test(label: str, parameters: dict[str, Any], *, refresh: bool = False) -> dict[str, Any]:
    source_dir = RAW_SOURCE_DIRS["alfred"]
    data_path = source_dir / f"{safe_name(label)}.json"
    auth_classification = (
        EXISTING_PROJECT_KEY_AVAILABLE
        if get_credential(FRED_API_KEY)
        else EXISTING_PROJECT_KEY_MISSING
    )
    cached, metadata = valid_cache(data_path)
    if cached and not refresh:
        return {**(metadata or {}), "cache_reused": True}
    try:
        payload = request_fred_json(FRED_OBSERVATIONS_URL, parameters)
        observations = payload.get("observations") or []
        frame = pd.DataFrame(observations)
        if frame.empty:
            raise RuntimeError("Vintage capability test returned no observations")
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        if "value" in frame.columns:
            frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        write_json_atomic(data_path, payload)
        metadata = summarize_tabular(
            frame,
            dataset_name=f"alfred_capability_{label}",
            source="FRED/ALFRED vintage-capable observations endpoint",
            data_path=data_path,
            frequency="test request",
            units=payload.get("units"),
            request_parameters=parameters,
            authentication_classification=auth_classification,
            credential_environment_variable=FRED_API_KEY,
            source_library_version=_library_version("fredapi"),
            date_column="date",
            vintage_parameters={
                "realtime_start": payload.get("realtime_start"),
                "realtime_end": payload.get("realtime_end"),
                "vintage_dates": parameters.get("vintage_dates"),
                "output_type": parameters.get("output_type"),
            },
            license_note="FRED/ALFRED citation and series-specific usage notes must be retained.",
            source_notes="Controlled capability test only; not a point-in-time model matrix.",
        )
        metadata["cache_reused"] = False
        return persist_metadata(data_path, metadata)
    except Exception as exc:
        error = _safe_error(exc, get_credential(FRED_API_KEY))
        failure_path = record_failure(
            data_path,
            dataset_name=f"alfred_capability_{label}",
            source="FRED/ALFRED vintage-capable observations endpoint",
            authentication_classification=auth_classification,
            credential_environment_variable=FRED_API_KEY,
            request_parameters=parameters,
            error=error,
        )
        return {
            "dataset_name": f"alfred_capability_{label}",
            "source": "FRED/ALFRED vintage-capable observations endpoint",
            "retrieval_status": "SOURCE_UNAVAILABLE",
            "error_details": error,
            "failure_file": str(failure_path),
            "file_path": str(data_path),
        }


def run_capability_tests(*, refresh: bool = False) -> list[dict[str, Any]]:
    return [
        run_capability_test(label, parameters, refresh=refresh)
        for label, parameters in CAPABILITY_TESTS.items()
    ]


def _vintage_auth_classification() -> str:
    return (
        EXISTING_PROJECT_KEY_AVAILABLE
        if get_credential(FRED_API_KEY)
        else EXISTING_PROJECT_KEY_MISSING
    )


def _vintage_dates_path(series_id: str):
    return RAW_SOURCE_DIRS["alfred"] / f"{safe_name(series_id)}.vintagedates.json"


def _load_vintage_dates(series_id: str) -> list[str]:
    path = _vintage_dates_path(series_id)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [str(value) for value in payload.get("vintage_dates", [])]
    except (OSError, json.JSONDecodeError, AttributeError):
        return []


def download_vintage_dates(series_id: str, *, refresh: bool = False) -> dict[str, Any]:
    data_path = _vintage_dates_path(series_id)
    request_parameters = {"series_id": series_id}
    auth_classification = _vintage_auth_classification()
    cached, metadata = valid_cache(data_path)
    if cached and not refresh:
        return {**(metadata or {}), "cache_reused": True}
    try:
        payload = request_fred_json(FRED_VINTAGEDATES_URL, request_parameters)
        vintage_dates = payload.get("vintage_dates") or []
        if not vintage_dates:
            raise RuntimeError(f"FRED returned no vintage dates for {series_id}")
        write_json_atomic(data_path, payload)
        frame = pd.DataFrame({"vintage_date": pd.to_datetime(vintage_dates, errors="coerce")})
        metadata = summarize_tabular(
            frame,
            dataset_name=f"alfred_{series_id}_vintage_dates",
            source="FRED/ALFRED vintage dates endpoint",
            data_path=data_path,
            frequency="irregular vintage calendar",
            units="vintage date",
            request_parameters=request_parameters,
            authentication_classification=auth_classification,
            credential_environment_variable=FRED_API_KEY,
            source_library_version=_library_version("fredapi"),
            date_column="vintage_date",
            vintage_parameters={"series_id": series_id, "output_type": 2},
            license_note="FRED source citation and series-specific usage notes must be retained.",
            source_notes="Complete vintage-date index preserved before requesting point-in-time observations.",
        )
        metadata["dataset_status"] = "STRUCTURED_RAW_DATA"
        metadata["cache_reused"] = False
        return persist_metadata(data_path, metadata)
    except Exception as exc:
        error = _safe_error(exc, get_credential(FRED_API_KEY))
        failure_path = record_failure(
            data_path,
            dataset_name=f"alfred_{series_id}_vintage_dates",
            source="FRED/ALFRED vintage dates endpoint",
            authentication_classification=auth_classification,
            credential_environment_variable=FRED_API_KEY,
            request_parameters=safe_request_parameters(request_parameters),
            error=error,
        )
        return {
            "dataset_name": f"alfred_{series_id}_vintage_dates",
            "source": "FRED/ALFRED vintage dates endpoint",
            "retrieval_status": "SOURCE_UNAVAILABLE",
            "error_details": error,
            "failure_file": str(failure_path),
            "file_path": str(data_path),
        }


def _series_units(series_id: str) -> str | None:
    path = RAW_SOURCE_DIRS["fred"] / f"{safe_name(series_id)}.series.json"
    try:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
        series = (payload.get("seriess") or [{}])[0]
        return series.get("units") or series.get("units_short")
    except Exception:
        return None


def _vintage_snapshot_path(series_id: str, chunk_number: int) -> Any:
    return RAW_SOURCE_DIRS["alfred"] / f"{safe_name(series_id)}.vintage_{chunk_number:03d}.json"


def download_vintage_snapshots(series_id: str, *, refresh: bool = False) -> list[dict[str, Any]]:
    vintage_dates = _load_vintage_dates(series_id)
    if not vintage_dates:
        return []
    results: list[dict[str, Any]] = []
    auth_classification = _vintage_auth_classification()
    for chunk_number, start in enumerate(range(0, len(vintage_dates), VINTAGE_DATE_CHUNK_SIZE), start=1):
        chunk = vintage_dates[start : start + VINTAGE_DATE_CHUNK_SIZE]
        data_path = _vintage_snapshot_path(series_id, chunk_number)
        request_parameters = {
            "series_id": series_id,
            "observation_start": "1950-01-01",
            "observation_end": DEFAULT_END_DATE,
            "output_type": 2,
            "vintage_dates": ",".join(chunk),
            "sort_order": "asc",
        }
        cached, metadata = valid_cache(data_path)
        if cached and not refresh:
            results.append({**(metadata or {}), "cache_reused": True})
            continue
        try:
            payload = request_fred_json(FRED_OBSERVATIONS_URL, request_parameters, timeout=60)
            observations = payload.get("observations") or []
            if not observations:
                raise RuntimeError(f"FRED returned no vintage observations for {series_id} chunk {chunk_number}")
            frame = pd.DataFrame(observations)
            if "date" in frame.columns:
                frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            if "value" in frame.columns:
                frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
            write_json_atomic(data_path, payload)
            missing_count = int(frame["value"].isna().sum()) if "value" in frame.columns else 0
            status = "DOWNLOADED_WITH_GAPS" if missing_count else "SUCCESSFULLY_DOWNLOADED"
            metadata = summarize_tabular(
                frame,
                dataset_name=f"alfred_{series_id}_vintage_{chunk_number:03d}",
                source="FRED/ALFRED vintage observations endpoint",
                data_path=data_path,
                frequency="source series frequency",
                units=_series_units(series_id),
                request_parameters=request_parameters,
                authentication_classification=auth_classification,
                credential_environment_variable=FRED_API_KEY,
                source_library_version=_library_version("fredapi"),
                date_column="date",
                vintage_parameters={
                    "series_id": series_id,
                    "output_type": 2,
                    "vintage_dates": chunk,
                    "vintage_date_count": len(chunk),
                },
                license_note="FRED source citation and series-specific usage notes must be retained.",
                source_notes="Point-in-time observations preserved by vintage date; no revisions are collapsed here.",
                retrieval_status=status,
            )
            metadata["dataset_status"] = "STRUCTURED_RAW_DATA"
            metadata["missing_value_count_total"] = missing_count
            metadata["cache_reused"] = False
            results.append(persist_metadata(data_path, metadata))
        except Exception as exc:
            error = _safe_error(exc, get_credential(FRED_API_KEY))
            failure_path = record_failure(
                data_path,
                dataset_name=f"alfred_{series_id}_vintage_{chunk_number:03d}",
                source="FRED/ALFRED vintage observations endpoint",
                authentication_classification=auth_classification,
                credential_environment_variable=FRED_API_KEY,
                request_parameters=safe_request_parameters(request_parameters),
                error=error,
            )
            results.append(
                {
                    "dataset_name": f"alfred_{series_id}_vintage_{chunk_number:03d}",
                    "source": "FRED/ALFRED vintage observations endpoint",
                    "retrieval_status": "SOURCE_UNAVAILABLE",
                    "error_details": error,
                    "failure_file": str(failure_path),
                    "file_path": str(data_path),
                }
            )
    return results


def download_vintage_history(*, refresh: bool = False) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for series_id in VINTAGE_SERIES:
        date_result = download_vintage_dates(series_id, refresh=refresh)
        results.append(date_result)
        if date_result.get("retrieval_status") in {"SUCCESSFULLY_DOWNLOADED", "DOWNLOADED_WITH_GAPS"}:
            results.extend(download_vintage_snapshots(series_id, refresh=refresh))
    return results
