from __future__ import annotations

from importlib.metadata import version
from typing import Any

import pandas as pd
import requests

from ..cache import record_failure, valid_cache, write_json_atomic
from ..config import NO_KEY_REQUIRED, RAW_SOURCE_DIRS, safe_request_parameters
from ..metadata import persist_metadata, summarize_tabular

from .public import download_document


DBNOMICS_PMI_URL = "https://api.db.nomics.world/v22/series/ISM/pmi/pm?observations=1"
REJECTED_OBSERVATIONS = {
    "2025-09-01": 11.1,
    "2025-10-01": 10.0,
    "2025-11-01": 10.0,
    "2025-12-01": 10.3,
}


def _mark_rejected(metadata: dict[str, Any]) -> dict[str, Any]:
    metadata.update(
        {
            "selection_status": "REJECTED_PRIMARY_CANDIDATE",
            "accepted_for_modeling": False,
            "trustworthy_observation_count": 64,
            "trustworthy_minimum_observation_date": "2020-05-01",
            "trustworthy_maximum_observation_date": "2025-08-01",
            "rejected_observations": REJECTED_OBSERVATIONS,
            "rejection_reasons": [
                "September through December 2025 contain corrupted or economically implausible parser output.",
                "Only 64 trustworthy monthly observations remain, which is too short for the intended chronological XGBoost benchmark.",
                "The cache contains observation-period dates, not structured historical release dates.",
                "The DBnomics mirror does not resolve the underlying ISM licensing and production-use restrictions.",
            ],
            "replacement_primary_source": (
                "Federal Reserve Bank of Kansas City Tenth District manufacturing "
                "composite, stored as an explicitly named regional PMI-style proxy"
            ),
        }
    )
    return metadata


def _download_official_ism_document(
    *,
    dataset_name: str,
    filename: str,
    url: str,
    note: str,
    license_note: str,
    refresh: bool,
) -> dict[str, Any]:
    result = download_document(
        source_key="ism",
        dataset_name=dataset_name,
        filename=filename,
        url=url,
        note=note,
        license_note=license_note,
        refresh=refresh,
    )
    data_path = RAW_SOURCE_DIRS["ism"] / filename
    if data_path.exists():
        text = data_path.read_text(encoding="utf-8", errors="replace").lower()
        if "captcha_form" in text or "grecaptcha.execute" in text:
            error = (
                "The direct request returned an ISM CAPTCHA shell rather than the "
                "requested official page content."
            )
            failure_path = record_failure(
                data_path,
                dataset_name=dataset_name,
                source="Institute for Supply Management",
                authentication_classification=NO_KEY_REQUIRED,
                credential_environment_variable=None,
                request_parameters={"method": "GET", "url": url},
                error=error,
            )
            blocked = {
                **result,
                "retrieval_status": "SOURCE_UNAVAILABLE",
                "dataset_status": "WEBPAGE_ACCESS_BLOCKED_BY_CAPTCHA",
                "selection_status": "REJECTED_REFERENCE_DOCUMENT_ACCESS",
                "source_notes": (
                    "Direct retrieval returned a CAPTCHA shell, not the requested "
                    "official document. This probe is retained only as failed access "
                    "evidence and is not a structured data source."
                ),
                "error_details": error,
                "failure_file": str(failure_path),
                "cache_reused": False,
            }
            return persist_metadata(data_path, blocked)
    return result


def _library_version(package: str) -> str | None:
    try:
        return version(package)
    except Exception:
        return None


def download_dbnomics_pmi(*, refresh: bool = False) -> dict[str, Any]:
    data_path = RAW_SOURCE_DIRS["ism"] / "dbnomics_ism_pmi.json"
    request_parameters = {"url": DBNOMICS_PMI_URL, "observations": 1}
    cached, metadata = valid_cache(data_path)
    if cached and not refresh:
        rejected_metadata = _mark_rejected(dict(metadata or {}))
        rejected_metadata["cache_reused"] = True
        return persist_metadata(data_path, rejected_metadata)
    try:
        response = requests.get(
            DBNOMICS_PMI_URL,
            timeout=60,
            headers={"User-Agent": "InsightPulse-ScenarioModeling/step3"},
        )
        response.raise_for_status()
        payload = response.json()
        docs = ((payload.get("series") or {}).get("docs") or [])
        if not docs:
            raise RuntimeError("DBnomics returned no ISM PMI series document")
        document = docs[0]
        dates = document.get("period_start_day") or document.get("period") or []
        values = document.get("value") or []
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(dates, errors="coerce"),
                "value": pd.to_numeric(values, errors="coerce"),
            }
        )
        frame = frame.dropna(subset=["date"]).sort_values("date")
        if frame.empty:
            raise RuntimeError("DBnomics returned no dated ISM PMI observations")
        write_json_atomic(data_path, payload)
        missing_count = int(frame["value"].isna().sum())
        status = "DOWNLOADED_WITH_GAPS" if missing_count else "SUCCESSFULLY_DOWNLOADED"
        metadata = summarize_tabular(
            frame,
            dataset_name="ism_manufacturing_pmi_dbnomics",
            source="DBnomics mirror of Institute for Supply Management",
            data_path=data_path,
            frequency="Monthly",
            units="ISM PMI index points",
            request_parameters=request_parameters,
            authentication_classification=NO_KEY_REQUIRED,
            credential_environment_variable=None,
            source_library_version=_library_version("requests"),
            date_column="date",
            vintage_parameters={"source_series": "ISM/pmi/pm", "observations": 1},
            license_note=(
                "Rejected for modeling. DBnomics attribution does not remove the "
                "underlying ISM restrictions on bulk historical use, derivative time "
                "series, or production redistribution."
            ),
            source_notes=(
                "Preserved unchanged as source-faithful evidence only. It is not an "
                "accepted model source because four post-2025-08 values are corrupted, "
                "the trustworthy history is short, release dates are absent, and "
                "underlying ISM licensing remains restrictive."
            ),
            retrieval_status=status,
        )
        metadata["dbnomics_provider"] = "ISM"
        metadata["dbnomics_dataset"] = "pmi"
        metadata["dbnomics_series"] = "pm"
        metadata["cache_reused"] = False
        return persist_metadata(data_path, _mark_rejected(metadata))
    except Exception as exc:
        failure_path = record_failure(
            data_path,
            dataset_name="ism_manufacturing_pmi_dbnomics",
            source="DBnomics mirror of Institute for Supply Management",
            authentication_classification=NO_KEY_REQUIRED,
            credential_environment_variable=None,
            request_parameters=safe_request_parameters(request_parameters),
            error=str(exc),
        )
        return {
            "dataset_name": "ism_manufacturing_pmi_dbnomics",
            "source": "DBnomics mirror of Institute for Supply Management",
            "retrieval_status": "SOURCE_UNAVAILABLE",
            "error_details": str(exc),
            "failure_file": str(failure_path),
            "file_path": str(data_path),
        }


def download_ism_documents(*, refresh: bool = False) -> list[dict[str, Any]]:
    return [
        download_dbnomics_pmi(refresh=refresh),
        _download_official_ism_document(
            dataset_name="ism_pmi_release_calendar",
            filename="pmi_release_calendar.html",
            url="https://www.ismworld.org/supply-management-news-and-reports/reports/rob-report-calendar/",
            note="Official ISM release calendar retrieved for release timing; the structured PMI values are cached separately from the DBnomics/ISM endpoint above.",
            license_note=(
                "Comparison-only official calendar. ISM terms do not authorize this "
                "project to build or redistribute a bulk historical ISM time series."
            ),
            refresh=refresh,
        ),
        _download_official_ism_document(
            dataset_name="ism_terms_of_use",
            filename="terms_of_use.html",
            url="https://www.ismworld.org/footer/terms-of-use/",
            note=(
                "Official ISM terms retained as source-faithful evidence for rejecting "
                "an unlicensed bulk historical/derivative time-series path."
            ),
            license_note=(
                "Terms evidence only. This cache is not permission to copy, archive, "
                "create derivative time series from, or redistribute ISM PMI history."
            ),
            refresh=refresh,
        ),
    ]
