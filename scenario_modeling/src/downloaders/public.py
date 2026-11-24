from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
from typing import Any

import requests

from ..cache import record_failure, valid_cache, write_bytes_atomic
from ..config import (
    ACCESS_METHOD_UNVERIFIED,
    NO_KEY_REQUIRED,
    RAW_SOURCE_DIRS,
    safe_request_parameters,
)
from ..metadata import persist_metadata, summarize_document


def _library_version(package: str) -> str | None:
    try:
        return version(package)
    except Exception:
        return None


def download_document(
    *,
    source_key: str,
    dataset_name: str,
    filename: str,
    url: str,
    note: str,
    license_note: str | None = None,
    refresh: bool = False,
    authentication_classification: str = NO_KEY_REQUIRED,
    credential_environment_variable: str | None = None,
    dataset_status: str = ACCESS_METHOD_UNVERIFIED,
    expected_file_signature: bytes | None = None,
) -> dict[str, Any]:
    source_dir = RAW_SOURCE_DIRS[source_key]
    data_path = source_dir / filename
    request_parameters = {
        "method": "GET",
        "url": url,
        "timeout_seconds": 30,
    }
    cached, metadata = valid_cache(data_path)
    if cached and not refresh:
        return {**(metadata or {}), "cache_reused": True}
    try:
        response = requests.get(
            url,
            timeout=30,
            headers={"User-Agent": "InsightPulse-ScenarioModeling/step3"},
        )
        response.raise_for_status()
        if expected_file_signature and not response.content.startswith(expected_file_signature):
            raise RuntimeError(
                f"Downloaded content for {dataset_name} did not match the expected file signature"
            )
        write_bytes_atomic(data_path, response.content)
        metadata = summarize_document(
            dataset_name=dataset_name,
            source=source_key,
            data_path=data_path,
            request_parameters=request_parameters,
            authentication_classification=authentication_classification,
            credential_environment_variable=credential_environment_variable,
            source_library_version=_library_version("requests"),
            note=note,
            license_note=license_note,
            dataset_status=dataset_status,
        )
        metadata["http_status"] = response.status_code
        metadata["content_type"] = response.headers.get("content-type")
        metadata["cache_reused"] = False
        return persist_metadata(data_path, metadata)
    except Exception as exc:
        failure_path = record_failure(
            data_path,
            dataset_name=dataset_name,
            source=source_key,
            authentication_classification=authentication_classification,
            credential_environment_variable=credential_environment_variable,
            request_parameters=safe_request_parameters(request_parameters),
            error=str(exc),
        )
        return {
            "dataset_name": dataset_name,
            "source": source_key,
            "retrieval_status": "SOURCE_UNAVAILABLE",
            "error_details": str(exc),
            "failure_file": str(failure_path),
            "file_path": str(data_path),
        }
