from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import (
    DOWNLOADED_WITH_GAPS,
    SUCCESSFULLY_DOWNLOADED,
    safe_request_parameters,
)


VALID_CACHE_STATUSES = {SUCCESSFULLY_DOWNLOADED, DOWNLOADED_WITH_GAPS}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "dataset"


def metadata_path(data_path: Path) -> Path:
    return data_path.with_suffix(data_path.suffix + ".metadata.json")


def failure_dir(data_path: Path) -> Path:
    return data_path.parent / "failures"


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def valid_cache(data_path: Path) -> tuple[bool, dict[str, Any] | None]:
    meta_path = metadata_path(data_path)
    metadata = read_json(meta_path) if meta_path.exists() else None
    is_valid = (
        data_path.exists()
        and data_path.stat().st_size > 0
        and metadata is not None
        and metadata.get("retrieval_status") in VALID_CACHE_STATUSES
    )
    return is_valid, metadata


def write_bytes_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    partial.write_bytes(content)
    partial.replace(path)


def write_text_atomic(path: Path, content: str) -> None:
    write_bytes_atomic(path, content.encode("utf-8"))


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def write_metadata(data_path: Path, metadata: dict[str, Any]) -> Path:
    path = metadata_path(data_path)
    write_json_atomic(path, metadata)
    return path


def record_failure(
    data_path: Path,
    *,
    dataset_name: str,
    source: str,
    authentication_classification: str,
    credential_environment_variable: str | None,
    request_parameters: dict[str, Any],
    error: str,
) -> Path:
    directory = failure_dir(data_path)
    directory.mkdir(parents=True, exist_ok=True)
    failure_path = directory / f"{safe_name(data_path.stem)}.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.failure.json"
    payload = {
        "dataset_name": dataset_name,
        "source": source,
        "retrieval_status": "SOURCE_UNAVAILABLE",
        "retrieved_at": utc_now(),
        "request_parameters": safe_request_parameters(request_parameters),
        "authentication_classification": authentication_classification,
        "credential_environment_variable": credential_environment_variable,
        "error_details": error,
        "preserved_previous_cache": data_path.exists(),
        "data_file": str(data_path),
    }
    payload["failure_file"] = str(failure_path)
    write_json_atomic(failure_path, payload)
    return failure_path
