from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from dotenv import dotenv_values


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ENV_PATH = PROJECT_ROOT / "backend" / ".env"

DATA_ROOT = Path(
    os.getenv("SCENARIO_DATA_ROOT", str(PROJECT_ROOT / "data" / "scenario_playground"))
).resolve()
RAW_ROOT = DATA_ROOT / "raw"
INTERIM_ROOT = DATA_ROOT / "interim"
PROCESSED_ROOT = DATA_ROOT / "processed"
REPORT_PATH = PROJECT_ROOT / "markdown_files" / "scenario_modeling" / "03_RAW_DATA_DOWNLOAD_REPORT.md"

RAW_SOURCE_DIRS = {
    "yahoo": RAW_ROOT / "yahoo",
    "fred": RAW_ROOT / "fred",
    "alfred": RAW_ROOT / "alfred",
    "bls": RAW_ROOT / "bls",
    "bea": RAW_ROOT / "bea",
    "federal_reserve": RAW_ROOT / "federal_reserve",
    "ism": RAW_ROOT / "ism",
    "kansas_city_fed": RAW_ROOT / "kansas_city_fed",
    "cleveland_fed": RAW_ROOT / "cleveland_fed",
    "atlanta_fed": RAW_ROOT / "atlanta_fed",
    "philadelphia_fed": RAW_ROOT / "philadelphia_fed",
}

DEFAULT_START_DATE = os.getenv("SCENARIO_START_DATE", "1950-01-01")
DEFAULT_END_DATE = os.getenv(
    "SCENARIO_END_DATE",
    (date.today() + timedelta(days=1)).isoformat(),
)

FRED_API_KEY = "FRED_API_KEY"
BEA_API_KEY = "BEA_API_KEY"

NO_KEY_REQUIRED = "NO_KEY_REQUIRED"
EXISTING_PROJECT_KEY_AVAILABLE = "EXISTING_PROJECT_KEY_AVAILABLE"
EXISTING_PROJECT_KEY_MISSING = "EXISTING_PROJECT_KEY_MISSING"
NEW_FREE_KEY_REQUIRED = "NEW_FREE_KEY_REQUIRED"
NEW_PAID_OR_RESTRICTED_ACCESS_REQUIRED = "NEW_PAID_OR_RESTRICTED_ACCESS_REQUIRED"
MANUAL_DOWNLOAD_REQUIRED = "MANUAL_DOWNLOAD_REQUIRED"
LICENSING_REVIEW_REQUIRED = "LICENSING_REVIEW_REQUIRED"
ACCESS_METHOD_UNVERIFIED = "ACCESS_METHOD_UNVERIFIED"

SUCCESSFULLY_DOWNLOADED = "SUCCESSFULLY_DOWNLOADED"
DOWNLOADED_WITH_GAPS = "DOWNLOADED_WITH_GAPS"
SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"


def project_env_values() -> dict[str, str]:
    """Read project configuration without printing or persisting secret values."""
    values: dict[str, str] = {}
    if BACKEND_ENV_PATH.exists():
        for key, value in dotenv_values(BACKEND_ENV_PATH).items():
            if value is not None:
                values[key] = value
    return values


def get_credential(name: str) -> str | None:
    """Use an active-process value first, then the existing backend .env value."""
    return os.getenv(name) or project_env_values().get(name)


def credential_is_configured(name: str) -> bool:
    return bool(get_credential(name))


def source_dir(source: str) -> Path:
    if source not in RAW_SOURCE_DIRS:
        raise KeyError(f"Unknown raw source directory: {source}")
    return RAW_SOURCE_DIRS[source]


def ensure_storage() -> None: # creates the expected data folders
    for path in RAW_SOURCE_DIRS.values():
        path.mkdir(parents=True, exist_ok=True)
    (INTERIM_ROOT / "source_comparisons").mkdir(parents=True, exist_ok=True)
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)


def request_dates() -> dict[str, str]:
    return {
        "start_date": DEFAULT_START_DATE,
        "end_date": DEFAULT_END_DATE,
    }


def safe_request_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    """Remove credential values from metadata while preserving request shape."""
    redacted: dict[str, Any] = {}
    for key, value in parameters.items():
        if key.lower() in {"api_key", "token", "key", "password", "secret"}:
            redacted[key] = "<redacted>"
        else:
            redacted[key] = value
    return redacted
