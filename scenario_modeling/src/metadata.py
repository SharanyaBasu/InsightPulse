from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .cache import utc_now, write_metadata
from .config import safe_request_parameters


def _date_series(df: pd.DataFrame, date_column: str) -> pd.Series:
    if date_column not in df.columns:
        return pd.Series(dtype="datetime64[ns]")
    return pd.to_datetime(df[date_column], errors="coerce", utc=True).dt.tz_localize(None)


def _gap_summary(dates: pd.Series, frequency: str | None) -> dict[str, Any]:
    valid_dates = dates.dropna().sort_values().drop_duplicates()
    if len(valid_dates) < 2:
        return {"largest_calendar_gap_days": None, "gap_count_over_one_period": None}

    differences = valid_dates.diff().dt.total_seconds().div(86400).dropna()
    largest = float(differences.max())
    if frequency and frequency.lower().startswith("daily"):
        expected = 1.0
    elif frequency and frequency.lower().startswith("weekly"):
        expected = 7.0
    elif frequency and frequency.lower().startswith("monthly"):
        expected = 31.0
    elif frequency and frequency.lower().startswith("quarter"):
        expected = 93.0
    else:
        expected = None
    return {
        "largest_calendar_gap_days": largest,
        "gap_count_over_one_period": int((differences > (expected or 1)).sum()),
        "gap_note": "Calendar gaps may be expected for weekends, holidays, release schedules, or source publication calendars.",
    }


def summarize_tabular(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    source: str,
    data_path: Path,
    frequency: str | None,
    units: str | None,
    request_parameters: dict[str, Any],
    authentication_classification: str,
    credential_environment_variable: str | None,
    source_library_version: str | None,
    date_column: str = "Date",
    adjustment_policy: str | None = None,
    vintage_parameters: dict[str, Any] | None = None,
    license_note: str | None = None,
    source_notes: str | None = None,
    retrieval_status: str = "SUCCESSFULLY_DOWNLOADED",
) -> dict[str, Any]:
    dates = _date_series(df, date_column)
    min_date = dates.min().date().isoformat() if dates.notna().any() else None
    max_date = dates.max().date().isoformat() if dates.notna().any() else None
    row_count = int(len(df))
    missing_by_column = {
        str(column): int(count)
        for column, count in df.isna().sum().items()
    }
    duplicate_count = int(dates.duplicated().sum()) if date_column in df.columns else None
    calendar_years = None
    if min_date and max_date:
        calendar_years = round(
            (pd.Timestamp(max_date) - pd.Timestamp(min_date)).days / 365.25,
            3,
        )
    approx_trading_years = (
        round(row_count / 252, 3)
        if frequency and frequency.lower().startswith("daily")
        else None
    )
    gaps = _gap_summary(dates, frequency)
    return {
        "dataset_name": dataset_name,
        "source": source,
        "retrieval_status": retrieval_status,
        "retrieved_at": utc_now(),
        "request_parameters": safe_request_parameters(request_parameters),
        "authentication_classification": authentication_classification,
        "credential_environment_variable": credential_environment_variable,
        "file_path": str(data_path),
        "file_format": data_path.suffix.lstrip("."),
        "row_count": row_count,
        "column_names": [str(column) for column in df.columns],
        "minimum_observation_date": min_date,
        "maximum_observation_date": max_date,
        "calendar_span_years": calendar_years,
        "approximate_trading_years": approx_trading_years,
        "frequency": frequency,
        "duplicate_timestamp_count": duplicate_count,
        "missing_value_count_by_column": missing_by_column,
        "missing_value_count_total": int(df.isna().sum().sum()),
        "unexpected_gaps": gaps,
        "timezone_information": "Normalized date column to timezone-naive UTC-derived timestamps for local inspection.",
        "units": units,
        "adjustment_policy": adjustment_policy,
        "vintage_parameters": vintage_parameters or {},
        "source_library_version": source_library_version,
        "license_or_redistribution_note": license_note,
        "source_notes": source_notes,
    }


def summarize_document(
    *,
    dataset_name: str,
    source: str,
    data_path: Path,
    request_parameters: dict[str, Any],
    authentication_classification: str,
    credential_environment_variable: str | None,
    source_library_version: str | None,
    note: str,
    license_note: str | None = None,
    retrieval_status: str = "SUCCESSFULLY_DOWNLOADED",
    dataset_status: str = "ACCESS_METHOD_UNVERIFIED",
) -> dict[str, Any]:
    return {
        "dataset_name": dataset_name,
        "source": source,
        "retrieval_status": retrieval_status,
        "dataset_status": dataset_status,
        "retrieved_at": utc_now(),
        "request_parameters": safe_request_parameters(request_parameters),
        "authentication_classification": authentication_classification,
        "credential_environment_variable": credential_environment_variable,
        "file_path": str(data_path),
        "file_format": data_path.suffix.lstrip("."),
        "row_count": None,
        "column_names": [],
        "minimum_observation_date": None,
        "maximum_observation_date": None,
        "calendar_span_years": None,
        "approximate_trading_years": None,
        "frequency": "document",
        "duplicate_timestamp_count": None,
        "missing_value_count_by_column": {},
        "missing_value_count_total": None,
        "unexpected_gaps": {"note": note},
        "timezone_information": None,
        "units": None,
        "adjustment_policy": None,
        "vintage_parameters": {},
        "source_library_version": source_library_version,
        "license_or_redistribution_note": license_note,
        "source_notes": note,
    }


def persist_metadata(data_path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    write_metadata(data_path, metadata)
    return metadata
