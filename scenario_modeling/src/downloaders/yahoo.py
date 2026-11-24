from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from ..cache import (
    record_failure,
    safe_name,
    valid_cache,
    write_text_atomic,
)
from ..config import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    NO_KEY_REQUIRED,
    RAW_SOURCE_DIRS,
    safe_request_parameters,
)
from ..metadata import persist_metadata, summarize_tabular


MARKET_CANDIDATES = [
    "SPY",
    "^GSPC",
    "^IXIC",
    "QQQ",
    "DX-Y.NYB",
    "GLD",
    "GC=F",
    "CL=F",
    "USO",
    "XLK",
    "XLE",
    "XLF",
    "XLU",
    "XLV",
    "XLY",
    "^VIX",
    "^TNX",
]


def _library_version(package: str) -> str | None:
    try:
        return version(package)
    except Exception:
        return None


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if isinstance(result.columns, pd.MultiIndex):
        result.columns = [
            str(column[0]) if isinstance(column, tuple) else str(column)
            for column in result.columns
        ]
    else:
        result.columns = [str(column) for column in result.columns]
    return result


def _download_frame(ticker: str) -> pd.DataFrame:
    cache_location = RAW_SOURCE_DIRS["yahoo"] / "yfinance_cache"
    cache_location.mkdir(parents=True, exist_ok=True)
    try:
        yf.set_tz_cache_location(str(cache_location))
    except AttributeError:
        pass
    kwargs: dict[str, Any] = {
        "tickers": ticker,
        "start": DEFAULT_START_DATE,
        "end": DEFAULT_END_DATE,
        "interval": "1d",
        "auto_adjust": False,
        "actions": True,
        "progress": False,
        "threads": False,
        "group_by": "column",
        "multi_level_index": False,
    }
    try:
        frame = yf.download(**kwargs)
    except TypeError:
        kwargs.pop("multi_level_index", None)
        frame = yf.download(**kwargs)
    if frame is None or frame.empty:
        raise RuntimeError("Yahoo returned no rows")
    frame = _flatten_columns(frame)
    frame.index.name = "Date"
    frame = frame.reset_index()
    if "Date" not in frame.columns:
        frame = frame.rename(columns={frame.columns[0]: "Date"})
    return frame


def download_ticker(ticker: str, *, refresh: bool = False) -> dict[str, Any]:
    source_dir = RAW_SOURCE_DIRS["yahoo"]
    data_path = source_dir / f"{safe_name(ticker)}.csv"
    request_parameters = {
        "ticker": ticker,
        "start": DEFAULT_START_DATE,
        "end": DEFAULT_END_DATE,
        "interval": "1d",
        "auto_adjust": False,
        "actions": True,
        "threads": False,
    }

    cached, metadata = valid_cache(data_path)
    if cached and not refresh:
        return {**(metadata or {}), "cache_reused": True}

    try:
        frame = _download_frame(ticker)
        csv_text = frame.to_csv(index=False)
        write_text_atomic(data_path, csv_text)
        metadata = summarize_tabular(
            frame,
            dataset_name=f"yahoo_{ticker}",
            source="Yahoo Finance via yfinance",
            data_path=data_path,
            frequency="Daily",
            units="ticker-native market fields",
            request_parameters=request_parameters,
            authentication_classification=NO_KEY_REQUIRED,
            credential_environment_variable=None,
            source_library_version=_library_version("yfinance"),
            date_column="Date",
            adjustment_policy="Raw OHLC fields requested with auto_adjust=False; Adj Close preserved when supplied.",
            license_note="Unofficial yfinance wrapper; Yahoo Finance terms and intended use require separate review.",
            source_notes="Ticker downloaded independently; no complete-row intersection was applied.",
        )
        metadata["cache_reused"] = False
        return persist_metadata(data_path, metadata)
    except Exception as exc:
        failure_path = record_failure(
            data_path,
            dataset_name=f"yahoo_{ticker}",
            source="Yahoo Finance via yfinance",
            authentication_classification=NO_KEY_REQUIRED,
            credential_environment_variable=None,
            request_parameters=safe_request_parameters(request_parameters),
            error=str(exc),
        )
        return {
            "dataset_name": f"yahoo_{ticker}",
            "source": "Yahoo Finance via yfinance",
            "retrieval_status": "SOURCE_UNAVAILABLE",
            "error_details": str(exc),
            "failure_file": str(failure_path),
            "file_path": str(data_path),
        }


def download_all(*, refresh: bool = False) -> list[dict[str, Any]]:
    return [download_ticker(ticker, refresh=refresh) for ticker in MARKET_CANDIDATES]
