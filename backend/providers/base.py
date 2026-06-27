"""Shared provider result types for future ingestion providers."""

from dataclasses import dataclass
from typing import Any


@dataclass
class ProviderResult:
    source: str
    data: Any
    rows_fetched: int
