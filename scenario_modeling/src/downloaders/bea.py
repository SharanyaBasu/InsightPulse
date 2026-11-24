from __future__ import annotations

from typing import Any

from .public import download_document


def download_bea_documents(*, refresh: bool = False) -> list[dict[str, Any]]:
    return [
        download_document(
            source_key="bea",
            dataset_name="bea_release_schedule",
            filename="release_schedule.html",
            url="https://www.bea.gov/news/schedule",
            note="Official BEA release schedule retrieved; the linked machine-readable release-date feed is downloaded separately below.",
            license_note="BEA citation and API-use requirements must be retained.",
            refresh=refresh,
        ),
        download_document(
            source_key="bea",
            dataset_name="bea_machine_readable_release_dates",
            filename="release_dates.json",
            url="https://apps.bea.gov/API/signup/release_dates.json",
            note="Official BEA machine-readable release-date feed linked from the BEA release schedule page.",
            license_note="BEA citation and API-use requirements must be retained.",
            dataset_status="STRUCTURED_RAW_DATA",
            refresh=refresh,
        ),
    ]
