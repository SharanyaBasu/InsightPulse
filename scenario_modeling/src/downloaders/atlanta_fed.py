from __future__ import annotations

from typing import Any

from .public import download_document


def download_atlanta_fed_documents(*, refresh: bool = False) -> list[dict[str, Any]]:
    results = [
        download_document(
            source_key="atlanta_fed",
            dataset_name="gdpnow_page",
            filename="gdpnow.html",
            url="https://www.atlantafed.org/cqer/research/gdpnow",
            note="Atlanta Fed GDPNow page retrieved; the linked model-data and release-date workbooks are downloaded separately below for later parsing.",
            license_note="Atlanta Fed citation and data-use terms must be retained.",
            refresh=refresh,
        )
    ]
    structured_specs = [
        (
            "gdpnow_model_data_and_historical_forecasts",
            "GDPTrackingModelDataAndForecasts.xlsx",
            "https://www.atlantafed.org/-/media/Project/Atlanta/FRBA/Documents/cqer/researchcq/gdpnow/GDPTrackingModelDataAndForecasts.xlsx",
            "Official Atlanta Fed GDPNow workbook containing model data and historical forecasts.",
        ),
        (
            "gdpnow_release_dates",
            "GDPNowcastDataReleaseDates.xlsx",
            "https://www.atlantafed.org/-/media/Project/Atlanta/FRBA/Documents/cqer/researchcq/gdpnow/GDPNowcastDataReleaseDates.xlsx",
            "Official Atlanta Fed GDPNow release-date workbook.",
        ),
    ]
    results.extend(
        download_document(
            source_key="atlanta_fed",
            dataset_name=name,
            filename=filename,
            url=url,
            note=note,
            license_note="Atlanta Fed citation and data-use terms must be retained.",
            dataset_status="STRUCTURED_RAW_DATA",
            expected_file_signature=b"PK\x03\x04",
            refresh=refresh,
        )
        for name, filename, url, note in structured_specs
    )
    return results
