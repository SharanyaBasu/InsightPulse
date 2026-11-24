from __future__ import annotations

from typing import Any

from .public import download_document


def download_cleveland_fed_documents(*, refresh: bool = False) -> list[dict[str, Any]]:
    results = [
        download_document(
            source_key="cleveland_fed",
            dataset_name="inflation_nowcasting_page",
            filename="inflation_nowcasting.html",
            url="https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting",
            note="Cleveland Fed Inflation Nowcasting page retrieved; the linked monthly, quarterly, and annual machine-readable nowcast files are downloaded separately below.",
            license_note="Cleveland Fed citation and data-use terms must be retained.",
            refresh=refresh,
        )
    ]
    structured_specs = [
        (
            "inflation_nowcasting_monthly_json",
            "nowcast_month.json",
            "https://www.clevelandfed.org/-/media/files/webcharts/inflationnowcasting/nowcast_month.json?sc_lang=en",
            "Machine-readable monthly Cleveland Fed CPI/PCE nowcast data referenced by the official nowcasting page.",
        ),
        (
            "inflation_nowcasting_quarterly_json",
            "nowcast_quarter.json",
            "https://www.clevelandfed.org/-/media/files/webcharts/inflationnowcasting/nowcast_quarter.json?sc_lang=en",
            "Machine-readable quarterly Cleveland Fed nowcast data referenced by the official nowcasting page.",
        ),
        (
            "inflation_nowcasting_annual_json",
            "nowcast_year.json",
            "https://www.clevelandfed.org/-/media/files/webcharts/inflationnowcasting/nowcast_year.json?sc_lang=en",
            "Machine-readable annual Cleveland Fed nowcast data referenced by the official nowcasting page.",
        ),
    ]
    results.extend(
        download_document(
            source_key="cleveland_fed",
            dataset_name=name,
            filename=filename,
            url=url,
            note=note,
            license_note="Cleveland Fed citation and data-use terms must be retained.",
            dataset_status="STRUCTURED_RAW_DATA",
            refresh=refresh,
        )
        for name, filename, url, note in structured_specs
    )
    return results
