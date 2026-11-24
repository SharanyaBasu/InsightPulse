from __future__ import annotations

from typing import Any

from .public import download_document


def download_philadelphia_fed_documents(*, refresh: bool = False) -> list[dict[str, Any]]:
    results = [
        download_document(
            source_key="philadelphia_fed",
            dataset_name="spf_and_real_time_data_index",
            filename="real_time_data_index.html",
            url="https://www.philadelphiafed.org/surveys-and-data/real-time-data-research",
            note="Philadelphia Fed real-time data index retrieved; SPF release-specific expectation alignment remains unverified.",
            license_note="Philadelphia Fed citation and data-use terms must be retained.",
            refresh=refresh,
        )
    ]
    structured_specs = [
        (
            "spf_mean_growth_history",
            "meanGrowth.xlsx",
            "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/survey-of-professional-forecasters/historical-data/meanGrowth.xlsx?hash=0F651C29D8FE2E04AB86BC7DD7EDAD2E&sc_lang=en",
            "Official Philadelphia Fed SPF mean growth forecasts since 1968; backup expectation source for GDP and inflation comparisons.",
        ),
        (
            "spf_mean_level_history",
            "meanLevel.xlsx",
            "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/survey-of-professional-forecasters/historical-data/meanLevel.xlsx?hash=A51C49A6FCF80FE18F2CE81D903F4970&sc_lang=en",
            "Official Philadelphia Fed SPF mean level forecasts since 1968; backup expectation source.",
        ),
        (
            "spf_median_growth_history",
            "medianGrowth.xlsx",
            "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/survey-of-professional-forecasters/historical-data/medianGrowth.xlsx?hash=44EE1B3E0569944F81475E78F1C7FEED&sc_lang=en",
            "Official Philadelphia Fed SPF median growth forecasts since 1968; backup expectation source.",
        ),
        (
            "spf_median_level_history",
            "medianLevel.xlsx",
            "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/survey-of-professional-forecasters/historical-data/medianLevel.xlsx?hash=E52F5DEF4551C3EBD24031A04CE6E98A&sc_lang=en",
            "Official Philadelphia Fed SPF median level forecasts since 1968; backup expectation source.",
        ),
    ]
    results.extend(
        download_document(
            source_key="philadelphia_fed",
            dataset_name=name,
            filename=filename,
            url=url,
            note=note,
            license_note="Philadelphia Fed citation and data-use terms must be retained.",
            dataset_status="STRUCTURED_RAW_DATA",
            expected_file_signature=b"PK\x03\x04",
            refresh=refresh,
        )
        for name, filename, url, note in structured_specs
    )
    return results
