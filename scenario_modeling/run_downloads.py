"""
Main entry point for downloading and caching raw sources for the Scenario Playground.

Can run all sources, selected sources, or an explicit refresh.

Its job is: choose source downloaders, run them, cache raw outputs, write metadata, and regenerate the coverage report.
"""

from __future__ import annotations

import argparse
from collections import Counter
from typing import Any, Callable

from .src.config import ensure_storage
from .src.coverage import write_report
from .src.downloaders.alfred import run_capability_tests, download_vintage_history
from .src.downloaders.atlanta_fed import download_atlanta_fed_documents
from .src.downloaders.bea import download_bea_documents
from .src.downloaders.bls import download_bls_documents
from .src.downloaders.cleveland_fed import download_cleveland_fed_documents
from .src.downloaders.federal_reserve import download_federal_reserve_documents
from .src.downloaders.fred import download_all as download_fred
from .src.downloaders.ism import download_ism_documents
from .src.downloaders.kansas_city_fed import download_kansas_city_fed_manufacturing
from .src.downloaders.philadelphia_fed import download_philadelphia_fed_documents
from .src.downloaders.yahoo import download_all as download_yahoo


DOWNLOADERS: dict[str, Callable[..., list[dict[str, Any]]]] = {
    "yahoo": download_yahoo,
    "fred": download_fred,
    "alfred": run_capability_tests,
    "federal_reserve": download_federal_reserve_documents,
    "bls": download_bls_documents,
    "bea": download_bea_documents,
    "cleveland_fed": download_cleveland_fed_documents,
    "atlanta_fed": download_atlanta_fed_documents,
    "ism": download_ism_documents,
    "kansas_city_fed": download_kansas_city_fed_manufacturing,
    "philadelphia_fed": download_philadelphia_fed_documents,
}


def run_downloads(
    *,
    refresh: bool = False,
    only: list[str] | None = None,
) -> list[dict[str, Any]]:
    ensure_storage()
    selected = only or list(DOWNLOADERS)
    results: list[dict[str, Any]] = []
    for source in selected:
        if source not in DOWNLOADERS:
            raise ValueError(f"Unknown source: {source}")
        results.extend(DOWNLOADERS[source](refresh=refresh))
        if source == "alfred":
            results.extend(download_vintage_history(refresh=refresh))
    write_report()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Download/cache Scenario Playground raw sources.")
    parser.add_argument("--refresh", action="store_true", help="Ignore valid local caches and refresh.")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(DOWNLOADERS),
        help="Run only the selected source downloaders.",
    )
    args = parser.parse_args()
    results = run_downloads(refresh=args.refresh, only=args.only)
    counts = Counter(str(result.get("retrieval_status", "UNKNOWN")) for result in results)
    print(f"retrieval_status_counts={dict(counts)}")
    print(f"report_path={write_report()}")


if __name__ == "__main__":
    main()
