"""Shared, offline source loaders for Scenario Playground feature construction.

The Step 6 and Step 7 builders consume only cached Step 3 data and the selected
Step 4 event-aware index.  This module centralizes source parsing and as-of
lookups so shock and state features apply the same point-in-time rules.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from xml.etree import ElementTree

import pandas as pd

from .config import INTERIM_ROOT, PROJECT_ROOT, RAW_ROOT
from .targets import load_selected_sampling_index


SELECTED_SAMPLING_PATH = (
    INTERIM_ROOT / "sampling" / "selected_sampling_index.csv"
)
NORMALIZED_EVENTS_PATH = (
    INTERIM_ROOT / "sampling" / "normalized_event_records.csv"
)


def relative_path(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_manifest(paths: Iterable[Path]) -> list[dict[str, str]]:
    return [
        {"path": relative_path(path), "sha256": sha256_file(path)}
        for path in sorted(set(path.resolve() for path in paths))
    ]


def frame_to_csv_text(frame: pd.DataFrame) -> str:
    buffer = io.StringIO(newline="")
    frame.to_csv(
        buffer,
        index=False,
        lineterminator="\n",
        float_format="%.12g",
    )
    return buffer.getvalue()


def read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def load_feature_index(
    path: Path = SELECTED_SAMPLING_PATH,
) -> pd.DataFrame:
    """Load the selected event-aware index without adding targets or splits."""
    frame = load_selected_sampling_index(path)
    required = {
        "shock_window_start_date",
        "shock_window_end_date",
        "event_count",
        "event_types_json",
        "event_ids_json",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Selected sampling index lacks columns: {missing}")
    for column in ("shock_window_start_date", "shock_window_end_date"):
        parsed = pd.to_datetime(frame[column], errors="coerce")
        if parsed.isna().any():
            raise ValueError(f"Selected index has invalid dates in {column}.")
        frame[column] = parsed.dt.strftime("%Y-%m-%d")
    if not (
        pd.to_datetime(frame["shock_window_start_date"])
        <= pd.to_datetime(frame["shock_window_end_date"])
    ).all():
        raise ValueError("Shock-window end precedes its start.")
    return frame


def load_normalized_event_records(
    path: Path = NORMALIZED_EVENTS_PATH,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Normalized Step 4 events are missing: {path}")
    frame = pd.read_csv(path, keep_default_na=False)
    required = {
        "event_id",
        "event_type",
        "event_observation_period",
        "original_release_date",
        "model_availability_date",
        "state_cutoff_date",
        "event_source",
        "event_source_file",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Normalized event table lacks columns: {missing}")
    if frame["event_id"].duplicated().any():
        raise ValueError("Normalized event table has duplicate event IDs.")
    for column in (
        "original_release_date",
        "model_availability_date",
        "state_cutoff_date",
    ):
        parsed = pd.to_datetime(frame[column], errors="coerce")
        if parsed.isna().any():
            raise ValueError(f"Normalized events have invalid {column}.")
        frame[column] = parsed.dt.strftime("%Y-%m-%d")
    return frame.set_index("event_id", drop=False)


def load_fred_series(
    series_id: str,
    *,
    raw_root: Path = RAW_ROOT,
) -> tuple[pd.Series, Path]:
    path = raw_root / "fred" / f"{series_id}.observations.json"
    payload = read_json_object(path)
    frame = pd.DataFrame(payload.get("observations", []))
    if not {"date", "value"}.issubset(frame.columns):
        raise ValueError(f"Malformed FRED observations in {path}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["date", "value"]).sort_values("date")
    if frame["date"].duplicated().any():
        raise ValueError(f"Duplicate FRED dates in {path}")
    values = frame.set_index("date")["value"].astype(float)
    values.name = series_id
    return values, path


def load_yahoo_series(
    filename: str,
    value_field: str,
    *,
    raw_root: Path = RAW_ROOT,
) -> tuple[pd.Series, Path]:
    path = raw_root / "yahoo" / filename
    frame = pd.read_csv(path, usecols=["Date", value_field])
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame[value_field] = pd.to_numeric(frame[value_field], errors="coerce")
    frame = frame.dropna(subset=["Date", value_field]).sort_values("Date")
    if frame["Date"].duplicated().any():
        raise ValueError(f"Duplicate Yahoo dates in {path}")
    values = frame.set_index("Date")[value_field].astype(float)
    values.name = f"{path.stem}:{value_field}"
    return values, path


@dataclass(frozen=True)
class AsOfResult:
    values: pd.Series
    source_dates: pd.Series
    age_calendar_days: pd.Series


def asof_lookup(values: pd.Series, cutoffs: pd.Series) -> AsOfResult:
    """Return the latest source value at or before each cutoff."""
    source = pd.DataFrame(
        {
            "source_date": pd.to_datetime(values.index, errors="coerce"),
            "source_value": pd.to_numeric(values.to_numpy(), errors="coerce"),
        }
    ).dropna()
    source = source.sort_values("source_date")
    if source["source_date"].duplicated().any():
        raise ValueError("As-of source has duplicate dates.")

    requests = pd.DataFrame(
        {
            "position": range(len(cutoffs)),
            "cutoff_date": pd.to_datetime(cutoffs, errors="coerce"),
        }
    )
    if requests["cutoff_date"].isna().any():
        raise ValueError("As-of requests contain invalid cutoff dates.")
    merged = pd.merge_asof(
        requests.sort_values("cutoff_date"),
        source,
        left_on="cutoff_date",
        right_on="source_date",
        direction="backward",
        allow_exact_matches=True,
    ).sort_values("position")
    if merged["source_value"].isna().any():
        raise ValueError("No historical source value exists before a cutoff.")
    if (merged["source_date"] > merged["cutoff_date"]).any():
        raise ValueError("As-of lookup selected a future source value.")
    output_index = cutoffs.index
    output_values = pd.Series(
        merged["source_value"].to_numpy(dtype=float),
        index=output_index,
        dtype=float,
    )
    output_dates = pd.Series(
        merged["source_date"].dt.strftime("%Y-%m-%d").to_numpy(),
        index=output_index,
        dtype="object",
    )
    ages = pd.Series(
        (
            merged["cutoff_date"] - merged["source_date"]
        ).dt.days.to_numpy(dtype=int),
        index=output_index,
        dtype=int,
    )
    return AsOfResult(output_values, output_dates, ages)


@dataclass
class AlfredVintagePanel:
    series_id: str
    values: dict[tuple[str, str], float]
    cell_sources: dict[tuple[str, str], Path]
    source_paths: list[Path]

    def value(self, observation_date: str, vintage_date: str) -> float:
        key = (observation_date, vintage_date)
        if key not in self.values:
            raise KeyError(
                f"{self.series_id} has no value for observation "
                f"{observation_date} at vintage {vintage_date}."
            )
        return self.values[key]

    def first_release_table(self) -> pd.DataFrame:
        earliest: dict[str, tuple[str, float, Path]] = {}
        for (observation_date, vintage_date), value in self.values.items():
            candidate = (
                vintage_date,
                value,
                self.cell_sources[(observation_date, vintage_date)],
            )
            current = earliest.get(observation_date)
            if current is None or candidate[0] < current[0]:
                earliest[observation_date] = candidate
        records = [
            {
                "observation_date": observation_date,
                "release_date": vintage_date,
                "value": value,
                "source_file": relative_path(source),
            }
            for observation_date, (vintage_date, value, source) in earliest.items()
        ]
        return (
            pd.DataFrame.from_records(records)
            .sort_values(["observation_date", "release_date"])
            .reset_index(drop=True)
        )


def load_alfred_vintage_panel(
    series_id: str,
    *,
    raw_root: Path = RAW_ROOT,
) -> AlfredVintagePanel:
    directory = raw_root / "alfred"
    paths = [
        path
        for path in sorted(directory.glob(f"{series_id}.vintage_*.json"))
        if ".metadata." not in path.name
    ]
    if not paths:
        raise FileNotFoundError(f"No ALFRED chunks found for {series_id}")
    pattern = re.compile(rf"^{re.escape(series_id)}_(\d{{8}})$")
    values: dict[tuple[str, str], float] = {}
    cell_sources: dict[tuple[str, str], Path] = {}
    for path in paths:
        payload = read_json_object(path)
        for observation in payload.get("observations", []):
            observation_date = str(observation["date"])
            for key, raw_value in observation.items():
                match = pattern.match(key)
                if not match or raw_value in (None, "", "."):
                    continue
                vintage_date = datetime.strptime(
                    match.group(1), "%Y%m%d"
                ).date().isoformat()
                value = float(raw_value)
                cell_key = (observation_date, vintage_date)
                if (
                    cell_key in values
                    and abs(values[cell_key] - value) > 1e-12
                ):
                    raise ValueError(
                        f"Conflicting {series_id} ALFRED cell {cell_key}."
                    )
                values[cell_key] = value
                cell_sources[cell_key] = path
    if not values:
        raise ValueError(f"No usable ALFRED values found for {series_id}")
    return AlfredVintagePanel(series_id, values, cell_sources, paths)


def previous_month_start(value: str) -> str:
    period = pd.Period(pd.Timestamp(value), freq="M") - 1
    return period.start_time.date().isoformat()


def months_before_start(value: str, months: int) -> str:
    period = pd.Period(pd.Timestamp(value), freq="M") - months
    return period.start_time.date().isoformat()


def _nearest_label_date(
    label: str,
    *,
    reference_period: pd.Period,
) -> pd.Timestamp | None:
    match = re.fullmatch(r"(\d{1,2})/(\d{1,2})", label.strip())
    if not match:
        return None
    month, day = (int(match.group(1)), int(match.group(2)))
    release_anchor = (reference_period + 1).start_time + pd.Timedelta(days=14)
    candidates: list[pd.Timestamp] = []
    for year in (
        release_anchor.year - 1,
        release_anchor.year,
        release_anchor.year + 1,
    ):
        try:
            candidates.append(pd.Timestamp(year=year, month=month, day=day))
        except ValueError:
            continue
    if not candidates:
        return None
    return min(candidates, key=lambda item: abs(item - release_anchor))


def load_cleveland_cpi_nowcasts(
    *,
    raw_root: Path = RAW_ROOT,
) -> tuple[pd.DataFrame, Path]:
    """Parse headline month-over-month CPI nowcasts from the cached JSON."""
    path = raw_root / "cleveland_fed" / "nowcast_month.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON array in {path}")
    records: list[dict[str, Any]] = []
    for chart_record in payload:
        chart = chart_record.get("chart", {})
        subcaption = str(chart.get("subcaption", ""))
        match = re.fullmatch(r"(\d{4})-(\d{1,2})", subcaption)
        if not match:
            continue
        reference_period = pd.Period(
            year=int(match.group(1)),
            month=int(match.group(2)),
            freq="M",
        )
        series = next(
            (
                item
                for item in chart_record.get("dataset", [])
                if item.get("seriesname") == "CPI Inflation"
            ),
            None,
        )
        if series is None:
            continue
        data = series.get("data", [])
        for point in data:
            raw_value = point.get("value")
            if raw_value in (None, ""):
                continue
            tooltext = str(point.get("tooltext", ""))
            label_match = re.search(
                r"\{br\}(\d{1,2}/\d{1,2})\{br\}", tooltext
            )
            if label_match is None:
                continue
            label = label_match.group(1)
            forecast_date = _nearest_label_date(
                label, reference_period=reference_period
            )
            if forecast_date is None:
                continue
            records.append(
                {
                    "reference_period": reference_period.start_time.date().isoformat(),
                    "forecast_date": forecast_date.date().isoformat(),
                    "expected_cpi_mom_pct": float(raw_value),
                }
            )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise ValueError("No Cleveland headline CPI nowcasts were parsed.")
    conflicts = (
        frame.groupby(["reference_period", "forecast_date"])[
            "expected_cpi_mom_pct"
        ]
        .nunique()
        .gt(1)
    )
    if conflicts.any():
        raise ValueError("Conflicting Cleveland CPI nowcasts on one date.")
    frame = (
        frame.drop_duplicates(["reference_period", "forecast_date"])
        .sort_values(["reference_period", "forecast_date"])
        .reset_index(drop=True)
    )
    return frame, path


def load_cleveland_cpi_actuals(
    *,
    raw_root: Path = RAW_ROOT,
) -> tuple[pd.DataFrame, Path]:
    """Parse release-date headline CPI actuals embedded in the nowcast cache."""
    path = raw_root / "cleveland_fed" / "nowcast_month.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON array in {path}")
    records: list[dict[str, Any]] = []
    for chart_record in payload:
        subcaption = str(
            chart_record.get("chart", {}).get("subcaption", "")
        )
        match = re.fullmatch(r"(\d{4})-(\d{1,2})", subcaption)
        if not match:
            continue
        reference_period = pd.Period(
            year=int(match.group(1)),
            month=int(match.group(2)),
            freq="M",
        )
        series = next(
            (
                item
                for item in chart_record.get("dataset", [])
                if item.get("seriesname") == "Actual CPI Inflation"
            ),
            None,
        )
        if series is None:
            continue
        for point in series.get("data", []):
            raw_value = point.get("value")
            if raw_value in (None, ""):
                continue
            tooltext = str(point.get("tooltext", ""))
            label_match = re.search(
                r"\{br\}(\d{1,2}/\d{1,2})\{br\}", tooltext
            )
            if label_match is None:
                continue
            actual_date = _nearest_label_date(
                label_match.group(1),
                reference_period=reference_period,
            )
            if actual_date is None:
                continue
            records.append(
                {
                    "reference_period": (
                        reference_period.start_time.date().isoformat()
                    ),
                    "actual_release_date": actual_date.date().isoformat(),
                    "actual_cpi_mom_pct": float(raw_value),
                }
            )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise ValueError("No Cleveland headline CPI actuals were parsed.")
    conflicts = (
        frame.groupby(["reference_period", "actual_release_date"])[
            "actual_cpi_mom_pct"
        ]
        .nunique()
        .gt(1)
    )
    if conflicts.any():
        raise ValueError("Conflicting Cleveland CPI actuals on one date.")
    frame = (
        frame.drop_duplicates(["reference_period", "actual_release_date"])
        .sort_values(["reference_period", "actual_release_date"])
        .reset_index(drop=True)
    )
    return frame, path


def _normalized_xlsx_target(target: str) -> str:
    path = PurePosixPath(target)
    if not str(path).startswith("xl/"):
        path = PurePosixPath("xl") / path
    parts: list[str] = []
    for part in path.parts:
        if part == "..":
            if parts:
                parts.pop()
        elif part not in ("", "."):
            parts.append(part)
    return "/".join(parts)


def read_xlsx_sheet_rows(
    path: Path,
    sheet_name: str,
) -> list[dict[str, str]]:
    """Read cached XLSX cell values using only the Python standard library."""
    namespace = {
        "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    }
    relationship_namespace = (
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    )
    with zipfile.ZipFile(path) as workbook:
        workbook_xml = ElementTree.fromstring(workbook.read("xl/workbook.xml"))
        relationships = ElementTree.fromstring(
            workbook.read("xl/_rels/workbook.xml.rels")
        )
        targets = {
            relationship.attrib["Id"]: relationship.attrib["Target"]
            for relationship in relationships
        }
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            shared_xml = ElementTree.fromstring(
                workbook.read("xl/sharedStrings.xml")
            )
            for item in shared_xml.findall("m:si", namespace):
                shared_strings.append(
                    "".join(
                        node.text or ""
                        for node in item.findall(".//m:t", namespace)
                    )
                )

        target: str | None = None
        sheets = workbook_xml.find("m:sheets", namespace)
        if sheets is None:
            raise ValueError(f"Workbook has no sheets list: {path}")
        for sheet in sheets:
            if sheet.attrib.get("name") == sheet_name:
                relationship_id = sheet.attrib[
                    f"{{{relationship_namespace}}}id"
                ]
                target = _normalized_xlsx_target(targets[relationship_id])
                break
        if target is None:
            raise ValueError(f"Workbook has no {sheet_name} sheet: {path}")
        sheet_xml = ElementTree.fromstring(workbook.read(target))

    rows: list[dict[str, str]] = []
    for row in sheet_xml.findall(".//m:sheetData/m:row", namespace):
        values: dict[str, str] = {}
        for cell in row.findall("m:c", namespace):
            match = re.match(r"[A-Z]+", cell.attrib["r"])
            if match is None:
                continue
            column = match.group()
            cell_type = cell.attrib.get("t")
            value_node = cell.find("m:v", namespace)
            if value_node is None:
                inline = cell.find("m:is/m:t", namespace)
                if inline is not None:
                    values[column] = inline.text or ""
                continue
            raw_value = value_node.text or ""
            if cell_type == "s":
                values[column] = shared_strings[int(raw_value)]
            else:
                values[column] = raw_value
        rows.append(values)
    return rows


def excel_serial_date(value: str) -> pd.Timestamp:
    return pd.Timestamp(datetime(1899, 12, 30) + timedelta(days=float(value)))


def load_atlanta_gdp_track_record(
    *,
    raw_root: Path = RAW_ROOT,
) -> tuple[pd.DataFrame, Path]:
    path = raw_root / "atlanta_fed" / "GDPTrackingModelDataAndForecasts.xlsx"
    rows = read_xlsx_sheet_rows(path, "TrackRecord")
    records: list[dict[str, Any]] = []
    for row in rows:
        if not all(column in row for column in ("A", "B", "C", "D")):
            continue
        try:
            quarter_end = excel_serial_date(row["A"])
            forecast = float(row["B"])
            actual = float(row["C"])
            release_date = excel_serial_date(row["D"])
        except (TypeError, ValueError):
            continue
        if quarter_end.year < 2000 or release_date.year < 2000:
            continue
        quarter = ((quarter_end.month - 1) // 3) + 1
        records.append(
            {
                "observation_period": f"{quarter_end.year}-Q{quarter}",
                "quarter_end_date": quarter_end.date().isoformat(),
                "forecast_publication_date": "",
                "forecast_timing_rule": (
                    "Workbook column B is explicitly the model forecast right "
                    "before BEA's advance estimate; no separate timestamp is "
                    "provided in the cached TrackRecord sheet."
                ),
                "release_date": release_date.date().isoformat(),
                "gdpnow_forecast_qoq_saar_pct": forecast,
                "bea_advance_qoq_saar_pct": actual,
                "gdp_growth_surprise_pct": actual - forecast,
            }
        )
    frame = (
        pd.DataFrame.from_records(records)
        .drop_duplicates("observation_period", keep="first")
        .sort_values("release_date")
        .reset_index(drop=True)
    )
    if frame.empty:
        raise ValueError("No Atlanta GDP TrackRecord rows were parsed.")
    if frame["observation_period"].duplicated().any():
        raise ValueError("GDP TrackRecord has duplicate quarters.")
    return frame, path


def load_kansas_city_pmi_events(
    *,
    raw_root: Path = RAW_ROOT,
) -> tuple[pd.DataFrame, Path]:
    path = (
        raw_root
        / "kansas_city_fed"
        / "manufacturing_pmi_proxy_events.json"
    )
    payload = read_json_object(path)
    frame = pd.DataFrame(payload.get("observations", []))
    required = {
        "observation_month",
        "release_date",
        "model_availability_date",
        "pmi_style_level",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"Malformed Kansas City PMI cache: {path}")
    frame["pmi_style_level"] = pd.to_numeric(
        frame["pmi_style_level"], errors="coerce"
    )
    frame = frame.dropna(subset=["pmi_style_level"]).sort_values(
        "observation_month"
    )
    if frame["observation_month"].duplicated().any():
        raise ValueError("Kansas City PMI cache has duplicate months.")
    return frame.reset_index(drop=True), path


def load_fomc_decisions(
    *,
    raw_root: Path = RAW_ROOT,
) -> tuple[pd.DataFrame, Path]:
    path = raw_root / "federal_reserve" / "fomc_decisions.json"
    payload = read_json_object(path)
    frame = pd.DataFrame(payload.get("events", []))
    required = {
        "announcement_date",
        "model_availability_date",
        "policy_change_bps",
        "statement_source_file",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"Malformed FOMC decision cache: {path}")
    frame["policy_change_bps"] = pd.to_numeric(
        frame["policy_change_bps"], errors="coerce"
    )
    if frame["policy_change_bps"].isna().any():
        raise ValueError("FOMC decision cache has missing policy changes.")

    def identity_source(row: pd.Series) -> str:
        statement = row.get("statement_source_file")
        if pd.notna(statement) and str(statement):
            return str(statement)
        calendar = row.get("calendar_source_file")
        if pd.notna(calendar) and str(calendar):
            return str(calendar)
        raise ValueError("FOMC decision lacks an identity source file.")

    frame["event_id"] = frame.apply(
        lambda row: (
            f"fomc:{row['announcement_date']}:"
            f"{identity_source(row)}"
        ),
        axis=1,
    )
    if frame["event_id"].duplicated().any():
        raise ValueError("FOMC decision cache produces duplicate event IDs.")
    return frame.set_index("event_id", drop=False), path
