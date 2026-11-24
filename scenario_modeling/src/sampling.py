"""Deterministic Step 4 sampling indexes built only from Step 3 caches.

This module defines timing metadata; it deliberately does not calculate shock
features, state features, target values, data splits, or model inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import statistics
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence
from xml.etree import ElementTree

import pandas as pd

from .cache import write_text_atomic
from .config import INTERIM_ROOT, PROJECT_ROOT, RAW_ROOT
from .trading_calendar import is_us_equity_trading_session


SAMPLING_ROOT = INTERIM_ROOT / "sampling"
SHARED_START = date(2013, 8, 1)
TARGET_HORIZON_SESSIONS = 21
EVENT_TYPE_ORDER = ("fomc", "cpi", "unemployment", "gdp", "pmi")
EVENT_TYPE_RANK = {name: position for position, name in enumerate(EVENT_TYPE_ORDER)}


@dataclass(frozen=True)
class TargetSpec:
    target: str
    relative_path: str
    value_field: str
    source: str
    value_policy: str
    file_format: str = "yahoo_csv"


TARGET_SPECS = (
    TargetSpec(
        "sp500",
        "yahoo/GSPC.csv",
        "Close",
        "Yahoo ^GSPC",
        "unadjusted index Close",
    ),
    TargetSpec(
        "nasdaq",
        "yahoo/IXIC.csv",
        "Close",
        "Yahoo ^IXIC",
        "unadjusted index Close",
    ),
    TargetSpec(
        "ten_year_yield_bps",
        "fred/DGS10.observations.json",
        "value",
        "FRED DGS10",
        "Treasury yield in percentage points; later target is a basis-point difference",
        "fred_json",
    ),
    TargetSpec(
        "dxy",
        "yahoo/DX-Y.NYB.csv",
        "Close",
        "Yahoo DX-Y.NYB",
        "unadjusted index Close",
    ),
    TargetSpec(
        "gold",
        "yahoo/GLD.csv",
        "Adj Close",
        "Yahoo GLD",
        "distribution-adjusted ETF close",
    ),
    TargetSpec(
        "oil",
        "yahoo/CL_F.csv",
        "Close",
        "Yahoo CL=F",
        "native continuous-futures Close; no dividend adjustment",
    ),
    TargetSpec(
        "technology",
        "yahoo/XLK.csv",
        "Adj Close",
        "Yahoo XLK",
        "distribution-adjusted ETF close",
    ),
    TargetSpec(
        "energy",
        "yahoo/XLE.csv",
        "Adj Close",
        "Yahoo XLE",
        "distribution-adjusted ETF close",
    ),
    TargetSpec(
        "financials",
        "yahoo/XLF.csv",
        "Adj Close",
        "Yahoo XLF",
        "distribution-adjusted ETF close",
    ),
    TargetSpec(
        "utilities",
        "yahoo/XLU.csv",
        "Adj Close",
        "Yahoo XLU",
        "distribution-adjusted ETF close",
    ),
    TargetSpec(
        "healthcare",
        "yahoo/XLV.csv",
        "Adj Close",
        "Yahoo XLV",
        "distribution-adjusted ETF close",
    ),
    TargetSpec(
        "consumer_discretionary",
        "yahoo/XLY.csv",
        "Adj Close",
        "Yahoo XLY",
        "distribution-adjusted ETF close",
    ),
)


NORMALIZED_EVENT_COLUMNS = (
    "event_id",
    "event_type",
    "event_source",
    "event_source_file",
    "event_observation_period",
    "original_release_date",
    "original_release_time",
    "original_release_timezone",
    "release_date_kind",
    "source_model_availability_date",
    "model_availability_date",
    "availability_rule",
    "next_session_shift_applied",
    "conservative_next_session_shift",
    "calendar_normalization_applied",
    "state_cutoff_date",
    "shock_window_start_date",
    "shock_window_end_date",
    "target_baseline_date",
    "target_end_date",
    "target_horizon_sessions",
    "has_21_future_sessions",
    "full_horizon_within_shared_coverage",
    "all_target_values_available",
    "missing_target_fields_json",
    "label_eligible",
    "ineligibility_reason",
)

WEEKLY_INDEX_COLUMNS = (
    "sampling_design",
    "sample_date",
    "anchor_date",
    "iso_year",
    "iso_week",
    "state_cutoff_date",
    "interval_start_exclusive",
    "interval_end_inclusive",
    "shock_window_start_date",
    "shock_window_end_date",
    "target_baseline_date",
    "target_end_date",
    "target_horizon_sessions",
    "has_21_future_sessions",
    "full_horizon_within_shared_coverage",
    "all_target_values_available",
    "missing_target_fields_json",
    "label_eligible",
    "ineligibility_reason",
    "anchor_is_shifted",
    "anchor_shift_reason",
    "initial_partial_interval",
    "event_count",
    "event_types_json",
    "event_ids_json",
    "event_observation_periods_json",
    "event_release_dates_json",
    "event_availability_dates_json",
    "event_sources_json",
    "multiple_same_type_event_types_json",
)

EVENT_AWARE_INDEX_COLUMNS = (
    "sampling_design",
    "sample_date",
    "model_availability_date",
    "state_cutoff_date",
    "shock_window_start_date",
    "shock_window_end_date",
    "target_baseline_date",
    "target_end_date",
    "target_horizon_sessions",
    "has_21_future_sessions",
    "full_horizon_within_shared_coverage",
    "all_target_values_available",
    "missing_target_fields_json",
    "label_eligible",
    "ineligibility_reason",
    "event_count",
    "collision_count",
    "duplicate_candidates_removed",
    "event_types_json",
    "event_ids_json",
    "event_observation_periods_json",
    "event_release_dates_json",
    "event_release_times_json",
    "event_timezones_json",
    "event_sources_json",
    "event_release_date_kinds_json",
    "next_session_shift_applied",
    "conservative_next_session_shift",
    "calendar_normalization_applied",
)


def _as_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return pd.Timestamp(value).date()


def _date_text(value: date | None) -> str:
    return value.isoformat() if value is not None else ""


def _json_list(values: Iterable[Any]) -> str:
    return json.dumps(list(values), ensure_ascii=True, separators=(",", ":"))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frame_to_csv_text(frame: pd.DataFrame) -> str:
    buffer = io.StringIO(newline="")
    frame.to_csv(buffer, index=False, lineterminator="\n")
    return buffer.getvalue()


def _frame_from_records(
    records: Sequence[dict[str, Any]], columns: Sequence[str]
) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(records, columns=list(columns))
    return frame.fillna("")


def _load_yahoo_dates(path: Path, value_field: str) -> set[date]:
    frame = pd.read_csv(path, usecols=["Date", value_field])
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame[value_field] = pd.to_numeric(frame[value_field], errors="coerce")
    frame = frame.dropna(subset=["Date", value_field])
    return {timestamp.date() for timestamp in frame["Date"]}


def _load_fred_dates(path: Path) -> set[date]:
    payload = _read_json(path)
    values: set[date] = set()
    for row in payload.get("observations", []):
        raw_value = row.get("value")
        if raw_value in (None, "", "."):
            continue
        try:
            float(raw_value)
        except (TypeError, ValueError):
            continue
        values.add(_as_date(row["date"]))
    return values


def load_target_availability(
    raw_root: Path = RAW_ROOT,
) -> tuple[dict[str, set[date]], list[Path]]:
    """Load only the selected target fields and return their usable dates."""
    availability: dict[str, set[date]] = {}
    paths: list[Path] = []
    for spec in TARGET_SPECS:
        path = raw_root / spec.relative_path
        if not path.exists():
            raise FileNotFoundError(f"Required selected target cache is missing: {path}")
        paths.append(path)
        if spec.file_format == "fred_json":
            dates = _load_fred_dates(path)
        else:
            dates = _load_yahoo_dates(path, spec.value_field)
        if not dates:
            raise ValueError(f"No usable {spec.value_field!r} values in {path}")
        availability[spec.target] = dates
    return availability, paths


def load_canonical_calendar(raw_root: Path = RAW_ROOT) -> tuple[date, ...]:
    """Use observed non-missing ^GSPC Close dates as canonical sessions."""
    path = raw_root / "yahoo" / "GSPC.csv"
    sessions = sorted(_load_yahoo_dates(path, "Close"))
    if not sessions:
        raise ValueError("The selected ^GSPC cache contains no usable Close dates.")
    if len(sessions) != len(set(sessions)):
        raise ValueError("The selected ^GSPC cache contains duplicate session dates.")
    return tuple(sessions)


def previous_observed_session(
    sessions: Sequence[date], value: date, *, inclusive: bool = False
) -> date:
    """Return the previous observed session from a sorted session sequence."""
    index = pd.DatetimeIndex(sessions)
    side = "right" if inclusive else "left"
    position = int(index.searchsorted(pd.Timestamp(value), side=side)) - 1
    if position < 0:
        raise ValueError(f"No previous observed session is available for {value}")
    return sessions[position]


def next_observed_session(
    sessions: Sequence[date], value: date, *, strictly_after: bool = True
) -> date:
    """Return the next observed session from a sorted session sequence."""
    index = pd.DatetimeIndex(sessions)
    side = "right" if strictly_after else "left"
    position = int(index.searchsorted(pd.Timestamp(value), side=side))
    if position >= len(sessions):
        raise ValueError(f"No next observed session is available for {value}")
    return sessions[position]


def nth_future_session(
    sessions: Sequence[date], baseline: date, horizon: int
) -> date | None:
    if horizon < 1:
        raise ValueError("The target horizon must be positive.")
    positions = {session: position for position, session in enumerate(sessions)}
    if baseline not in positions:
        raise ValueError(f"Target baseline is not a canonical session: {baseline}")
    future_position = positions[baseline] + horizon
    return sessions[future_position] if future_position < len(sessions) else None


def weekly_anchors(
    sessions: Sequence[date], start: date, end: date
) -> list[dict[str, Any]]:
    """Return the final observed session of every ISO week in the range."""
    grouped: dict[tuple[int, int], list[date]] = defaultdict(list)
    for session in sessions:
        if start <= session <= end:
            iso = session.isocalendar()
            grouped[(iso.year, iso.week)].append(session)

    anchors: list[dict[str, Any]] = []
    for (iso_year, iso_week), values in sorted(grouped.items()):
        anchor = max(values)
        friday = date.fromisocalendar(iso_year, iso_week, 5)
        shifted = anchor != friday
        if not shifted:
            reason = ""
        elif friday > end:
            reason = "shared_period_end_partial_week"
        elif not is_us_equity_trading_session(friday):
            reason = "friday_market_holiday"
        else:
            reason = "friday_absent_from_observed_calendar"
        anchors.append(
            {
                "anchor_date": anchor,
                "iso_year": iso_year,
                "iso_week": iso_week,
                "anchor_is_shifted": shifted,
                "anchor_shift_reason": reason,
            }
        )
    return anchors


def _normalize_source_availability(
    sessions: Sequence[date], source_date: date
) -> tuple[date, bool]:
    if source_date in set(sessions):
        return source_date, False
    # A structured source can contain a scheduled event later than the last
    # cached market observation. It will be filtered by the shared-data end
    # before candidate construction, so no projected session is invented here.
    if source_date > sessions[-1]:
        return source_date, False
    return next_observed_session(sessions, source_date, strictly_after=False), True


def _state_cutoff_for_release(
    sessions: Sequence[date],
    release_date: date,
    release_time: str | None,
) -> date:
    """Use the latest completed close that still precedes the event."""
    parsed_time: time | None = None
    if release_time:
        parsed_time = time.fromisoformat(release_time)
    if (
        release_date in set(sessions)
        and parsed_time is not None
        and parsed_time >= time(16, 0)
    ):
        return release_date
    return previous_observed_session(sessions, release_date)


def _event_record(
    *,
    event_id: str,
    event_type: str,
    event_source: str,
    event_source_file: str,
    observation_period: str,
    release_date: date,
    release_time: str | None,
    release_timezone: str | None,
    release_date_kind: str,
    source_model_availability_date: date | None,
    availability_rule: str,
    conservative_next_session_shift: bool,
    sessions: Sequence[date],
) -> dict[str, Any]:
    if source_model_availability_date is None:
        source_availability = next_observed_session(
            sessions, release_date, strictly_after=True
        )
    else:
        source_availability = source_model_availability_date
    availability, calendar_normalized = _normalize_source_availability(
        sessions, source_availability
    )
    state_cutoff = _state_cutoff_for_release(
        sessions, release_date, release_time
    )
    return {
        "event_id": event_id,
        "event_type": event_type,
        "event_source": event_source,
        "event_source_file": event_source_file,
        "event_observation_period": observation_period,
        "original_release_date": release_date,
        "original_release_time": release_time or "",
        "original_release_timezone": release_timezone or "",
        "release_date_kind": release_date_kind,
        "source_model_availability_date": source_availability,
        "model_availability_date": availability,
        "availability_rule": availability_rule,
        "next_session_shift_applied": availability > release_date,
        "conservative_next_session_shift": conservative_next_session_shift,
        "calendar_normalization_applied": calendar_normalized,
        "state_cutoff_date": state_cutoff,
        "shock_window_start_date": state_cutoff,
        "shock_window_end_date": availability,
        "target_baseline_date": state_cutoff,
    }


def _load_fomc_events(
    sessions: Sequence[date], raw_root: Path
) -> tuple[list[dict[str, Any]], list[Path]]:
    path = raw_root / "federal_reserve" / "fomc_decisions.json"
    payload = _read_json(path)
    records: list[dict[str, Any]] = []
    for position, row in enumerate(payload.get("events", []), start=1):
        release_date = _as_date(row["announcement_date"])
        release_time = row.get("announcement_time")
        source_availability = _as_date(row["model_availability_date"])
        source_file = row.get("statement_source_file") or row.get(
            "calendar_source_file", ""
        )
        identity_piece = source_file or f"record-{position:03d}"
        record = _event_record(
            event_id=f"fomc:{release_date.isoformat()}:{identity_piece}",
            event_type="fomc",
            event_source="Federal Reserve FOMC decision history",
            event_source_file=f"data/scenario_playground/raw/federal_reserve/{source_file}",
            observation_period=str(row.get("official_event_heading") or release_date),
            release_date=release_date,
            release_time=release_time,
            release_timezone=row.get("announcement_timezone"),
            release_date_kind="official_fomc_announcement",
            source_model_availability_date=source_availability,
            availability_rule=str(row.get("model_availability_rule", "")),
            conservative_next_session_shift=release_time in (None, ""),
            sessions=sessions,
        )
        records.append(record)
    return records, [path]


def _load_alfred_first_appearance_events(
    *,
    series_id: str,
    event_type: str,
    source_name: str,
    sessions: Sequence[date],
    raw_root: Path,
) -> tuple[list[dict[str, Any]], list[Path]]:
    directory = raw_root / "alfred"
    paths = [
        path
        for path in sorted(directory.glob(f"{series_id}.vintage_*.json"))
        if ".metadata." not in path.name
    ]
    if not paths:
        raise FileNotFoundError(f"No ALFRED vintage chunks found for {series_id}")

    earliest: dict[str, tuple[date, Path]] = {}
    suffix_pattern = re.compile(rf"^{re.escape(series_id)}_(\d{{8}})$")
    for path in paths:
        payload = _read_json(path)
        for observation in payload.get("observations", []):
            observation_period = str(observation["date"])
            for key, raw_value in observation.items():
                match = suffix_pattern.match(key)
                if not match or raw_value in (None, "", "."):
                    continue
                vintage_date = datetime.strptime(match.group(1), "%Y%m%d").date()
                current = earliest.get(observation_period)
                candidate = (vintage_date, path)
                if current is None or candidate < current:
                    earliest[observation_period] = candidate

    records: list[dict[str, Any]] = []
    for observation_period, (release_date, source_path) in sorted(
        earliest.items(), key=lambda item: (item[1][0], item[0])
    ):
        record = _event_record(
            event_id=f"{event_type}:{observation_period}:{release_date.isoformat()}",
            event_type=event_type,
            event_source=source_name,
            event_source_file=_relative(source_path),
            observation_period=observation_period,
            release_date=release_date,
            release_time=None,
            release_timezone=None,
            release_date_kind="ALFRED_INITIAL_APPEARANCE",
            source_model_availability_date=None,
            availability_rule=(
                "Earliest non-missing ALFRED output_type=2 vintage; exact release "
                "time is unavailable, so use the next observed equity session."
            ),
            conservative_next_session_shift=True,
            sessions=sessions,
        )
        records.append(record)
    return records, paths


def _excel_serial_date(value: str) -> date:
    return (datetime(1899, 12, 30) + timedelta(days=float(value))).date()


def _load_gdp_events(
    sessions: Sequence[date], raw_root: Path
) -> tuple[list[dict[str, Any]], list[Path]]:
    path = raw_root / "atlanta_fed" / "GDPTrackingModelDataAndForecasts.xlsx"
    namespace = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
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
        sheet_target: str | None = None
        sheets = workbook_xml.find("m:sheets", namespace)
        if sheets is None:
            raise ValueError("Atlanta Fed workbook does not contain a sheets list.")
        for sheet in sheets:
            if sheet.attrib.get("name") == "TrackRecord":
                relationship_id = sheet.attrib[
                    f"{{{relationship_namespace}}}id"
                ]
                sheet_target = targets[relationship_id]
                break
        if sheet_target is None:
            raise ValueError("Atlanta Fed workbook has no TrackRecord worksheet.")
        sheet_path = (
            sheet_target
            if sheet_target.startswith("xl/")
            else f"xl/{sheet_target.lstrip('/')}"
        )
        sheet_xml = ElementTree.fromstring(workbook.read(sheet_path))

    records: list[dict[str, Any]] = []
    for row in sheet_xml.findall(".//m:sheetData/m:row", namespace):
        values: dict[str, str] = {}
        for cell in row.findall("m:c", namespace):
            value_node = cell.find("m:v", namespace)
            if value_node is None:
                continue
            column = re.match(r"[A-Z]+", cell.attrib["r"])
            if column:
                values[column.group()] = value_node.text or ""
        if not all(key in values for key in ("A", "B", "C", "D")):
            continue
        try:
            quarter_end = _excel_serial_date(values["A"])
            release_date = _excel_serial_date(values["D"])
            float(values["B"])
            float(values["C"])
        except (TypeError, ValueError):
            continue
        # Exclude the workbook's numeric summary row, which looks like an
        # early-1900s Excel date when interpreted as a quarter.
        if quarter_end.year < 2000 or release_date.year < 2000:
            continue
        quarter = ((quarter_end.month - 1) // 3) + 1
        observation_period = f"{quarter_end.year}-Q{quarter}"
        records.append(
            _event_record(
                event_id=f"gdp:{observation_period}:{release_date.isoformat()}",
                event_type="gdp",
                event_source="Atlanta Fed GDPNow TrackRecord / BEA advance estimate",
                event_source_file=(
                    "data/scenario_playground/raw/atlanta_fed/"
                    "GDPTrackingModelDataAndForecasts.xlsx#TrackRecord"
                ),
                observation_period=observation_period,
                release_date=release_date,
                release_time=None,
                release_timezone=None,
                release_date_kind="official_bea_advance_date_in_atlanta_track_record",
                source_model_availability_date=None,
                availability_rule=(
                    "TrackRecord BEA advance release date; exact cached time is "
                    "unavailable, so use the next observed equity session."
                ),
                conservative_next_session_shift=True,
                sessions=sessions,
            )
        )
    return records, [path]


def _load_pmi_events(
    sessions: Sequence[date], raw_root: Path
) -> tuple[list[dict[str, Any]], list[Path]]:
    path = (
        raw_root
        / "kansas_city_fed"
        / "manufacturing_pmi_proxy_events.json"
    )
    payload = _read_json(path)
    records: list[dict[str, Any]] = []
    for row in payload.get("observations", []):
        observation_period = str(row["observation_month"])
        release_date = _as_date(row["release_date"])
        source_availability = _as_date(row["model_availability_date"])
        records.append(
            _event_record(
                event_id=f"pmi:{observation_period}:{release_date.isoformat()}",
                event_type="pmi",
                event_source=(
                    "Federal Reserve Bank of Kansas City manufacturing "
                    "PMI-style proxy"
                ),
                event_source_file=_relative(path),
                observation_period=observation_period,
                release_date=release_date,
                release_time=row.get("release_time"),
                release_timezone=row.get("release_timezone"),
                release_date_kind=str(row.get("release_source_kind", "")),
                source_model_availability_date=source_availability,
                availability_rule=str(row.get("model_availability_rule", "")),
                conservative_next_session_shift=True,
                sessions=sessions,
            )
        )
    return records, [path]


def _label_metadata(
    baseline: date,
    *,
    sessions: Sequence[date],
    shared_end: date,
    target_availability: dict[str, set[date]],
) -> dict[str, Any]:
    target_end = nth_future_session(
        sessions, baseline, TARGET_HORIZON_SESSIONS
    )
    has_future_sessions = target_end is not None
    within_shared_coverage = bool(
        target_end is not None and target_end <= shared_end
    )
    missing_targets: list[str] = []
    if target_end is not None:
        for spec in TARGET_SPECS:
            available = target_availability[spec.target]
            if baseline not in available or target_end not in available:
                missing_targets.append(spec.target)
    all_targets_available = bool(target_end is not None and not missing_targets)
    eligible = (
        has_future_sessions
        and within_shared_coverage
        and all_targets_available
    )

    reasons: list[str] = []
    if not has_future_sessions:
        reasons.append("fewer_than_21_future_gspc_sessions")
    elif not within_shared_coverage:
        reasons.append("target_end_after_shared_target_coverage")
    if missing_targets:
        reasons.append("missing_selected_target_values")
    return {
        "target_end_date": _date_text(target_end),
        "target_horizon_sessions": TARGET_HORIZON_SESSIONS,
        "has_21_future_sessions": has_future_sessions,
        "full_horizon_within_shared_coverage": within_shared_coverage,
        "all_target_values_available": all_targets_available,
        "missing_target_fields_json": _json_list(missing_targets),
        "label_eligible": eligible,
        "ineligibility_reason": "|".join(reasons),
    }


def load_normalized_events(
    *,
    sessions: Sequence[date],
    target_availability: dict[str, set[date]],
    shared_start: date,
    shared_end: date,
    raw_root: Path = RAW_ROOT,
) -> tuple[pd.DataFrame, list[Path]]:
    """Load the five selected structured event paths and normalize timing."""
    records: list[dict[str, Any]] = []
    source_paths: list[Path] = []

    loaders = (
        lambda: _load_fomc_events(sessions, raw_root),
        lambda: _load_alfred_first_appearance_events(
            series_id="CPIAUCSL",
            event_type="cpi",
            source_name="FRED/ALFRED CPIAUCSL first vintage",
            sessions=sessions,
            raw_root=raw_root,
        ),
        lambda: _load_alfred_first_appearance_events(
            series_id="UNRATE",
            event_type="unemployment",
            source_name="FRED/ALFRED UNRATE first vintage",
            sessions=sessions,
            raw_root=raw_root,
        ),
        lambda: _load_gdp_events(sessions, raw_root),
        lambda: _load_pmi_events(sessions, raw_root),
    )
    for loader in loaders:
        loaded_records, loaded_paths = loader()
        records.extend(loaded_records)
        source_paths.extend(loaded_paths)

    filtered: list[dict[str, Any]] = []
    for record in records:
        availability = _as_date(record["model_availability_date"])
        state_cutoff = _as_date(record["state_cutoff_date"])
        if not (shared_start <= availability <= shared_end):
            continue
        if state_cutoff < shared_start:
            continue
        label = _label_metadata(
            state_cutoff,
            sessions=sessions,
            shared_end=shared_end,
            target_availability=target_availability,
        )
        normalized = dict(record)
        for column in (
            "original_release_date",
            "source_model_availability_date",
            "model_availability_date",
            "state_cutoff_date",
            "shock_window_start_date",
            "shock_window_end_date",
            "target_baseline_date",
        ):
            normalized[column] = _date_text(_as_date(normalized[column]))
        normalized.update(label)
        filtered.append(normalized)

    filtered.sort(
        key=lambda row: (
            row["model_availability_date"],
            EVENT_TYPE_RANK[row["event_type"]],
            row["original_release_date"],
            row["event_id"],
        )
    )
    event_ids = [row["event_id"] for row in filtered]
    if len(event_ids) != len(set(event_ids)):
        duplicates = [
            event_id
            for event_id, count in Counter(event_ids).items()
            if count > 1
        ]
        raise ValueError(f"Duplicate normalized event identities: {duplicates}")
    frame = _frame_from_records(filtered, NORMALIZED_EVENT_COLUMNS)
    return frame, sorted(set(source_paths))


def construct_weekly_candidate(
    *,
    sessions: Sequence[date],
    events: pd.DataFrame,
    target_availability: dict[str, set[date]],
    shared_start: date,
    shared_end: date,
) -> pd.DataFrame:
    """Build one row at the final observed session of every ISO week."""
    anchor_records = weekly_anchors(sessions, shared_start, shared_end)
    event_records = events.to_dict("records")
    rows: list[dict[str, Any]] = []
    first_session = next_observed_session(
        sessions, shared_start, strictly_after=False
    )

    previous_anchor: date | None = None
    for anchor_record in anchor_records:
        anchor = _as_date(anchor_record["anchor_date"])
        state_cutoff = previous_anchor or first_session
        interval_events = [
            row
            for row in event_records
            if state_cutoff < _as_date(row["model_availability_date"]) <= anchor
        ]
        interval_events.sort(
            key=lambda row: (
                row["model_availability_date"],
                EVENT_TYPE_RANK[row["event_type"]],
                row["original_release_date"],
                row["event_id"],
            )
        )
        type_counts = Counter(row["event_type"] for row in interval_events)
        repeated_types = sorted(
            (name for name, count in type_counts.items() if count > 1),
            key=EVENT_TYPE_RANK.get,
        )
        label = _label_metadata(
            state_cutoff,
            sessions=sessions,
            shared_end=shared_end,
            target_availability=target_availability,
        )
        row = {
            "sampling_design": "WEEKLY",
            "sample_date": _date_text(anchor),
            "anchor_date": _date_text(anchor),
            "iso_year": anchor_record["iso_year"],
            "iso_week": anchor_record["iso_week"],
            "state_cutoff_date": _date_text(state_cutoff),
            "interval_start_exclusive": _date_text(state_cutoff),
            "interval_end_inclusive": _date_text(anchor),
            "shock_window_start_date": _date_text(state_cutoff),
            "shock_window_end_date": _date_text(anchor),
            "target_baseline_date": _date_text(state_cutoff),
            **label,
            "anchor_is_shifted": anchor_record["anchor_is_shifted"],
            "anchor_shift_reason": anchor_record["anchor_shift_reason"],
            "initial_partial_interval": previous_anchor is None,
            "event_count": len(interval_events),
            "event_types_json": _json_list(
                row["event_type"] for row in interval_events
            ),
            "event_ids_json": _json_list(
                row["event_id"] for row in interval_events
            ),
            "event_observation_periods_json": _json_list(
                row["event_observation_period"] for row in interval_events
            ),
            "event_release_dates_json": _json_list(
                row["original_release_date"] for row in interval_events
            ),
            "event_availability_dates_json": _json_list(
                row["model_availability_date"] for row in interval_events
            ),
            "event_sources_json": _json_list(
                row["event_source"] for row in interval_events
            ),
            "multiple_same_type_event_types_json": _json_list(repeated_types),
        }
        rows.append(row)
        previous_anchor = anchor
    return _frame_from_records(rows, WEEKLY_INDEX_COLUMNS)


def construct_event_aware_candidate(
    *,
    sessions: Sequence[date],
    events: pd.DataFrame,
    target_availability: dict[str, set[date]],
    shared_end: date,
) -> pd.DataFrame:
    """Deduplicate normalized events to one row per availability session."""
    rows: list[dict[str, Any]] = []
    for availability_text, group in events.groupby(
        "model_availability_date", sort=True
    ):
        records = group.to_dict("records")
        records.sort(
            key=lambda row: (
                EVENT_TYPE_RANK[row["event_type"]],
                row["original_release_date"],
                row["event_id"],
            )
        )
        state_cutoff = min(
            _as_date(row["state_cutoff_date"]) for row in records
        )
        availability = _as_date(availability_text)
        label = _label_metadata(
            state_cutoff,
            sessions=sessions,
            shared_end=shared_end,
            target_availability=target_availability,
        )
        rows.append(
            {
                "sampling_design": "EVENT_AWARE",
                "sample_date": _date_text(availability),
                "model_availability_date": _date_text(availability),
                "state_cutoff_date": _date_text(state_cutoff),
                "shock_window_start_date": _date_text(state_cutoff),
                "shock_window_end_date": _date_text(availability),
                "target_baseline_date": _date_text(state_cutoff),
                **label,
                "event_count": len(records),
                "collision_count": len(records) if len(records) > 1 else 0,
                "duplicate_candidates_removed": len(records) - 1,
                "event_types_json": _json_list(
                    row["event_type"] for row in records
                ),
                "event_ids_json": _json_list(
                    row["event_id"] for row in records
                ),
                "event_observation_periods_json": _json_list(
                    row["event_observation_period"] for row in records
                ),
                "event_release_dates_json": _json_list(
                    row["original_release_date"] for row in records
                ),
                "event_release_times_json": _json_list(
                    row["original_release_time"] for row in records
                ),
                "event_timezones_json": _json_list(
                    row["original_release_timezone"] for row in records
                ),
                "event_sources_json": _json_list(
                    row["event_source"] for row in records
                ),
                "event_release_date_kinds_json": _json_list(
                    row["release_date_kind"] for row in records
                ),
                "next_session_shift_applied": any(
                    bool(row["next_session_shift_applied"]) for row in records
                ),
                "conservative_next_session_shift": any(
                    bool(row["conservative_next_session_shift"])
                    for row in records
                ),
                "calendar_normalization_applied": any(
                    bool(row["calendar_normalization_applied"])
                    for row in records
                ),
            }
        )
    return _frame_from_records(rows, EVENT_AWARE_INDEX_COLUMNS)


def _year_counts(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    raw_years = pd.to_datetime(frame["sample_date"]).dt.year.astype(str)
    eligible = frame[frame["label_eligible"].astype(bool)]
    eligible_years = pd.to_datetime(eligible["sample_date"]).dt.year.astype(str)
    raw_counts = Counter(raw_years)
    eligible_counts = Counter(eligible_years)
    years = sorted(set(raw_counts) | set(eligible_counts))
    return {
        year: {
            "raw_rows": int(raw_counts.get(year, 0)),
            "label_eligible_rows": int(eligible_counts.get(year, 0)),
        }
        for year in years
    }


def _planning_periods(sessions: Sequence[date], start: date, end: date) -> dict[str, Any]:
    supported = [session for session in sessions if start <= session <= end]
    if len(supported) < 5:
        raise ValueError("Too few supported sessions for planning-period estimates.")
    train_last_position = max(0, int(len(supported) * 0.60) - 1)
    validation_last_position = max(
        train_last_position + 1, int(len(supported) * 0.80) - 1
    )
    return {
        "train_estimate": {
            "start": _date_text(supported[0]),
            "end": _date_text(supported[train_last_position]),
        },
        "validation_estimate": {
            "start": _date_text(supported[train_last_position + 1]),
            "end": _date_text(supported[validation_last_position]),
        },
        "test_estimate": {
            "start": _date_text(supported[validation_last_position + 1]),
            "end": _date_text(supported[-1]),
        },
        "status": (
            "planning-only common-calendar 60/20/20 periods; not final data "
            "splits and not embargoed"
        ),
    }


def _period_counts(
    frame: pd.DataFrame, periods: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    eligible = frame[frame["label_eligible"].astype(bool)].copy()
    dates = pd.to_datetime(eligible["sample_date"]).dt.date
    result: dict[str, dict[str, Any]] = {}
    for name in ("train_estimate", "validation_estimate", "test_estimate"):
        start = _as_date(periods[name]["start"])
        end = _as_date(periods[name]["end"])
        selected = eligible[(dates >= start) & (dates <= end)]
        result[name] = {
            "rows": int(len(selected)),
            "first_sample": (
                str(selected["sample_date"].iloc[0]) if len(selected) else None
            ),
            "last_sample": (
                str(selected["sample_date"].iloc[-1]) if len(selected) else None
            ),
        }
    return result


def _spacing_metrics(
    frame: pd.DataFrame, sessions: Sequence[date]
) -> dict[str, Any]:
    eligible = frame[frame["label_eligible"].astype(bool)]
    dates = sorted(_as_date(value) for value in eligible["sample_date"])
    positions = {session: position for position, session in enumerate(sessions)}
    session_gaps = [
        positions[current] - positions[previous]
        for previous, current in zip(dates, dates[1:])
    ]
    calendar_gaps = [
        (current - previous).days
        for previous, current in zip(dates, dates[1:])
    ]
    buckets = {
        "1_session": 0,
        "2_to_3_sessions": 0,
        "4_to_5_sessions": 0,
        "6_to_10_sessions": 0,
        "11_to_20_sessions": 0,
        "over_20_sessions": 0,
    }
    for gap in session_gaps:
        if gap == 1:
            buckets["1_session"] += 1
        elif gap <= 3:
            buckets["2_to_3_sessions"] += 1
        elif gap <= 5:
            buckets["4_to_5_sessions"] += 1
        elif gap <= 10:
            buckets["6_to_10_sessions"] += 1
        elif gap <= 20:
            buckets["11_to_20_sessions"] += 1
        else:
            buckets["over_20_sessions"] += 1
    if not session_gaps:
        return {
            "pair_count": 0,
            "trading_session_gap": {},
            "calendar_day_gap": {},
            "distribution": buckets,
            "consecutive_session_rows": 0,
            "long_no_event_gaps_over_21_sessions": 0,
        }
    maximum_position = session_gaps.index(max(session_gaps))
    return {
        "pair_count": len(session_gaps),
        "trading_session_gap": {
            "minimum": min(session_gaps),
            "median": statistics.median(session_gaps),
            "maximum": max(session_gaps),
        },
        "calendar_day_gap": {
            "minimum": min(calendar_gaps),
            "median": statistics.median(calendar_gaps),
            "maximum": max(calendar_gaps),
        },
        "distribution": buckets,
        "consecutive_session_rows": sum(gap == 1 for gap in session_gaps),
        "long_no_event_gaps_over_21_sessions": sum(
            gap > TARGET_HORIZON_SESSIONS for gap in session_gaps
        ),
        "largest_gap_between": [
            _date_text(dates[maximum_position]),
            _date_text(dates[maximum_position + 1]),
        ],
    }


def _target_overlap_metrics(
    frame: pd.DataFrame, sessions: Sequence[date]
) -> dict[str, Any]:
    eligible = frame[frame["label_eligible"].astype(bool)].copy()
    eligible = eligible.sort_values("sample_date")
    baselines = [_as_date(value) for value in eligible["target_baseline_date"]]
    positions = {session: position for position, session in enumerate(sessions)}
    overlap_sessions: list[int] = []
    for previous, current in zip(baselines, baselines[1:]):
        gap = positions[current] - positions[previous]
        overlap_sessions.append(max(0, TARGET_HORIZON_SESSIONS - max(gap, 0)))
    overlap_pairs = sum(value > 0 for value in overlap_sessions)
    pair_count = len(overlap_sessions)
    return {
        "consecutive_row_pairs": pair_count,
        "overlapping_window_pairs": overlap_pairs,
        "overlap_proportion": (
            round(overlap_pairs / pair_count, 6) if pair_count else 0.0
        ),
        "mean_overlapping_future_sessions": (
            round(statistics.mean(overlap_sessions), 6)
            if overlap_sessions
            else 0.0
        ),
        "maximum_overlapping_future_sessions": (
            max(overlap_sessions) if overlap_sessions else 0
        ),
        "definition": (
            "Consecutive target windows overlap when their pre-shock target "
            "baselines are fewer than 21 canonical sessions apart."
        ),
    }


def _candidate_summary(
    frame: pd.DataFrame,
    *,
    sessions: Sequence[date],
    periods: dict[str, Any],
) -> dict[str, Any]:
    eligible = frame[frame["label_eligible"].astype(bool)]
    boundary_removed = frame[
        ~frame["full_horizon_within_shared_coverage"].astype(bool)
    ]
    target_missing = frame[
        frame["full_horizon_within_shared_coverage"].astype(bool)
        & ~frame["all_target_values_available"].astype(bool)
    ]
    return {
        "raw_rows": int(len(frame)),
        "label_eligible_rows": int(len(eligible)),
        "rows_removed_by_21_session_boundary": int(len(boundary_removed)),
        "rows_removed_by_endpoint_target_missingness": int(len(target_missing)),
        "first_sample": str(frame["sample_date"].iloc[0]) if len(frame) else None,
        "last_raw_sample": str(frame["sample_date"].iloc[-1]) if len(frame) else None,
        "first_label_eligible_sample": (
            str(eligible["sample_date"].iloc[0]) if len(eligible) else None
        ),
        "last_label_eligible_sample": (
            str(eligible["sample_date"].iloc[-1]) if len(eligible) else None
        ),
        "first_target_baseline": (
            str(eligible["target_baseline_date"].iloc[0])
            if len(eligible)
            else None
        ),
        "last_label_eligible_target_end": (
            str(eligible["target_end_date"].iloc[-1]) if len(eligible) else None
        ),
        "by_calendar_year": _year_counts(frame),
        "planning_period_estimates": _period_counts(frame, periods),
        "spacing": _spacing_metrics(frame, sessions),
        "target_overlap": _target_overlap_metrics(frame, sessions),
    }


def _event_representation(
    events: pd.DataFrame,
    weekly: pd.DataFrame,
    event_aware: pd.DataFrame,
) -> dict[str, Any]:
    weekly_assignments: dict[str, tuple[int, bool]] = {}
    for row in weekly.to_dict("records"):
        ids = json.loads(row["event_ids_json"])
        combined = len(ids) > 1
        for event_id in ids:
            weekly_assignments[event_id] = (len(ids), combined)

    event_group_sizes: dict[str, int] = {}
    for row in event_aware.to_dict("records"):
        ids = json.loads(row["event_ids_json"])
        for event_id in ids:
            event_group_sizes[event_id] = len(ids)

    result: dict[str, Any] = {}
    for event_type in EVENT_TYPE_ORDER:
        subset = events[events["event_type"] == event_type]
        ids = list(subset["event_id"])
        result[event_type] = {
            "total_available_events": len(ids),
            "weekly_events_represented": sum(
                event_id in weekly_assignments for event_id in ids
            ),
            "weekly_events_combined_with_another_event": sum(
                weekly_assignments.get(event_id, (0, False))[1]
                for event_id in ids
            ),
            "weekly_events_omitted": sum(
                event_id not in weekly_assignments for event_id in ids
            ),
            "event_aware_events_represented": sum(
                event_id in event_group_sizes for event_id in ids
            ),
            "event_aware_events_combined_with_another_event": sum(
                event_group_sizes.get(event_id, 0) > 1 for event_id in ids
            ),
            "event_aware_events_omitted": sum(
                event_id not in event_group_sizes for event_id in ids
            ),
            "events_shifted_to_later_session": int(
                subset["next_session_shift_applied"].astype(bool).sum()
            ),
            "events_conservatively_shifted_for_unknown_time": int(
                subset["conservative_next_session_shift"].astype(bool).sum()
            ),
        }
    return result


def _events_by_year(events: pd.DataFrame) -> dict[str, dict[str, int]]:
    records: dict[str, Counter[str]] = defaultdict(Counter)
    for row in events.to_dict("records"):
        year = str(_as_date(row["model_availability_date"]).year)
        records[year]["total_events"] += 1
        records[year][row["event_type"]] += 1
    return {
        year: {
            "total_events": int(counts["total_events"]),
            **{
                event_type: int(counts[event_type])
                for event_type in EVENT_TYPE_ORDER
            },
        }
        for year, counts in sorted(records.items())
    }


def _events_by_period(
    events: pd.DataFrame, periods: dict[str, Any]
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for name in ("train_estimate", "validation_estimate", "test_estimate"):
        start = _as_date(periods[name]["start"])
        end = _as_date(periods[name]["end"])
        subset = events[
            events["model_availability_date"].map(
                lambda value: start <= _as_date(value) <= end
            )
        ]
        result[name] = {
            "total_events": int(len(subset)),
            **{
                event_type: int(
                    (subset["event_type"] == event_type).sum()
                )
                for event_type in EVENT_TYPE_ORDER
            },
        }
    return result


def _collision_summary(
    events: pd.DataFrame, event_aware: pd.DataFrame
) -> dict[str, Any]:
    collision_rows = event_aware[event_aware["event_count"].astype(int) > 1]
    combinations: Counter[str] = Counter()
    collision_details: list[dict[str, Any]] = []
    for row in collision_rows.to_dict("records"):
        event_types = json.loads(row["event_types_json"])
        combination = "+".join(event_types)
        combinations[combination] += 1
        collision_details.append(
            {
                "sample_date": row["sample_date"],
                "event_count": int(row["event_count"]),
                "event_types": event_types,
                "event_ids": json.loads(row["event_ids_json"]),
            }
        )
    return {
        "total_events_before_deduplication": int(len(events)),
        "unique_event_aware_dates": int(len(event_aware)),
        "same_day_collision_dates": int(len(collision_rows)),
        "events_on_collision_dates": int(
            collision_rows["event_count"].astype(int).sum()
        ),
        "duplicate_candidate_dates_removed": int(
            event_aware["duplicate_candidates_removed"].astype(int).sum()
        ),
        "event_identity_loss_count": (
            int(len(events))
            - sum(
                len(json.loads(value))
                for value in event_aware["event_ids_json"]
            )
        ),
        "event_type_combinations": dict(sorted(combinations.items())),
        "collision_details": collision_details,
    }


def _point_in_time_checks(
    *,
    events: pd.DataFrame,
    weekly: pd.DataFrame,
    event_aware: pd.DataFrame,
    selected: pd.DataFrame,
    sessions: Sequence[date],
) -> dict[str, bool]:
    session_set = set(sessions)
    positions = {session: position for position, session in enumerate(sessions)}
    event_records = events.to_dict("records")

    availability_on_calendar = all(
        _as_date(row["model_availability_date"]) in session_set
        for row in event_records
    )
    release_before_availability = all(
        _as_date(row["original_release_date"])
        <= _as_date(row["model_availability_date"])
        for row in event_records
    )
    state_precedes_event = True
    for row in event_records:
        state = _as_date(row["state_cutoff_date"])
        release = _as_date(row["original_release_date"])
        raw_time = str(row["original_release_time"])
        if state < release:
            continue
        if state == release and raw_time:
            if time.fromisoformat(raw_time) >= time(16, 0):
                continue
        state_precedes_event = False
        break

    shifts_follow_calendar = True
    for row in event_records:
        if not bool(row["next_session_shift_applied"]):
            continue
        release = _as_date(row["original_release_date"])
        expected = next_observed_session(sessions, release, strictly_after=True)
        if _as_date(row["model_availability_date"]) != expected:
            shifts_follow_calendar = False
            break

    target_ends_are_exact = True
    for frame in (weekly, event_aware):
        for row in frame.to_dict("records"):
            if not row["target_end_date"]:
                continue
            baseline = _as_date(row["target_baseline_date"])
            target_end = _as_date(row["target_end_date"])
            if (
                positions[target_end] - positions[baseline]
                != TARGET_HORIZON_SESSIONS
            ):
                target_ends_are_exact = False
                break

    release_kinds = set(events["release_date_kind"])
    initial_release_path_only = (
        "ALFRED_INITIAL_APPEARANCE" in release_kinds
        and "official_bea_advance_date_in_atlanta_track_record" in release_kinds
        and all(
            row["release_date_kind"] == "ALFRED_INITIAL_APPEARANCE"
            for row in event_records
            if row["event_type"] in {"cpi", "unemployment"}
        )
    )
    observation_month_not_release_date = all(
        row["release_date_kind"]
        in {
            "official_fomc_announcement",
            "ALFRED_INITIAL_APPEARANCE",
            "official_bea_advance_date_in_atlanta_track_record",
            "official_yearly_archive_entry",
            "official_supplemental_release_page",
            "official_release_page_missing_from_yearly_archive",
        }
        for row in event_records
    )
    no_future_market_close = all(
        _as_date(row["shock_window_end_date"])
        <= _as_date(row["sample_date"])
        for row in event_aware.to_dict("records")
    ) and all(
        _as_date(row["shock_window_end_date"])
        <= _as_date(row["sample_date"])
        for row in weekly.to_dict("records")
    )
    return {
        "all_event_availability_dates_are_observed_sessions": availability_on_calendar,
        "no_event_used_before_release_date": release_before_availability,
        "all_state_cutoffs_precede_the_modeled_event": state_precedes_event,
        "all_next_session_shifts_follow_observed_calendar": shifts_follow_calendar,
        "no_observation_month_used_as_release_date": observation_month_not_release_date,
        "initial_vintage_or_advance_release_paths_preserved": initial_release_path_only,
        "all_nonblank_target_ends_are_exactly_21_sessions": target_ends_are_exact,
        "no_market_close_after_sample_date_used_by_shock_window": no_future_market_close,
        "event_aware_dates_are_unique": not event_aware["sample_date"].duplicated().any(),
        "selected_dates_are_unique": not selected["sample_date"].duplicated().any(),
        "selected_rows_are_all_label_eligible": bool(
            selected["label_eligible"].astype(bool).all()
        ),
    }


def _source_manifest(paths: Iterable[Path]) -> list[dict[str, str]]:
    records = []
    for path in sorted(set(path.resolve() for path in paths)):
        records.append(
            {
                "path": _relative(path),
                "sha256": _sha256(path),
            }
        )
    return records


def build_sampling_artifacts(
    *,
    raw_root: Path = RAW_ROOT,
) -> dict[str, Any]:
    """Construct all Step 4 indexes in memory without external access."""
    sessions = load_canonical_calendar(raw_root)
    target_availability, target_paths = load_target_availability(raw_root)
    shared_end = min(max(values) for values in target_availability.values())
    if shared_end <= SHARED_START:
        raise ValueError("Selected target caches do not overlap the Step 3 period.")

    events, event_source_paths = load_normalized_events(
        sessions=sessions,
        target_availability=target_availability,
        shared_start=SHARED_START,
        shared_end=shared_end,
        raw_root=raw_root,
    )
    weekly = construct_weekly_candidate(
        sessions=sessions,
        events=events,
        target_availability=target_availability,
        shared_start=SHARED_START,
        shared_end=shared_end,
    )
    event_aware = construct_event_aware_candidate(
        sessions=sessions,
        events=events,
        target_availability=target_availability,
        shared_end=shared_end,
    )
    selected = event_aware[event_aware["label_eligible"].astype(bool)].copy()
    selected = selected.reset_index(drop=True)

    periods = _planning_periods(sessions, SHARED_START, shared_end)
    weekly_summary = _candidate_summary(
        weekly, sessions=sessions, periods=periods
    )
    event_summary = _candidate_summary(
        event_aware, sessions=sessions, periods=periods
    )
    collisions = _collision_summary(events, event_aware)
    event_representation = _event_representation(events, weekly, event_aware)
    checks = _point_in_time_checks(
        events=events,
        weekly=weekly,
        event_aware=event_aware,
        selected=selected,
        sessions=sessions,
    )
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"Point-in-time sampling validation failed: {failed}")

    selected_sessions = [
        session for session in sessions if SHARED_START <= session <= shared_end
    ]
    observed_rule_mismatches = [
        session
        for session in selected_sessions
        if not is_us_equity_trading_session(session)
    ]
    target_spec_records = [
        {
            "target": spec.target,
            "path": f"data/scenario_playground/raw/{spec.relative_path}",
            "field": spec.value_field,
            "source": spec.source,
            "value_policy": spec.value_policy,
            "usable_observation_count": len(target_availability[spec.target]),
            "first_usable_date": _date_text(
                min(target_availability[spec.target])
            ),
            "last_usable_date": _date_text(
                max(target_availability[spec.target])
            ),
        }
        for spec in TARGET_SPECS
    ]
    summary: dict[str, Any] = {
        "schema_version": "scenario_sampling_v1",
        "construction_mode": "offline_cached_data_only",
        "selected_design": "EVENT_AWARE",
        "target_horizon_trading_sessions": TARGET_HORIZON_SESSIONS,
        "shared_period": {
            "start": _date_text(SHARED_START),
            "end": _date_text(shared_end),
            "start_limiter": "Cleveland Fed CPI nowcast coverage decision from Step 3",
            "end_limiter": "FRED DGS10 selected-target usable coverage",
        },
        "canonical_calendar": {
            "source": "Yahoo ^GSPC non-missing Close observation dates",
            "source_path": "data/scenario_playground/raw/yahoo/GSPC.csv",
            "selected_period_session_count": len(selected_sessions),
            "first_selected_session": _date_text(selected_sessions[0]),
            "last_selected_session": _date_text(selected_sessions[-1]),
            "observed_sessions_not_recognized_by_existing_rule_utility": (
                len(observed_rule_mismatches)
            ),
            "observed_rule_mismatch_dates": [
                _date_text(value) for value in observed_rule_mismatches
            ],
            "policy": (
                "Observed ^GSPC sessions are authoritative. Weekend and holiday "
                "releases move to an observed session; target dates are never "
                "calendar-day approximations."
            ),
        },
        "model_row_timing": {
            "conceptual_t": (
                "the completed canonical market close immediately before the "
                "modeled shock or shock interval"
            ),
            "sample_date": (
                "the end-of-day row key when all attached shocks are available; "
                "weekly anchor date or event model-availability date"
            ),
            "state_cutoff": "the pre-shock completed close",
            "target_baseline": "the same pre-shock close as the state cutoff",
            "target_end": "the 21st canonical session after target baseline",
            "target_start_policy": (
                "pre-event/pre-window close, so the cumulative target can include "
                "the imposed initial shock as required by the model contract"
            ),
            "market_close_availability": (
                "a close becomes usable only after that session closes"
            ),
            "pre_market_or_during_market_release": (
                "state uses the prior close; known-time FOMC events can use that "
                "session as model availability"
            ),
            "after_market_release": (
                "state may use that day's completed close; availability moves to "
                "the next observed session"
            ),
            "unknown_release_time": (
                "state uses the prior close and availability moves conservatively "
                "to the next observed session"
            ),
        },
        "market_shock_window_rule": {
            "oil_price_change_pct": (
                "later calculate DCOILWTICO percent change from state_cutoff_date "
                "through shock_window_end_date"
            ),
            "dxy_change_pct": (
                "later calculate DX-Y.NYB Close percent change from "
                "state_cutoff_date through shock_window_end_date"
            ),
            "vix_change_points": (
                "later calculate VIXCLS point change from state_cutoff_date "
                "through shock_window_end_date"
            ),
            "boundary": (
                "shock_window_end_date is the weekly anchor or event availability "
                "session and never follows sample_date; Step 4 stores boundaries "
                "but calculates no shock values"
            ),
        },
        "target_specs": target_spec_records,
        "normalized_event_schema": list(NORMALIZED_EVENT_COLUMNS),
        "weekly_candidate_schema": list(WEEKLY_INDEX_COLUMNS),
        "event_aware_candidate_schema": list(EVENT_AWARE_INDEX_COLUMNS),
        "selected_sampling_schema": list(EVENT_AWARE_INDEX_COLUMNS),
        "weekly": {
            **weekly_summary,
            "anchor_rule": (
                "final observed ^GSPC session of each ISO week; normally Friday"
            ),
            "shifted_anchors": int(
                weekly["anchor_is_shifted"].astype(bool).sum()
            ),
            "shift_reasons": dict(
                sorted(
                    Counter(
                        value
                        for value in weekly["anchor_shift_reason"]
                        if value
                    ).items()
                )
            ),
            "weeks_with_any_event": int(
                (weekly["event_count"].astype(int) > 0).sum()
            ),
            "weeks_with_multiple_events": int(
                (weekly["event_count"].astype(int) > 1).sum()
            ),
            "multiple_same_type_rule_for_later_features": {
                "fed_funds_rate_change_bps": "sum signed decision changes",
                "cpi_surprise_pct": "sum signed release surprises within the interval",
                "gdp_growth_surprise_pct": "sum signed release surprises within the interval",
                "unemployment_change_pct": (
                    "net latest release minus the last release before the interval"
                ),
                "pmi_change_points": (
                    "net latest release minus the last release before the interval"
                ),
                "market_inputs": (
                    "one endpoint change across the complete weekly shock window"
                ),
            },
        },
        "event_aware": {
            **event_summary,
            "construction_rule": (
                "one row per unique observed model-availability session, with all "
                "same-session event identities retained in fixed type order"
            ),
            "event_counts_by_type": {
                event_type: int(
                    (events["event_type"] == event_type).sum()
                )
                for event_type in EVENT_TYPE_ORDER
            },
            "events_by_calendar_year": _events_by_year(events),
            "events_by_planning_period": _events_by_period(events, periods),
            "total_events_before_deduplication": int(len(events)),
            "same_day_collisions": collisions,
        },
        "event_representation": event_representation,
        "planning_periods": periods,
        "point_in_time_validation": checks,
        "model_suitability": {
            "weekly": {
                "ridge": (
                    "Enough chronological rows and regular spacing, but interval "
                    "aggregation blurs individual macro shocks."
                ),
                "xgboost": (
                    "Enough rows only for shallow, strongly regularized trees; "
                    "weekly aggregation creates heterogeneous shock combinations."
                ),
            },
            "event_aware": {
                "ridge": (
                    "Enough rows; precise event alignment preserves interpretable "
                    "signed shock coefficients, subject to overlapping-label controls."
                ),
                "xgboost": (
                    "Enough rows for a shallow, strongly regularized benchmark; "
                    "sparse event types and overlap require conservative tuning."
                ),
            },
        },
        "final_decision": {
            "selected": "EVENT_AWARE",
            "rejected": "WEEKLY",
            "reason": (
                "The event-aware index retains a comparable sample size while "
                "matching the contract's pre-shock state and event-specific shock "
                "semantics. Weekly rows combine releases, blur timing, and require "
                "extra aggregation rules. The measured simplicity advantage is not "
                "large enough to justify that semantic loss."
            ),
            "accepted_limitations": [
                "Irregular row spacing",
                "Substantial overlap among 21-session target windows",
                "Conservative next-session timing where exact release times are absent",
                "Same-day rows can carry more than one event type",
                "Later chronological splitting needs an embargo of at least the target horizon",
            ],
            "later_feature_consequence": (
                "Build state as of state_cutoff_date and shocks only from attached "
                "events or the stored market-shock window; do not use post-event "
                "market data as state."
            ),
            "later_target_consequence": (
                "Calculate all twelve targets from target_baseline_date to the exact "
                "stored target_end_date without shortening the horizon."
            ),
            "later_split_consequence": (
                "Use chronological splits with a 21-session purge/embargo; the "
                "planning periods in this summary are counts only."
            ),
        },
        "source_manifest": _source_manifest(
            [raw_root / "yahoo" / "GSPC.csv"]
            + target_paths
            + event_source_paths
        ),
    }
    return {
        "normalized_events": events,
        "weekly": weekly,
        "event_aware": event_aware,
        "selected": selected,
        "summary": summary,
    }


def persist_sampling_artifacts(
    artifacts: dict[str, Any],
    *,
    output_dir: Path = SAMPLING_ROOT,
) -> dict[str, str]:
    """Persist deterministic CSV/JSON Step 4 artifacts atomically."""
    output_dir = output_dir.resolve()
    outputs = {
        "normalized_events": output_dir / "normalized_event_records.csv",
        "weekly": output_dir / "weekly_candidate_index.csv",
        "event_aware": output_dir / "event_aware_candidate_index.csv",
        "selected": output_dir / "selected_sampling_index.csv",
        "summary": output_dir / "sampling_comparison_summary.json",
    }
    csv_text = {
        name: _frame_to_csv_text(artifacts[name])
        for name in ("normalized_events", "weekly", "event_aware", "selected")
    }
    output_hashes = {
        outputs[name].name: hashlib.sha256(text.encode("utf-8")).hexdigest()
        for name, text in csv_text.items()
    }
    summary = dict(artifacts["summary"])
    summary["output_files"] = {
        name: {
            "path": _relative(path),
            "sha256": output_hashes.get(path.name),
        }
        for name, path in outputs.items()
    }
    summary_text = (
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    )
    for name, text in csv_text.items():
        write_text_atomic(outputs[name], text)
    write_text_atomic(outputs["summary"], summary_text)
    return {name: str(path) for name, path in outputs.items()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic weekly and event-aware Step 4 sampling indexes "
            "from local Step 3 caches only."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Construct and validate in memory without writing interim outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SAMPLING_ROOT,
        help="Output directory (default: data/scenario_playground/interim/sampling).",
    )
    args = parser.parse_args(argv)

    artifacts = build_sampling_artifacts()
    summary = artifacts["summary"]
    if args.dry_run:
        paths: dict[str, str] = {}
    else:
        paths = persist_sampling_artifacts(
            artifacts, output_dir=args.output_dir
        )
    result = {
        "selected_design": summary["selected_design"],
        "weekly_raw_rows": summary["weekly"]["raw_rows"],
        "weekly_label_eligible_rows": summary["weekly"][
            "label_eligible_rows"
        ],
        "event_aware_raw_rows": summary["event_aware"]["raw_rows"],
        "event_aware_label_eligible_rows": summary["event_aware"][
            "label_eligible_rows"
        ],
        "outputs": paths,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
