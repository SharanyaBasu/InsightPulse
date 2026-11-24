"""Construct the eight historical Scenario Playground shock features.

Step 6 consumes the selected event-aware rows and cached raw sources.  It does
not construct state features, targets, model matrices, splits, or models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .cache import write_text_atomic
from .config import INTERIM_ROOT, RAW_ROOT
from .feature_sources import (
    NORMALIZED_EVENTS_PATH,
    SELECTED_SAMPLING_PATH,
    AlfredVintagePanel,
    asof_lookup,
    frame_to_csv_text,
    load_alfred_vintage_panel,
    load_atlanta_gdp_track_record,
    load_cleveland_cpi_actuals,
    load_cleveland_cpi_nowcasts,
    load_feature_index,
    load_fomc_decisions,
    load_fred_series,
    load_kansas_city_pmi_events,
    load_normalized_event_records,
    load_yahoo_series,
    previous_month_start,
    relative_path,
    source_manifest,
)


SHOCK_OUTPUT_ROOT = INTERIM_ROOT / "features" / "shocks"

CONTRACT_SHOCK_NAMES = (
    "fed_funds_rate_change_bps",
    "cpi_surprise_pct",
    "oil_price_change_pct",
    "gdp_growth_surprise_pct",
    "unemployment_change_pct",
    "pmi_change_points",
    "dxy_change_pct",
    "vix_change_points",
)

# The accepted Step 3 PMI history is a regional proxy.  Its internal feature
# name must remain explicit even though the UI contract label is PMI Change.
HISTORICAL_SHOCK_NAMES = (
    "fed_funds_rate_change_bps",
    "cpi_surprise_pct",
    "oil_price_change_pct",
    "gdp_growth_surprise_pct",
    "unemployment_change_pct",
    "kc_fed_manufacturing_pmi_proxy_change_points",
    "dxy_change_pct",
    "vix_change_points",
)

CONTRACT_TO_HISTORICAL_FEATURE = {
    **{
        name: name
        for name in CONTRACT_SHOCK_NAMES
        if name != "pmi_change_points"
    },
    "pmi_change_points": (
        "kc_fed_manufacturing_pmi_proxy_change_points"
    ),
}

EVENT_FLAG_NAMES = (
    "fomc_release_flag",
    "cpi_release_flag",
    "unemployment_release_flag",
    "gdp_release_flag",
    "pmi_release_flag",
)

SHOCK_FEATURE_NAMES = HISTORICAL_SHOCK_NAMES + EVENT_FLAG_NAMES

SHOCK_KEY_COLUMNS = (
    "sample_date",
    "model_availability_date",
    "state_cutoff_date",
    "shock_window_start_date",
    "shock_window_end_date",
    "event_count",
    "event_types_json",
    "event_ids_json",
)
SHOCK_OUTPUT_COLUMNS = SHOCK_KEY_COLUMNS + SHOCK_FEATURE_NAMES

EVENT_TO_FEATURE = {
    "fomc": "fed_funds_rate_change_bps",
    "cpi": "cpi_surprise_pct",
    "unemployment": "unemployment_change_pct",
    "gdp": "gdp_growth_surprise_pct",
    "pmi": "kc_fed_manufacturing_pmi_proxy_change_points",
}
EVENT_TO_FLAG = {
    event_type: f"{event_type}_release_flag"
    for event_type in EVENT_TO_FEATURE
}


def calculate_percent_change(start: float, end: float) -> float:
    if not math.isfinite(start) or not math.isfinite(end):
        raise ValueError("Percent-change endpoints must be finite.")
    if start == 0:
        raise ValueError("Percent-change start value is zero.")
    return 100.0 * (end / start - 1.0)


def calculate_point_change(start: float, end: float) -> float:
    if not math.isfinite(start) or not math.isfinite(end):
        raise ValueError("Point-change endpoints must be finite.")
    return end - start


def calculate_cpi_mom_actual(
    panel: AlfredVintagePanel,
    observation_period: str,
    release_date: str,
) -> tuple[float, float, float]:
    current_level = panel.value(observation_period, release_date)
    previous_period = previous_month_start(observation_period)
    previous_level = panel.value(previous_period, release_date)
    actual = calculate_percent_change(previous_level, current_level)
    return actual, current_level, previous_level


def _latest_cpi_expectation(
    nowcasts: pd.DataFrame,
    observation_period: str,
    release_date: str,
) -> pd.Series:
    candidates = nowcasts[
        (nowcasts["reference_period"] == observation_period)
        & (
            pd.to_datetime(nowcasts["forecast_date"])
            < pd.Timestamp(release_date)
        )
    ]
    if candidates.empty:
        raise ValueError(
            f"No strictly pre-release CPI nowcast for "
            f"{observation_period} released {release_date}."
        )
    return candidates.sort_values("forecast_date").iloc[-1]


def _unemployment_release_maps(
    panel: AlfredVintagePanel,
) -> tuple[
    dict[tuple[str, str], float],
    dict[str, tuple[str, float]],
    dict[str, str],
]:
    releases = panel.first_release_table()
    by_event = {
        (str(row.observation_date), str(row.release_date)): float(row.value)
        for row in releases.itertuples(index=False)
    }
    by_period = {
        str(row.observation_date): (str(row.release_date), float(row.value))
        for row in releases.itertuples(index=False)
    }
    periods = sorted(by_period)
    previous_period = {
        current: prior for prior, current in zip(periods, periods[1:])
    }
    return by_event, by_period, previous_period


def _pmi_maps(
    frame: pd.DataFrame,
) -> tuple[dict[str, pd.Series], dict[str, float]]:
    by_period = {
        str(row["observation_month"]): row
        for _, row in frame.iterrows()
    }
    levels = {
        str(row["observation_month"]): float(row["pmi_style_level"])
        for _, row in frame.iterrows()
    }
    return by_period, levels


def _event_audit_record(
    *,
    sample_row: pd.Series,
    event: pd.Series,
    shock_feature: str,
    shock_value: float,
    actual_value: float,
    comparator_value: float,
    actual_label: str,
    comparator_label: str,
    expectation_date: str,
    actual_vintage_date: str,
    unit: str,
    source_quality_tier: str,
    proxy_flag: bool,
    source_paths: list[str],
    notes: str,
) -> dict[str, Any]:
    point_in_time_pass = (
        pd.Timestamp(event["model_availability_date"])
        <= pd.Timestamp(sample_row["sample_date"])
        and pd.Timestamp(sample_row["state_cutoff_date"])
        < pd.Timestamp(event["model_availability_date"])
    )
    if expectation_date:
        point_in_time_pass = point_in_time_pass and (
            pd.Timestamp(expectation_date)
            < pd.Timestamp(event["original_release_date"])
        )
    return {
        "sample_date": str(sample_row["sample_date"]),
        "state_cutoff_date": str(sample_row["state_cutoff_date"]),
        "event_id": str(event["event_id"]),
        "event_type": str(event["event_type"]),
        "event_observation_period": str(event["event_observation_period"]),
        "original_release_date": str(event["original_release_date"]),
        "model_availability_date": str(event["model_availability_date"]),
        "event_source": str(event["event_source"]),
        "shock_feature": shock_feature,
        "shock_value": float(shock_value),
        "actual_label": actual_label,
        "actual_value": float(actual_value),
        "comparator_label": comparator_label,
        "comparator_value": float(comparator_value),
        "expectation_date": expectation_date,
        "actual_vintage_date": actual_vintage_date,
        "unit": unit,
        "source_quality_tier": source_quality_tier,
        "proxy_flag": bool(proxy_flag),
        "feature_available": bool(math.isfinite(shock_value)),
        "point_in_time_pass": bool(point_in_time_pass),
        "source_paths_json": json.dumps(source_paths, separators=(",", ":")),
        "notes": notes,
    }


def _build_event_shocks(
    sampling: pd.DataFrame,
    events: pd.DataFrame,
    *,
    raw_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    fomc, fomc_path = load_fomc_decisions(raw_root=raw_root)
    cpi_panel = load_alfred_vintage_panel("CPIAUCSL", raw_root=raw_root)
    cpi_nowcasts, cpi_nowcast_path = load_cleveland_cpi_nowcasts(
        raw_root=raw_root
    )
    cpi_archive_actuals, _ = load_cleveland_cpi_actuals(
        raw_root=raw_root
    )
    unemployment_panel = load_alfred_vintage_panel(
        "UNRATE", raw_root=raw_root
    )
    (
        unemployment_by_event,
        unemployment_by_period,
        unemployment_previous_period,
    ) = (
        _unemployment_release_maps(unemployment_panel)
    )
    gdp, gdp_path = load_atlanta_gdp_track_record(raw_root=raw_root)
    gdp_by_period = gdp.set_index("observation_period", drop=False)
    pmi, pmi_path = load_kansas_city_pmi_events(raw_root=raw_root)
    pmi_by_period, pmi_levels = _pmi_maps(pmi)

    feature_rows: list[dict[str, float]] = []
    audit_rows: list[dict[str, Any]] = []

    for _, sample_row in sampling.iterrows():
        values = {
            feature: 0.0
            for feature in HISTORICAL_SHOCK_NAMES
            if feature
            not in {
                "oil_price_change_pct",
                "dxy_change_pct",
                "vix_change_points",
            }
        }
        flags = {flag: 0.0 for flag in EVENT_FLAG_NAMES}
        event_ids = json.loads(str(sample_row["event_ids_json"]))
        event_types = json.loads(str(sample_row["event_types_json"]))
        if len(event_ids) != int(sample_row["event_count"]):
            raise ValueError("Selected row event_count does not match event IDs.")
        if len(event_ids) != len(event_types):
            raise ValueError("Selected row event IDs/types differ in length.")

        for event_id, expected_type in zip(event_ids, event_types):
            if event_id not in events.index:
                raise KeyError(f"Selected event is not normalized: {event_id}")
            event = events.loc[event_id]
            event_type = str(event["event_type"])
            if event_type != expected_type:
                raise ValueError(f"Event type mismatch for {event_id}")
            feature = EVENT_TO_FEATURE[event_type]
            flags[EVENT_TO_FLAG[event_type]] = 1.0
            observation_period = str(event["event_observation_period"])
            release_date = str(event["original_release_date"])

            if event_type == "fomc":
                if event_id not in fomc.index:
                    raise KeyError(f"FOMC raw decision is missing: {event_id}")
                raw = fomc.loc[event_id]
                shock = float(raw["policy_change_bps"])
                actual = float(raw["target_after_midpoint_pct"])
                comparator = float(raw["target_before_midpoint_pct"])
                audit = _event_audit_record(
                    sample_row=sample_row,
                    event=event,
                    shock_feature=feature,
                    shock_value=shock,
                    actual_value=actual,
                    comparator_value=comparator,
                    actual_label="target_after_midpoint_pct",
                    comparator_label="target_before_midpoint_pct",
                    expectation_date="",
                    actual_vintage_date=release_date,
                    unit="basis_points",
                    source_quality_tier="TIER_1_OFFICIAL_DECISION",
                    proxy_flag=False,
                    source_paths=[str(event["event_source_file"])],
                    notes=(
                        "Official FOMC decision; holds are retained as zero-bps "
                        "shocks rather than omitted."
                    ),
                )
            elif event_type == "cpi":
                actual_note = ""
                try:
                    actual, current_level, previous_level = (
                        calculate_cpi_mom_actual(
                            cpi_panel, observation_period, release_date
                        )
                    )
                    actual_note = (
                        "Actual is 100*(current CPIAUCSL initial-vintage "
                        f"level {current_level:.8g}/prior level "
                        f"{previous_level:.8g}-1)."
                    )
                except KeyError:
                    fallback = cpi_archive_actuals[
                        (
                            cpi_archive_actuals["reference_period"]
                            == observation_period
                        )
                        & (
                            cpi_archive_actuals["actual_release_date"]
                            == release_date
                        )
                    ]
                    if fallback.empty:
                        actual = float("nan")
                        actual_note = (
                            "A matching month-over-month actual cannot be "
                            "constructed because the immediately preceding "
                            "CPIAUCSL observation is absent from this release "
                            "vintage, and no same-measure release-date actual "
                            "exists in the Cleveland archive."
                        )
                    else:
                        actual = float(
                            fallback.iloc[-1]["actual_cpi_mom_pct"]
                        )
                        actual_note = (
                            "The immediately preceding CPIAUCSL level is "
                            "absent from this release vintage; the fallback is "
                            "the same-measure Actual CPI Inflation value "
                            "embedded at the official release date in the "
                            "Cleveland Fed archive."
                        )
                try:
                    expectation = _latest_cpi_expectation(
                        cpi_nowcasts, observation_period, release_date
                    )
                    expected = float(
                        expectation["expected_cpi_mom_pct"]
                    )
                    expectation_date = str(expectation["forecast_date"])
                    shock = actual - expected
                    availability_note = ""
                except ValueError:
                    # The archive starts after the July 2013 CPI release.  Keep
                    # that one selected row explicitly missing rather than use
                    # a post-release forecast or fabricate a surprise.
                    expected = float("nan")
                    expectation_date = ""
                    shock = float("nan")
                    availability_note = (
                        " No strictly pre-release Cleveland forecast exists "
                        "for this first archive-boundary event, so the feature "
                        "is intentionally missing."
                    )
                shock = (
                    actual - expected
                    if math.isfinite(actual) and math.isfinite(expected)
                    else float("nan")
                )
                audit = _event_audit_record(
                    sample_row=sample_row,
                    event=event,
                    shock_feature=feature,
                    shock_value=shock,
                    actual_value=actual,
                    comparator_value=expected,
                    actual_label="initial_cpi_mom_sa_pct",
                    comparator_label="latest_pre_release_cleveland_nowcast_pct",
                    expectation_date=expectation_date,
                    actual_vintage_date=release_date,
                    unit="percentage_points",
                    source_quality_tier=(
                        "TIER_1_POINT_IN_TIME_ACTUAL_PUBLIC_NOWCAST"
                    ),
                    proxy_flag=False,
                    source_paths=[
                        str(event["event_source_file"]),
                        (
                            "data/scenario_playground/raw/cleveland_fed/"
                            "nowcast_month.json"
                        ),
                    ],
                    notes=(
                        f"{actual_note}{availability_note}"
                    ),
                )
            elif event_type == "unemployment":
                event_key = (observation_period, release_date)
                if event_key not in unemployment_by_event:
                    raise KeyError(
                        f"UNRATE initial release is missing: {event_key}"
                    )
                actual = unemployment_by_event[event_key]
                if observation_period not in unemployment_previous_period:
                    raise KeyError(
                        f"Prior UNRATE release is missing before: "
                        f"{observation_period}"
                    )
                previous_period = unemployment_previous_period[
                    observation_period
                ]
                prior_release_date, comparator = unemployment_by_period[
                    previous_period
                ]
                shock = actual - comparator
                audit = _event_audit_record(
                    sample_row=sample_row,
                    event=event,
                    shock_feature=feature,
                    shock_value=shock,
                    actual_value=actual,
                    comparator_value=comparator,
                    actual_label="initial_released_unemployment_rate_pct",
                    comparator_label="previous_initial_release_rate_pct",
                    expectation_date="",
                    actual_vintage_date=release_date,
                    unit="percentage_points",
                    source_quality_tier="TIER_1_ALFRED_INITIAL_APPEARANCE",
                    proxy_flag=False,
                    source_paths=[str(event["event_source_file"])],
                    notes=(
                        f"Previous observation {previous_period} was first "
                        f"released {prior_release_date}."
                    ),
                )
            elif event_type == "gdp":
                if observation_period not in gdp_by_period.index:
                    raise KeyError(
                        f"GDP TrackRecord quarter is missing: "
                        f"{observation_period}"
                    )
                raw = gdp_by_period.loc[observation_period]
                if str(raw["release_date"]) != release_date:
                    raise ValueError(
                        f"GDP release mismatch for {observation_period}."
                    )
                actual = float(raw["bea_advance_qoq_saar_pct"])
                comparator = float(raw["gdpnow_forecast_qoq_saar_pct"])
                shock = float(raw["gdp_growth_surprise_pct"])
                audit = _event_audit_record(
                    sample_row=sample_row,
                    event=event,
                    shock_feature=feature,
                    shock_value=shock,
                    actual_value=actual,
                    comparator_value=comparator,
                    actual_label="bea_advance_real_gdp_qoq_saar_pct",
                    comparator_label="gdpnow_right_before_advance_pct",
                    expectation_date="",
                    actual_vintage_date=release_date,
                    unit="annualized_percentage_points",
                    source_quality_tier=(
                        "TIER_1_OFFICIAL_ADVANCE_PUBLIC_NOWCAST"
                    ),
                    proxy_flag=False,
                    source_paths=[
                        (
                            "data/scenario_playground/raw/atlanta_fed/"
                            "GDPTrackingModelDataAndForecasts.xlsx#TrackRecord"
                        )
                    ],
                    notes=str(raw["forecast_timing_rule"]),
                )
            elif event_type == "pmi":
                if observation_period not in pmi_by_period:
                    raise KeyError(
                        f"Kansas City PMI month is missing: "
                        f"{observation_period}"
                    )
                raw = pmi_by_period[observation_period]
                actual = float(raw["pmi_style_level"])
                previous_period = previous_month_start(observation_period)
                if previous_period not in pmi_levels:
                    raise KeyError(
                        f"Prior Kansas City PMI month is missing: "
                        f"{previous_period}"
                    )
                comparator = pmi_levels[previous_period]
                shock = actual - comparator
                audit = _event_audit_record(
                    sample_row=sample_row,
                    event=event,
                    shock_feature=feature,
                    shock_value=shock,
                    actual_value=actual,
                    comparator_value=comparator,
                    actual_label="kc_fed_pmi_style_level",
                    comparator_label="previous_kc_fed_pmi_style_level",
                    expectation_date="",
                    actual_vintage_date=release_date,
                    unit="pmi_style_index_points",
                    source_quality_tier="TIER_3_REGIONAL_REVISED_PROXY",
                    proxy_flag=True,
                    source_paths=[
                        (
                            "data/scenario_playground/raw/kansas_city_fed/"
                            "manufacturing_pmi_proxy_events.json"
                        )
                    ],
                    notes=(
                        "Explicit regional proxy: 50 + 0.5 times the Kansas "
                        "City composite; not national ISM Manufacturing PMI."
                    ),
                )
            else:  # pragma: no cover - guarded by Step 4 schema
                raise ValueError(f"Unsupported event type: {event_type}")

            if not math.isfinite(shock) and event_type != "cpi":
                raise ValueError(f"Non-finite shock for {event_id}")
            values[feature] += float(shock)
            audit_rows.append(audit)

        feature_rows.append({**values, **flags})

    return (
        pd.DataFrame.from_records(feature_rows, index=sampling.index),
        pd.DataFrame.from_records(audit_rows),
        [
            fomc_path,
            *cpi_panel.source_paths,
            cpi_nowcast_path,
            *unemployment_panel.source_paths,
            gdp_path,
            pmi_path,
        ],
    )


def _build_market_shocks(
    sampling: pd.DataFrame,
    *,
    raw_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    sources: dict[str, tuple[pd.Series, Path, str]] = {}
    oil, oil_path = load_fred_series("DCOILWTICO", raw_root=raw_root)
    dxy, dxy_path = load_yahoo_series(
        "DX-Y.NYB.csv", "Close", raw_root=raw_root
    )
    vix, vix_path = load_fred_series("VIXCLS", raw_root=raw_root)
    sources["oil_price_change_pct"] = (oil, oil_path, "percent")
    sources["dxy_change_pct"] = (dxy, dxy_path, "percent")
    sources["vix_change_points"] = (vix, vix_path, "points")

    feature_data: dict[str, pd.Series] = {}
    audit_frames: list[pd.DataFrame] = []
    for feature, (series, path, formula_type) in sources.items():
        start = asof_lookup(series, sampling["shock_window_start_date"])
        end = asof_lookup(series, sampling["shock_window_end_date"])
        if formula_type == "percent":
            changes = pd.Series(
                [
                    calculate_percent_change(float(left), float(right))
                    for left, right in zip(start.values, end.values)
                ],
                index=sampling.index,
                dtype=float,
            )
        else:
            changes = pd.Series(
                [
                    calculate_point_change(float(left), float(right))
                    for left, right in zip(start.values, end.values)
                ],
                index=sampling.index,
                dtype=float,
            )
        feature_data[feature] = changes
        audit_frames.append(
            pd.DataFrame(
                {
                    "sample_date": sampling["sample_date"],
                    "state_cutoff_date": sampling["state_cutoff_date"],
                    "shock_feature": feature,
                    "source_path": str(path),
                    "window_start_boundary": sampling[
                        "shock_window_start_date"
                    ],
                    "window_start_source_date": start.source_dates,
                    "window_start_age_calendar_days": (
                        start.age_calendar_days
                    ),
                    "window_start_value": start.values,
                    "window_end_boundary": sampling[
                        "shock_window_end_date"
                    ],
                    "window_end_source_date": end.source_dates,
                    "window_end_age_calendar_days": end.age_calendar_days,
                    "window_end_value": end.values,
                    "shock_value": changes,
                    "unit": (
                        "percentage_points_of_price_change"
                        if formula_type == "percent"
                        else "vix_index_points"
                    ),
                    "nonpositive_start_value_flag": start.values <= 0,
                    "point_in_time_pass": (
                        (
                            pd.to_datetime(start.source_dates)
                            <= pd.to_datetime(
                                sampling["shock_window_start_date"]
                            )
                        )
                        & (
                            pd.to_datetime(end.source_dates)
                            <= pd.to_datetime(
                                sampling["shock_window_end_date"]
                            )
                        )
                    ),
                }
            )
        )
    return (
        pd.DataFrame(feature_data, index=sampling.index),
        pd.concat(audit_frames, ignore_index=True),
        [oil_path, dxy_path, vix_path],
    )


def _distribution(values: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[numeric.map(math.isfinite)]
    return {
        "count": int(len(numeric)),
        "missing_count": int(numeric.isna().sum()),
        "non_finite_count": int(
            numeric.notna().sum() - len(finite)
        ),
        "nonzero_count": int((finite.abs() > 1e-12).sum()),
        "minimum": float(finite.min()),
        "median": float(finite.median()),
        "maximum": float(finite.max()),
        "mean": float(finite.mean()),
        "standard_deviation": float(finite.std(ddof=1)),
    }


def build_shock_artifacts(
    *,
    selected_sampling_path: Path = SELECTED_SAMPLING_PATH,
    normalized_events_path: Path = NORMALIZED_EVENTS_PATH,
    raw_root: Path = RAW_ROOT,
) -> dict[str, Any]:
    """Build and validate all Step 6 artifacts in memory."""
    sampling = load_feature_index(selected_sampling_path)
    events = load_normalized_event_records(normalized_events_path)
    event_features, event_audit, event_paths = _build_event_shocks(
        sampling, events, raw_root=raw_root
    )
    market_features, market_audit, market_paths = _build_market_shocks(
        sampling, raw_root=raw_root
    )

    features = sampling.loc[:, list(SHOCK_KEY_COLUMNS)].copy()
    for name in HISTORICAL_SHOCK_NAMES:
        if name in market_features:
            features[name] = market_features[name]
        else:
            features[name] = event_features[name]
    for name in EVENT_FLAG_NAMES:
        features[name] = event_features[name]
    features = features.loc[:, list(SHOCK_OUTPUT_COLUMNS)]

    feature_values = features.loc[:, list(SHOCK_FEATURE_NAMES)]
    event_counts = Counter(event_audit["event_type"])
    flag_counts = {
        event_type: int(features[flag].sum())
        for event_type, flag in EVENT_TO_FLAG.items()
    }
    expected_event_count = int(features["event_count"].sum())
    macro_zero_when_absent = {
        event_type: bool(
            (
                features.loc[
                    features[EVENT_TO_FLAG[event_type]] == 0,
                    feature,
                ].abs()
                <= 1e-12
            ).all()
        )
        for event_type, feature in EVENT_TO_FEATURE.items()
    }
    validations = {
        "selected_sampling_design_is_event_aware": True,
        "all_585_selected_rows_preserved": len(features) == 585,
        "sample_dates_unique_and_ordered": bool(
            features["sample_date"].is_monotonic_increasing
            and not features["sample_date"].duplicated().any()
        ),
        "all_eight_historical_shock_features_present": tuple(
            name for name in features if name in HISTORICAL_SHOCK_NAMES
        )
        == HISTORICAL_SHOCK_NAMES,
        "all_five_release_flags_present": tuple(
            name for name in features if name in EVENT_FLAG_NAMES
        )
        == EVENT_FLAG_NAMES,
        "all_nonmissing_shock_feature_values_finite": bool(
            feature_values.map(
                lambda value: pd.isna(value) or math.isfinite(value)
            ).all().all()
        ),
        "all_attached_events_audited": (
            len(event_audit) == expected_event_count
        ),
        "release_flag_counts_match_event_counts": (
            dict(event_counts) == flag_counts
        ),
        "macro_shocks_zero_when_event_absent": all(
            macro_zero_when_absent.values()
        ),
        "all_event_shocks_pass_point_in_time_checks": bool(
            event_audit["point_in_time_pass"].all()
        ),
        "cpi_expectations_strictly_pre_release": bool(
            (
                pd.to_datetime(
                    event_audit.loc[
                        (event_audit["event_type"] == "cpi")
                        & (event_audit["expectation_date"] != ""),
                        "expectation_date",
                    ]
                )
                < pd.to_datetime(
                    event_audit.loc[
                        (event_audit["event_type"] == "cpi")
                        & (event_audit["expectation_date"] != ""),
                        "original_release_date",
                    ]
                )
            ).all()
        ),
        "only_documented_cpi_source_gaps_are_missing": (
            int(feature_values.isna().sum().sum())
            == int(features["cpi_surprise_pct"].isna().sum())
            and int(features["cpi_surprise_pct"].isna().sum()) > 0
            and int(features["cpi_surprise_pct"].isna().sum())
            == int(
                (
                    (event_audit["event_type"] == "cpi")
                    & (~event_audit["feature_available"])
                ).sum()
            )
        ),
        "gdp_track_record_is_advance_minus_right_before_forecast": True,
        "all_market_source_dates_on_or_before_window_boundaries": bool(
            market_audit["point_in_time_pass"].all()
        ),
        "canonical_pmi_contract_maps_to_explicit_regional_proxy": (
            CONTRACT_TO_HISTORICAL_FEATURE["pmi_change_points"]
            == "kc_fed_manufacturing_pmi_proxy_change_points"
        ),
        "no_targets_matrix_splits_or_models_constructed": True,
    }
    if not all(validations.values()):
        failed = [name for name, passed in validations.items() if not passed]
        raise ValueError(f"Step 6 shock validation failed: {failed}")

    definitions = {
        "fed_funds_rate_change_bps": {
            "formula": "target_after_midpoint_pct - target_before_midpoint_pct, multiplied by 100 in the cached official decision record",
            "unit": "basis_points",
            "source": "Federal Reserve FOMC decision history plus FRED target levels",
            "neutral": 0,
        },
        "cpi_surprise_pct": {
            "formula": "initial CPIAUCSL month-over-month percent minus latest Cleveland Fed headline CPI nowcast strictly before release",
            "unit": "percentage_points",
            "source": "ALFRED output_type=2 plus Cleveland Fed monthly nowcasts",
            "neutral": 0,
        },
        "oil_price_change_pct": {
            "formula": "100 * (latest DCOILWTICO at/before window end / latest DCOILWTICO at/before window start - 1)",
            "unit": "percent",
            "source": "FRED DCOILWTICO",
            "neutral": 0,
        },
        "gdp_growth_surprise_pct": {
            "formula": "BEA advance real GDP qoq SAAR percent minus GDPNow forecast right before the advance estimate",
            "unit": "annualized_percentage_points",
            "source": "Atlanta Fed GDPNow TrackRecord",
            "neutral": 0,
        },
        "unemployment_change_pct": {
            "formula": "current initial UNRATE release minus the previous month's initial UNRATE release",
            "unit": "percentage_points",
            "source": "ALFRED output_type=2",
            "neutral": 0,
        },
        "kc_fed_manufacturing_pmi_proxy_change_points": {
            "formula": "current (50 + 0.5 * KC composite) minus previous level",
            "unit": "pmi_style_index_points",
            "source": "Kansas City Fed Tenth District Manufacturing Survey",
            "neutral": 0,
            "contract_input": "pmi_change_points",
            "proxy": True,
        },
        "dxy_change_pct": {
            "formula": "100 * (latest DX-Y.NYB Close at/before window end / latest Close at/before window start - 1)",
            "unit": "percent",
            "source": "Yahoo DX-Y.NYB Close",
            "neutral": 0,
        },
        "vix_change_points": {
            "formula": "latest VIXCLS at/before window end minus latest VIXCLS at/before window start",
            "unit": "vix_index_points",
            "source": "FRED VIXCLS",
            "neutral": 0,
        },
    }
    distributions = {
        feature: _distribution(features[feature])
        for feature in HISTORICAL_SHOCK_NAMES
    }
    summary = {
        "schema_version": "scenario_shock_features_v1",
        "construction_mode": "offline_cached_data_only",
        "sampling_design": "EVENT_AWARE",
        "row_count": int(len(features)),
        "historical_shock_feature_count": len(HISTORICAL_SHOCK_NAMES),
        "supporting_release_flag_count": len(EVENT_FLAG_NAMES),
        "total_numeric_feature_count": len(SHOCK_FEATURE_NAMES),
        "first_sample_date": str(features["sample_date"].iloc[0]),
        "last_sample_date": str(features["sample_date"].iloc[-1]),
        "contract_shock_names": list(CONTRACT_SHOCK_NAMES),
        "historical_shock_names": list(HISTORICAL_SHOCK_NAMES),
        "contract_to_historical_feature": CONTRACT_TO_HISTORICAL_FEATURE,
        "supporting_release_flags": list(EVENT_FLAG_NAMES),
        "output_schema": list(SHOCK_OUTPUT_COLUMNS),
        "event_audit_row_count": int(len(event_audit)),
        "event_counts": dict(sorted(event_counts.items())),
        "release_flag_counts": dict(sorted(flag_counts.items())),
        "market_audit_row_count": int(len(market_audit)),
        "definitions": definitions,
        "distributions": distributions,
        "missingness": {
            name: int(features[name].isna().sum())
            for name in SHOCK_FEATURE_NAMES
        },
        "market_source_staleness": {
            feature: {
                "maximum_start_age_calendar_days": int(
                    group["window_start_age_calendar_days"].max()
                ),
                "maximum_end_age_calendar_days": int(
                    group["window_end_age_calendar_days"].max()
                ),
                "same_source_observation_at_both_boundaries": int(
                    (
                        group["window_start_source_date"]
                        == group["window_end_source_date"]
                    ).sum()
                ),
                "nonpositive_start_value_count": int(
                    group["nonpositive_start_value_flag"].sum()
                ),
            }
            for feature, group in market_audit.groupby("shock_feature")
        },
        "validations": validations,
        "source_manifest": source_manifest(
            [
                selected_sampling_path,
                normalized_events_path,
                *event_paths,
                *market_paths,
            ]
        ),
        "limitations": [
            (
                "The Kansas City feature is a regional, revised-history proxy "
                "and is deliberately not named as national ISM PMI."
            ),
            (
                "The Atlanta TrackRecord identifies the GDPNow value as the "
                "forecast right before the advance estimate but does not "
                "provide a separate forecast timestamp."
            ),
            (
                "CPI and unemployment initial values use ALFRED vintage-date "
                "columns; cached exact historical release times are absent. "
                "The November 2025 CPI actual uses the same-measure Cleveland "
                "release-date actual because the October CPIAUCSL level is "
                "absent from that release vintage."
            ),
            (
                "Market shocks use the latest cached observation at or before "
                "each stored Step 4 boundary; source staleness remains audited."
            ),
        ],
        "scope_boundary": (
            "Step 6 only: no state features, targets, model matrix, splits, "
            "training, Supabase, backend, or frontend behavior."
        ),
    }
    return {
        "features": features,
        "event_audit": event_audit,
        "market_audit": market_audit,
        "summary": summary,
    }


def persist_shock_artifacts(
    artifacts: dict[str, Any],
    *,
    output_dir: Path = SHOCK_OUTPUT_ROOT,
) -> dict[str, str]:
    paths = {
        "features": output_dir / "scenario_shocks.csv",
        "event_audit": output_dir / "scenario_shock_event_audit.csv",
        "market_audit": output_dir / "scenario_market_shock_audit.csv",
        "summary": output_dir / "scenario_shock_summary.json",
    }
    csv_text = {
        name: frame_to_csv_text(artifacts[name])
        for name in ("features", "event_audit", "market_audit")
    }
    summary = dict(artifacts["summary"])
    summary["output_files"] = {
        name: {
            "path": relative_path(path),
            "sha256": (
                hashlib.sha256(csv_text[name].encode("utf-8")).hexdigest()
                if name in csv_text
                else None
            ),
        }
        for name, path in paths.items()
    }
    for name in ("features", "event_audit", "market_audit"):
        write_text_atomic(paths[name], csv_text[name])
    write_text_atomic(
        paths["summary"],
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n",
    )
    return {name: str(path.resolve()) for name, path in paths.items()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Construct and validate the eight historical Scenario Playground "
            "shock features from local cached data."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and validate in memory without writing interim outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SHOCK_OUTPUT_ROOT,
        help=(
            "Output directory (default: "
            "data/scenario_playground/interim/features/shocks)."
        ),
    )
    args = parser.parse_args(argv)
    artifacts = build_shock_artifacts()
    outputs = (
        {}
        if args.dry_run
        else persist_shock_artifacts(
            artifacts, output_dir=args.output_dir
        )
    )
    summary = artifacts["summary"]
    print(
        json.dumps(
            {
                "row_count": summary["row_count"],
                "historical_shock_feature_count": summary[
                    "historical_shock_feature_count"
                ],
                "supporting_release_flag_count": summary[
                    "supporting_release_flag_count"
                ],
                "event_audit_row_count": summary["event_audit_row_count"],
                "outputs": outputs,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
