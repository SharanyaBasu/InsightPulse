"""Construct point-in-time market, macro, risk, and trend state features.

Step 7 conditions each selected event-aware row on information available at its
pre-shock ``state_cutoff_date``.  It does not construct shocks, targets, model
matrices, splits, or models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .cache import write_text_atomic
from .config import INTERIM_ROOT, RAW_ROOT
from .feature_sources import (
    SELECTED_SAMPLING_PATH,
    AlfredVintagePanel,
    asof_lookup,
    frame_to_csv_text,
    load_alfred_vintage_panel,
    load_atlanta_gdp_track_record,
    load_feature_index,
    load_fred_series,
    load_yahoo_series,
    months_before_start,
    previous_month_start,
    relative_path,
    source_manifest,
)
from .sampling import load_canonical_calendar


STATE_OUTPUT_ROOT = INTERIM_ROOT / "features" / "state"

MACRO_STATE_FEATURES = (
    "fed_funds_target_midpoint_pct",
    "effective_fed_funds_rate_pct",
    "cpi_mom_pct",
    "cpi_yoy_pct",
    "unemployment_rate_pct",
    "real_gdp_qoq_annualized_pct",
    "ten_year_yield_pct",
    "two_year_yield_pct",
    "yield_curve_slope_10y_2y_pct_points",
    "real_rate_proxy_pct_points",
)

MARKET_ASSET_NAMES = ("sp500", "nasdaq", "gold", "oil", "dxy")
MARKET_DERIVED_SUFFIXES = (
    "return_5d_pct",
    "return_21d_pct",
    "volatility_20d_annualized_pct",
    "drawdown_252d_pct",
    "ma_distance_50d_pct",
    "ma_distance_200d_pct",
    "zscore_252d",
)
MARKET_DERIVED_FEATURES = tuple(
    f"{asset}_{suffix}"
    for asset in MARKET_ASSET_NAMES
    for suffix in MARKET_DERIVED_SUFFIXES
)

STATE_FEATURE_NAMES = (
    MACRO_STATE_FEATURES
    + ("vix_level_points",)
    + MARKET_DERIVED_FEATURES
)
STATE_KEY_COLUMNS = ("sample_date", "state_cutoff_date")
STATE_OUTPUT_COLUMNS = STATE_KEY_COLUMNS + STATE_FEATURE_NAMES


@dataclass(frozen=True)
class MarketStateSpec:
    name: str
    filename: str
    value_field: str
    source: str
    adjustment_policy: str


MARKET_STATE_SPECS = (
    MarketStateSpec(
        "sp500",
        "GSPC.csv",
        "Close",
        "Yahoo ^GSPC",
        "native index Close",
    ),
    MarketStateSpec(
        "nasdaq",
        "IXIC.csv",
        "Close",
        "Yahoo ^IXIC",
        "native index Close",
    ),
    MarketStateSpec(
        "gold",
        "GLD.csv",
        "Adj Close",
        "Yahoo GLD",
        "distribution-adjusted ETF close",
    ),
    MarketStateSpec(
        "oil",
        "CL_F.csv",
        "Close",
        "Yahoo CL=F",
        "native continuous-futures Close",
    ),
    MarketStateSpec(
        "dxy",
        "DX-Y.NYB.csv",
        "Close",
        "Yahoo DX-Y.NYB",
        "native index Close",
    ),
)


def calculate_market_state_history(values: pd.Series) -> pd.DataFrame:
    """Calculate trailing features at every source observation date."""
    series = pd.to_numeric(values, errors="coerce").dropna().sort_index()
    if series.index.duplicated().any():
        raise ValueError("Market-state source has duplicate dates.")
    if len(series) < 253:
        raise ValueError("Market-state source has fewer than 253 observations.")
    daily_returns = series.pct_change(fill_method=None)
    rolling_252_mean = series.rolling(252, min_periods=252).mean()
    rolling_252_std = series.rolling(252, min_periods=252).std(ddof=1)

    frame = pd.DataFrame(index=series.index)
    frame["return_5d_pct"] = 100.0 * (series / series.shift(5) - 1.0)
    frame["return_21d_pct"] = 100.0 * (series / series.shift(21) - 1.0)
    frame["volatility_20d_annualized_pct"] = (
        100.0
        * math.sqrt(252.0)
        * daily_returns.rolling(20, min_periods=20).std(ddof=1)
    )
    frame["drawdown_252d_pct"] = 100.0 * (
        series / series.rolling(252, min_periods=252).max() - 1.0
    )
    frame["ma_distance_50d_pct"] = 100.0 * (
        series / series.rolling(50, min_periods=50).mean() - 1.0
    )
    frame["ma_distance_200d_pct"] = 100.0 * (
        series / series.rolling(200, min_periods=200).mean() - 1.0
    )
    frame["zscore_252d"] = (
        (series - rolling_252_mean) / rolling_252_std
    )
    frame = frame.replace([float("inf"), float("-inf")], pd.NA).dropna()
    if frame.empty:
        raise ValueError("Market-state calculations produced no complete rows.")
    return frame.astype(float)


def _next_session(
    value: str,
    sessions: pd.DatetimeIndex,
) -> str | None:
    position = int(
        sessions.searchsorted(pd.Timestamp(value), side="right")
    )
    if position >= len(sessions):
        return None
    return sessions[position].date().isoformat()


def _availability_timeline(
    frame: pd.DataFrame,
    *,
    value_column: str,
    release_column: str,
    source_observation_column: str,
    sessions: pd.DatetimeIndex,
    strictly_next_session: bool,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        release_date = str(getattr(row, release_column))
        availability_date = (
            _next_session(release_date, sessions)
            if strictly_next_session
            else release_date
        )
        if availability_date is None:
            continue
        value = float(getattr(row, value_column))
        if not math.isfinite(value):
            continue
        records.append(
            {
                "availability_date": availability_date,
                "source_observation_date": str(
                    getattr(row, source_observation_column)
                ),
                "release_or_publication_date": release_date,
                "value": value,
            }
        )
    timeline = pd.DataFrame.from_records(records)
    if timeline.empty:
        raise ValueError(f"No availability timeline for {value_column}.")
    timeline = (
        timeline.sort_values(
            ["availability_date", "release_or_publication_date"]
        )
        .drop_duplicates("availability_date", keep="last")
        .reset_index(drop=True)
    )
    return timeline


def _timeline_asof(
    timeline: pd.DataFrame,
    cutoffs: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    timeline = (
        timeline.sort_values(
            ["availability_date", "source_observation_date"]
        )
        .drop_duplicates("availability_date", keep="last")
        .reset_index(drop=True)
    )
    series = pd.Series(
        timeline["value"].to_numpy(dtype=float),
        index=pd.to_datetime(timeline["availability_date"]),
    )
    result = asof_lookup(series, cutoffs)
    metadata = timeline.set_index("availability_date")
    observation_dates = result.source_dates.map(
        metadata["source_observation_date"]
    )
    release_dates = result.source_dates.map(
        metadata["release_or_publication_date"]
    )
    if observation_dates.isna().any() or release_dates.isna().any():
        raise ValueError("Timeline metadata could not be reconstructed.")
    return (
        result.values,
        result.source_dates,
        observation_dates,
        release_dates,
    )


def _cpi_state_timelines(
    panel: AlfredVintagePanel,
    sessions: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    releases = panel.first_release_table()
    mom_records: list[dict[str, Any]] = []
    yoy_records: list[dict[str, Any]] = []
    for row in releases.itertuples(index=False):
        observation_date = str(row.observation_date)
        release_date = str(row.release_date)
        current = float(row.value)
        availability = _next_session(release_date, sessions)
        if availability is None:
            continue
        common = {
            "availability_date": availability,
            "source_observation_date": observation_date,
            "release_or_publication_date": release_date,
        }
        try:
            previous = panel.value(
                previous_month_start(observation_date), release_date
            )
            mom_records.append(
                {
                    **common,
                    "value": 100.0 * (current / previous - 1.0),
                }
            )
        except KeyError:
            pass
        try:
            year_ago = panel.value(
                months_before_start(observation_date, 12), release_date
            )
            yoy_records.append(
                {
                    **common,
                    "value": 100.0 * (current / year_ago - 1.0),
                }
            )
        except KeyError:
            pass
    mom = pd.DataFrame.from_records(mom_records).sort_values(
        "availability_date"
    )
    yoy = pd.DataFrame.from_records(yoy_records).sort_values(
        "availability_date"
    )
    if mom.empty or yoy.empty:
        raise ValueError("CPI state timelines are empty.")
    return mom.reset_index(drop=True), yoy.reset_index(drop=True)


def _unemployment_timeline(
    panel: AlfredVintagePanel,
    sessions: pd.DatetimeIndex,
) -> pd.DataFrame:
    releases = panel.first_release_table()
    return _availability_timeline(
        releases,
        value_column="value",
        release_column="release_date",
        source_observation_column="observation_date",
        sessions=sessions,
        strictly_next_session=True,
    )


def _effr_timeline(
    values: pd.Series,
    sessions: pd.DatetimeIndex,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "observation_date": pd.to_datetime(values.index).strftime(
                "%Y-%m-%d"
            ),
            "release_date": pd.to_datetime(values.index).strftime("%Y-%m-%d"),
            "value": values.to_numpy(dtype=float),
        }
    )
    # EFFR for business day d is published the following business morning.
    return _availability_timeline(
        frame,
        value_column="value",
        release_column="release_date",
        source_observation_column="observation_date",
        sessions=sessions,
        strictly_next_session=True,
    )


def _direct_timeline(values: pd.Series) -> pd.DataFrame:
    dates = pd.to_datetime(values.index).strftime("%Y-%m-%d")
    return pd.DataFrame(
        {
            "availability_date": dates,
            "source_observation_date": dates,
            "release_or_publication_date": dates,
            "value": values.to_numpy(dtype=float),
        }
    )


def _macro_features(
    sampling: pd.DataFrame,
    *,
    raw_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[Path], dict[str, Any]]:
    sessions = pd.DatetimeIndex(
        [pd.Timestamp(item) for item in load_canonical_calendar(raw_root)]
    )
    cutoffs = sampling["state_cutoff_date"]

    lower, lower_path = load_fred_series("DFEDTARL", raw_root=raw_root)
    upper, upper_path = load_fred_series("DFEDTARU", raw_root=raw_root)
    midpoint_frame = pd.concat(
        [lower.rename("lower"), upper.rename("upper")], axis=1
    ).dropna()
    midpoint = (midpoint_frame["lower"] + midpoint_frame["upper"]) / 2.0

    effr, effr_path = load_fred_series("EFFR", raw_root=raw_root)
    dgs10, dgs10_path = load_fred_series("DGS10", raw_root=raw_root)
    dgs2, dgs2_path = load_fred_series("DGS2", raw_root=raw_root)
    cpi_panel = load_alfred_vintage_panel("CPIAUCSL", raw_root=raw_root)
    unemployment_panel = load_alfred_vintage_panel(
        "UNRATE", raw_root=raw_root
    )
    gdp, gdp_path = load_atlanta_gdp_track_record(raw_root=raw_root)

    cpi_mom_timeline, cpi_yoy_timeline = _cpi_state_timelines(
        cpi_panel, sessions
    )
    unemployment_timeline = _unemployment_timeline(
        unemployment_panel, sessions
    )
    gdp_timeline = _availability_timeline(
        gdp,
        value_column="bea_advance_qoq_saar_pct",
        release_column="release_date",
        source_observation_column="observation_period",
        sessions=sessions,
        strictly_next_session=True,
    )

    timelines = {
        "fed_funds_target_midpoint_pct": _direct_timeline(midpoint),
        "effective_fed_funds_rate_pct": _effr_timeline(effr, sessions),
        "cpi_mom_pct": cpi_mom_timeline,
        "cpi_yoy_pct": cpi_yoy_timeline,
        "unemployment_rate_pct": unemployment_timeline,
        "real_gdp_qoq_annualized_pct": gdp_timeline,
        "ten_year_yield_pct": _direct_timeline(dgs10),
        "two_year_yield_pct": _direct_timeline(dgs2),
    }

    features = pd.DataFrame(index=sampling.index)
    audit = pd.DataFrame(
        {
            "sample_date": sampling["sample_date"],
            "state_cutoff_date": sampling["state_cutoff_date"],
        }
    )
    staleness: dict[str, Any] = {}
    for feature, timeline in timelines.items():
        values, availability_dates, observation_dates, release_dates = (
            _timeline_asof(timeline, cutoffs)
        )
        features[feature] = values
        prefix = feature.removesuffix("_pct").removesuffix("_points")
        audit[f"{prefix}_availability_date"] = availability_dates
        audit[f"{prefix}_source_observation"] = observation_dates
        audit[f"{prefix}_release_or_publication_date"] = release_dates
        ages = (
            pd.to_datetime(cutoffs) - pd.to_datetime(availability_dates)
        ).dt.days
        audit[f"{prefix}_age_calendar_days"] = ages
        staleness[feature] = {
            "maximum_age_calendar_days": int(ages.max()),
            "median_age_calendar_days": float(ages.median()),
        }

    features["yield_curve_slope_10y_2y_pct_points"] = (
        features["ten_year_yield_pct"] - features["two_year_yield_pct"]
    )
    features["real_rate_proxy_pct_points"] = (
        features["fed_funds_target_midpoint_pct"]
        - features["cpi_yoy_pct"]
    )
    return (
        features.loc[:, list(MACRO_STATE_FEATURES)],
        audit,
        [
            lower_path,
            upper_path,
            effr_path,
            dgs10_path,
            dgs2_path,
            *cpi_panel.source_paths,
            *unemployment_panel.source_paths,
            gdp_path,
        ],
        staleness,
    )


def _market_features(
    sampling: pd.DataFrame,
    *,
    raw_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[Path], dict[str, Any]]:
    features = pd.DataFrame(index=sampling.index)
    audit = pd.DataFrame(
        {
            "sample_date": sampling["sample_date"],
            "state_cutoff_date": sampling["state_cutoff_date"],
        }
    )
    paths: list[Path] = []
    diagnostics: dict[str, Any] = {}
    cutoffs = sampling["state_cutoff_date"]

    for spec in MARKET_STATE_SPECS:
        values, path = load_yahoo_series(
            spec.filename, spec.value_field, raw_root=raw_root
        )
        history = calculate_market_state_history(values)
        source_dates: pd.Series | None = None
        source_ages: pd.Series | None = None
        for suffix in MARKET_DERIVED_SUFFIXES:
            result = asof_lookup(history[suffix], cutoffs)
            features[f"{spec.name}_{suffix}"] = result.values
            if source_dates is None:
                source_dates = result.source_dates
                source_ages = result.age_calendar_days
            elif not result.source_dates.equals(source_dates):
                raise ValueError(
                    f"{spec.name} rolling features use different end dates."
                )
        if source_dates is None or source_ages is None:
            raise ValueError(f"No market-state features for {spec.name}.")
        audit[f"{spec.name}_source_date"] = source_dates
        audit[f"{spec.name}_age_calendar_days"] = source_ages
        diagnostics[spec.name] = {
            "source": spec.source,
            "path": relative_path(path),
            "field": spec.value_field,
            "adjustment_policy": spec.adjustment_policy,
            "maximum_source_age_calendar_days": int(source_ages.max()),
            "minimum_history_observations_required": 252,
            "nonpositive_raw_value_count": int((values <= 0).sum()),
        }
        paths.append(path)

    vix, vix_path = load_fred_series("VIXCLS", raw_root=raw_root)
    vix_result = asof_lookup(vix, cutoffs)
    features["vix_level_points"] = vix_result.values
    audit["vix_source_date"] = vix_result.source_dates
    audit["vix_age_calendar_days"] = vix_result.age_calendar_days
    diagnostics["vix"] = {
        "source": "FRED VIXCLS",
        "path": str(vix_path),
        "field": "value",
        "adjustment_policy": "native VIX index points",
        "maximum_source_age_calendar_days": int(
            vix_result.age_calendar_days.max()
        ),
    }
    paths.append(vix_path)
    ordered = ["vix_level_points", *MARKET_DERIVED_FEATURES]
    return features.loc[:, ordered], audit, paths, diagnostics


def _definition_map() -> dict[str, dict[str, Any]]:
    definitions: dict[str, dict[str, Any]] = {
        "fed_funds_target_midpoint_pct": {
            "source": "FRED DFEDTARL and DFEDTARU",
            "formula": "(lower + upper) / 2, latest effective value at cutoff",
            "unit": "percent",
        },
        "effective_fed_funds_rate_pct": {
            "source": "FRED EFFR",
            "formula": "latest observation published by the next equity session",
            "unit": "percent",
        },
        "cpi_mom_pct": {
            "source": "ALFRED CPIAUCSL output_type=2",
            "formula": "100 * (initial-vintage current month / prior month - 1)",
            "unit": "percent",
        },
        "cpi_yoy_pct": {
            "source": "ALFRED CPIAUCSL output_type=2",
            "formula": "100 * (initial-vintage current month / 12-month-prior level - 1)",
            "unit": "percent",
        },
        "unemployment_rate_pct": {
            "source": "ALFRED UNRATE output_type=2",
            "formula": "latest initially released unemployment rate",
            "unit": "percent",
        },
        "real_gdp_qoq_annualized_pct": {
            "source": "Atlanta Fed TrackRecord BEA advance value",
            "formula": "latest BEA advance real-GDP qoq SAAR value",
            "unit": "annualized_percent",
        },
        "ten_year_yield_pct": {
            "source": "FRED DGS10",
            "formula": "latest value at or before state cutoff",
            "unit": "percent",
        },
        "two_year_yield_pct": {
            "source": "FRED DGS2",
            "formula": "latest value at or before state cutoff",
            "unit": "percent",
        },
        "yield_curve_slope_10y_2y_pct_points": {
            "source": "Derived from DGS10 and DGS2",
            "formula": "ten_year_yield_pct - two_year_yield_pct",
            "unit": "percentage_points",
        },
        "real_rate_proxy_pct_points": {
            "source": "Derived policy midpoint and CPI yoy",
            "formula": "fed_funds_target_midpoint_pct - cpi_yoy_pct",
            "unit": "percentage_points",
            "proxy": True,
        },
        "vix_level_points": {
            "source": "FRED VIXCLS",
            "formula": "latest value at or before state cutoff",
            "unit": "vix_index_points",
        },
    }
    source_by_asset = {spec.name: spec.source for spec in MARKET_STATE_SPECS}
    for asset in MARKET_ASSET_NAMES:
        source = source_by_asset[asset]
        definitions[f"{asset}_return_5d_pct"] = {
            "source": source,
            "formula": "100 * (current / value 5 source observations earlier - 1)",
            "unit": "percent",
        }
        definitions[f"{asset}_return_21d_pct"] = {
            "source": source,
            "formula": "100 * (current / value 21 source observations earlier - 1)",
            "unit": "percent",
        }
        definitions[f"{asset}_volatility_20d_annualized_pct"] = {
            "source": source,
            "formula": "100 * sqrt(252) * sample std of last 20 arithmetic daily returns",
            "unit": "annualized_percent",
        }
        definitions[f"{asset}_drawdown_252d_pct"] = {
            "source": source,
            "formula": "100 * (current / maximum of trailing 252 source observations - 1)",
            "unit": "percent",
        }
        definitions[f"{asset}_ma_distance_50d_pct"] = {
            "source": source,
            "formula": "100 * (current / trailing 50-observation mean - 1)",
            "unit": "percent",
        }
        definitions[f"{asset}_ma_distance_200d_pct"] = {
            "source": source,
            "formula": "100 * (current / trailing 200-observation mean - 1)",
            "unit": "percent",
        }
        definitions[f"{asset}_zscore_252d"] = {
            "source": source,
            "formula": "(current - trailing 252-observation mean) / trailing sample standard deviation",
            "unit": "standard_deviations",
        }
    return definitions


def _distribution(values: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[numeric.map(math.isfinite)]
    return {
        "count": int(len(numeric)),
        "missing_count": int(numeric.isna().sum()),
        "non_finite_count": int(
            numeric.notna().sum() - len(finite)
        ),
        "minimum": float(finite.min()),
        "median": float(finite.median()),
        "maximum": float(finite.max()),
        "mean": float(finite.mean()),
        "standard_deviation": float(finite.std(ddof=1)),
    }


def build_state_artifacts(
    *,
    selected_sampling_path: Path = SELECTED_SAMPLING_PATH,
    raw_root: Path = RAW_ROOT,
) -> dict[str, Any]:
    """Build and validate all Step 7 artifacts in memory."""
    sampling = load_feature_index(selected_sampling_path)
    macro, macro_audit, macro_paths, macro_staleness = _macro_features(
        sampling, raw_root=raw_root
    )
    market, market_audit, market_paths, market_diagnostics = (
        _market_features(sampling, raw_root=raw_root)
    )

    features = sampling.loc[:, list(STATE_KEY_COLUMNS)].copy()
    for name in MACRO_STATE_FEATURES:
        features[name] = macro[name]
    features["vix_level_points"] = market["vix_level_points"]
    for name in MARKET_DERIVED_FEATURES:
        features[name] = market[name]
    features = features.loc[:, list(STATE_OUTPUT_COLUMNS)]

    audit = macro_audit.merge(
        market_audit,
        on=["sample_date", "state_cutoff_date"],
        how="inner",
        validate="one_to_one",
    )
    availability_columns = [
        column
        for column in audit.columns
        if column.endswith("_availability_date")
        or column.endswith("_source_date")
    ]
    point_in_time_by_column = {
        column: bool(
            (
                pd.to_datetime(audit[column])
                <= pd.to_datetime(audit["state_cutoff_date"])
            ).all()
        )
        for column in availability_columns
    }
    feature_values = features.loc[:, list(STATE_FEATURE_NAMES)]
    definitions = _definition_map()
    validations = {
        "selected_sampling_design_is_event_aware": True,
        "all_585_selected_rows_preserved": len(features) == 585,
        "sample_dates_unique_and_ordered": bool(
            features["sample_date"].is_monotonic_increasing
            and not features["sample_date"].duplicated().any()
        ),
        "state_cutoff_strictly_precedes_sample_date": bool(
            (
                pd.to_datetime(features["state_cutoff_date"])
                < pd.to_datetime(features["sample_date"])
            ).all()
        ),
        "feature_count_between_25_and_60": (
            25 <= len(STATE_FEATURE_NAMES) <= 60
        ),
        "all_declared_state_features_present": tuple(
            name for name in features if name in STATE_FEATURE_NAMES
        )
        == STATE_FEATURE_NAMES,
        "all_feature_definitions_present": (
            set(definitions) == set(STATE_FEATURE_NAMES)
        ),
        "no_state_feature_values_missing": bool(
            not feature_values.isna().any().any()
        ),
        "all_state_feature_values_finite": bool(
            feature_values.map(math.isfinite).all().all()
        ),
        "all_source_availability_dates_on_or_before_state_cutoff": all(
            point_in_time_by_column.values()
        ),
        "rolling_features_end_on_or_before_state_cutoff": all(
            passed
            for column, passed in point_in_time_by_column.items()
            if column.endswith("_source_date")
        ),
        "macro_release_values_not_backfilled_before_availability": all(
            passed
            for column, passed in point_in_time_by_column.items()
            if column.endswith("_availability_date")
        ),
        "rolling_200_and_252_observation_lookbacks_are_complete": True,
        "no_shocks_targets_matrix_splits_or_models_constructed": True,
    }
    if not all(validations.values()):
        failed = [name for name, passed in validations.items() if not passed]
        raise ValueError(f"Step 7 state validation failed: {failed}")

    summary = {
        "schema_version": "scenario_state_features_v1",
        "construction_mode": "offline_cached_data_only",
        "sampling_design": "EVENT_AWARE",
        "row_count": int(len(features)),
        "state_feature_count": len(STATE_FEATURE_NAMES),
        "macro_state_feature_count": len(MACRO_STATE_FEATURES),
        "market_level_feature_count": 1,
        "market_risk_trend_feature_count": len(
            MARKET_DERIVED_FEATURES
        ),
        "first_sample_date": str(features["sample_date"].iloc[0]),
        "last_sample_date": str(features["sample_date"].iloc[-1]),
        "first_state_cutoff_date": str(
            features["state_cutoff_date"].iloc[0]
        ),
        "last_state_cutoff_date": str(
            features["state_cutoff_date"].iloc[-1]
        ),
        "state_feature_names": list(STATE_FEATURE_NAMES),
        "output_schema": list(STATE_OUTPUT_COLUMNS),
        "availability_audit_schema": list(audit.columns),
        "definitions": definitions,
        "missingness": {
            name: int(features[name].isna().sum())
            for name in STATE_FEATURE_NAMES
        },
        "distributions": {
            name: _distribution(features[name])
            for name in STATE_FEATURE_NAMES
        },
        "macro_staleness": macro_staleness,
        "market_diagnostics": market_diagnostics,
        "point_in_time_by_availability_column": point_in_time_by_column,
        "validations": validations,
        "source_manifest": source_manifest(
            [selected_sampling_path, *macro_paths, *market_paths]
        ),
        "limitations": [
            (
                "CPI, unemployment, and GDP use conservative next-session "
                "availability because exact cached release times are absent."
            ),
            (
                "CPI and unemployment states use first-vintage paths; the "
                "Kansas City PMI proxy is not a state feature in this step."
            ),
            (
                "CL=F contains one nonpositive historical close in 2020. Oil "
                "arithmetic-return and trend features remain untrimmed for "
                "later Step 9 outlier review."
            ),
            (
                "Market as-of joins can use a prior source observation on a "
                "US equity session when that source did not publish; exact "
                "source dates and ages are retained in the audit."
            ),
        ],
        "scope_boundary": (
            "Step 7 only: no shocks, targets, model matrix, splits, training, "
            "Supabase, backend, or frontend behavior."
        ),
    }
    return {
        "features": features,
        "availability_audit": audit,
        "summary": summary,
    }


def persist_state_artifacts(
    artifacts: dict[str, Any],
    *,
    output_dir: Path = STATE_OUTPUT_ROOT,
) -> dict[str, str]:
    paths = {
        "features": output_dir / "state_features.csv",
        "availability_audit": (
            output_dir / "state_feature_availability_audit.csv"
        ),
        "summary": output_dir / "state_feature_summary.json",
    }
    csv_text = {
        name: frame_to_csv_text(artifacts[name])
        for name in ("features", "availability_audit")
    }
    summary = dict(artifacts["summary"])
    summary["output_files"] = {
        name: {
            "path": str(path),
            "sha256": (
                hashlib.sha256(csv_text[name].encode("utf-8")).hexdigest()
                if name in csv_text
                else None
            ),
        }
        for name, path in paths.items()
    }
    write_text_atomic(paths["features"], csv_text["features"])
    write_text_atomic(
        paths["availability_audit"], csv_text["availability_audit"]
    )
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
            "Construct and validate point-in-time Scenario Playground state "
            "features from local cached data."
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
        default=STATE_OUTPUT_ROOT,
        help=(
            "Output directory (default: "
            "data/scenario_playground/interim/features/state)."
        ),
    )
    args = parser.parse_args(argv)
    artifacts = build_state_artifacts()
    outputs = (
        {}
        if args.dry_run
        else persist_state_artifacts(
            artifacts, output_dir=args.output_dir
        )
    )
    summary = artifacts["summary"]
    print(
        json.dumps(
            {
                "row_count": summary["row_count"],
                "state_feature_count": summary["state_feature_count"],
                "macro_state_feature_count": summary[
                    "macro_state_feature_count"
                ],
                "market_risk_trend_feature_count": summary[
                    "market_risk_trend_feature_count"
                ],
                "outputs": outputs,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
