"""Assemble the raw point-in-time Scenario Playground model matrix.

Step 8 joins the already point-in-time Step 5-7 outputs.  It keeps model
values, provenance, and audit-only columns in separate artifacts and performs
no imputation, scaling, clipping, winsorization, or model fitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from .cache import write_text_atomic
from .config import INTERIM_ROOT, PROCESSED_ROOT, RAW_ROOT
from .feature_sources import frame_to_csv_text, relative_path, source_manifest
from .sampling import (
    TARGET_HORIZON_SESSIONS,
    TARGET_SPECS,
    load_canonical_calendar,
)
from .shock_features import (
    EVENT_FLAG_NAMES,
    HISTORICAL_SHOCK_NAMES,
    SHOCK_FEATURE_NAMES,
)
from .splitting import build_split_manifest
from .state_features import (
    MACRO_STATE_FEATURES,
    MARKET_DERIVED_FEATURES,
    STATE_FEATURE_NAMES,
)
from .targets import TARGET_NAMES


SAMPLING_PATH = (
    INTERIM_ROOT / "sampling" / "selected_sampling_index.csv"
)
SHOCKS_PATH = (
    INTERIM_ROOT / "features" / "shocks" / "scenario_shocks.csv"
)
SHOCK_SUMMARY_PATH = (
    INTERIM_ROOT
    / "features"
    / "shocks"
    / "scenario_shock_summary.json"
)
STATE_PATH = (
    INTERIM_ROOT / "features" / "state" / "state_features.csv"
)
STATE_SUMMARY_PATH = (
    INTERIM_ROOT
    / "features"
    / "state"
    / "state_feature_summary.json"
)
TARGETS_PATH = INTERIM_ROOT / "targets" / "market_targets.csv"
TARGET_SUMMARY_PATH = (
    INTERIM_ROOT / "targets" / "market_target_summary.json"
)
MATRIX_OUTPUT_ROOT = PROCESSED_ROOT / "model_matrix"

MATRIX_KEY_COLUMNS = ("sample_date",)
INITIAL_FEATURE_NAMES = SHOCK_FEATURE_NAMES + STATE_FEATURE_NAMES

# Step 9's structural outlier audit found that this percentage-return
# volatility crosses the one negative CL=F settlement in April 2020.  The
# source state table remains unchanged; only this unsafe model input is omitted
# from the final training matrix.
OIL_VOLATILITY_FEATURE = "oil_volatility_20d_annualized_pct"
FINAL_AUDIT_EXCLUDED_FEATURES = (OIL_VOLATILITY_FEATURE,)
FINAL_FEATURE_NAMES = tuple(
    name
    for name in INITIAL_FEATURE_NAMES
    if name not in FINAL_AUDIT_EXCLUDED_FEATURES
)

PROVENANCE_COLUMNS = (
    "sample_date",
    "sampling_design",
    "model_availability_date",
    "state_cutoff_date",
    "shock_window_start_date",
    "shock_window_end_date",
    "target_baseline_date",
    "target_end_date",
    "target_horizon_sessions",
    "event_count",
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

SOURCE_ROW_AUDIT_COLUMNS = (
    "sample_date",
    "has_21_future_sessions",
    "full_horizon_within_shared_coverage",
    "all_target_values_available",
    "missing_target_fields_json",
    "label_eligible",
    "ineligibility_reason",
    "collision_count",
    "duplicate_candidates_removed",
)

ROW_EXCLUSION_COLUMNS = (
    "sample_date",
    "column_name",
    "event_types_json",
    "event_ids_json",
    "decision",
    "reason",
    "imputation_performed",
)

FEATURE_EXCLUSION_COLUMNS = (
    "feature_name",
    "source_artifact",
    "decision",
    "reason",
    "rows_removed",
    "transformation_performed",
)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required Step 5-7 artifact is missing: {path}")
    return pd.read_csv(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required summary is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _same_text(left: pd.Series, right: pd.Series) -> bool:
    return left.fillna("").astype(str).tolist() == right.fillna("").astype(
        str
    ).tolist()


def _assert_matching_columns(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    columns: Iterable[str],
    *,
    candidate_name: str,
) -> None:
    for column in columns:
        if column not in reference or column not in candidate:
            raise ValueError(
                f"Cannot verify {candidate_name} key column: {column}"
            )
        if not _same_text(reference[column], candidate[column]):
            raise ValueError(
                f"{candidate_name} disagrees with the selected sampling index "
                f"for {column}."
            )


def _feature_family(name: str) -> str:
    if name in HISTORICAL_SHOCK_NAMES:
        return "scenario_shock"
    if name in EVENT_FLAG_NAMES:
        return "shock_release_flag"
    if name in MACRO_STATE_FEATURES:
        return "macro_state"
    if name == "vix_level_points":
        return "market_state_level"
    if name in MARKET_DERIVED_FEATURES:
        return "market_state_risk_trend"
    raise KeyError(f"Unknown feature family: {name}")


def _revision_risk(name: str) -> str:
    if name in {
        "kc_fed_manufacturing_pmi_proxy_change_points",
        "real_gdp_qoq_annualized_pct",
        "gdp_growth_surprise_pct",
    }:
        return "MODERATE_TO_HIGH"
    if name.startswith("oil_"):
        return "HIGH_CONTINUOUS_FUTURES_STRUCTURE"
    if name in {
        "cpi_surprise_pct",
        "cpi_mom_pct",
        "cpi_yoy_pct",
        "unemployment_change_pct",
        "unemployment_rate_pct",
        "real_rate_proxy_pct_points",
    }:
        return "MODERATE_POINT_IN_TIME_OR_RELEASE_LIMITATION"
    return "LOW_DOCUMENTED_SOURCE_LIMITATION"


def _point_in_time_rule(name: str) -> str:
    if name in {
        "oil_price_change_pct",
        "dxy_change_pct",
        "vix_change_points",
    }:
        return (
            "Use only source observations at or before the stored shock-window "
            "boundaries; the completed shock is known by sample_date."
        )
    if name in HISTORICAL_SHOCK_NAMES or name in EVENT_FLAG_NAMES:
        return (
            "Use only the attached release event and information available by "
            "model_availability_date; absent macro events are structural zero."
        )
    return (
        "Use the latest released or observed value at or before "
        "state_cutoff_date; rolling calculations end at that cutoff."
    )


def _column_catalog(
    *,
    matrix_columns: Sequence[str],
    feature_names: Sequence[str],
    first_sample: str,
    last_sample: str,
    shock_summary: dict[str, Any],
    state_summary: dict[str, Any],
    target_summary: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(
        storage_artifact: str,
        column_name: str,
        column_role: str,
        family: str,
        source: str,
        source_artifact: str,
        definition: str,
        unit: str,
        point_in_time_rule: str,
        *,
        proxy: bool = False,
        revision_risk: str = "NONE",
        included: bool = False,
        notes: str = "",
    ) -> None:
        rows.append(
            {
                "storage_artifact": storage_artifact,
                "column_name": column_name,
                "column_role": column_role,
                "feature_family": family,
                "source": source,
                "source_artifact": source_artifact,
                "definition_or_formula": definition,
                "unit": unit,
                "point_in_time_rule": point_in_time_rule,
                "proxy": bool(proxy),
                "revision_risk": revision_risk,
                "included_in_training_matrix": bool(included),
                "earliest_matrix_sample": first_sample,
                "latest_matrix_sample": last_sample,
                "notes": notes,
            }
        )

    add(
        "scenario_model_matrix_raw.csv",
        "sample_date",
        "row_key",
        "row_identity",
        "Selected Step 4 EVENT_AWARE index",
        relative_path(SAMPLING_PATH),
        "Operational date on which the historical shock row is available.",
        "ISO_date",
        "Observed canonical US equity session.",
        included=True,
    )

    shock_definitions = shock_summary["definitions"]
    state_definitions = state_summary["definitions"]
    for name in feature_names:
        family = _feature_family(name)
        if name in EVENT_FLAG_NAMES:
            event_type = name.removesuffix("_release_flag")
            definition = {
                "source": "Selected normalized Step 4 event attachments",
                "formula": (
                    f"1 when a {event_type} event is attached to the row; "
                    "otherwise 0"
                ),
                "unit": "binary_indicator",
            }
            source_artifact = relative_path(SHOCKS_PATH)
            proxy = False
        elif name in shock_definitions:
            definition = shock_definitions[name]
            source_artifact = relative_path(SHOCKS_PATH)
            proxy = bool(definition.get("proxy", False))
        else:
            definition = state_definitions[name]
            source_artifact = relative_path(STATE_PATH)
            proxy = bool(definition.get("proxy", False))
        add(
            "scenario_model_matrix_raw.csv",
            name,
            "feature",
            family,
            str(definition["source"]),
            source_artifact,
            str(definition["formula"]),
            str(definition["unit"]),
            _point_in_time_rule(name),
            proxy=proxy,
            revision_risk=_revision_risk(name),
            included=True,
            notes=(
                "Historical regional proxy for the pmi_change_points contract."
                if name
                == "kc_fed_manufacturing_pmi_proxy_change_points"
                else ""
            ),
        )

    target_specs = {
        str(record["target"]): record
        for record in target_summary["target_specs"]
    }
    for name in TARGET_NAMES:
        spec = target_specs[name]
        add(
            "scenario_model_matrix_raw.csv",
            name,
            "target",
            "forward_21_session_target",
            str(spec["source"]),
            str(spec["path"]),
            str(spec["formula"]),
            str(spec["unit"]),
            (
                "Label only: exact value at target_baseline_date and exact "
                "value at the stored 21st canonical target_end_date. Never a "
                "feature."
            ),
            revision_risk=(
                "HIGH_CONTINUOUS_FUTURES_STRUCTURE"
                if name == "oil"
                else "LOW_DOCUMENTED_SOURCE_LIMITATION"
            ),
            included=True,
            notes=str(spec["value_policy"]),
        )

    provenance_definitions = {
        "sample_date": "Row join key and operational model-availability session.",
        "sampling_design": "Frozen Step 4 sampling design identifier.",
        "model_availability_date": "First canonical session when the shock row is usable.",
        "state_cutoff_date": "Completed market close strictly before the modeled shock.",
        "shock_window_start_date": "Stored start boundary for market shock construction.",
        "shock_window_end_date": "Stored end boundary for market shock construction.",
        "target_baseline_date": "Canonical pre-shock target baseline session.",
        "target_end_date": "Exact 21st canonical session after the target baseline.",
        "target_horizon_sessions": "Number of future canonical sessions in the label.",
        "event_count": "Number of normalized events attached to the row.",
        "event_types_json": "Ordered JSON list of attached event types.",
        "event_ids_json": "Ordered JSON list of immutable normalized event IDs.",
        "event_observation_periods_json": "JSON list of economic observation periods.",
        "event_release_dates_json": "JSON list of original release dates.",
        "event_release_times_json": "JSON list of release times when cached.",
        "event_timezones_json": "JSON list of release time zones when cached.",
        "event_sources_json": "JSON list of structured event sources.",
        "event_release_date_kinds_json": "JSON list describing official, vintage, or inferred dates.",
        "next_session_shift_applied": "Whether availability was shifted to the next equity session.",
        "conservative_next_session_shift": "Whether unknown timing required a conservative shift.",
        "calendar_normalization_applied": "Whether a non-session date was normalized to a session.",
    }
    for name in PROVENANCE_COLUMNS:
        add(
            "scenario_model_matrix_provenance.csv",
            name,
            "provenance",
            "timing_and_event_provenance",
            "Selected Step 4 EVENT_AWARE index",
            relative_path(SAMPLING_PATH),
            provenance_definitions[name],
            (
                "ISO_date"
                if name.endswith("_date")
                else "metadata"
            ),
            "Retained for audit and splitting; excluded from model features.",
            included=False,
        )

    catalog = pd.DataFrame(rows)
    expected_matrix = set(matrix_columns)
    cataloged_matrix = set(
        catalog.loc[
            catalog["storage_artifact"] == "scenario_model_matrix_raw.csv",
            "column_name",
        ]
    )
    if cataloged_matrix != expected_matrix:
        raise ValueError(
            "Column catalog does not exactly describe the raw model matrix."
        )
    return catalog


def build_model_matrix_artifacts(
    *,
    excluded_features: Sequence[str] = FINAL_AUDIT_EXCLUDED_FEATURES,
    sampling_path: Path = SAMPLING_PATH,
    shocks_path: Path = SHOCKS_PATH,
    state_path: Path = STATE_PATH,
    targets_path: Path = TARGETS_PATH,
    raw_root: Path = RAW_ROOT,
) -> dict[str, Any]:
    """Build all Step 8 artifacts in memory from completed local outputs."""
    excluded = tuple(excluded_features)
    unknown_exclusions = sorted(set(excluded) - set(INITIAL_FEATURE_NAMES))
    if unknown_exclusions:
        raise ValueError(f"Unknown feature exclusions: {unknown_exclusions}")
    unsupported_exclusions = sorted(
        set(excluded) - set(FINAL_AUDIT_EXCLUDED_FEATURES)
    )
    if unsupported_exclusions:
        raise ValueError(
            "Only the documented Step 9 oil-volatility correction is supported: "
            f"{unsupported_exclusions}"
        )

    sampling = _read_csv(sampling_path)
    shocks = _read_csv(shocks_path)
    state = _read_csv(state_path)
    targets = _read_csv(targets_path)
    shock_summary = _read_json(SHOCK_SUMMARY_PATH)
    state_summary = _read_json(STATE_SUMMARY_PATH)
    target_summary = _read_json(TARGET_SUMMARY_PATH)

    expected_rows = 585
    for name, frame in {
        "sampling": sampling,
        "shocks": shocks,
        "state": state,
        "targets": targets,
    }.items():
        if len(frame) != expected_rows:
            raise ValueError(
                f"{name} has {len(frame)} rows; expected {expected_rows}."
            )
        if frame["sample_date"].duplicated().any():
            raise ValueError(f"{name} contains duplicate sample dates.")
        if not frame["sample_date"].is_monotonic_increasing:
            raise ValueError(f"{name} sample dates are not chronological.")

    if not (
        sampling["sample_date"].tolist()
        == shocks["sample_date"].tolist()
        == state["sample_date"].tolist()
        == targets["sample_date"].tolist()
    ):
        raise ValueError("Step 5-7 sample-date indexes do not match exactly.")

    _assert_matching_columns(
        sampling,
        shocks,
        (
            "sample_date",
            "model_availability_date",
            "state_cutoff_date",
            "shock_window_start_date",
            "shock_window_end_date",
            "event_count",
            "event_types_json",
            "event_ids_json",
        ),
        candidate_name="shock features",
    )
    _assert_matching_columns(
        sampling,
        state,
        ("sample_date", "state_cutoff_date"),
        candidate_name="state features",
    )
    _assert_matching_columns(
        sampling,
        targets,
        (
            "sample_date",
            "model_availability_date",
            "state_cutoff_date",
            "target_baseline_date",
            "target_end_date",
            "target_horizon_sessions",
            "event_types_json",
            "event_ids_json",
        ),
        candidate_name="targets",
    )

    combined = sampling.loc[:, ["sample_date"]].copy()
    combined = combined.merge(
        shocks.loc[:, ["sample_date", *SHOCK_FEATURE_NAMES]],
        on="sample_date",
        how="inner",
        validate="one_to_one",
    )
    combined = combined.merge(
        state.loc[:, ["sample_date", *STATE_FEATURE_NAMES]],
        on="sample_date",
        how="inner",
        validate="one_to_one",
    )
    combined = combined.merge(
        targets.loc[:, ["sample_date", *TARGET_NAMES]],
        on="sample_date",
        how="inner",
        validate="one_to_one",
    )

    all_value_columns = [*INITIAL_FEATURE_NAMES, *TARGET_NAMES]
    missing_locations = combined[all_value_columns].isna()
    missing_pairs = [
        (
            str(combined.iloc[row_position]["sample_date"]),
            all_value_columns[column_position],
        )
        for row_position, column_position in zip(
            *missing_locations.to_numpy().nonzero()
        )
    ]
    expected_missing = [("2013-08-16", "cpi_surprise_pct")]
    if missing_pairs != expected_missing:
        raise ValueError(
            "The Step 8 missingness policy expected only the genuine CPI "
            f"archive-boundary gap, but found: {missing_pairs}"
        )

    missing_sample = expected_missing[0][0]
    missing_sampling_row = sampling.loc[
        sampling["sample_date"] == missing_sample
    ].iloc[0]
    row_exclusions = pd.DataFrame(
        [
            {
                "sample_date": missing_sample,
                "column_name": "cpi_surprise_pct",
                "event_types_json": missing_sampling_row["event_types_json"],
                "event_ids_json": missing_sampling_row["event_ids_json"],
                "decision": "EXCLUDE_ROW",
                "reason": (
                    "The initial CPI actual exists, but no compatible Cleveland "
                    "Fed expectation is available strictly before the release. "
                    "Zero-filling would falsely mean no surprise and future-filling "
                    "would leak."
                ),
                "imputation_performed": False,
            }
        ],
        columns=ROW_EXCLUSION_COLUMNS,
    )
    retained = combined[combined["sample_date"] != missing_sample].copy()
    selected_features = tuple(
        name for name in INITIAL_FEATURE_NAMES if name not in excluded
    )

    feature_exclusion_rows: list[dict[str, Any]] = []
    for name in excluded:
        if name == OIL_VOLATILITY_FEATURE:
            reason = (
                "Twenty-day arithmetic percentage-return volatility crosses "
                "the April 20, 2020 negative CL=F close. The resulting values "
                "near 1,196% are not a stable percentage-volatility quantity."
            )
        else:
            reason = "Excluded by the approved Step 9 data-quality audit."
        feature_exclusion_rows.append(
            {
                "feature_name": name,
                "source_artifact": relative_path(state_path),
                "decision": "EXCLUDE_FEATURE_RETAIN_ROWS",
                "reason": reason,
                "rows_removed": 0,
                "transformation_performed": False,
            }
        )
    feature_exclusions = pd.DataFrame(
        feature_exclusion_rows,
        columns=FEATURE_EXCLUSION_COLUMNS,
    )

    matrix_columns = (
        MATRIX_KEY_COLUMNS + selected_features + TARGET_NAMES
    )
    matrix = retained.loc[:, list(matrix_columns)].reset_index(drop=True)

    retained_samples = set(matrix["sample_date"])
    provenance = sampling[
        sampling["sample_date"].isin(retained_samples)
    ].loc[:, list(PROVENANCE_COLUMNS)].reset_index(drop=True)
    source_row_audit = sampling[
        sampling["sample_date"].isin(retained_samples)
    ].loc[:, list(SOURCE_ROW_AUDIT_COLUMNS)].reset_index(drop=True)

    canonical_sessions = load_canonical_calendar(raw_root)
    split_manifest, split_summary = build_split_manifest(
        provenance, canonical_sessions
    )

    feature_values = matrix.loc[:, list(selected_features)]
    target_values = matrix.loc[:, list(TARGET_NAMES)]
    if feature_values.isna().any().any() or target_values.isna().any().any():
        raise ValueError("The final matrix still contains missing model values.")
    if not feature_values.map(math.isfinite).all().all():
        raise ValueError("The final matrix contains non-finite feature values.")
    if not target_values.map(math.isfinite).all().all():
        raise ValueError("The final matrix contains non-finite target values.")

    source_feature_values = retained.loc[
        :, ["sample_date", *selected_features]
    ].reset_index(drop=True)
    source_target_values = retained.loc[
        :, ["sample_date", *TARGET_NAMES]
    ].reset_index(drop=True)
    raw_features_unchanged = matrix[
        ["sample_date", *selected_features]
    ].equals(source_feature_values)
    raw_targets_unchanged = matrix[
        ["sample_date", *TARGET_NAMES]
    ].equals(source_target_values)

    session_positions = {
        session: position for position, session in enumerate(canonical_sessions)
    }
    horizon_distances = []
    for row in provenance.itertuples(index=False):
        baseline = pd.Timestamp(row.target_baseline_date).date()
        endpoint = pd.Timestamp(row.target_end_date).date()
        horizon_distances.append(
            session_positions[endpoint] - session_positions[baseline]
        )

    catalog = _column_catalog(
        matrix_columns=matrix.columns,
        feature_names=selected_features,
        first_sample=str(matrix["sample_date"].iloc[0]),
        last_sample=str(matrix["sample_date"].iloc[-1]),
        shock_summary=shock_summary,
        state_summary=state_summary,
        target_summary=target_summary,
    )

    proxy_features = sorted(
        catalog.loc[
            (catalog["storage_artifact"] == "scenario_model_matrix_raw.csv")
            & (catalog["column_role"] == "feature")
            & catalog["proxy"].astype(bool),
            "column_name",
        ].tolist()
    )
    revision_risk_counts = dict(
        sorted(
            Counter(
                catalog.loc[
                    (catalog["storage_artifact"]
                     == "scenario_model_matrix_raw.csv")
                    & (catalog["column_role"] == "feature"),
                    "revision_risk",
                ]
            ).items()
        )
    )
    feature_family_counts = dict(
        sorted(Counter(_feature_family(name) for name in selected_features).items())
    )

    validations = {
        "all_step_5_to_7_inputs_have_585_rows": True,
        "all_input_sample_indexes_match_one_to_one": True,
        "all_shared_timing_and_event_keys_match": True,
        "selected_sampling_design_is_event_aware": bool(
            (provenance["sampling_design"] == "EVENT_AWARE").all()
        ),
        "only_genuine_missing_cpi_row_removed": bool(
            len(row_exclusions) == 1
            and row_exclusions["sample_date"].iloc[0] == "2013-08-16"
        ),
        "no_imputation_scaling_clipping_or_winsorization": True,
        "model_matrix_values_are_raw_source_values": bool(
            raw_features_unchanged and raw_targets_unchanged
        ),
        "no_feature_values_missing": bool(
            not feature_values.isna().any().any()
        ),
        "no_target_values_missing": bool(
            not target_values.isna().any().any()
        ),
        "all_feature_and_target_values_finite": bool(
            feature_values.map(math.isfinite).all().all()
            and target_values.map(math.isfinite).all().all()
        ),
        "state_cutoff_precedes_sample_date": bool(
            (
                pd.to_datetime(provenance["state_cutoff_date"])
                < pd.to_datetime(provenance["sample_date"])
            ).all()
        ),
        "shock_window_ends_on_or_before_sample_date": bool(
            (
                pd.to_datetime(provenance["shock_window_end_date"])
                <= pd.to_datetime(provenance["sample_date"])
            ).all()
        ),
        "target_baseline_equals_state_cutoff": bool(
            (
                provenance["target_baseline_date"]
                == provenance["state_cutoff_date"]
            ).all()
        ),
        "all_targets_span_exactly_21_canonical_sessions": bool(
            all(
                distance == TARGET_HORIZON_SESSIONS
                for distance in horizon_distances
            )
        ),
        "matrix_contains_only_key_features_and_targets": tuple(
            matrix.columns
        )
        == matrix_columns,
        "provenance_columns_are_not_model_features": not bool(
            (set(PROVENANCE_COLUMNS) - {"sample_date"})
            & set(selected_features)
        ),
        "audit_columns_are_not_model_features": not bool(
            (set(SOURCE_ROW_AUDIT_COLUMNS) - {"sample_date"})
            & set(selected_features)
        ),
        "every_matrix_column_has_catalog_provenance": set(matrix.columns)
        == set(
            catalog.loc[
                catalog["storage_artifact"]
                == "scenario_model_matrix_raw.csv",
                "column_name",
            ]
        ),
        "split_manifest_rows_align_with_matrix": bool(
            split_manifest["sample_date"].tolist()
            == matrix["sample_date"].tolist()
        ),
        "split_purge_checks_pass": all(
            split_summary["validations"].values()
        ),
        "upstream_shock_point_in_time_checks_pass": all(
            shock_summary["validations"].values()
        ),
        "upstream_state_point_in_time_checks_pass": all(
            state_summary["validations"].values()
        ),
        "upstream_target_checks_pass": all(
            target_summary["validations"].values()
        ),
    }
    if not all(validations.values()):
        failed = [name for name, passed in validations.items() if not passed]
        raise ValueError(f"Step 8 model-matrix validation failed: {failed}")

    target_availability = {
        name: {
            "usable_rows": int(matrix[name].notna().sum()),
            "first_sample_date": str(matrix["sample_date"].iloc[0]),
            "last_sample_date": str(matrix["sample_date"].iloc[-1]),
            "source": next(
                spec.source for spec in TARGET_SPECS if spec.target == name
            ),
            "field": next(
                spec.value_field for spec in TARGET_SPECS if spec.target == name
            ),
        }
        for name in TARGET_NAMES
    }
    summary = {
        "schema_version": "scenario_point_in_time_matrix_v1",
        "construction_mode": "offline_completed_artifacts_only",
        "sampling_design": "EVENT_AWARE",
        "initial_input_row_count": expected_rows,
        "row_exclusion_count": int(len(row_exclusions)),
        "final_row_count": int(len(matrix)),
        "first_sample_date": str(matrix["sample_date"].iloc[0]),
        "last_sample_date": str(matrix["sample_date"].iloc[-1]),
        "earliest_common_date": str(matrix["sample_date"].iloc[0]),
        "initial_feature_count": len(INITIAL_FEATURE_NAMES),
        "final_feature_count": len(selected_features),
        "historical_scenario_shock_count": len(HISTORICAL_SHOCK_NAMES),
        "supporting_release_flag_count": len(EVENT_FLAG_NAMES),
        "selected_state_feature_count": sum(
            name in STATE_FEATURE_NAMES for name in selected_features
        ),
        "target_count": len(TARGET_NAMES),
        "matrix_key_column_count": len(MATRIX_KEY_COLUMNS),
        "matrix_column_count": len(matrix.columns),
        "provenance_column_count": len(PROVENANCE_COLUMNS),
        "source_row_audit_column_count": len(SOURCE_ROW_AUDIT_COLUMNS),
        "feature_names": list(selected_features),
        "target_names": list(TARGET_NAMES),
        "matrix_schema": list(matrix.columns),
        "provenance_schema": list(PROVENANCE_COLUMNS),
        "source_row_audit_schema": list(SOURCE_ROW_AUDIT_COLUMNS),
        "feature_family_counts": feature_family_counts,
        "proxy_feature_count": len(proxy_features),
        "proxy_features": proxy_features,
        "revision_risk_feature_counts": revision_risk_counts,
        "missingness": {
            name: int(matrix[name].isna().sum())
            for name in [*selected_features, *TARGET_NAMES]
        },
        "missing_rates": {
            name: float(matrix[name].isna().mean())
            for name in [*selected_features, *TARGET_NAMES]
        },
        "rows_removed": row_exclusions.to_dict("records"),
        "features_removed_by_audit": feature_exclusions.to_dict("records"),
        "target_availability": target_availability,
        "split": split_summary,
        "alignment_decision": (
            "Steps 6 and 7 already performed source-level as-of alignment. "
            "Step 8 therefore uses strict one-to-one sample-row joins and "
            "verifies every shared cutoff, boundary, event ID, and horizon "
            "rather than repeating or weakening those as-of decisions."
        ),
        "raw_matrix_policy": (
            "Raw and unscaled. No learned preprocessing is fit here; later "
            "imputation or scaling must be fit on training data only."
        ),
        "validations": validations,
        "source_manifest": source_manifest(
            [
                sampling_path,
                shocks_path,
                SHOCK_SUMMARY_PATH,
                state_path,
                STATE_SUMMARY_PATH,
                targets_path,
                TARGET_SUMMARY_PATH,
                raw_root / "yahoo" / "GSPC.csv",
            ]
        ),
        "scope_boundary": (
            "Steps 8-9 data assembly only: no preprocessing fit, model "
            "training, predictions, Supabase, backend, frontend, or server."
        ),
    }
    return {
        "matrix": matrix,
        "provenance": provenance,
        "source_row_audit": source_row_audit,
        "column_catalog": catalog,
        "row_exclusions": row_exclusions,
        "feature_exclusions": feature_exclusions,
        "split_manifest": split_manifest,
        "summary": summary,
    }


def persist_model_matrix_artifacts(
    artifacts: dict[str, Any],
    *,
    output_dir: Path = MATRIX_OUTPUT_ROOT,
) -> dict[str, str]:
    """Persist deterministic Step 8 CSV and JSON artifacts."""
    paths = {
        "matrix": output_dir / "scenario_model_matrix_raw.csv",
        "provenance": (
            output_dir / "scenario_model_matrix_provenance.csv"
        ),
        "source_row_audit": (
            output_dir / "scenario_model_matrix_source_row_audit.csv"
        ),
        "column_catalog": (
            output_dir / "scenario_model_matrix_column_catalog.csv"
        ),
        "row_exclusions": (
            output_dir / "scenario_model_matrix_row_exclusions.csv"
        ),
        "feature_exclusions": (
            output_dir / "scenario_model_matrix_feature_exclusions.csv"
        ),
        "split_manifest": (
            output_dir / "scenario_model_matrix_split_manifest.csv"
        ),
        "summary": output_dir / "scenario_model_matrix_summary.json",
    }
    dataframe_names = (
        "matrix",
        "provenance",
        "source_row_audit",
        "column_catalog",
        "row_exclusions",
        "feature_exclusions",
        "split_manifest",
    )
    csv_text = {
        name: frame_to_csv_text(artifacts[name])
        for name in dataframe_names
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
    for name in dataframe_names:
        write_text_atomic(paths[name], csv_text[name])
    write_text_atomic(
        paths["summary"],
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True)
        + "\n",
    )
    return {name: str(path.resolve()) for name, path in paths.items()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble and validate the raw point-in-time Scenario Playground "
            "model matrix from completed local Step 5-7 artifacts."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and validate in memory without writing processed outputs.",
    )
    parser.add_argument(
        "--include-rejected-oil-volatility",
        action="store_true",
        help=(
            "Reconstruct the pre-audit 59-feature matrix for review only. "
            "The persisted/default matrix excludes the rejected CL=F "
            "percentage-volatility feature."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MATRIX_OUTPUT_ROOT,
        help=(
            "Output directory (default: "
            "data/scenario_playground/processed/model_matrix)."
        ),
    )
    args = parser.parse_args(argv)
    excluded = (
        ()
        if args.include_rejected_oil_volatility
        else FINAL_AUDIT_EXCLUDED_FEATURES
    )
    artifacts = build_model_matrix_artifacts(excluded_features=excluded)
    outputs = (
        {}
        if args.dry_run
        else persist_model_matrix_artifacts(
            artifacts, output_dir=args.output_dir
        )
    )
    summary = artifacts["summary"]
    print(
        json.dumps(
            {
                "row_count": summary["final_row_count"],
                "feature_count": summary["final_feature_count"],
                "target_count": summary["target_count"],
                "matrix_column_count": summary["matrix_column_count"],
                "excluded_features": list(excluded),
                "outputs": outputs,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
