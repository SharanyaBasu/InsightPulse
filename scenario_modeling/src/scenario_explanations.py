"""Step 15 Ridge explanations and scenario-sensitivity diagnostics.

The module explains the corrected Step 14 all-target Ridge engine without
changing its models or inference semantics. Contributions are calculated in
the exact fitted model space:

    prediction = ridge_intercept + sum(standardized_feature * coefficient)

All language describes predictive associations rather than causal effects.
The locked final-test target rows remain unavailable through the Step 14
bounded reader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd

from .cache import write_text_atomic
from .config import INTERIM_ROOT
from .feature_sources import frame_to_csv_text, relative_path
from .scenario_inference import (
    ALL_RIDGE_TARGETS,
    CONTRACT_SHOCK_NAMES,
    CONTRACT_TO_HISTORICAL_FEATURE,
    EVENT_FLAG_NAMES,
    EVENT_INPUT_TO_FLAG,
    EXAMPLE_SCENARIO,
    NEUTRAL_SCENARIO,
    ResearchInferenceEngine,
    SHOCK_UNITS,
    build_research_inference_engine,
)


EXPLANATION_OUTPUT_ROOT = INTERIM_ROOT / "explanations"
STEP13_ROOT = INTERIM_ROOT / "models" / "all_targets"
STEP13_SENSITIVITY_SUMMARY_PATH = (
    STEP13_ROOT / "all_targets_sensitivity_summary.csv"
)

EXPLANATION_SCHEMA_VERSION = "scenario_explanation_metadata_v1"
EXPLANATION_SUMMARY_VERSION = "scenario_explanation_summary_v1"
RECONCILIATION_TOLERANCE = 1e-9
TOP_DRIVER_COUNT = 3
SENSITIVITY_QUANTILES = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)

# These deliberately conservative references are plausibility checks, not
# causal rules. Ambiguous target/shock pairs have no reference and cannot be
# labelled counterintuitive by this diagnostic.
DIRECTION_REFERENCES: dict[tuple[str, str], str] = {
    ("sp500", "vix_change_points"): "negative",
    ("nasdaq", "vix_change_points"): "negative",
    ("technology", "vix_change_points"): "negative",
    ("energy", "vix_change_points"): "negative",
    ("financials", "vix_change_points"): "negative",
    ("utilities", "vix_change_points"): "negative",
    ("healthcare", "vix_change_points"): "negative",
    ("consumer_discretionary", "vix_change_points"): "negative",
    ("oil", "oil_price_change_pct"): "positive",
    ("energy", "oil_price_change_pct"): "positive",
    ("dxy", "dxy_change_pct"): "positive",
    ("gold", "dxy_change_pct"): "negative",
    ("ten_year_yield_bps", "fed_funds_rate_change_bps"): "positive",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    return value


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _target_unit(target: str) -> str:
    return "basis_points" if target == "ten_year_yield_bps" else "percent"


def _meaningful_response_threshold(target: str) -> float:
    return 1.0 if target == "ten_year_yield_bps" else 0.05


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() == "true"


def _split_pipe_values(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    return [
        item.strip()
        for item in str(value).split("|")
        if item.strip()
    ]


def _feature_role_lookup(engine: ResearchInferenceEngine) -> pd.DataFrame:
    mapping = engine.feature_mapping.copy()
    required = {
        "feature_position",
        "feature_name",
        "inference_role",
        "contract_input",
        "unit",
    }
    missing = sorted(required - set(mapping.columns))
    if missing:
        raise ValueError(f"Feature mapping is missing columns: {missing}.")
    mapping = mapping.sort_values("feature_position", kind="stable")
    if tuple(mapping["feature_name"]) != engine.feature_names:
        raise ValueError("Explanation feature mapping order changed.")
    return mapping.set_index("feature_name", drop=False)


def _resolve_explanation_rows(
    engine: ResearchInferenceEngine,
    scenario_shocks: Mapping[str, Any],
    *,
    current_state: Mapping[str, Any] | None,
    release_flags: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    result = engine.predict(
        scenario_shocks,
        current_state=current_state,
        release_flags=release_flags,
    )
    baseline, scenario, isolation = engine._assemble_pair(
        result["scenario_shocks"],
        result["context"]["state_features"],
        result["release_flags"],
    )
    if isolation != result["feature_assembly"]:
        raise ValueError("Explanation rows do not match Step 14 inference rows.")
    return result, baseline, scenario


def _transformed_feature_values(
    engine: ResearchInferenceEngine,
    target: str,
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    pipeline = engine.pipelines[target]
    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]
    ridge = pipeline.named_steps["ridge"]
    imputed = np.asarray(imputer.transform(frame), dtype=float)
    standardized = np.asarray(scaler.transform(imputed), dtype=float)
    coefficients = np.asarray(ridge.coef_, dtype=float).reshape(-1)
    if imputed.shape[1] != len(engine.feature_names):
        raise ValueError(f"Imputer changed the feature count for {target}.")
    if standardized.shape[1] != len(engine.feature_names):
        raise ValueError(f"Scaler changed the feature count for {target}.")
    if coefficients.shape != (len(engine.feature_names),):
        raise ValueError(f"Ridge coefficient shape changed for {target}.")
    contributions = standardized * coefficients
    return (
        imputed,
        standardized,
        contributions,
        coefficients,
        float(ridge.intercept_),
    )


def _training_contribution_thresholds(
    engine: ResearchInferenceEngine,
    target: str,
) -> np.ndarray:
    training = engine.training.loc[:, list(engine.feature_names)]
    _, _, contributions, _, _ = _transformed_feature_values(
        engine,
        target,
        training,
    )
    return np.quantile(np.abs(contributions), 0.99, axis=0)


def _contribution_artifacts(
    engine: ResearchInferenceEngine,
    baseline: pd.DataFrame,
    scenario: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mapping = _feature_role_lookup(engine)
    contribution_rows: list[dict[str, Any]] = []
    reconciliation_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []

    for target in engine.target_names:
        threshold = _training_contribution_thresholds(engine, target)
        row_contributions: dict[str, pd.DataFrame] = {}
        row_predictions: dict[str, float] = {}
        row_intercepts: dict[str, float] = {}
        for row_type, frame in (
            ("baseline", baseline),
            ("scenario", scenario),
        ):
            (
                imputed,
                standardized,
                contributions,
                coefficients,
                intercept,
            ) = _transformed_feature_values(engine, target, frame)
            model_prediction = float(
                engine.pipelines[target].predict(frame)[0]
            )
            records: list[dict[str, Any]] = []
            for position, feature in enumerate(engine.feature_names):
                metadata = mapping.loc[feature]
                raw_value = float(frame.iloc[0, position])
                standardized_value = float(standardized[0, position])
                contribution = float(contributions[0, position])
                p99 = float(threshold[position])
                record = {
                    "target": target,
                    "unit": _target_unit(target),
                    "row_type": row_type,
                    "feature_position": position,
                    "feature_name": feature,
                    "inference_role": str(metadata["inference_role"]),
                    "contract_input": (
                        ""
                        if pd.isna(metadata["contract_input"])
                        else str(metadata["contract_input"])
                    ),
                    "feature_unit": str(metadata["unit"]),
                    "raw_value": raw_value,
                    "imputed_value": float(imputed[0, position]),
                    "scaler_mean": float(
                        engine.pipelines[target]
                        .named_steps["scaler"]
                        .mean_[position]
                    ),
                    "scaler_scale": float(
                        engine.pipelines[target]
                        .named_steps["scaler"]
                        .scale_[position]
                    ),
                    "standardized_value": standardized_value,
                    "ridge_coefficient": float(coefficients[position]),
                    "feature_contribution": contribution,
                    "absolute_contribution": abs(contribution),
                    "training_absolute_contribution_p99": p99,
                    "unusually_large_contribution": bool(
                        abs(contribution) > max(p99, 1e-12)
                        + RECONCILIATION_TOLERANCE
                    ),
                    "raw_zero_but_standardized_nonzero": bool(
                        abs(raw_value) <= 1e-12
                        and abs(standardized_value) > 1e-12
                    ),
                    "causal_effect_claimed": False,
                }
                records.append(record)
                contribution_rows.append(record)
            row_frame = pd.DataFrame(records)
            row_contributions[row_type] = row_frame
            row_predictions[row_type] = model_prediction
            row_intercepts[row_type] = intercept
            role_sums = (
                row_frame.groupby("inference_role")[
                    "feature_contribution"
                ]
                .sum()
                .to_dict()
            )
            contribution_sum = float(
                row_frame["feature_contribution"].sum()
            )
            reconciled = intercept + contribution_sum
            error = reconciled - model_prediction
            reconciliation_rows.append(
                {
                    "target": target,
                    "unit": _target_unit(target),
                    "row_type": row_type,
                    "model_intercept": intercept,
                    "current_state_contribution": float(
                        role_sums.get("current_state", 0.0)
                    ),
                    "scenario_shock_contribution": float(
                        role_sums.get("scenario_shock", 0.0)
                    ),
                    "release_flag_contribution": float(
                        role_sums.get("release_flag", 0.0)
                    ),
                    "feature_contribution_sum": contribution_sum,
                    "reconciled_prediction": reconciled,
                    "model_prediction": model_prediction,
                    "reconciliation_error": error,
                    "absolute_reconciliation_error": abs(error),
                    "tolerance": RECONCILIATION_TOLERANCE,
                    "reconciliation_passed": (
                        abs(error) <= RECONCILIATION_TOLERANCE
                    ),
                }
            )

        baseline_rows = row_contributions["baseline"].set_index(
            "feature_name"
        )
        scenario_rows = row_contributions["scenario"].set_index(
            "feature_name"
        )
        for position, feature in enumerate(engine.feature_names):
            baseline_row = baseline_rows.loc[feature]
            scenario_row = scenario_rows.loc[feature]
            delta_rows.append(
                {
                    "target": target,
                    "unit": _target_unit(target),
                    "feature_position": position,
                    "feature_name": feature,
                    "inference_role": baseline_row["inference_role"],
                    "contract_input": baseline_row["contract_input"],
                    "feature_unit": baseline_row["feature_unit"],
                    "baseline_raw_value": baseline_row["raw_value"],
                    "scenario_raw_value": scenario_row["raw_value"],
                    "raw_value_change": (
                        scenario_row["raw_value"]
                        - baseline_row["raw_value"]
                    ),
                    "baseline_standardized_value": baseline_row[
                        "standardized_value"
                    ],
                    "scenario_standardized_value": scenario_row[
                        "standardized_value"
                    ],
                    "standardized_value_change": (
                        scenario_row["standardized_value"]
                        - baseline_row["standardized_value"]
                    ),
                    "ridge_coefficient": baseline_row[
                        "ridge_coefficient"
                    ],
                    "baseline_contribution": baseline_row[
                        "feature_contribution"
                    ],
                    "scenario_contribution": scenario_row[
                        "feature_contribution"
                    ],
                    "contribution_change": (
                        scenario_row["feature_contribution"]
                        - baseline_row["feature_contribution"]
                    ),
                    "causal_effect_claimed": False,
                }
            )

        target_deltas = pd.DataFrame(
            delta_rows[-len(engine.feature_names) :]
        )
        delta_role_sums = (
            target_deltas.groupby("inference_role")[
                "contribution_change"
            ]
            .sum()
            .to_dict()
        )
        model_incremental = (
            row_predictions["scenario"] - row_predictions["baseline"]
        )
        contribution_incremental = float(
            target_deltas["contribution_change"].sum()
        )
        incremental_error = contribution_incremental - model_incremental
        reconciliation_rows.append(
            {
                "target": target,
                "unit": _target_unit(target),
                "row_type": "incremental_difference",
                "model_intercept": (
                    row_intercepts["scenario"]
                    - row_intercepts["baseline"]
                ),
                "current_state_contribution": float(
                    delta_role_sums.get("current_state", 0.0)
                ),
                "scenario_shock_contribution": float(
                    delta_role_sums.get("scenario_shock", 0.0)
                ),
                "release_flag_contribution": float(
                    delta_role_sums.get("release_flag", 0.0)
                ),
                "feature_contribution_sum": contribution_incremental,
                "reconciled_prediction": contribution_incremental,
                "model_prediction": model_incremental,
                "reconciliation_error": incremental_error,
                "absolute_reconciliation_error": abs(incremental_error),
                "tolerance": RECONCILIATION_TOLERANCE,
                "reconciliation_passed": (
                    abs(incremental_error)
                    <= RECONCILIATION_TOLERANCE
                ),
            }
        )

    return (
        pd.DataFrame(contribution_rows),
        pd.DataFrame(delta_rows),
        pd.DataFrame(reconciliation_rows),
    )


def _driver_text(
    *,
    feature: str,
    role: str,
    value: float,
    unit: str,
    contribution: float,
    target_unit: str,
    driver_scope: str,
) -> str:
    return (
        f"{feature} ({role}, raw {value:.6g} {unit}) has a "
        f"{contribution:+.6g} {target_unit} standardized Ridge contribution "
        f"in the {driver_scope}; this is a predictive association, not a "
        "causal effect."
    )


def _top_driver_artifacts(
    contributions: pd.DataFrame,
    deltas: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    target_payload: dict[str, Any] = {}
    for target in ALL_RIDGE_TARGETS:
        target_payload[target] = {}
        for row_type in ("baseline", "scenario"):
            subset = contributions.loc[
                contributions["target"].eq(target)
                & contributions["row_type"].eq(row_type)
            ].copy()
            driver_groups = {
                f"{row_type}_positive": subset.loc[
                    subset["feature_contribution"] > 0
                ].sort_values(
                    ["feature_contribution", "feature_position"],
                    ascending=[False, True],
                    kind="stable",
                ),
                f"{row_type}_negative": subset.loc[
                    subset["feature_contribution"] < 0
                ].sort_values(
                    ["feature_contribution", "feature_position"],
                    ascending=[True, True],
                    kind="stable",
                ),
            }
            for driver_kind, group in driver_groups.items():
                payload_rows: list[dict[str, Any]] = []
                for rank, (_, item) in enumerate(
                    group.head(TOP_DRIVER_COUNT).iterrows(),
                    1,
                ):
                    record = {
                        "target": target,
                        "unit": _target_unit(target),
                        "driver_kind": driver_kind,
                        "rank": rank,
                        "feature_name": item["feature_name"],
                        "inference_role": item["inference_role"],
                        "raw_value": item["raw_value"],
                        "feature_unit": item["feature_unit"],
                        "standardized_value": item["standardized_value"],
                        "contribution": item["feature_contribution"],
                        "explanation": _driver_text(
                            feature=item["feature_name"],
                            role=item["inference_role"],
                            value=float(item["raw_value"]),
                            unit=item["feature_unit"],
                            contribution=float(
                                item["feature_contribution"]
                            ),
                            target_unit=_target_unit(target),
                            driver_scope=f"{row_type} row",
                        ),
                        "causal_effect_claimed": False,
                    }
                    rows.append(record)
                    payload_rows.append(record)
                target_payload[target][driver_kind] = payload_rows

        target_delta = deltas.loc[deltas["target"].eq(target)].copy()
        delta_groups = {
            "incremental_positive": target_delta.loc[
                target_delta["contribution_change"] > 0
            ].sort_values(
                ["contribution_change", "feature_position"],
                ascending=[False, True],
                kind="stable",
            ),
            "incremental_negative": target_delta.loc[
                target_delta["contribution_change"] < 0
            ].sort_values(
                ["contribution_change", "feature_position"],
                ascending=[True, True],
                kind="stable",
            ),
        }
        for driver_kind, group in delta_groups.items():
            payload_rows = []
            for rank, (_, item) in enumerate(
                group.head(TOP_DRIVER_COUNT).iterrows(),
                1,
            ):
                record = {
                    "target": target,
                    "unit": _target_unit(target),
                    "driver_kind": driver_kind,
                    "rank": rank,
                    "feature_name": item["feature_name"],
                    "inference_role": item["inference_role"],
                    "raw_value": item["raw_value_change"],
                    "feature_unit": item["feature_unit"],
                    "standardized_value": item[
                        "standardized_value_change"
                    ],
                    "contribution": item["contribution_change"],
                    "explanation": _driver_text(
                        feature=item["feature_name"],
                        role=item["inference_role"],
                        value=float(item["raw_value_change"]),
                        unit=item["feature_unit"],
                        contribution=float(item["contribution_change"]),
                        target_unit=_target_unit(target),
                        driver_scope="scenario-minus-baseline difference",
                    ),
                    "causal_effect_claimed": False,
                }
                rows.append(record)
                payload_rows.append(record)
            target_payload[target][driver_kind] = payload_rows
    return pd.DataFrame(rows), target_payload


def _representative_states(
    engine: ResearchInferenceEngine,
) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    vix = engine.training["vix_level_points"].astype(float)
    for name, quantile in (
        ("low_vix", 0.10),
        ("median_vix", 0.50),
        ("high_vix", 0.90),
    ):
        target_value = float(vix.quantile(quantile))
        index = (vix - target_value).abs().idxmin()
        row = engine.training.loc[index]
        states.append(
            {
                "representative_state": name,
                "quantile": quantile,
                "sample_date": str(row["sample_date"]),
                "vix_level_points": float(row["vix_level_points"]),
                "state": {
                    feature: float(row[feature])
                    for feature in engine.state_feature_names
                },
            }
        )
    return states


def _fixed_sensitivity_flags(contract_input: str) -> dict[str, float]:
    flags = {name: 0.0 for name in EVENT_FLAG_NAMES}
    flag = EVENT_INPUT_TO_FLAG.get(contract_input)
    if flag is not None:
        flags[flag] = 1.0
    return flags


def _sensitivity_grid(values: pd.Series) -> np.ndarray:
    numeric = values.to_numpy(dtype=float)
    quantiles = np.quantile(numeric, SENSITIVITY_QUANTILES)
    evenly_spaced = np.linspace(
        float(numeric.min()),
        float(numeric.max()),
        7,
    )
    return np.unique(
        np.concatenate([quantiles, evenly_spaced, np.array([0.0])])
    ).astype(float)


def _slope_diagnostics(
    values: np.ndarray,
    predictions: np.ndarray,
) -> tuple[int, float, bool]:
    value_changes = np.diff(values)
    prediction_changes = np.diff(predictions)
    slopes = prediction_changes / value_changes
    active = slopes[np.abs(slopes) > 1e-12]
    signs = np.sign(active)
    direction_changes = (
        int(np.sum(signs[1:] != signs[:-1]))
        if len(signs) > 1
        else 0
    )
    maximum_slope_change = (
        float(np.max(np.abs(np.diff(slopes))))
        if len(slopes) > 1
        else 0.0
    )
    slope_scale = max(float(np.max(np.abs(slopes))), 1.0)
    discontinuity = maximum_slope_change > 1e-9 * slope_scale
    return direction_changes, maximum_slope_change, discontinuity


def _build_sensitivity(
    engine: ResearchInferenceEngine,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_summary = pd.read_csv(STEP13_SENSITIVITY_SUMMARY_PATH)
    fold_lookup = fold_summary.set_index(["target", "contract_input"])
    curve_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for state_metadata in _representative_states(engine):
        state = state_metadata["state"]
        for contract_input in CONTRACT_SHOCK_NAMES:
            feature = CONTRACT_TO_HISTORICAL_FEATURE[contract_input]
            grid = _sensitivity_grid(engine.training[feature])
            flags = _fixed_sensitivity_flags(contract_input)
            baseline_shocks = dict(NEUTRAL_SCENARIO)
            baseline, _, isolation = engine._assemble_pair(
                baseline_shocks,
                state,
                flags,
            )
            if not (
                isolation["state_identical"]
                and isolation["release_flags_identical"]
                and isolation["feature_order_exact"]
            ):
                raise ValueError("Sensitivity baseline isolation failed.")
            scenario_frames: list[pd.DataFrame] = []
            for value in grid:
                shocks = dict(NEUTRAL_SCENARIO)
                shocks[contract_input] = float(value)
                _, scenario, pair_isolation = engine._assemble_pair(
                    shocks,
                    state,
                    flags,
                )
                if not (
                    pair_isolation["state_identical"]
                    and pair_isolation["release_flags_identical"]
                    and pair_isolation["feature_order_exact"]
                ):
                    raise ValueError("Sensitivity scenario isolation failed.")
                scenario_frames.append(scenario)
            scenarios = pd.concat(scenario_frames, ignore_index=True)

            for target in engine.target_names:
                pipeline = engine.pipelines[target]
                baseline_prediction = float(pipeline.predict(baseline)[0])
                predictions = np.asarray(
                    pipeline.predict(
                        scenarios.loc[:, list(engine.feature_names)]
                    ),
                    dtype=float,
                )
                incremental = predictions - baseline_prediction
                differences = np.diff(predictions)
                (
                    direction_changes,
                    maximum_slope_change,
                    discontinuity,
                ) = _slope_diagnostics(grid, predictions)
                end_to_end = float(predictions[-1] - predictions[0])
                span = float(predictions.max() - predictions.min())
                direction = (
                    "flat"
                    if abs(end_to_end) <= 1e-12
                    else ("positive" if end_to_end > 0 else "negative")
                )
                threshold = _meaningful_response_threshold(target)
                reference = DIRECTION_REFERENCES.get(
                    (target, contract_input)
                )
                fold_row = fold_lookup.loc[(target, contract_input)]
                fold_directions = _split_pipe_values(
                    fold_row["meaningful_directions_observed"]
                )
                fold_consistent = _as_bool(
                    fold_row["direction_consistent_when_meaningful"]
                )
                counterintuitive = bool(
                    reference is not None
                    and direction != "flat"
                    and direction != reference
                )
                unstable = bool(
                    int(fold_row["folds_with_meaningful_response"]) > 0
                    and not fold_consistent
                )
                observed_target_range = float(
                    engine.training[target].max()
                    - engine.training[target].min()
                )
                maximum_sensible_span = max(
                    50.0 if target == "ten_year_yield_bps" else 10.0,
                    observed_target_range,
                )
                for grid_index, value in enumerate(grid):
                    curve_rows.append(
                        {
                            "target": target,
                            "unit": _target_unit(target),
                            "representative_state": state_metadata[
                                "representative_state"
                            ],
                            "state_sample_date": state_metadata[
                                "sample_date"
                            ],
                            "state_vix_level_points": state_metadata[
                                "vix_level_points"
                            ],
                            "contract_input": contract_input,
                            "model_feature": feature,
                            "input_unit": SHOCK_UNITS[contract_input],
                            "grid_index": grid_index,
                            "input_value": float(value),
                            "baseline_prediction": baseline_prediction,
                            "scenario_prediction": float(
                                predictions[grid_index]
                            ),
                            "incremental_effect": float(
                                incremental[grid_index]
                            ),
                            "release_flag_policy": (
                                "RELEVANT_EVENT_FLAG_FIXED_AT_ONE"
                                if contract_input in EVENT_INPUT_TO_FLAG
                                else "ALL_EVENT_FLAGS_FIXED_AT_ZERO"
                            ),
                            "state_fixed": True,
                            "release_flags_fixed": True,
                            "causal_effect_claimed": False,
                        }
                    )
                summary_rows.append(
                    {
                        "target": target,
                        "unit": _target_unit(target),
                        "representative_state": state_metadata[
                            "representative_state"
                        ],
                        "state_sample_date": state_metadata["sample_date"],
                        "state_vix_level_points": state_metadata[
                            "vix_level_points"
                        ],
                        "contract_input": contract_input,
                        "model_feature": feature,
                        "input_unit": SHOCK_UNITS[contract_input],
                        "grid_minimum": float(grid.min()),
                        "grid_maximum": float(grid.max()),
                        "grid_points": len(grid),
                        "prediction_span": span,
                        "end_to_end_effect": end_to_end,
                        "response_direction": direction,
                        "maximum_adjacent_jump": (
                            float(np.max(np.abs(differences)))
                            if len(differences)
                            else 0.0
                        ),
                        "direction_change_count": direction_changes,
                        "maximum_slope_change": maximum_slope_change,
                        "monotonic": direction_changes == 0,
                        "discontinuity_detected": discontinuity,
                        "meaningful_response_threshold": threshold,
                        "negligible_response": span < threshold,
                        "finite_predictions": bool(
                            np.isfinite(predictions).all()
                        ),
                        "maximum_sensible_span": maximum_sensible_span,
                        "unusually_large_response": (
                            span > maximum_sensible_span
                        ),
                        "direction_reference": reference or "not_defined",
                        "direction_reference_basis": (
                            "CONSERVATIVE_PLAUSIBILITY_HEURISTIC_NOT_CAUSAL"
                            if reference is not None
                            else "AMBIGUOUS_NO_REFERENCE_APPLIED"
                        ),
                        "counterintuitive_direction_flag": counterintuitive,
                        "step13_meaningful_directions": "|".join(
                            fold_directions
                        ),
                        "step13_direction_consistent_when_meaningful": (
                            fold_consistent
                        ),
                        "step13_folds_with_meaningful_response": int(
                            fold_row["folds_with_meaningful_response"]
                        ),
                        "unstable_across_walk_forward_folds": unstable,
                        "state_fixed": True,
                        "release_flags_fixed": True,
                        "causal_effect_claimed": False,
                    }
                )

    curves = pd.DataFrame(curve_rows)
    state_summary = pd.DataFrame(summary_rows)
    aggregate_rows: list[dict[str, Any]] = []
    for (target, contract_input), group in state_summary.groupby(
        ["target", "contract_input"],
        sort=True,
    ):
        directions = sorted(set(group["response_direction"]))
        aggregate_rows.append(
            {
                "target": target,
                "unit": _target_unit(target),
                "contract_input": contract_input,
                "model_feature": group["model_feature"].iloc[0],
                "input_unit": group["input_unit"].iloc[0],
                "representative_state_count": len(group),
                "response_directions": "|".join(directions),
                "all_states_same_direction": len(directions) == 1,
                "minimum_prediction_span": float(
                    group["prediction_span"].min()
                ),
                "median_prediction_span": float(
                    group["prediction_span"].median()
                ),
                "maximum_prediction_span": float(
                    group["prediction_span"].max()
                ),
                "negligible_in_all_states": bool(
                    group["negligible_response"].all()
                ),
                "all_monotonic": bool(group["monotonic"].all()),
                "any_discontinuity": bool(
                    group["discontinuity_detected"].any()
                ),
                "all_finite": bool(group["finite_predictions"].all()),
                "any_unusually_large_response": bool(
                    group["unusually_large_response"].any()
                ),
                "direction_reference": group[
                    "direction_reference"
                ].iloc[0],
                "counterintuitive_direction_flag": bool(
                    group["counterintuitive_direction_flag"].any()
                ),
                "unstable_across_walk_forward_folds": bool(
                    group[
                        "unstable_across_walk_forward_folds"
                    ].any()
                ),
                "step13_folds_with_meaningful_response": int(
                    group[
                        "step13_folds_with_meaningful_response"
                    ].iloc[0]
                ),
                "step13_direction_consistent_when_meaningful": bool(
                    group[
                        "step13_direction_consistent_when_meaningful"
                    ].iloc[0]
                ),
                "causal_effect_claimed": False,
            }
        )
    return curves, state_summary, pd.DataFrame(aggregate_rows)


def _target_explanation_payload(
    engine: ResearchInferenceEngine,
    inference_result: Mapping[str, Any],
    reconciliation: pd.DataFrame,
    top_driver_payload: Mapping[str, Any],
    sensitivity_summary: pd.DataFrame,
    contributions: pd.DataFrame,
) -> dict[str, Any]:
    targets: dict[str, Any] = {}
    for target in engine.target_names:
        output = inference_result["targets"][target]
        target_reconciliation = reconciliation.loc[
            reconciliation["target"].eq(target)
        ].set_index("row_type")
        target_sensitivity = sensitivity_summary.loc[
            sensitivity_summary["target"].eq(target)
        ]
        negligible = sorted(
            target_sensitivity.loc[
                target_sensitivity["negligible_in_all_states"],
                "contract_input",
            ].tolist()
        )
        unstable = sorted(
            target_sensitivity.loc[
                target_sensitivity[
                    "unstable_across_walk_forward_folds"
                ],
                "contract_input",
            ].tolist()
        )
        counterintuitive = sorted(
            target_sensitivity.loc[
                target_sensitivity[
                    "counterintuitive_direction_flag"
                ],
                "contract_input",
            ].tolist()
        )
        large_features = sorted(
            contributions.loc[
                contributions["target"].eq(target)
                & contributions["unusually_large_contribution"],
                "feature_name",
            ].unique()
        )
        baseline_row = target_reconciliation.loc["baseline"]
        scenario_row = target_reconciliation.loc["scenario"]
        incremental_row = target_reconciliation.loc[
            "incremental_difference"
        ]
        targets[target] = {
            "target": target,
            "unit": output["unit"],
            "model_family": "ridge",
            "selected_alpha": output["selected_alpha"],
            "confidence_level": output["confidence_level"],
            "step13_evaluation_classification": output[
                "step13_evaluation_classification"
            ],
            "baseline_prediction": output["baseline_prediction"],
            "scenario_prediction": output["scenario_prediction"],
            "incremental_scenario_effect": output[
                "incremental_scenario_effect"
            ],
            "contribution_groups": {
                "baseline": {
                    "model_intercept": baseline_row["model_intercept"],
                    "current_state_contribution": baseline_row[
                        "current_state_contribution"
                    ],
                    "scenario_shock_contribution": baseline_row[
                        "scenario_shock_contribution"
                    ],
                    "release_flag_contribution": baseline_row[
                        "release_flag_contribution"
                    ],
                },
                "scenario": {
                    "model_intercept": scenario_row["model_intercept"],
                    "current_state_contribution": scenario_row[
                        "current_state_contribution"
                    ],
                    "scenario_shock_contribution": scenario_row[
                        "scenario_shock_contribution"
                    ],
                    "release_flag_contribution": scenario_row[
                        "release_flag_contribution"
                    ],
                },
                "incremental": {
                    "model_intercept_change": incremental_row[
                        "model_intercept"
                    ],
                    "current_state_contribution_change": incremental_row[
                        "current_state_contribution"
                    ],
                    "scenario_shock_contribution_change": incremental_row[
                        "scenario_shock_contribution"
                    ],
                    "release_flag_contribution_change": incremental_row[
                        "release_flag_contribution"
                    ],
                },
            },
            "top_drivers": top_driver_payload[target],
            "diagnostics": {
                "negligible_scenario_inputs": negligible,
                "unstable_walk_forward_directions": unstable,
                "counterintuitive_reference_flags": counterintuitive,
                "unusually_large_contribution_features": large_features,
                "raw_and_standardized_values_reported_separately": True,
            },
            "warning_codes": output["applicable_warnings"],
            "known_limitations": output["known_limitations"],
            "explanation_warning": (
                "Contributions decompose a standardized Ridge predictive "
                "association. They do not estimate causal effects."
            ),
            "causal_effect_claimed": False,
        }
    return {
        "schema_version": EXPLANATION_SCHEMA_VERSION,
        "horizon_trading_days": 21,
        "scenario_shocks": inference_result["scenario_shocks"],
        "release_flags": inference_result["release_flags"],
        "context": inference_result["context"],
        "targets": targets,
        "global_warning": (
            "Research-only Ridge contribution explanations. Coefficients, "
            "contributions, sensitivity slopes, and direction references are "
            "predictive diagnostics rather than causal financial effects."
        ),
    }


def explanation_metadata_schema() -> dict[str, Any]:
    return {
        "schema_version": EXPLANATION_SCHEMA_VERSION,
        "target_payload_required_fields": [
            "target",
            "unit",
            "model_family",
            "selected_alpha",
            "confidence_level",
            "baseline_prediction",
            "scenario_prediction",
            "incremental_scenario_effect",
            "contribution_groups",
            "top_drivers",
            "diagnostics",
            "warning_codes",
            "known_limitations",
            "explanation_warning",
            "causal_effect_claimed",
        ],
        "contribution_row_required_fields": [
            "target",
            "row_type",
            "feature_position",
            "feature_name",
            "inference_role",
            "raw_value",
            "imputed_value",
            "scaler_mean",
            "scaler_scale",
            "standardized_value",
            "ridge_coefficient",
            "feature_contribution",
        ],
        "sensitivity_required_dimensions": {
            "targets": 12,
            "scenario_inputs": 8,
            "representative_states": 3,
        },
        "reconciliation_tolerance": RECONCILIATION_TOLERANCE,
        "confidence_boundary": (
            "Step 15 preserves Step 13 confidence evidence and does not "
            "promote target quality."
        ),
        "causal_effect_claimed": False,
    }


def _validate_explanation_schema(payload: Mapping[str, Any]) -> bool:
    schema = explanation_metadata_schema()
    if payload.get("schema_version") != schema["schema_version"]:
        return False
    if tuple(payload.get("targets", {})) != ALL_RIDGE_TARGETS:
        return False
    required = set(schema["target_payload_required_fields"])
    return all(
        required <= set(target_payload)
        for target_payload in payload["targets"].values()
    )


def build_explanation_artifacts(
    *,
    scenario_shocks: Mapping[str, Any] = EXAMPLE_SCENARIO,
    current_state: Mapping[str, Any] | None = None,
    release_flags: Mapping[str, Any] | None = None,
    engine: ResearchInferenceEngine | None = None,
) -> dict[str, Any]:
    """Build all Step 15 explanation evidence without exporting models."""
    active_engine = engine or build_research_inference_engine()
    inference_result, baseline, scenario = _resolve_explanation_rows(
        active_engine,
        scenario_shocks,
        current_state=current_state,
        release_flags=release_flags,
    )
    (
        contributions,
        contribution_deltas,
        reconciliation,
    ) = _contribution_artifacts(active_engine, baseline, scenario)
    top_drivers, top_driver_payload = _top_driver_artifacts(
        contributions,
        contribution_deltas,
    )
    (
        sensitivity_curves,
        sensitivity_state_summary,
        sensitivity_summary,
    ) = _build_sensitivity(active_engine)
    target_explanations = _target_explanation_payload(
        active_engine,
        inference_result,
        reconciliation,
        top_driver_payload,
        sensitivity_summary,
        contributions,
    )
    metadata_schema = explanation_metadata_schema()
    neutral = active_engine.predict(NEUTRAL_SCENARIO)
    repeated = active_engine.predict(
        inference_result["scenario_shocks"],
        current_state=inference_result["context"]["state_features"],
        release_flags=inference_result["release_flags"],
    )

    numeric_columns = [
        "raw_value",
        "imputed_value",
        "standardized_value",
        "ridge_coefficient",
        "feature_contribution",
    ]
    expected_sensitivity_pairs = (
        len(active_engine.target_names) * len(CONTRACT_SHOCK_NAMES)
    )
    validations = {
        "exactly_twelve_targets_explained": (
            tuple(target_explanations["targets"]) == ALL_RIDGE_TARGETS
        ),
        "exact_58_feature_order_preserved": (
            active_engine.feature_names
            == tuple(active_engine.feature_mapping["feature_name"])
        ),
        "baseline_and_scenario_feature_contributions_complete": (
            len(contributions)
            == len(active_engine.target_names)
            * 2
            * len(active_engine.feature_names)
        ),
        "contribution_differences_complete": (
            len(contribution_deltas)
            == len(active_engine.target_names)
            * len(active_engine.feature_names)
        ),
        "all_contribution_values_finite": bool(
            np.isfinite(
                contributions[numeric_columns].to_numpy(dtype=float)
            ).all()
        ),
        "all_prediction_reconciliations_pass": bool(
            reconciliation["reconciliation_passed"].all()
        ),
        "maximum_reconciliation_error_within_tolerance": (
            float(
                reconciliation["absolute_reconciliation_error"].max()
            )
            <= RECONCILIATION_TOLERANCE
        ),
        "incremental_change_is_scenario_shock_only": bool(
            reconciliation.loc[
                reconciliation["row_type"].eq(
                    "incremental_difference"
                ),
                [
                    "current_state_contribution",
                    "release_flag_contribution",
                    "model_intercept",
                ],
            ]
            .abs()
            .le(RECONCILIATION_TOLERANCE)
            .all()
            .all()
        ),
        "all_eight_inputs_sensitivity_tested_for_all_targets": (
            len(sensitivity_summary) == expected_sensitivity_pairs
            and set(sensitivity_summary["contract_input"])
            == set(CONTRACT_SHOCK_NAMES)
        ),
        "three_representative_states_tested": (
            sensitivity_state_summary[
                "representative_state"
            ].nunique()
            == 3
            and len(sensitivity_state_summary)
            == expected_sensitivity_pairs * 3
        ),
        "all_sensitivity_predictions_finite": bool(
            sensitivity_summary["all_finite"].all()
        ),
        "all_ridge_sensitivity_curves_monotonic": bool(
            sensitivity_summary["all_monotonic"].all()
        ),
        "no_ridge_sensitivity_discontinuities": bool(
            ~sensitivity_summary["any_discontinuity"].any()
        ),
        "state_and_release_flags_fixed_in_sensitivity": bool(
            sensitivity_curves["state_fixed"].all()
            and sensitivity_curves["release_flags_fixed"].all()
        ),
        "neutral_incremental_predictions_zero": all(
            neutral["targets"][target][
                "incremental_scenario_effect"
            ]
            == 0.0
            for target in active_engine.target_names
        ),
        "repeated_explanatory_inference_deterministic": all(
            repeated["targets"][target]["baseline_prediction"]
            == inference_result["targets"][target][
                "baseline_prediction"
            ]
            and repeated["targets"][target]["scenario_prediction"]
            == inference_result["targets"][target][
                "scenario_prediction"
            ]
            for target in active_engine.target_names
        ),
        "naive_benchmarks_not_substituted": all(
            not inference_result["targets"][target][
                "naive_benchmark_substituted"
            ]
            for target in active_engine.target_names
        ),
        "locked_final_test_target_rows_parsed_zero": (
            active_engine.guard[
                "locked_final_test_target_rows_parsed"
            ]
            == 0
        ),
        "locked_final_test_access_count_zero": (
            active_engine.guard["final_test_access_count"] == 0
        ),
        "metadata_schema_valid": _validate_explanation_schema(
            target_explanations
        ),
        "all_outputs_noncausal": all(
            not payload["causal_effect_claimed"]
            for payload in target_explanations["targets"].values()
        ),
    }
    if not all(validations.values()):
        failures = [
            name for name, passed in validations.items() if not passed
        ]
        raise ValueError(f"Step 15 validation failed: {failures}")

    summary = {
        "schema_version": EXPLANATION_SUMMARY_VERSION,
        "completion_gate": (
            "STEP_15_EXPLANATIONS_AND_SENSITIVITY_COMPLETE"
        ),
        "target_count": len(active_engine.target_names),
        "feature_count": len(active_engine.feature_names),
        "scenario_input_count": len(CONTRACT_SHOCK_NAMES),
        "representative_state_count": 3,
        "training_rows": len(active_engine.training),
        "training_first_sample": str(
            active_engine.training["sample_date"].iloc[0]
        ),
        "training_last_sample": str(
            active_engine.training["sample_date"].iloc[-1]
        ),
        "contribution_rows": len(contributions),
        "contribution_delta_rows": len(contribution_deltas),
        "reconciliation_rows": len(reconciliation),
        "top_driver_rows": len(top_drivers),
        "sensitivity_curve_rows": len(sensitivity_curves),
        "sensitivity_state_summary_rows": len(
            sensitivity_state_summary
        ),
        "sensitivity_target_input_rows": len(sensitivity_summary),
        "negligible_target_input_pairs": int(
            sensitivity_summary["negligible_in_all_states"].sum()
        ),
        "unstable_walk_forward_direction_pairs": int(
            sensitivity_summary[
                "unstable_across_walk_forward_folds"
            ].sum()
        ),
        "counterintuitive_reference_pairs": int(
            sensitivity_summary[
                "counterintuitive_direction_flag"
            ].sum()
        ),
        "unusually_large_response_pairs": int(
            sensitivity_summary[
                "any_unusually_large_response"
            ].sum()
        ),
        "unusually_large_contribution_rows": int(
            contributions["unusually_large_contribution"].sum()
        ),
        "raw_zero_standardized_nonzero_rows": int(
            contributions[
                "raw_zero_but_standardized_nonzero"
            ].sum()
        ),
        "maximum_absolute_reconciliation_error": float(
            reconciliation["absolute_reconciliation_error"].max()
        ),
        "explanation_method": (
            "Ridge intercept plus exact standardized feature contributions"
        ),
        "sensitivity_method": (
            "One shock varied over historical quantiles plus neutral zero; "
            "all other shocks, state features, and release flags fixed."
        ),
        "direction_reference_policy": (
            "Limited conservative plausibility heuristics; ambiguous pairs "
            "are not classified and no reference affects model fitting."
        ),
        "locked_final_test": {
            "target_rows_parsed": 0,
            "access_count": 0,
            "target_values_available": False,
        },
        "validations": validations,
        "scope_boundary": (
            "Step 15 only: no Step 17 artifact export, all-target XGBoost, "
            "final-test access, API, Supabase, backend, frontend, or servers."
        ),
    }
    return {
        "engine": active_engine,
        "inference_result": inference_result,
        "baseline_row": baseline,
        "scenario_row": scenario,
        "feature_contributions": contributions,
        "contribution_deltas": contribution_deltas,
        "prediction_reconciliation": reconciliation,
        "top_drivers": top_drivers,
        "sensitivity_curves": sensitivity_curves,
        "sensitivity_state_summary": sensitivity_state_summary,
        "sensitivity_summary": sensitivity_summary,
        "target_explanations": target_explanations,
        "metadata_schema": metadata_schema,
        "summary": summary,
    }


def _svg_polyline(
    points: Sequence[tuple[float, float]],
    *,
    color: str,
) -> str:
    values = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return (
        f'<polyline points="{values}" fill="none" stroke="{color}" '
        'stroke-width="2" />'
    )


def sensitivity_svg(
    curves: pd.DataFrame,
    target: str,
) -> str:
    """Render deterministic, dependency-free eight-panel sensitivity SVG."""
    subset = curves.loc[curves["target"].eq(target)].copy()
    if subset.empty:
        raise ValueError(f"No sensitivity curves found for {target}.")
    width, height = 1200, 900
    panel_width, panel_height = 280, 185
    left, top = 28, 65
    x_gap, y_gap = 12, 22
    colors = {
        "low_vix": "#2563eb",
        "median_vix": "#16a34a",
        "high_vix": "#dc2626",
    }
    elements = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<text x="28" y="32" font-size="20" font-family="sans-serif" '
            f'font-weight="bold">{escape(target)} Ridge sensitivity</text>'
        ),
        (
            '<text x="28" y="51" font-size="12" font-family="sans-serif" '
            'fill="#475569">Incremental 21-session prediction; state and '
            "release flags fixed. Predictive association, not causal.</text>"
        ),
    ]
    for panel_index, contract_input in enumerate(CONTRACT_SHOCK_NAMES):
        column = panel_index % 4
        row = panel_index // 4
        x0 = left + column * (panel_width + x_gap)
        y0 = top + row * (panel_height + y_gap)
        panel = subset.loc[
            subset["contract_input"].eq(contract_input)
        ]
        x_min = float(panel["input_value"].min())
        x_max = float(panel["input_value"].max())
        y_min = float(panel["incremental_effect"].min())
        y_max = float(panel["incremental_effect"].max())
        if math.isclose(x_min, x_max):
            x_max = x_min + 1.0
        if math.isclose(y_min, y_max):
            y_min -= 0.5
            y_max += 0.5
        plot_left = x0 + 36
        plot_top = y0 + 28
        plot_width = panel_width - 48
        plot_height = panel_height - 62
        elements.extend(
            [
                (
                    f'<rect x="{x0}" y="{y0}" width="{panel_width}" '
                    f'height="{panel_height}" fill="#f8fafc" '
                    'stroke="#cbd5e1"/>'
                ),
                (
                    f'<text x="{x0 + 8}" y="{y0 + 18}" font-size="11" '
                    f'font-family="sans-serif">{escape(contract_input)}</text>'
                ),
                (
                    f'<line x1="{plot_left}" y1="{plot_top + plot_height}" '
                    f'x2="{plot_left + plot_width}" '
                    f'y2="{plot_top + plot_height}" stroke="#64748b"/>'
                ),
                (
                    f'<line x1="{plot_left}" y1="{plot_top}" '
                    f'x2="{plot_left}" y2="{plot_top + plot_height}" '
                    'stroke="#64748b"/>'
                ),
            ]
        )
        if y_min <= 0 <= y_max:
            zero_y = plot_top + plot_height * (
                1.0 - (0.0 - y_min) / (y_max - y_min)
            )
            elements.append(
                f'<line x1="{plot_left}" y1="{zero_y:.2f}" '
                f'x2="{plot_left + plot_width}" y2="{zero_y:.2f}" '
                'stroke="#94a3b8" stroke-dasharray="3 3"/>'
            )
        for state_name, group in panel.groupby(
            "representative_state",
            sort=False,
        ):
            group = group.sort_values("input_value", kind="stable")
            points = []
            for _, item in group.iterrows():
                px = plot_left + plot_width * (
                    (float(item["input_value"]) - x_min)
                    / (x_max - x_min)
                )
                py = plot_top + plot_height * (
                    1.0
                    - (float(item["incremental_effect"]) - y_min)
                    / (y_max - y_min)
                )
                points.append((px, py))
            elements.append(
                _svg_polyline(points, color=colors[state_name])
            )
        elements.extend(
            [
                (
                    f'<text x="{plot_left}" '
                    f'y="{plot_top + plot_height + 15}" font-size="9" '
                    f'font-family="sans-serif">{x_min:.3g}</text>'
                ),
                (
                    f'<text x="{plot_left + plot_width - 25}" '
                    f'y="{plot_top + plot_height + 15}" font-size="9" '
                    f'font-family="sans-serif">{x_max:.3g}</text>'
                ),
                (
                    f'<text x="{plot_left + 3}" y="{plot_top + 10}" '
                    f'font-size="9" font-family="sans-serif">{y_max:.3g}</text>'
                ),
                (
                    f'<text x="{plot_left + 3}" '
                    f'y="{plot_top + plot_height - 3}" font-size="9" '
                    f'font-family="sans-serif">{y_min:.3g}</text>'
                ),
            ]
        )
    legend_y = height - 36
    for index, state_name in enumerate(
        ("low_vix", "median_vix", "high_vix")
    ):
        x = 30 + index * 150
        elements.append(
            f'<line x1="{x}" y1="{legend_y}" x2="{x + 24}" '
            f'y2="{legend_y}" stroke="{colors[state_name]}" '
            'stroke-width="3"/>'
        )
        elements.append(
            f'<text x="{x + 31}" y="{legend_y + 4}" font-size="11" '
            f'font-family="sans-serif">{state_name}</text>'
        )
    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def persist_explanation_artifacts(
    artifacts: Mapping[str, Any],
    output_dir: Path = EXPLANATION_OUTPUT_ROOT,
) -> dict[str, str]:
    """Persist Step 15 research evidence; never serialize fitted models."""
    paths = {
        "feature_contributions": output_dir / "feature_contributions.csv",
        "contribution_deltas": output_dir / "contribution_deltas.csv",
        "prediction_reconciliation": (
            output_dir / "prediction_reconciliation.csv"
        ),
        "top_drivers": output_dir / "top_drivers.csv",
        "sensitivity_curves": output_dir / "sensitivity_curves.csv",
        "sensitivity_state_summary": (
            output_dir / "sensitivity_state_summary.csv"
        ),
        "sensitivity_summary": output_dir / "sensitivity_summary.csv",
        "target_explanations": output_dir / "target_explanations.json",
        "metadata_schema": (
            output_dir / "explanation_metadata_schema.json"
        ),
        "summary": output_dir / "scenario_explanations_summary.json",
    }
    csv_names = (
        "feature_contributions",
        "contribution_deltas",
        "prediction_reconciliation",
        "top_drivers",
        "sensitivity_curves",
        "sensitivity_state_summary",
        "sensitivity_summary",
    )
    csv_texts = {
        name: frame_to_csv_text(artifacts[name]) for name in csv_names
    }
    json_texts = {
        name: json.dumps(
            _json_safe(artifacts[name]),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
        for name in ("target_explanations", "metadata_schema")
    }
    figure_dir = output_dir / "figures"
    figure_texts = {
        target: sensitivity_svg(artifacts["sensitivity_curves"], target)
        for target in ALL_RIDGE_TARGETS
    }
    for name, text in csv_texts.items():
        write_text_atomic(paths[name], text)
    for name, text in json_texts.items():
        write_text_atomic(paths[name], text)
    figure_paths: dict[str, Path] = {}
    for target, text in figure_texts.items():
        path = figure_dir / f"{target}_sensitivity.svg"
        write_text_atomic(path, text)
        figure_paths[target] = path

    content_hashes = {
        **{name: _sha256_text(text) for name, text in csv_texts.items()},
        **{name: _sha256_text(text) for name, text in json_texts.items()},
    }
    summary = dict(artifacts["summary"])
    summary["output_files"] = {
        name: {
            "path": relative_path(path),
            "sha256": content_hashes[name],
        }
        for name, path in paths.items()
        if name != "summary"
    }
    summary["output_files"]["summary"] = {
        "path": relative_path(paths["summary"]),
        "sha256": None,
    }
    summary["sensitivity_figures"] = {
        target: {
            "path": relative_path(path),
            "sha256": _sha256_text(figure_texts[target]),
        }
        for target, path in figure_paths.items()
    }
    summary_text = (
        json.dumps(
            _json_safe(summary),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    write_text_atomic(paths["summary"], summary_text)
    return {
        **{name: str(path.resolve()) for name, path in paths.items()},
        **{
            f"figure_{target}": str(path.resolve())
            for target, path in figure_paths.items()
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build exact all-target Ridge explanations and sensitivity "
            "evidence without exporting fitted model artifacts."
        )
    )
    parser.add_argument(
        "--scenario-json",
        default=json.dumps(EXAMPLE_SCENARIO),
        help="JSON object containing exactly the eight scenario inputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate in memory without writing Step 15 evidence.",
    )
    args = parser.parse_args(argv)
    scenario = json.loads(args.scenario_json)
    artifacts = build_explanation_artifacts(
        scenario_shocks=scenario
    )
    paths = (
        {}
        if args.dry_run
        else persist_explanation_artifacts(artifacts)
    )
    print(
        json.dumps(
            {
                "completion_gate": artifacts["summary"][
                    "completion_gate"
                ],
                "targets": artifacts["summary"]["target_count"],
                "features": artifacts["summary"]["feature_count"],
                "maximum_reconciliation_error": artifacts["summary"][
                    "maximum_absolute_reconciliation_error"
                ],
                "locked_final_test": artifacts["summary"][
                    "locked_final_test"
                ],
                "outputs": paths,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
