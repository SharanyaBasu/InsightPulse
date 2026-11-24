"""Step 16 confidence and reliability diagnostics for all-target Ridge.

Confidence is rule based and non-compensatory. Weak predictive evidence,
unfamiliar inputs, stale state, and regime-specific errors remain separate;
they are never averaged into an unjustifiably strong score. All-target
XGBoost agreement is explicitly unavailable, excluded, and neutral.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from .cache import write_text_atomic
from .config import INTERIM_ROOT
from .feature_sources import frame_to_csv_text, relative_path
from .scenario_explanations import (
    EXPLANATION_SCHEMA_VERSION,
    build_explanation_artifacts,
)
from .scenario_inference import (
    ALL_RIDGE_TARGETS,
    CONTRACT_SHOCK_NAMES,
    CONTRACT_TO_HISTORICAL_FEATURE,
    EVENT_FLAG_NAMES,
    EXAMPLE_SCENARIO,
    NEUTRAL_SCENARIO,
    ResearchInferenceEngine,
    build_research_inference_engine,
)


CONFIDENCE_OUTPUT_ROOT = INTERIM_ROOT / "confidence"
STEP13_ROOT = INTERIM_ROOT / "models" / "all_targets"
STEP13_FOLD_METRICS_PATH = STEP13_ROOT / "all_targets_fold_metrics.csv"
STEP13_REGIME_METRICS_PATH = STEP13_ROOT / "all_targets_regime_metrics.csv"
STEP13_RESIDUAL_SUMMARY_PATH = (
    STEP13_ROOT / "all_targets_residual_summary.csv"
)

CONFIDENCE_SCHEMA_VERSION = "scenario_confidence_metadata_v1"
CONFIDENCE_SUMMARY_VERSION = "scenario_confidence_summary_v1"
XGBOOST_AGREEMENT_STATUS = (
    "UNAVAILABLE_INTENTIONALLY_POSTPONED"
)
WARNING_CATEGORIES = (
    "input_range",
    "state_extrapolation",
    "model_performance",
    "scenario_responsiveness",
    "regime",
    "state_freshness",
)


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
    if isinstance(value, (pd.Timestamp, date)):
        return value.isoformat()
    return value


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() == "true"


def _empirical_percentile(values: np.ndarray, value: float) -> float:
    below = float(np.sum(values < value))
    equal = float(np.sum(values == value))
    return 100.0 * (below + 0.5 * equal) / float(len(values))


def _warning(
    *,
    target: str,
    category: str,
    code: str,
    severity: str,
    message: str,
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if category not in WARNING_CATEGORIES:
        raise ValueError(f"Unknown warning category: {category}.")
    return {
        "target": target,
        "category": category,
        "code": code,
        "severity": severity,
        "message": message,
        "evidence": dict(evidence or {}),
        "causal_effect_claimed": False,
    }


def _inference_rows(
    engine: ResearchInferenceEngine,
    inference_result: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline, scenario, isolation = engine._assemble_pair(
        inference_result["scenario_shocks"],
        inference_result["context"]["state_features"],
        inference_result["release_flags"],
    )
    if not (
        isolation["feature_order_exact"]
        and isolation["only_shock_magnitudes_changed"]
        and isolation["state_identical"]
        and isolation["release_flags_identical"]
    ):
        raise ValueError("Step 16 inference-row isolation failed.")
    return baseline, scenario


def _input_diagnostics(
    engine: ResearchInferenceEngine,
    inference_result: Mapping[str, Any],
) -> pd.DataFrame:
    mapping = engine.feature_mapping.set_index("feature_name")
    scenario = inference_result["scenario_shocks"]
    state = inference_result["context"]["state_features"]
    first_pipeline = engine.pipelines[engine.target_names[0]]
    scaler = first_pipeline.named_steps["scaler"]
    rows: list[dict[str, Any]] = []

    input_specs: list[tuple[str, str, str, float]] = []
    for contract_input in CONTRACT_SHOCK_NAMES:
        feature = CONTRACT_TO_HISTORICAL_FEATURE[contract_input]
        input_specs.append(
            (
                "scenario_shock",
                contract_input,
                feature,
                float(scenario[contract_input]),
            )
        )
    for feature in engine.state_feature_names:
        input_specs.append(
            ("current_state", "", feature, float(state[feature]))
        )

    feature_positions = {
        feature: position
        for position, feature in enumerate(engine.feature_names)
    }
    for input_role, contract_input, feature, value in input_specs:
        values = engine.training[feature].to_numpy(dtype=float)
        position = feature_positions[feature]
        minimum = float(values.min())
        maximum = float(values.max())
        percentile = _empirical_percentile(values, value)
        outside = value < minimum or value > maximum
        near_edge = bool(
            not outside and (percentile < 1.0 or percentile > 99.0)
        )
        standardized = (
            value - float(scaler.mean_[position])
        ) / float(scaler.scale_[position])
        warning_status = (
            "OUTSIDE_TRAINING_RANGE"
            if outside
            else (
                "EXTREME_WITHIN_TRAINING_RANGE"
                if near_edge
                else "WITHIN_TRAINING_RANGE"
            )
        )
        rows.append(
            {
                "input_role": input_role,
                "contract_input": contract_input,
                "feature_name": feature,
                "feature_unit": str(mapping.loc[feature, "unit"]),
                "value": value,
                "historical_minimum": minimum,
                "historical_p01": float(np.quantile(values, 0.01)),
                "historical_p05": float(np.quantile(values, 0.05)),
                "historical_p25": float(np.quantile(values, 0.25)),
                "historical_median": float(np.quantile(values, 0.50)),
                "historical_p75": float(np.quantile(values, 0.75)),
                "historical_p95": float(np.quantile(values, 0.95)),
                "historical_p99": float(np.quantile(values, 0.99)),
                "historical_maximum": maximum,
                "empirical_percentile": percentile,
                "standardized_distance_from_training_mean": standardized,
                "absolute_standardized_distance": abs(standardized),
                "outside_training_range": outside,
                "extreme_within_range": near_edge,
                "warning_status": warning_status,
                "clipped": False,
                "winsorized": False,
            }
        )
    return pd.DataFrame(rows)


def _joint_distance_diagnostics(
    engine: ResearchInferenceEngine,
    baseline: pd.DataFrame,
    scenario: pd.DataFrame,
) -> dict[str, Any]:
    pipeline = engine.pipelines[engine.target_names[0]]
    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]
    training_raw = engine.training.loc[:, list(engine.feature_names)]
    training = scaler.transform(imputer.transform(training_raw))
    pairwise = pairwise_distances(training, metric="euclidean")
    np.fill_diagonal(pairwise, np.inf)
    reference_nearest = pairwise.min(axis=1)
    reference = {
        "training_rows": len(training),
        "feature_count": training.shape[1],
        "nearest_neighbor_distance_minimum": float(
            reference_nearest.min()
        ),
        "nearest_neighbor_distance_median": float(
            np.quantile(reference_nearest, 0.50)
        ),
        "nearest_neighbor_distance_p95": float(
            np.quantile(reference_nearest, 0.95)
        ),
        "nearest_neighbor_distance_p99": float(
            np.quantile(reference_nearest, 0.99)
        ),
        "nearest_neighbor_distance_maximum": float(
            reference_nearest.max()
        ),
        "distance_space": (
            "58-feature median-imputed and training-standardized Euclidean"
        ),
    }
    rows: dict[str, Any] = {}
    for row_name, frame in (
        ("baseline", baseline),
        ("scenario", scenario),
    ):
        transformed = scaler.transform(imputer.transform(frame))
        distances = pairwise_distances(
            transformed,
            training,
            metric="euclidean",
        )[0]
        nearest = float(distances.min())
        percentile = _empirical_percentile(reference_nearest, nearest)
        if nearest > reference[
            "nearest_neighbor_distance_maximum"
        ]:
            status = "OUTSIDE_OBSERVED_NEIGHBORHOOD"
        elif percentile > 99.0:
            status = "EXTREME_UNFAMILIAR_COMBINATION"
        elif percentile > 95.0:
            status = "UNUSUAL_COMBINATION"
        else:
            status = "WITHIN_OBSERVED_COMBINATION_SUPPORT"
        rows[row_name] = {
            "nearest_training_distance": nearest,
            "distance_percentile_vs_training_neighbors": percentile,
            "distance_status": status,
            "nearest_training_sample_date": str(
                engine.training.iloc[int(np.argmin(distances))][
                    "sample_date"
                ]
            ),
            "included_in_confidence": True,
        }
    return {
        "schema_version": "joint_feature_distance_v1",
        "reference_distribution": reference,
        "rows": rows,
    }


def _state_freshness(
    engine: ResearchInferenceEngine,
    as_of_date: str | date | None,
) -> dict[str, Any]:
    evaluation_date = (
        date.today()
        if as_of_date is None
        else pd.Timestamp(as_of_date).date()
    )
    state_date = pd.Timestamp(
        engine.latest_state_metadata["sample_date"]
    ).date()
    age_days = (evaluation_date - state_date).days
    if age_days < 0:
        status = "STATE_DATE_AFTER_DIAGNOSTIC_DATE"
    elif age_days <= 7:
        status = "FRESH"
    elif age_days <= 30:
        status = "AGING"
    else:
        status = "STALE"
    return {
        "diagnostic_as_of_date": evaluation_date.isoformat(),
        "state_sample_date": state_date.isoformat(),
        "state_cutoff_date": engine.latest_state_metadata[
            "state_cutoff_date"
        ],
        "calendar_age_days": age_days,
        "freshness_status": status,
        "thresholds_calendar_days": {
            "fresh_maximum": 7,
            "aging_maximum": 30,
            "stale_minimum": 31,
        },
        "included_in_confidence": True,
    }


def _current_vix_regime(vix_level: float) -> str:
    if vix_level < 20.0:
        return "low_vix_lt_20"
    if vix_level < 30.0:
        return "medium_vix_20_to_30"
    return "high_vix_ge_30"


def _target_confidence_artifacts(
    engine: ResearchInferenceEngine,
    inference_result: Mapping[str, Any],
    input_diagnostics: pd.DataFrame,
    joint_distance: Mapping[str, Any],
    freshness: Mapping[str, Any],
    sensitivity_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    fold_metrics = pd.read_csv(STEP13_FOLD_METRICS_PATH)
    fold_metrics = fold_metrics.loc[
        fold_metrics["strategy"].eq("ridge_full_history")
    ]
    regime_metrics = pd.read_csv(STEP13_REGIME_METRICS_PATH)
    regime_metrics = regime_metrics.loc[
        regime_metrics["strategy"].eq("ridge_full_history")
    ]
    residuals = pd.read_csv(STEP13_RESIDUAL_SUMMARY_PATH).set_index(
        "target"
    )
    current_vix = float(
        inference_result["context"]["state_features"][
            "vix_level_points"
        ]
    )
    current_regime = _current_vix_regime(current_vix)
    shock_outside = input_diagnostics.loc[
        input_diagnostics["input_role"].eq("scenario_shock"),
        "outside_training_range",
    ]
    state_outside = input_diagnostics.loc[
        input_diagnostics["input_role"].eq("current_state"),
        "outside_training_range",
    ]
    shock_outside_names = sorted(
        input_diagnostics.loc[
            input_diagnostics["input_role"].eq("scenario_shock")
            & input_diagnostics["outside_training_range"],
            "contract_input",
        ].tolist()
    )
    state_outside_names = sorted(
        input_diagnostics.loc[
            input_diagnostics["input_role"].eq("current_state")
            & input_diagnostics["outside_training_range"],
            "feature_name",
        ].tolist()
    )
    scenario_distance_status = joint_distance["rows"]["scenario"][
        "distance_status"
    ]

    flattened_rows: list[dict[str, Any]] = []
    warning_rows: list[dict[str, Any]] = []
    target_payload: dict[str, Any] = {}

    for target in engine.target_names:
        output = inference_result["targets"][target]
        assessment = engine.assessment.loc[target]
        target_folds = fold_metrics.loc[
            fold_metrics["target"].eq(target)
        ].sort_values("fold")
        if len(target_folds) != 5:
            raise ValueError(
                f"Expected five Ridge folds for {target}, found "
                f"{len(target_folds)}."
            )
        target_regimes = regime_metrics.loc[
            regime_metrics["target"].eq(target)
        ].set_index("vix_regime")
        if current_regime not in target_regimes.index:
            raise ValueError(
                f"Missing {current_regime} Ridge evidence for {target}."
            )
        similar_regime = target_regimes.loc[current_regime]
        residual = residuals.loc[target]
        sensitivity = sensitivity_summary.loc[
            sensitivity_summary["target"].eq(target)
        ]
        negligible = sorted(
            sensitivity.loc[
                sensitivity["negligible_in_all_states"],
                "contract_input",
            ].tolist()
        )
        unstable = sorted(
            sensitivity.loc[
                sensitivity[
                    "unstable_across_walk_forward_folds"
                ],
                "contract_input",
            ].tolist()
        )
        counterintuitive = sorted(
            sensitivity.loc[
                sensitivity["counterintuitive_direction_flag"],
                "contract_input",
            ].tolist()
        )

        warnings = {category: [] for category in WARNING_CATEGORIES}
        if shock_outside_names:
            warnings["input_range"].append(
                _warning(
                    target=target,
                    category="input_range",
                    code="SCENARIO_SHOCK_OUTSIDE_TRAINING_RANGE",
                    severity="high",
                    message=(
                        "One or more user shocks are outside the authorized "
                        "pretest training range; values were not clipped."
                    ),
                    evidence={"fields": shock_outside_names},
                )
            )
        if state_outside_names:
            warnings["state_extrapolation"].append(
                _warning(
                    target=target,
                    category="state_extrapolation",
                    code="CURRENT_STATE_OUTSIDE_TRAINING_RANGE",
                    severity="high",
                    message=(
                        "The current state contains features outside the "
                        "authorized pretest training range."
                    ),
                    evidence={"fields": state_outside_names},
                )
            )
        if scenario_distance_status != (
            "WITHIN_OBSERVED_COMBINATION_SUPPORT"
        ):
            warnings["state_extrapolation"].append(
                _warning(
                    target=target,
                    category="state_extrapolation",
                    code=scenario_distance_status,
                    severity=(
                        "high"
                        if scenario_distance_status
                        in {
                            "OUTSIDE_OBSERVED_NEIGHBORHOOD",
                            "EXTREME_UNFAMILIAR_COMBINATION",
                        }
                        else "medium"
                    ),
                    message=(
                        "The complete scenario row is distant from common "
                        "historical 58-feature combinations."
                    ),
                    evidence=joint_distance["rows"]["scenario"],
                )
            )

        classification = str(
            output["step13_evaluation_classification"]
        )
        ridge_oos_r2 = float(assessment["ridge_oos_r_squared"])
        best_naive_both = bool(
            float(
                assessment[
                    "ridge_mae_improvement_vs_best_naive_pct"
                ]
            )
            > 0.0
            and float(
                assessment[
                    "ridge_rmse_improvement_vs_best_naive_pct"
                ]
            )
            > 0.0
        )
        if classification == "NAIVE_FALLBACK_REQUIRED":
            warnings["model_performance"].append(
                _warning(
                    target=target,
                    category="model_performance",
                    code="WEAK_EVIDENCE_EXPERIMENTAL_RIDGE",
                    severity="high",
                    message=(
                        "Step 13 found weak evidence and did not approve this "
                        "Ridge model beyond experimental research use."
                    ),
                    evidence={"classification": classification},
                )
            )
        else:
            warnings["model_performance"].append(
                _warning(
                    target=target,
                    category="model_performance",
                    code="LOW_CONFIDENCE_EXPERIMENTAL_RIDGE",
                    severity="medium",
                    message=(
                        "Step 13 approved only low-confidence experimental "
                        "research inference."
                    ),
                    evidence={"classification": classification},
                )
            )
        if ridge_oos_r2 <= 0.0:
            warnings["model_performance"].append(
                _warning(
                    target=target,
                    category="model_performance",
                    code="NONPOSITIVE_WALK_FORWARD_R_SQUARED",
                    severity="high",
                    message=(
                        "Pooled walk-forward out-of-sample R-squared is not "
                        "positive."
                    ),
                    evidence={"out_of_sample_r_squared": ridge_oos_r2},
                )
            )
        if not best_naive_both:
            warnings["model_performance"].append(
                _warning(
                    target=target,
                    category="model_performance",
                    code="RIDGE_NOT_BETTER_THAN_BEST_NAIVE_ON_BOTH_ERRORS",
                    severity="high",
                    message=(
                        "Ridge did not beat the best naive benchmark on both "
                        "MAE and RMSE."
                    ),
                    evidence={
                        "ridge_mae_improvement_pct": float(
                            assessment[
                                "ridge_mae_improvement_vs_best_naive_pct"
                            ]
                        ),
                        "ridge_rmse_improvement_pct": float(
                            assessment[
                                "ridge_rmse_improvement_vs_best_naive_pct"
                            ]
                        ),
                    },
                )
            )
        if negligible:
            warnings["scenario_responsiveness"].append(
                _warning(
                    target=target,
                    category="scenario_responsiveness",
                    code="NEGLIGIBLE_RIDGE_SCENARIO_RESPONSES",
                    severity="medium",
                    message=(
                        "One or more shocks have negligible final Ridge "
                        "response across all representative VIX states."
                    ),
                    evidence={"scenario_inputs": negligible},
                )
            )
        if unstable:
            warnings["scenario_responsiveness"].append(
                _warning(
                    target=target,
                    category="scenario_responsiveness",
                    code="UNSTABLE_WALK_FORWARD_RESPONSE_DIRECTIONS",
                    severity="high",
                    message=(
                        "Historical walk-forward folds disagree on the "
                        "direction of one or more meaningful responses."
                    ),
                    evidence={"scenario_inputs": unstable},
                )
            )
        if counterintuitive:
            warnings["scenario_responsiveness"].append(
                _warning(
                    target=target,
                    category="scenario_responsiveness",
                    code="CONSERVATIVE_DIRECTION_REFERENCE_MISMATCH",
                    severity="medium",
                    message=(
                        "One or more final Ridge slopes differ from a limited "
                        "noncausal plausibility reference."
                    ),
                    evidence={"scenario_inputs": counterintuitive},
                )
            )

        overall_mae = float(assessment["ridge_mae"])
        similar_regime_mae = float(similar_regime["mae"])
        regime_error_ratio = (
            similar_regime_mae / overall_mae
            if overall_mae > 0.0
            else math.inf
        )
        if regime_error_ratio > 1.25:
            warnings["regime"].append(
                _warning(
                    target=target,
                    category="regime",
                    code="CURRENT_VIX_REGIME_ERROR_ELEVATED",
                    severity="medium",
                    message=(
                        "Walk-forward MAE in the current VIX regime is more "
                        "than 25% above pooled Ridge MAE."
                    ),
                    evidence={
                        "current_vix_regime": current_regime,
                        "regime_mae": similar_regime_mae,
                        "overall_mae": overall_mae,
                        "ratio": regime_error_ratio,
                    },
                )
            )
        if freshness["freshness_status"] != "FRESH":
            warnings["state_freshness"].append(
                _warning(
                    target=target,
                    category="state_freshness",
                    code=(
                        "STATE_SNAPSHOT_STALE"
                        if freshness["freshness_status"] == "STALE"
                        else "STATE_SNAPSHOT_AGING"
                    ),
                    severity=(
                        "high"
                        if freshness["freshness_status"] == "STALE"
                        else "medium"
                    ),
                    message=(
                        "The default state snapshot is not current relative "
                        "to the diagnostic date."
                    ),
                    evidence=freshness,
                )
            )

        severe_distance = scenario_distance_status in {
            "OUTSIDE_OBSERVED_NEIGHBORHOOD",
            "EXTREME_UNFAMILIAR_COMBINATION",
        }
        if bool(shock_outside.any()) or severe_distance:
            overall = "OUTSIDE_HISTORICAL_SUPPORT"
        elif classification == "NAIVE_FALLBACK_REQUIRED":
            overall = "VERY_LOW_RESEARCH_CONFIDENCE"
        elif (
            bool(state_outside.any())
            or scenario_distance_status == "UNUSUAL_COMBINATION"
            or freshness["freshness_status"] != "FRESH"
        ):
            overall = "LOW_RESEARCH_CONFIDENCE_WITH_CONTEXT_WARNINGS"
        else:
            overall = "LOW_RESEARCH_CONFIDENCE"

        xgboost_agreement = {
            "status": XGBOOST_AGREEMENT_STATUS,
            "available": False,
            "included_in_confidence": False,
            "score": None,
            "penalty_or_bonus": "NONE",
            "reason": (
                "All-target XGBoost evaluation was intentionally postponed. "
                "This diagnostic may be added in a later model version."
            ),
        }
        flattened = {
            "target": target,
            "unit": output["unit"],
            "selected_alpha": output["selected_alpha"],
            "step13_evaluation_classification": classification,
            "overall_research_confidence": overall,
            "confidence_method": "RULE_BASED_NON_COMPENSATORY",
            "walk_forward_rows": int(assessment["walk_forward_rows"]),
            "walk_forward_mae": overall_mae,
            "walk_forward_rmse": float(assessment["ridge_rmse"]),
            "walk_forward_oos_r_squared": ridge_oos_r2,
            "fold_mae_standard_deviation": float(
                assessment["fold_mae_standard_deviation"]
            ),
            "worst_fold_mae": float(assessment["worst_fold_mae"]),
            "current_vix_level_points": current_vix,
            "current_vix_regime": current_regime,
            "similar_regime_observations": int(
                similar_regime["sample_count"]
            ),
            "similar_regime_mae": similar_regime_mae,
            "similar_regime_rmse": float(similar_regime["rmse"]),
            "similar_regime_error_ratio": regime_error_ratio,
            "maximum_absolute_walk_forward_error": float(
                residual["maximum_absolute_error"]
            ),
            "worst_error_sample_date": str(
                residual["worst_sample_date"]
            ),
            "negligible_scenario_input_count": len(negligible),
            "unstable_scenario_input_count": len(unstable),
            "counterintuitive_reference_flag_count": len(
                counterintuitive
            ),
            "input_range_warning_count": len(
                warnings["input_range"]
            ),
            "state_extrapolation_warning_count": len(
                warnings["state_extrapolation"]
            ),
            "model_performance_warning_count": len(
                warnings["model_performance"]
            ),
            "scenario_responsiveness_warning_count": len(
                warnings["scenario_responsiveness"]
            ),
            "regime_warning_count": len(warnings["regime"]),
            "state_freshness_warning_count": len(
                warnings["state_freshness"]
            ),
            "xgboost_agreement_status": XGBOOST_AGREEMENT_STATUS,
            "xgboost_agreement_included": False,
            "xgboost_agreement_score": None,
            "naive_benchmark_substituted": False,
            "production_ready": False,
            "causal_effect_claimed": False,
        }
        flattened_rows.append(flattened)
        for category in WARNING_CATEGORIES:
            warning_rows.extend(warnings[category])
        target_payload[target] = {
            **flattened,
            "input_range": {
                "outside_scenario_inputs": shock_outside_names,
                "diagnostic_rows": input_diagnostics.loc[
                    input_diagnostics["input_role"].eq(
                        "scenario_shock"
                    )
                ].to_dict(orient="records"),
            },
            "state_extrapolation": {
                "outside_state_features": state_outside_names,
                "joint_distance": joint_distance,
            },
            "model_performance": {
                "fold_metrics": target_folds.to_dict(orient="records"),
                "best_naive_comparison": output[
                    "naive_benchmark_comparison"
                ],
                "residual_extreme_period": {
                    key: _json_safe(value)
                    for key, value in residual.to_dict().items()
                },
            },
            "scenario_responsiveness": {
                "negligible_scenario_inputs": negligible,
                "unstable_walk_forward_directions": unstable,
                "counterintuitive_reference_flags": counterintuitive,
                "sensitivity_rows": sensitivity.to_dict(
                    orient="records"
                ),
            },
            "regime": {
                "current_vix_regime": current_regime,
                "current_regime_metrics": {
                    key: _json_safe(value)
                    for key, value in similar_regime.to_dict().items()
                },
            },
            "state_freshness": dict(freshness),
            "xgboost_agreement": xgboost_agreement,
            "warnings": warnings,
            "overall_research_confidence_result": {
                "status": overall,
                "method": "RULE_BASED_NON_COMPENSATORY",
                "maximum_possible_status": (
                    "LOW_RESEARCH_CONFIDENCE"
                    if classification
                    == "APPROVED_WITH_LOW_CONFIDENCE"
                    else "VERY_LOW_RESEARCH_CONFIDENCE"
                ),
                "no_unrelated_diagnostic_averaging": True,
                "xgboost_unavailability_did_not_change_status": True,
            },
        }
    return (
        pd.DataFrame(flattened_rows),
        pd.DataFrame(warning_rows),
        target_payload,
    )


def confidence_metadata_schema() -> dict[str, Any]:
    return {
        "schema_version": CONFIDENCE_SCHEMA_VERSION,
        "target_required_fields": [
            "target",
            "unit",
            "selected_alpha",
            "step13_evaluation_classification",
            "overall_research_confidence",
            "confidence_method",
            "input_range",
            "state_extrapolation",
            "model_performance",
            "scenario_responsiveness",
            "regime",
            "state_freshness",
            "xgboost_agreement",
            "warnings",
            "overall_research_confidence_result",
            "naive_benchmark_substituted",
            "production_ready",
            "causal_effect_claimed",
        ],
        "warning_categories": list(WARNING_CATEGORIES),
        "xgboost_agreement_contract": {
            "status": XGBOOST_AGREEMENT_STATUS,
            "available": False,
            "included_in_confidence": False,
            "score": None,
            "penalty_or_bonus": "NONE",
        },
        "confidence_method": "RULE_BASED_NON_COMPENSATORY",
        "allowed_overall_statuses": [
            "LOW_RESEARCH_CONFIDENCE",
            "LOW_RESEARCH_CONFIDENCE_WITH_CONTEXT_WARNINGS",
            "VERY_LOW_RESEARCH_CONFIDENCE",
            "OUTSIDE_HISTORICAL_SUPPORT",
        ],
        "strong_confidence_status_available": False,
        "causal_effect_claimed": False,
    }


def _validate_confidence_schema(payload: Mapping[str, Any]) -> bool:
    schema = confidence_metadata_schema()
    if payload.get("schema_version") != schema["schema_version"]:
        return False
    if tuple(payload.get("targets", {})) != ALL_RIDGE_TARGETS:
        return False
    required = set(schema["target_required_fields"])
    for target_payload in payload["targets"].values():
        if not required <= set(target_payload):
            return False
        if set(target_payload["warnings"]) != set(WARNING_CATEGORIES):
            return False
        agreement = target_payload["xgboost_agreement"]
        if (
            agreement["status"] != XGBOOST_AGREEMENT_STATUS
            or agreement["included_in_confidence"]
            or agreement["score"] is not None
        ):
            return False
    return True


def build_confidence_artifacts(
    *,
    scenario_shocks: Mapping[str, Any] = EXAMPLE_SCENARIO,
    diagnostic_as_of_date: str | date | None = None,
    engine: ResearchInferenceEngine | None = None,
    explanation_artifacts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build Step 16 diagnostics from accepted Step 15 Ridge evidence."""
    active_engine = engine or build_research_inference_engine()
    explanations = (
        explanation_artifacts
        if explanation_artifacts is not None
        else build_explanation_artifacts(
            scenario_shocks=scenario_shocks,
            engine=active_engine,
        )
    )
    if explanations["summary"]["completion_gate"] != (
        "STEP_15_EXPLANATIONS_AND_SENSITIVITY_COMPLETE"
    ):
        raise ValueError("Step 16 requires accepted Step 15 outputs.")
    inference_result = explanations["inference_result"]
    if dict(inference_result["scenario_shocks"]) != dict(
        scenario_shocks
    ):
        raise ValueError("Step 15 scenario does not match Step 16.")
    baseline, scenario = _inference_rows(
        active_engine,
        inference_result,
    )
    input_diagnostics = _input_diagnostics(
        active_engine,
        inference_result,
    )
    joint_distance = _joint_distance_diagnostics(
        active_engine,
        baseline,
        scenario,
    )
    freshness = _state_freshness(
        active_engine,
        diagnostic_as_of_date,
    )
    (
        target_diagnostics,
        warning_records,
        target_payload,
    ) = _target_confidence_artifacts(
        active_engine,
        inference_result,
        input_diagnostics,
        joint_distance,
        freshness,
        explanations["sensitivity_summary"],
    )
    payload = {
        "schema_version": CONFIDENCE_SCHEMA_VERSION,
        "diagnostic_as_of_date": freshness[
            "diagnostic_as_of_date"
        ],
        "horizon_trading_days": 21,
        "feature_count": len(active_engine.feature_names),
        "scenario_shocks": inference_result["scenario_shocks"],
        "release_flags": inference_result["release_flags"],
        "context": inference_result["context"],
        "state_freshness": freshness,
        "joint_feature_distance": joint_distance,
        "xgboost_agreement_policy": {
            "status": XGBOOST_AGREEMENT_STATUS,
            "available": False,
            "included_in_confidence": False,
            "score": None,
            "penalty_or_bonus": "NONE",
            "later_version_supported": True,
        },
        "targets": target_payload,
        "global_warning": (
            "Research confidence is non-compensatory and cannot exceed the "
            "frozen Step 13 target evidence. Diagnostics do not establish "
            "causal effects or production readiness."
        ),
    }
    metadata_schema = confidence_metadata_schema()
    numeric_target_columns = [
        "selected_alpha",
        "walk_forward_rows",
        "walk_forward_mae",
        "walk_forward_rmse",
        "walk_forward_oos_r_squared",
        "fold_mae_standard_deviation",
        "worst_fold_mae",
        "current_vix_level_points",
        "similar_regime_observations",
        "similar_regime_mae",
        "similar_regime_rmse",
        "similar_regime_error_ratio",
        "maximum_absolute_walk_forward_error",
    ]
    expected_metrics_match = all(
        math.isclose(
            float(
                target_diagnostics.set_index("target").loc[
                    target, "walk_forward_mae"
                ]
            ),
            float(active_engine.assessment.loc[target, "ridge_mae"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(
                target_diagnostics.set_index("target").loc[
                    target, "walk_forward_rmse"
                ]
            ),
            float(active_engine.assessment.loc[target, "ridge_rmse"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for target in active_engine.target_names
    )
    repeated = build_confidence_payload_signature(
        payload,
        target_diagnostics,
        input_diagnostics,
    )
    repeated_check = build_confidence_payload_signature(
        payload,
        target_diagnostics.copy(),
        input_diagnostics.copy(),
    )
    validations = {
        "accepted_step15_outputs_used": (
            explanations["summary"]["completion_gate"]
            == "STEP_15_EXPLANATIONS_AND_SENSITIVITY_COMPLETE"
        ),
        "exactly_twelve_target_diagnostics": (
            tuple(target_diagnostics["target"]) == ALL_RIDGE_TARGETS
        ),
        "all_53_continuous_inputs_diagnosed": (
            len(input_diagnostics) == 53
            and int(
                input_diagnostics["input_role"]
                .eq("scenario_shock")
                .sum()
            )
            == 8
            and int(
                input_diagnostics["input_role"]
                .eq("current_state")
                .sum()
            )
            == 45
        ),
        "all_input_diagnostic_numeric_values_finite": bool(
            np.isfinite(
                input_diagnostics[
                    [
                        "value",
                        "historical_minimum",
                        "historical_maximum",
                        "empirical_percentile",
                        "standardized_distance_from_training_mean",
                    ]
                ].to_numpy(dtype=float)
            ).all()
        ),
        "joint_distance_diagnostics_finite": all(
            math.isfinite(
                float(
                    joint_distance["rows"][row_name][
                        "nearest_training_distance"
                    ]
                )
            )
            for row_name in ("baseline", "scenario")
        ),
        "walk_forward_evidence_matches_step13": expected_metrics_match,
        "all_target_numeric_diagnostics_finite": bool(
            np.isfinite(
                target_diagnostics[
                    numeric_target_columns
                ].to_numpy(dtype=float)
            ).all()
        ),
        "all_warning_categories_structured": all(
            set(target_payload[target]["warnings"])
            == set(WARNING_CATEGORIES)
            for target in active_engine.target_names
        ),
        "xgboost_agreement_unavailable_for_all_targets": bool(
            target_diagnostics["xgboost_agreement_status"].eq(
                XGBOOST_AGREEMENT_STATUS
            ).all()
        ),
        "xgboost_agreement_excluded_for_all_targets": bool(
            not target_diagnostics[
                "xgboost_agreement_included"
            ].any()
            and target_diagnostics[
                "xgboost_agreement_score"
            ].isna()
            .all()
        ),
        "xgboost_unavailability_not_scored_or_penalized": all(
            target_payload[target]["xgboost_agreement"][
                "penalty_or_bonus"
            ]
            == "NONE"
            for target in active_engine.target_names
        ),
        "confidence_never_exceeds_step13_evidence": bool(
            target_diagnostics["overall_research_confidence"].isin(
                metadata_schema["allowed_overall_statuses"]
            ).all()
        ),
        "naive_benchmarks_never_substituted": bool(
            ~target_diagnostics["naive_benchmark_substituted"].any()
        ),
        "feature_order_and_pair_isolation_preserved": bool(
            inference_result["feature_assembly"][
                "feature_order_exact"
            ]
            and inference_result["feature_assembly"][
                "state_identical"
            ]
            and inference_result["feature_assembly"][
                "release_flags_identical"
            ]
        ),
        "neutral_shock_effects_remain_zero": all(
            active_engine.predict(NEUTRAL_SCENARIO)["targets"][target][
                "incremental_scenario_effect"
            ]
            == 0.0
            for target in active_engine.target_names
        ),
        "diagnostics_deterministic": repeated == repeated_check,
        "metadata_schema_valid": _validate_confidence_schema(payload),
        "locked_final_test_target_rows_parsed_zero": (
            active_engine.guard[
                "locked_final_test_target_rows_parsed"
            ]
            == 0
        ),
        "locked_final_test_access_count_zero": (
            active_engine.guard["final_test_access_count"] == 0
        ),
        "all_outputs_research_only_noncausal": bool(
            ~target_diagnostics["production_ready"].any()
            and ~target_diagnostics["causal_effect_claimed"].any()
        ),
    }
    if not all(validations.values()):
        failures = [
            name for name, passed in validations.items() if not passed
        ]
        raise ValueError(f"Step 16 validation failed: {failures}")

    summary = {
        "schema_version": CONFIDENCE_SUMMARY_VERSION,
        "completion_gate": (
            "STEP_16_CONFIDENCE_DIAGNOSTICS_COMPLETE"
        ),
        "step15_schema_version": EXPLANATION_SCHEMA_VERSION,
        "target_count": len(active_engine.target_names),
        "feature_count": len(active_engine.feature_names),
        "scenario_input_count": len(CONTRACT_SHOCK_NAMES),
        "state_feature_count": len(active_engine.state_feature_names),
        "input_diagnostic_rows": len(input_diagnostics),
        "warning_record_rows": len(warning_records),
        "confidence_method": "RULE_BASED_NON_COMPENSATORY",
        "overall_confidence_counts": {
            str(key): int(value)
            for key, value in target_diagnostics[
                "overall_research_confidence"
            ]
            .value_counts()
            .sort_index()
            .items()
        },
        "input_range_warning_targets": int(
            target_diagnostics[
                "input_range_warning_count"
            ].gt(0).sum()
        ),
        "state_extrapolation_warning_targets": int(
            target_diagnostics[
                "state_extrapolation_warning_count"
            ].gt(0).sum()
        ),
        "model_performance_warning_targets": int(
            target_diagnostics[
                "model_performance_warning_count"
            ].gt(0).sum()
        ),
        "scenario_responsiveness_warning_targets": int(
            target_diagnostics[
                "scenario_responsiveness_warning_count"
            ].gt(0).sum()
        ),
        "regime_warning_targets": int(
            target_diagnostics["regime_warning_count"].gt(0).sum()
        ),
        "state_freshness_warning_targets": int(
            target_diagnostics[
                "state_freshness_warning_count"
            ].gt(0).sum()
        ),
        "scenario_distance_status": joint_distance["rows"]["scenario"][
            "distance_status"
        ],
        "state_freshness": freshness,
        "xgboost_agreement": {
            "status": XGBOOST_AGREEMENT_STATUS,
            "target_count": len(active_engine.target_names),
            "included_in_confidence_count": 0,
            "scored_target_count": 0,
            "penalty_or_bonus": "NONE",
        },
        "locked_final_test": {
            "target_rows_parsed": 0,
            "access_count": 0,
            "target_values_available": False,
        },
        "validations": validations,
        "step17_readiness": {
            "explanation_schema_frozen": True,
            "confidence_schema_frozen": True,
            "twelve_ridge_pipelines_unchanged": True,
            "ready_for_ridge_artifact_export_after_acceptance": True,
        },
        "scope_boundary": (
            "Step 16 only: no Step 17 artifact export, all-target XGBoost, "
            "final-test access, API, Supabase, backend, frontend, or servers."
        ),
    }
    return {
        "engine": active_engine,
        "explanation_artifacts": explanations,
        "input_diagnostics": input_diagnostics,
        "target_confidence_diagnostics": target_diagnostics,
        "warning_records": warning_records,
        "joint_feature_distance": joint_distance,
        "state_freshness": freshness,
        "target_confidence": payload,
        "metadata_schema": metadata_schema,
        "summary": summary,
    }


def build_confidence_payload_signature(
    payload: Mapping[str, Any],
    target_diagnostics: pd.DataFrame,
    input_diagnostics: pd.DataFrame,
) -> str:
    serialized = json.dumps(
        {
            "payload": _json_safe(payload),
            "target_diagnostics": _json_safe(
                target_diagnostics.to_dict(orient="records")
            ),
            "input_diagnostics": _json_safe(
                input_diagnostics.to_dict(orient="records")
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return _sha256_text(serialized)


def persist_confidence_artifacts(
    artifacts: Mapping[str, Any],
    output_dir: Path = CONFIDENCE_OUTPUT_ROOT,
) -> dict[str, str]:
    """Persist Step 16 evidence; never serialize fitted model pipelines."""
    paths = {
        "input_diagnostics": output_dir / "input_diagnostics.csv",
        "target_confidence_diagnostics": (
            output_dir / "target_confidence_diagnostics.csv"
        ),
        "warning_records": output_dir / "warning_records.csv",
        "joint_feature_distance": (
            output_dir / "joint_feature_distance.json"
        ),
        "state_freshness": output_dir / "state_freshness.json",
        "target_confidence": output_dir / "target_confidence.json",
        "metadata_schema": (
            output_dir / "confidence_metadata_schema.json"
        ),
        "summary": output_dir / "confidence_diagnostics_summary.json",
    }
    csv_names = (
        "input_diagnostics",
        "target_confidence_diagnostics",
        "warning_records",
    )
    csv_texts = {
        name: frame_to_csv_text(artifacts[name]) for name in csv_names
    }
    json_names = (
        "joint_feature_distance",
        "state_freshness",
        "target_confidence",
        "metadata_schema",
    )
    json_texts = {
        name: json.dumps(
            _json_safe(artifacts[name]),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
        for name in json_names
    }
    for name, text in csv_texts.items():
        write_text_atomic(paths[name], text)
    for name, text in json_texts.items():
        write_text_atomic(paths[name], text)

    hashes = {
        **{name: _sha256_text(text) for name, text in csv_texts.items()},
        **{name: _sha256_text(text) for name, text in json_texts.items()},
    }
    summary = dict(artifacts["summary"])
    summary["output_files"] = {
        name: {
            "path": relative_path(path),
            "sha256": hashes[name],
        }
        for name, path in paths.items()
        if name != "summary"
    }
    summary["output_files"]["summary"] = {
        "path": relative_path(paths["summary"]),
        "sha256": None,
    }
    write_text_atomic(
        paths["summary"],
        json.dumps(
            _json_safe(summary),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )
    return {
        name: str(path.resolve()) for name, path in paths.items()
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build all-target Ridge confidence diagnostics without "
            "training XGBoost or exporting fitted artifacts."
        )
    )
    parser.add_argument(
        "--scenario-json",
        default=json.dumps(EXAMPLE_SCENARIO),
        help="JSON object containing exactly the eight scenario inputs.",
    )
    parser.add_argument(
        "--diagnostic-as-of-date",
        default=None,
        help="Optional YYYY-MM-DD date for reproducible freshness checks.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate in memory without writing Step 16 evidence.",
    )
    args = parser.parse_args(argv)
    scenario = json.loads(args.scenario_json)
    artifacts = build_confidence_artifacts(
        scenario_shocks=scenario,
        diagnostic_as_of_date=args.diagnostic_as_of_date,
    )
    paths = (
        {}
        if args.dry_run
        else persist_confidence_artifacts(artifacts)
    )
    print(
        json.dumps(
            {
                "completion_gate": artifacts["summary"][
                    "completion_gate"
                ],
                "targets": artifacts["summary"]["target_count"],
                "overall_confidence_counts": artifacts["summary"][
                    "overall_confidence_counts"
                ],
                "xgboost_agreement": artifacts["summary"][
                    "xgboost_agreement"
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
