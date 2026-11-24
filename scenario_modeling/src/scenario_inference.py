"""Step 14 all-target Ridge research inference for the Scenario Playground.

All 12 targets receive independent Ridge baseline/scenario predictions using
their frozen Step 13 alphas. Step 13 evaluation classifications remain visible
as confidence evidence and warnings, while naive strategies remain comparison
benchmarks and are never substituted for Ridge inference outputs.

No model binaries are read or written. All Ridge pipelines are refitted in
memory from the bounded pre-final-test rows and frozen Step 13 configurations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .all_target_models import load_all_pretest_rows
from .cache import write_text_atomic
from .config import INTERIM_ROOT, PROCESSED_ROOT
from .feature_sources import frame_to_csv_text, relative_path
from .model_matrix import FINAL_FEATURE_NAMES, STATE_PATH
from .ridge_baselines import make_ridge_pipeline
from .shock_features import (
    CONTRACT_SHOCK_NAMES,
    CONTRACT_TO_HISTORICAL_FEATURE,
    EVENT_FLAG_NAMES,
)


STEP13_ROOT = INTERIM_ROOT / "models" / "all_targets"
STEP13_CONFIGURATIONS_PATH = (
    STEP13_ROOT / "all_targets_final_configurations.csv"
)
STEP13_ASSESSMENT_PATH = (
    STEP13_ROOT / "all_targets_target_assessment.csv"
)
COLUMN_CATALOG_PATH = (
    PROCESSED_ROOT
    / "model_matrix"
    / "scenario_model_matrix_column_catalog.csv"
)
INFERENCE_OUTPUT_ROOT = INTERIM_ROOT / "inference"

LOW_CONFIDENCE_PROMOTED_TARGETS = (
    "sp500",
    "energy",
    "consumer_discretionary",
)
ASSET_TARGETS = (
    "sp500",
    "nasdaq",
    "ten_year_yield_bps",
    "dxy",
    "gold",
    "oil",
)
SECTOR_TARGETS = (
    "technology",
    "energy",
    "financials",
    "utilities",
    "healthcare",
    "consumer_discretionary",
)
ALL_RIDGE_TARGETS = ASSET_TARGETS + SECTOR_TARGETS
WEAK_EVIDENCE_RIDGE_TARGETS = tuple(
    target
    for target in ALL_RIDGE_TARGETS
    if target not in LOW_CONFIDENCE_PROMOTED_TARGETS
)
EXPECTED_TARGET_CONFIGURATIONS = {
    "sp500": {
        "alpha": 10_000.0,
        "unit": "percent",
        "classification": "APPROVED_WITH_LOW_CONFIDENCE",
    },
    "nasdaq": {
        "alpha": 10_000.0,
        "unit": "percent",
        "classification": "NAIVE_FALLBACK_REQUIRED",
    },
    "ten_year_yield_bps": {
        "alpha": 1_000_000.0,
        "unit": "basis_points",
        "classification": "NAIVE_FALLBACK_REQUIRED",
    },
    "dxy": {
        "alpha": 100.0,
        "unit": "percent",
        "classification": "NAIVE_FALLBACK_REQUIRED",
    },
    "gold": {
        "alpha": 10.0,
        "unit": "percent",
        "classification": "NAIVE_FALLBACK_REQUIRED",
    },
    "oil": {
        "alpha": 1_000_000.0,
        "unit": "percent",
        "classification": "NAIVE_FALLBACK_REQUIRED",
    },
    "technology": {
        "alpha": 1_000_000.0,
        "unit": "percent",
        "classification": "NAIVE_FALLBACK_REQUIRED",
    },
    "energy": {
        "alpha": 10_000.0,
        "unit": "percent",
        "classification": "APPROVED_WITH_LOW_CONFIDENCE",
    },
    "financials": {
        "alpha": 100_000.0,
        "unit": "percent",
        "classification": "NAIVE_FALLBACK_REQUIRED",
    },
    "utilities": {
        "alpha": 1_000.0,
        "unit": "percent",
        "classification": "NAIVE_FALLBACK_REQUIRED",
    },
    "healthcare": {
        "alpha": 100_000.0,
        "unit": "percent",
        "classification": "NAIVE_FALLBACK_REQUIRED",
    },
    "consumer_discretionary": {
        "alpha": 10_000.0,
        "unit": "percent",
        "classification": "APPROVED_WITH_LOW_CONFIDENCE",
    },
}

EVENT_INPUT_TO_FLAG = {
    "fed_funds_rate_change_bps": "fomc_release_flag",
    "cpi_surprise_pct": "cpi_release_flag",
    "unemployment_change_pct": "unemployment_release_flag",
    "gdp_growth_surprise_pct": "gdp_release_flag",
    "pmi_change_points": "pmi_release_flag",
}

SHOCK_UNITS = {
    "fed_funds_rate_change_bps": "basis_points",
    "cpi_surprise_pct": "percentage_points",
    "oil_price_change_pct": "percent",
    "gdp_growth_surprise_pct": "annualized_percentage_points",
    "unemployment_change_pct": "percentage_points",
    "pmi_change_points": "pmi_style_index_points",
    "dxy_change_pct": "percent",
    "vix_change_points": "vix_index_points",
}

NEUTRAL_SCENARIO = {
    name: 0.0 for name in CONTRACT_SHOCK_NAMES
}
EXAMPLE_SCENARIO = {
    "fed_funds_rate_change_bps": 0.0,
    "cpi_surprise_pct": 0.2,
    "oil_price_change_pct": 5.0,
    "gdp_growth_surprise_pct": 0.0,
    "unemployment_change_pct": 0.0,
    "pmi_change_points": 0.0,
    "dxy_change_pct": 1.0,
    "vix_change_points": 5.0,
}


class ScenarioValidationError(ValueError):
    """Raised when a scenario cannot be assembled safely."""


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }
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


def _ordered_finite_mapping(
    values: Mapping[str, Any],
    required_names: Sequence[str],
    *,
    label: str,
) -> dict[str, float]:
    if not isinstance(values, Mapping):
        raise ScenarioValidationError(f"{label} must be a mapping.")
    required = tuple(required_names)
    missing = [name for name in required if name not in values]
    extra = [name for name in values if name not in required]
    if missing or extra:
        raise ScenarioValidationError(
            f"{label} keys do not match the contract; "
            f"missing={missing}; extra={extra}."
        )
    ordered: dict[str, float] = {}
    for name in required:
        value = values[name]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ScenarioValidationError(
                f"{label}.{name} must be a finite number, not "
                f"{type(value).__name__}."
            )
        number = float(value)
        if not math.isfinite(number):
            raise ScenarioValidationError(
                f"{label}.{name} must be finite."
            )
        ordered[name] = number
    return ordered


def _ordered_release_flags(
    flags: Mapping[str, Any],
) -> dict[str, float]:
    if not isinstance(flags, Mapping):
        raise ScenarioValidationError(
            "release_flags must be a mapping."
        )
    missing = [name for name in EVENT_FLAG_NAMES if name not in flags]
    extra = [name for name in flags if name not in EVENT_FLAG_NAMES]
    if missing or extra:
        raise ScenarioValidationError(
            "release_flags keys do not match the five-feature schema; "
            f"missing={missing}; extra={extra}."
        )
    ordered: dict[str, float] = {}
    for name in EVENT_FLAG_NAMES:
        value = flags[name]
        if isinstance(value, (bool, np.bool_)):
            number = float(value)
        elif isinstance(value, Real):
            number = float(value)
        else:
            raise ScenarioValidationError(
                f"release_flags.{name} must be 0 or 1."
            )
        if number not in {0.0, 1.0}:
            raise ScenarioValidationError(
                f"release_flags.{name} must be exactly 0 or 1."
            )
        ordered[name] = number
    return ordered


def _derived_release_flags(
    scenario: Mapping[str, float],
) -> dict[str, float]:
    flags = {name: 0.0 for name in EVENT_FLAG_NAMES}
    for contract_name, flag_name in EVENT_INPUT_TO_FLAG.items():
        flags[flag_name] = float(scenario[contract_name] != 0.0)
    return flags


def _feature_row_hash(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
) -> str:
    payload = [
        [name, float(frame.iloc[0][name])]
        for name in feature_names
    ]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode()
    ).hexdigest()


def _as_bool(value: Any, *, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, Real) and float(value) in {0.0, 1.0}:
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "false"}:
        return text == "true"
    raise ScenarioValidationError(f"{label} is not a boolean value.")


def _finite_metric(value: Any, *, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ScenarioValidationError(f"{label} must be finite.")
    return number


def _split_limitations(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    return [
        item.strip()
        for item in str(value).split("|")
        if item.strip()
    ]


def _target_evidence_metadata(
    target: str,
    decision: pd.Series,
) -> dict[str, Any]:
    classification = str(decision["classification"])
    weak_evidence = classification == "NAIVE_FALLBACK_REQUIRED"
    if classification not in {
        "APPROVED_WITH_LOW_CONFIDENCE",
        "NAIVE_FALLBACK_REQUIRED",
    }:
        raise ScenarioValidationError(
            f"Unexpected Step 13 classification for {target}: "
            f"{classification}."
        )

    ridge_mae_improvement = _finite_metric(
        decision["ridge_mae_improvement_vs_best_naive_pct"],
        label=f"{target}.ridge_mae_improvement_vs_best_naive_pct",
    )
    ridge_rmse_improvement = _finite_metric(
        decision["ridge_rmse_improvement_vs_best_naive_pct"],
        label=f"{target}.ridge_rmse_improvement_vs_best_naive_pct",
    )
    known_limitations = _split_limitations(decision["known_limitations"])
    warning_codes = [
        "RESEARCH_ONLY_NOT_PRODUCTION_APPROVED",
        "PREDICTIVE_ASSOCIATION_NOT_CAUSAL_EFFECT",
        "PMI_INPUT_USES_KANSAS_CITY_FED_MANUFACTURING_PROXY",
    ]
    if weak_evidence:
        warning_codes.extend(
            [
                "WEAK_EVIDENCE_EXPERIMENTAL_RIDGE",
                (
                    "RIDGE_DID_NOT_RELIABLY_OUTPERFORM_BEST_"
                    "NAIVE_BENCHMARK"
                ),
                (
                    "STEP13_NAIVE_FALLBACK_REQUIRED_RETAINED_"
                    "AS_EVIDENCE_ONLY"
                ),
            ]
        )
        confidence_level = "WEAK_EVIDENCE_EXPERIMENTAL"
    else:
        warning_codes.append("LOW_CONFIDENCE_EXPERIMENTAL_RIDGE")
        confidence_level = "LOW_CONFIDENCE_EXPERIMENTAL"

    return {
        "confidence_classification": classification,
        "confidence_level": confidence_level,
        "validation_evidence": {
            "walk_forward_observations": int(
                decision["walk_forward_rows"]
            ),
            "mae": _finite_metric(
                decision["ridge_mae"],
                label=f"{target}.ridge_mae",
            ),
            "rmse": _finite_metric(
                decision["ridge_rmse"],
                label=f"{target}.ridge_rmse",
            ),
            "out_of_sample_r_squared": _finite_metric(
                decision["ridge_oos_r_squared"],
                label=f"{target}.ridge_oos_r_squared",
            ),
            "directional_accuracy": _finite_metric(
                decision["ridge_directional_accuracy"],
                label=f"{target}.ridge_directional_accuracy",
            ),
            "test_metrics_status": str(
                decision["test_metrics_status"]
            ),
        },
        "naive_benchmark_comparison": {
            "available_evaluation_strategies": [
                "naive_zero",
                "naive_training_mean",
                "naive_recent_trailing_21_observation",
            ],
            "best_mae_strategy": str(
                decision["best_naive_mae_strategy"]
            ),
            "best_mae": _finite_metric(
                decision["best_naive_mae"],
                label=f"{target}.best_naive_mae",
            ),
            "best_rmse_strategy": str(
                decision["best_naive_rmse_strategy"]
            ),
            "best_rmse": _finite_metric(
                decision["best_naive_rmse"],
                label=f"{target}.best_naive_rmse",
            ),
            "ridge_mae_improvement_pct": ridge_mae_improvement,
            "ridge_rmse_improvement_pct": ridge_rmse_improvement,
            "ridge_beat_best_naive_mae": ridge_mae_improvement > 0.0,
            "ridge_beat_best_naive_rmse": (
                ridge_rmse_improvement > 0.0
            ),
            "ridge_beat_best_naive_on_both_errors": (
                ridge_mae_improvement > 0.0
                and ridge_rmse_improvement > 0.0
            ),
            "serving_role": (
                "EVALUATION_BENCHMARK_ONLY_NOT_SUBSTITUTED"
            ),
            "substituted_for_ridge_output": False,
        },
        "scenario_responsiveness": {
            "responsive_inputs_any_fold": int(
                decision["responsive_inputs_any_fold"]
            ),
            "responsive_inputs_three_or_more_folds": int(
                decision["responsive_inputs_three_or_more_folds"]
            ),
            "responsive_inputs_all_folds": int(
                decision["responsive_inputs_all_folds"]
            ),
            "responsive_inputs_latest_fold": int(
                decision["responsive_inputs_latest_fold"]
            ),
            "directionally_consistent_responsive_inputs": int(
                decision["directionally_consistent_responsive_inputs"]
            ),
            "coefficient_sign_stable_all_folds": int(
                decision[
                    "scenario_coefficient_sign_stable_all_folds"
                ]
            ),
            "all_sensitivity_outputs_sensible": _as_bool(
                decision["all_sensitivity_outputs_sensible"],
                label=f"{target}.all_sensitivity_outputs_sensible",
            ),
            "limitations": [
                item
                for item in known_limitations
                if (
                    "shock" in item.lower()
                    or "responsive" in item.lower()
                    or "causal" in item.lower()
                )
            ],
        },
        "applicable_warnings": warning_codes,
        "known_limitations": known_limitations,
    }


@dataclass
class ResearchInferenceEngine:
    """In-memory, all-target Step 14 research interface."""

    feature_names: tuple[str, ...]
    state_feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    training: pd.DataFrame
    pipelines: dict[str, Pipeline]
    selected_alphas: dict[str, float]
    assessment: pd.DataFrame
    shock_ranges: dict[str, tuple[float, float]]
    state_ranges: dict[str, tuple[float, float]]
    target_ranges: dict[str, tuple[float, float]]
    latest_state: dict[str, float]
    latest_state_metadata: dict[str, Any]
    feature_mapping: pd.DataFrame
    guard: dict[str, Any]
    maximum_historical_active_flags: int

    def _resolve_release_flags(
        self,
        scenario: Mapping[str, float],
        release_flags: Mapping[str, Any] | None,
    ) -> tuple[dict[str, float], str]:
        if release_flags is None:
            return (
                _derived_release_flags(scenario),
                (
                    "INFERRED_ONCE_FROM_NONZERO_EVENT_SHOCKS_AND_FIXED_"
                    "BETWEEN_BASELINE_AND_SCENARIO"
                ),
            )
        return (
            _ordered_release_flags(release_flags),
            "CALLER_PROVIDED_AND_FIXED_BETWEEN_BASELINE_AND_SCENARIO",
        )

    def _input_range_warnings(
        self,
        scenario: Mapping[str, float],
        state: Mapping[str, float],
        release_flags: Mapping[str, float],
    ) -> list[dict[str, Any]]:
        warnings: list[dict[str, Any]] = []
        for contract_name in CONTRACT_SHOCK_NAMES:
            feature = CONTRACT_TO_HISTORICAL_FEATURE[contract_name]
            minimum, maximum = self.shock_ranges[contract_name]
            value = float(scenario[contract_name])
            if value < minimum or value > maximum:
                warnings.append(
                    {
                        "warning_type": "SCENARIO_SHOCK_OUTSIDE_TRAINING_RANGE",
                        "field": contract_name,
                        "historical_feature": feature,
                        "value": value,
                        "training_minimum": minimum,
                        "training_maximum": maximum,
                        "unit": SHOCK_UNITS[contract_name],
                        "action": "WARN_ONLY_NO_CLIPPING_OR_WINSORIZATION",
                    }
                )
        for feature in self.state_feature_names:
            minimum, maximum = self.state_ranges[feature]
            value = float(state[feature])
            if value < minimum or value > maximum:
                warnings.append(
                    {
                        "warning_type": "CURRENT_STATE_OUTSIDE_TRAINING_RANGE",
                        "field": feature,
                        "value": value,
                        "training_minimum": minimum,
                        "training_maximum": maximum,
                        "action": "WARN_ONLY_CURRENT_STATE_NOT_MODIFIED",
                    }
                )
        active_flags = int(sum(release_flags.values()))
        if active_flags > self.maximum_historical_active_flags:
            warnings.append(
                {
                    "warning_type": (
                        "RELEASE_FLAG_COMBINATION_OUTSIDE_TRAINING_SUPPORT"
                    ),
                    "active_release_flags": active_flags,
                    "historical_maximum": (
                        self.maximum_historical_active_flags
                    ),
                    "action": "WARN_ONLY_FLAGS_NOT_MODIFIED",
                }
            )
        for contract_name, flag_name in EVENT_INPUT_TO_FLAG.items():
            if (
                scenario[contract_name] != 0.0
                and release_flags[flag_name] == 0.0
            ):
                warnings.append(
                    {
                        "warning_type": "EVENT_SHOCK_WITHOUT_RELEASE_FLAG",
                        "field": contract_name,
                        "release_flag": flag_name,
                        "action": (
                            "WARN_ONLY_CALLER_PROVIDED_CONTEXT_PRESERVED"
                        ),
                    }
                )
        return warnings

    def _assemble_pair(
        self,
        scenario: Mapping[str, float],
        state: Mapping[str, float],
        release_flags: Mapping[str, float],
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        baseline_values: dict[str, float] = {}
        scenario_values: dict[str, float] = {}
        for contract_name in CONTRACT_SHOCK_NAMES:
            feature = CONTRACT_TO_HISTORICAL_FEATURE[contract_name]
            baseline_values[feature] = 0.0
            scenario_values[feature] = float(scenario[contract_name])
        baseline_values.update(release_flags)
        scenario_values.update(release_flags)
        baseline_values.update(state)
        scenario_values.update(state)

        if set(baseline_values) != set(self.feature_names):
            missing = sorted(set(self.feature_names) - set(baseline_values))
            extra = sorted(set(baseline_values) - set(self.feature_names))
            raise ScenarioValidationError(
                "Baseline feature assembly does not match the frozen schema; "
                f"missing={missing}; extra={extra}."
            )
        if set(scenario_values) != set(self.feature_names):
            raise ScenarioValidationError(
                "Scenario feature assembly does not match the frozen schema."
            )

        baseline = pd.DataFrame(
            [[baseline_values[name] for name in self.feature_names]],
            columns=list(self.feature_names),
        )
        scenario_frame = pd.DataFrame(
            [[scenario_values[name] for name in self.feature_names]],
            columns=list(self.feature_names),
        )
        changed = [
            name
            for name in self.feature_names
            if baseline.iloc[0][name] != scenario_frame.iloc[0][name]
        ]
        allowed_changes = set(CONTRACT_TO_HISTORICAL_FEATURE.values())
        isolation = {
            "feature_count": len(self.feature_names),
            "feature_order_exact": (
                tuple(baseline.columns) == self.feature_names
                and tuple(scenario_frame.columns) == self.feature_names
            ),
            "changed_features": changed,
            "only_shock_magnitudes_changed": (
                set(changed) <= allowed_changes
            ),
            "state_identical": bool(
                baseline[list(self.state_feature_names)].equals(
                    scenario_frame[list(self.state_feature_names)]
                )
            ),
            "release_flags_identical": bool(
                baseline[list(EVENT_FLAG_NAMES)].equals(
                    scenario_frame[list(EVENT_FLAG_NAMES)]
                )
            ),
            "baseline_row_sha256": _feature_row_hash(
                baseline,
                self.feature_names,
            ),
            "scenario_row_sha256": _feature_row_hash(
                scenario_frame,
                self.feature_names,
            ),
        }
        if not all(
            isolation[key]
            for key in (
                "feature_order_exact",
                "only_shock_magnitudes_changed",
                "state_identical",
                "release_flags_identical",
            )
        ):
            raise ScenarioValidationError(
                "Baseline/scenario isolation validation failed."
            )
        return baseline, scenario_frame, isolation

    def _prediction_range_warnings(
        self,
        target: str,
        baseline: float,
        scenario: float,
    ) -> list[dict[str, Any]]:
        row = self.assessment.loc[target]
        oos_minimum = float(row["prediction_minimum"])
        oos_maximum = float(row["prediction_maximum"])
        target_minimum, target_maximum = self.target_ranges[target]
        warnings: list[dict[str, Any]] = []
        for output_name, value in (
            ("baseline_prediction", baseline),
            ("scenario_prediction", scenario),
        ):
            if value < oos_minimum or value > oos_maximum:
                warnings.append(
                    {
                        "warning_type": (
                            "PREDICTION_OUTSIDE_STEP13_OOS_MODEL_RANGE"
                        ),
                        "target": target,
                        "output": output_name,
                        "value": value,
                        "step13_oos_minimum": oos_minimum,
                        "step13_oos_maximum": oos_maximum,
                    }
                )
            if value < target_minimum or value > target_maximum:
                warnings.append(
                    {
                        "warning_type": (
                            "PREDICTION_OUTSIDE_PRETEST_TARGET_RANGE"
                        ),
                        "target": target,
                        "output": output_name,
                        "value": value,
                        "pretest_target_minimum": target_minimum,
                        "pretest_target_maximum": target_maximum,
                    }
                )
        return warnings

    def predict(
        self,
        scenario_shocks: Mapping[str, Any],
        *,
        current_state: Mapping[str, Any] | None = None,
        release_flags: Mapping[str, Any] | None = None,
        out_of_range_policy: str = "warn",
    ) -> dict[str, Any]:
        """Return a complete all-target Ridge research result."""
        if out_of_range_policy not in {"warn", "error"}:
            raise ScenarioValidationError(
                "out_of_range_policy must be 'warn' or 'error'."
            )
        scenario = _ordered_finite_mapping(
            scenario_shocks,
            CONTRACT_SHOCK_NAMES,
            label="scenario_shocks",
        )
        if current_state is None:
            state = dict(self.latest_state)
            state_source = "LATEST_VALID_STEP7_FEATURE_ONLY_CACHE"
            state_metadata = dict(self.latest_state_metadata)
        else:
            state = _ordered_finite_mapping(
                current_state,
                self.state_feature_names,
                label="current_state",
            )
            state_source = "CALLER_PROVIDED_COMPLETE_STATE"
            state_metadata = {
                "sample_date": None,
                "state_cutoff_date": None,
            }
        flags, flag_source = self._resolve_release_flags(
            scenario,
            release_flags,
        )
        input_warnings = self._input_range_warnings(
            scenario,
            state,
            flags,
        )
        shock_range_warnings = [
            warning
            for warning in input_warnings
            if warning["warning_type"]
            == "SCENARIO_SHOCK_OUTSIDE_TRAINING_RANGE"
        ]
        if out_of_range_policy == "error" and shock_range_warnings:
            names = [warning["field"] for warning in shock_range_warnings]
            raise ScenarioValidationError(
                "Scenario shocks outside the model training range: "
                f"{names}. Values were not clipped."
            )

        baseline_row, scenario_row, isolation = self._assemble_pair(
            scenario,
            state,
            flags,
        )
        outputs: dict[str, dict[str, Any]] = {}
        prediction_warnings: list[dict[str, Any]] = []
        is_neutral = all(value == 0.0 for value in scenario.values())
        for target in self.target_names:
            decision = self.assessment.loc[target]
            unit = str(decision["unit"])
            evidence = _target_evidence_metadata(target, decision)
            common = {
                "target": target,
                "horizon_trading_days": 21,
                "unit": unit,
                "source": str(decision["source"]),
                "source_field": str(decision["source_field"]),
                "step13_gate": str(decision["classification"]),
                "step13_evaluation_classification": str(
                    decision["classification"]
                ),
                "confidence_classification": evidence[
                    "confidence_classification"
                ],
                "confidence_level": evidence["confidence_level"],
                "production_ready": False,
                "causal_effect_claimed": False,
            }
            pipeline = self.pipelines[target]
            baseline_prediction = float(
                pipeline.predict(baseline_row)[0]
            )
            scenario_prediction = float(
                pipeline.predict(scenario_row)[0]
            )
            incremental = scenario_prediction - baseline_prediction
            if not all(
                math.isfinite(value)
                for value in (
                    baseline_prediction,
                    scenario_prediction,
                    incremental,
                )
            ):
                raise ScenarioValidationError(
                    f"Nonfinite Ridge inference output for {target}."
                )
            if is_neutral:
                if abs(incremental) > 1e-12:
                    raise ScenarioValidationError(
                        f"Neutral scenario changed {target}."
                    )
                incremental = 0.0
            target_warnings = self._prediction_range_warnings(
                target,
                baseline_prediction,
                scenario_prediction,
            )
            prediction_warnings.extend(target_warnings)
            if target in LOW_CONFIDENCE_PROMOTED_TARGETS:
                inference_status = (
                    "LOW_CONFIDENCE_RIDGE_RESEARCH_INFERENCE"
                )
                confidence_warning = (
                    "Low-confidence experimental Ridge output. Step 13 "
                    "passed the prior minimum promotion gate, but the "
                    "prediction remains a noncausal research association."
                )
            else:
                inference_status = (
                    "WEAK_EVIDENCE_RIDGE_RESEARCH_INFERENCE"
                )
                confidence_warning = (
                    "Weak-evidence experimental Ridge output. Step 13 "
                    "classified this target as NAIVE_FALLBACK_REQUIRED "
                    "because Ridge did not reliably outperform its best "
                    "naive benchmark. Ridge remains callable for research; "
                    "the output is not validated for production or causal "
                    "interpretation."
                )
            outputs[target] = {
                **common,
                "inference_status": inference_status,
                "inference_supported": True,
                "model_name": str(decision["model_name"]),
                "model_family": "ridge",
                "model_structure": (
                    "independent_target_specific_pipeline"
                ),
                "history_strategy": "full_history",
                "selected_alpha": self.selected_alphas[target],
                "baseline_prediction": baseline_prediction,
                "scenario_prediction": scenario_prediction,
                "incremental_scenario_effect": incremental,
                "incremental_effect_status": (
                    "SUPPORTED_EXPERIMENTAL_PREDICTIVE_ASSOCIATION"
                ),
                "primary_output_strategy": "ridge",
                "naive_benchmark_substituted": False,
                "validation_evidence": evidence[
                    "validation_evidence"
                ],
                "naive_benchmark_comparison": evidence[
                    "naive_benchmark_comparison"
                ],
                "scenario_responsiveness": evidence[
                    "scenario_responsiveness"
                ],
                "applicable_warnings": evidence[
                    "applicable_warnings"
                ],
                "known_limitations": evidence["known_limitations"],
                "prediction_range_warnings": target_warnings,
                "warning": confidence_warning,
            }

        result = {
            "schema_version": "scenario_research_inference_v2",
            "inference_status": (
                "RESEARCH_ONLY_ALL_TARGET_RIDGE_PREDICTIONS"
            ),
            "research_only": True,
            "production_ready": False,
            "horizon_trading_days": 21,
            "scenario_shocks": scenario,
            "scenario_input_units": dict(SHOCK_UNITS),
            "release_flags": flags,
            "release_flag_source": flag_source,
            "release_flag_mapping": dict(EVENT_INPUT_TO_FLAG),
            "context": {
                **state_metadata,
                "state_source": state_source,
                "state_feature_count": len(self.state_feature_names),
                "state_features": state,
            },
            "definitions": {
                "baseline_prediction": (
                    "Latest fixed state plus neutral shock magnitudes and "
                    "the same release flags used by the scenario row."
                ),
                "scenario_prediction": (
                    "The same fixed state and release flags plus the eight "
                    "user-entered shock magnitudes."
                ),
                "incremental_scenario_effect": (
                    "Scenario prediction minus baseline prediction for each "
                    "of the 12 independent target-specific Ridge models."
                ),
            },
            "feature_assembly": isolation,
            "input_range_policy": {
                "mode": out_of_range_policy,
                "clipping_performed": False,
                "winsorization_performed": False,
            },
            "input_range_warnings": input_warnings,
            "prediction_range_warnings": prediction_warnings,
            "targets": outputs,
            "asset_outputs": {
                target: outputs[target] for target in ASSET_TARGETS
            },
            "sector_outputs": {
                target: outputs[target] for target in SECTOR_TARGETS
            },
            "target_confidence_counts": {
                "ridge_pipeline_count": len(self.pipelines),
                "low_confidence_promotion_gate_passed": len(
                    LOW_CONFIDENCE_PROMOTED_TARGETS
                ),
                "weak_evidence_ridge": len(
                    WEAK_EVIDENCE_RIDGE_TARGETS
                ),
                "naive_benchmark_substitutions": 0,
            },
            "locked_final_test": {
                "target_rows_parsed": 0,
                "access_count": 0,
                "target_values_available": False,
                "latest_state_reads_feature_only_cache": True,
            },
            "global_warning": (
                "Research only. All 12 outputs are Ridge predictive "
                "associations, not causal effects or production forecasts. "
                "Nine models did not reliably outperform their best naive "
                "evaluation benchmark."
            ),
        }
        return _json_safe(result)


def _load_latest_state(
    state_feature_names: Sequence[str],
    *,
    state_path: Path = STATE_PATH,
) -> tuple[dict[str, float], dict[str, Any]]:
    frame = pd.read_csv(state_path)
    required = {
        "sample_date",
        "state_cutoff_date",
        *state_feature_names,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ScenarioValidationError(
            f"Step 7 state cache is missing columns: {missing}."
        )
    if frame["sample_date"].duplicated().any():
        raise ScenarioValidationError(
            "Step 7 state cache contains duplicate sample dates."
        )
    frame = frame.sort_values("sample_date", kind="stable")
    numeric = frame[list(state_feature_names)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    complete = (
        numeric.notna().all(axis=1)
        & np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    )
    if not complete.any():
        raise ScenarioValidationError(
            "Step 7 state cache has no complete inference row."
        )
    position = int(np.flatnonzero(complete.to_numpy())[-1])
    row = frame.iloc[position]
    state = {
        name: float(numeric.iloc[position][name])
        for name in state_feature_names
    }
    metadata = {
        "sample_date": str(row["sample_date"]),
        "state_cutoff_date": str(row["state_cutoff_date"]),
        "state_cache_path": relative_path(state_path),
        "state_cache_contains_targets": False,
        "latest_valid_state_rule": (
            "Last chronological Step 7 row complete and finite for all 45 "
            "state features in the frozen model schema."
        ),
    }
    return state, metadata


def _load_feature_mapping(
    feature_names: Sequence[str],
    *,
    catalog_path: Path = COLUMN_CATALOG_PATH,
) -> pd.DataFrame:
    catalog = pd.read_csv(catalog_path)
    catalog = catalog.loc[
        catalog["column_role"].eq("feature")
        & catalog["storage_artifact"].eq(
            "scenario_model_matrix_raw.csv"
        )
    ].drop_duplicates("column_name")
    lookup = catalog.set_index("column_name")
    rows: list[dict[str, Any]] = []
    historical_to_contract = {
        feature: contract
        for contract, feature in CONTRACT_TO_HISTORICAL_FEATURE.items()
    }
    flag_to_contract = {
        flag: contract
        for contract, flag in EVENT_INPUT_TO_FLAG.items()
    }
    for position, feature in enumerate(feature_names):
        if feature not in lookup.index:
            raise ScenarioValidationError(
                f"Column catalog does not describe {feature}."
            )
        source = lookup.loc[feature]
        if feature in historical_to_contract:
            role = "scenario_shock"
            contract_input = historical_to_contract[feature]
            baseline_policy = "set to neutral 0"
            pair_invariant = False
        elif feature in EVENT_FLAG_NAMES:
            role = "release_flag"
            contract_input = flag_to_contract.get(feature, "")
            baseline_policy = "fixed between baseline and scenario"
            pair_invariant = True
        else:
            role = "current_state"
            contract_input = ""
            baseline_policy = "fixed between baseline and scenario"
            pair_invariant = True
        rows.append(
            {
                "feature_position": position,
                "feature_name": feature,
                "inference_role": role,
                "contract_input": contract_input,
                "unit": source["unit"],
                "source": source["source"],
                "definition_or_formula": source[
                    "definition_or_formula"
                ],
                "point_in_time_rule": source["point_in_time_rule"],
                "baseline_policy": baseline_policy,
                "pair_invariant": pair_invariant,
                "included_in_frozen_58_feature_schema": True,
            }
        )
    return pd.DataFrame(rows)


def build_research_inference_engine(
    *,
    configurations_path: Path = STEP13_CONFIGURATIONS_PATH,
    assessment_path: Path = STEP13_ASSESSMENT_PATH,
) -> ResearchInferenceEngine:
    """Fit all 12 frozen target-specific Ridge pipelines."""
    training, feature_names, target_names, guard = load_all_pretest_rows()
    if feature_names != tuple(FINAL_FEATURE_NAMES):
        raise ScenarioValidationError(
            "Frozen feature names or order changed."
        )
    if len(feature_names) != 58 or len(target_names) != 12:
        raise ScenarioValidationError(
            "Step 14 requires 58 features and 12 targets."
        )
    if len(training) != 464:
        raise ScenarioValidationError(
            "Step 14 requires exactly 464 authorized pretest rows."
        )
    if target_names != ALL_RIDGE_TARGETS:
        raise ScenarioValidationError(
            "Frozen target names or order changed."
        )
    if set(EXPECTED_TARGET_CONFIGURATIONS) != set(target_names):
        raise ScenarioValidationError(
            "Expected target configuration contract is incomplete."
        )
    configurations = pd.read_csv(configurations_path).set_index("target")
    assessment = pd.read_csv(assessment_path).set_index("target")
    if set(configurations.index) != set(target_names):
        raise ScenarioValidationError(
            "Step 13 configuration targets changed."
        )
    if set(assessment.index) != set(target_names):
        raise ScenarioValidationError(
            "Step 13 assessment targets changed."
        )

    selected_alphas: dict[str, float] = {}
    pipelines: dict[str, Pipeline] = {}
    for target in target_names:
        configuration = configurations.loc[target]
        decision = assessment.loc[target]
        expected = EXPECTED_TARGET_CONFIGURATIONS[target]
        if configuration["configuration_status"] != (
            "FROZEN_FROM_PRETEST_DATA_FOR_STEP_14_RESEARCH"
        ):
            raise ScenarioValidationError(
                f"Step 13 configuration is not frozen for {target}."
            )
        if _as_bool(
            configuration["model_binary_exported"],
            label=f"{target}.model_binary_exported",
        ):
            raise ScenarioValidationError(
                "Step 14 does not accept exported model binaries."
            )
        if _as_bool(
            configuration["locked_final_test_used"],
            label=f"{target}.locked_final_test_used",
        ):
            raise ScenarioValidationError(
                f"Step 13 configuration accessed locked data for {target}."
            )
        if str(configuration["model_family"]) != "ridge":
            raise ScenarioValidationError(
                f"Step 13 model family changed for {target}."
            )
        if str(configuration["history_strategy"]) != "full_history":
            raise ScenarioValidationError(
                f"Step 13 history strategy changed for {target}."
            )
        if int(configuration["training_rows"]) != len(training):
            raise ScenarioValidationError(
                f"Step 13 training-row count changed for {target}."
            )
        alpha = float(configuration["selected_alpha"])
        if alpha != float(expected["alpha"]):
            raise ScenarioValidationError(
                f"Frozen Ridge alpha changed for {target}: {alpha}."
            )
        if str(configuration["unit"]) != str(expected["unit"]):
            raise ScenarioValidationError(
                f"Frozen target unit changed for {target}."
            )
        if str(decision["unit"]) != str(expected["unit"]):
            raise ScenarioValidationError(
                f"Step 13 assessment unit changed for {target}."
            )
        if float(decision["final_pretest_alpha"]) != alpha:
            raise ScenarioValidationError(
                f"Configuration/assessment alpha mismatch for {target}."
            )
        if str(decision["classification"]) != str(
            expected["classification"]
        ):
            raise ScenarioValidationError(
                f"Step 13 classification changed for {target}."
            )
        pipeline = make_ridge_pipeline(alpha)
        pipeline.fit(
            training[list(feature_names)],
            training[target],
        )
        selected_alphas[target] = alpha
        pipelines[target] = pipeline
    if tuple(pipelines) != target_names:
        raise ScenarioValidationError(
            "All 12 Ridge pipelines were not fitted in frozen target order."
        )

    shock_ranges = {
        contract: (
            float(
                training[
                    CONTRACT_TO_HISTORICAL_FEATURE[contract]
                ].min()
            ),
            float(
                training[
                    CONTRACT_TO_HISTORICAL_FEATURE[contract]
                ].max()
            ),
        )
        for contract in CONTRACT_SHOCK_NAMES
    }
    state_feature_names = tuple(
        feature
        for feature in feature_names
        if feature
        not in {
            *CONTRACT_TO_HISTORICAL_FEATURE.values(),
            *EVENT_FLAG_NAMES,
        }
    )
    if len(state_feature_names) != 45:
        raise ScenarioValidationError(
            f"Expected 45 state features, found {len(state_feature_names)}."
        )
    state_ranges = {
        feature: (
            float(training[feature].min()),
            float(training[feature].max()),
        )
        for feature in state_feature_names
    }
    target_ranges = {
        target: (
            float(training[target].min()),
            float(training[target].max()),
        )
        for target in target_names
    }
    latest_state, latest_state_metadata = _load_latest_state(
        state_feature_names
    )
    feature_mapping = _load_feature_mapping(feature_names)
    maximum_historical_active_flags = int(
        training[list(EVENT_FLAG_NAMES)].sum(axis=1).max()
    )
    guard = {
        **guard,
        "final_test_target_values_available_to_step_14": False,
        "final_test_access_count": 0,
        "latest_state_source_contains_targets": False,
    }
    return ResearchInferenceEngine(
        feature_names=feature_names,
        state_feature_names=state_feature_names,
        target_names=target_names,
        training=training,
        pipelines=pipelines,
        selected_alphas=selected_alphas,
        assessment=assessment,
        shock_ranges=shock_ranges,
        state_ranges=state_ranges,
        target_ranges=target_ranges,
        latest_state=latest_state,
        latest_state_metadata=latest_state_metadata,
        feature_mapping=feature_mapping,
        guard=guard,
        maximum_historical_active_flags=(
            maximum_historical_active_flags
        ),
    )


def predict_scenario(
    scenario_shocks: Mapping[str, Any],
    *,
    current_state: Mapping[str, Any] | None = None,
    release_flags: Mapping[str, Any] | None = None,
    out_of_range_policy: str = "warn",
    engine: ResearchInferenceEngine | None = None,
) -> dict[str, Any]:
    """Simple callable Step 14 research interface."""
    active_engine = engine or build_research_inference_engine()
    return active_engine.predict(
        scenario_shocks,
        current_state=current_state,
        release_flags=release_flags,
        out_of_range_policy=out_of_range_policy,
    )


def _model_gate_audit(
    engine: ResearchInferenceEngine,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target in engine.target_names:
        assessment = engine.assessment.loc[target]
        evidence = _target_evidence_metadata(target, assessment)
        comparison = evidence["naive_benchmark_comparison"]
        rows.append(
            {
                "target": target,
                "unit": assessment["unit"],
                "step13_gate": assessment["classification"],
                "step14_inference_mode": "TARGET_SPECIFIC_RIDGE",
                "confidence_level": evidence["confidence_level"],
                "ridge_pipeline_fitted": True,
                "selected_alpha": engine.selected_alphas[target],
                "primary_output_strategy": "ridge",
                "best_naive_mae_benchmark": comparison[
                    "best_mae_strategy"
                ],
                "ridge_beat_best_naive_mae": comparison[
                    "ridge_beat_best_naive_mae"
                ],
                "ridge_beat_best_naive_rmse": comparison[
                    "ridge_beat_best_naive_rmse"
                ],
                "naive_benchmark_substituted": False,
                "baseline_numeric": True,
                "scenario_prediction_numeric": True,
                "incremental_effect_numeric": True,
                "warning_codes": "|".join(
                    evidence["applicable_warnings"]
                ),
                "production_ready": False,
                "causal_effect_claimed": False,
            }
        )
    return pd.DataFrame(rows)


def _input_range_table(
    engine: ResearchInferenceEngine,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for contract in CONTRACT_SHOCK_NAMES:
        minimum, maximum = engine.shock_ranges[contract]
        rows.append(
            {
                "input_role": "scenario_shock",
                "contract_input": contract,
                "feature_name": (
                    CONTRACT_TO_HISTORICAL_FEATURE[contract]
                ),
                "unit": SHOCK_UNITS[contract],
                "training_minimum": minimum,
                "training_maximum": maximum,
                "latest_state_value": np.nan,
                "out_of_range_action": (
                    "warn by default; optional error; never clip"
                ),
            }
        )
    mapping = engine.feature_mapping.set_index("feature_name")
    for feature in engine.state_feature_names:
        minimum, maximum = engine.state_ranges[feature]
        rows.append(
            {
                "input_role": "current_state",
                "contract_input": "",
                "feature_name": feature,
                "unit": mapping.loc[feature, "unit"],
                "training_minimum": minimum,
                "training_maximum": maximum,
                "latest_state_value": engine.latest_state[feature],
                "out_of_range_action": (
                    "warn; preserve observed current state"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_inference_artifacts(
    *,
    engine: ResearchInferenceEngine | None = None,
    example_scenario: Mapping[str, Any] = EXAMPLE_SCENARIO,
) -> dict[str, Any]:
    """Build and validate the Step 14 notebook/research outputs."""
    active_engine = engine or build_research_inference_engine()
    example = active_engine.predict(example_scenario)
    neutral = active_engine.predict(NEUTRAL_SCENARIO)
    repeated = active_engine.predict(example_scenario)

    out_of_range = dict(EXAMPLE_SCENARIO)
    fed_maximum = active_engine.shock_ranges[
        "fed_funds_rate_change_bps"
    ][1]
    out_of_range["fed_funds_rate_change_bps"] = fed_maximum + 1.0
    range_result = active_engine.predict(out_of_range)
    strict_range_rejected = False
    try:
        active_engine.predict(
            out_of_range,
            out_of_range_policy="error",
        )
    except ScenarioValidationError:
        strict_range_rejected = True

    missing_input_rejected = False
    incomplete = dict(EXAMPLE_SCENARIO)
    incomplete.pop("vix_change_points")
    try:
        active_engine.predict(incomplete)
    except ScenarioValidationError:
        missing_input_rejected = True

    ridge_targets = {
        target
        for target, output in example["targets"].items()
        if (
            output["inference_supported"]
            and output["model_family"] == "ridge"
            and output["primary_output_strategy"] == "ridge"
        )
    }
    validations = {
        "exact_eight_input_names": (
            tuple(example["scenario_shocks"])
            == tuple(CONTRACT_SHOCK_NAMES)
        ),
        "exact_five_release_flags": (
            tuple(example["release_flags"]) == tuple(EVENT_FLAG_NAMES)
        ),
        "exact_58_feature_order": bool(
            example["feature_assembly"]["feature_order_exact"]
            and active_engine.feature_names
            == tuple(FINAL_FEATURE_NAMES)
        ),
        "exact_45_state_features": (
            len(active_engine.state_feature_names) == 45
        ),
        "latest_state_is_feature_only": bool(
            example["context"]["state_cache_contains_targets"] is False
        ),
        "baseline_scenario_state_isolated": bool(
            example["feature_assembly"]["state_identical"]
        ),
        "baseline_scenario_flags_isolated": bool(
            example["feature_assembly"]["release_flags_identical"]
        ),
        "only_shock_magnitudes_change": bool(
            example["feature_assembly"][
                "only_shock_magnitudes_changed"
            ]
        ),
        "neutral_all_target_incremental_effects_zero": all(
            neutral["targets"][target][
                "incremental_scenario_effect"
            ]
            == 0.0
            for target in active_engine.target_names
        ),
        "exactly_twelve_independent_ridge_pipelines_fitted": (
            tuple(active_engine.pipelines)
            == active_engine.target_names
            and len(
                {
                    id(pipeline)
                    for pipeline in active_engine.pipelines.values()
                }
            )
            == 12
        ),
        "all_twelve_targets_use_ridge_inference": (
            ridge_targets == set(active_engine.target_names)
        ),
        "all_selected_alphas_match_frozen_configurations": all(
            active_engine.selected_alphas[target]
            == float(EXPECTED_TARGET_CONFIGURATIONS[target]["alpha"])
            and example["targets"][target]["selected_alpha"]
            == float(EXPECTED_TARGET_CONFIGURATIONS[target]["alpha"])
            for target in active_engine.target_names
        ),
        "all_target_units_match_frozen_configurations": all(
            example["targets"][target]["unit"]
            == EXPECTED_TARGET_CONFIGURATIONS[target]["unit"]
            for target in active_engine.target_names
        ),
        "all_preprocessing_fitted_on_authorized_rows_only": all(
            int(
                np.max(
                    np.asarray(
                        pipeline.named_steps[
                            "scaler"
                        ].n_samples_seen_
                    )
                )
            )
            == len(active_engine.training)
            for pipeline in active_engine.pipelines.values()
        ),
        "confidence_metadata_matches_step13_evidence": all(
            example["targets"][target][
                "confidence_classification"
            ]
            == EXPECTED_TARGET_CONFIGURATIONS[target][
                "classification"
            ]
            and example["targets"][target][
                "validation_evidence"
            ]["mae"]
            == float(
                active_engine.assessment.loc[target, "ridge_mae"]
            )
            and example["targets"][target][
                "validation_evidence"
            ]["rmse"]
            == float(
                active_engine.assessment.loc[target, "ridge_rmse"]
            )
            and example["targets"][target][
                "validation_evidence"
            ]["out_of_sample_r_squared"]
            == float(
                active_engine.assessment.loc[
                    target,
                    "ridge_oos_r_squared",
                ]
            )
            and example["targets"][target][
                "validation_evidence"
            ]["directional_accuracy"]
            == float(
                active_engine.assessment.loc[
                    target,
                    "ridge_directional_accuracy",
                ]
            )
            for target in active_engine.target_names
        ),
        "weak_models_preserve_explicit_warnings": all(
            (
                "WEAK_EVIDENCE_EXPERIMENTAL_RIDGE"
                in example["targets"][target][
                    "applicable_warnings"
                ]
                and "did not reliably outperform"
                in example["targets"][target]["warning"]
            )
            for target in WEAK_EVIDENCE_RIDGE_TARGETS
        ),
        "naive_benchmarks_are_comparison_only": all(
            output["naive_benchmark_substituted"] is False
            and output["naive_benchmark_comparison"][
                "substituted_for_ridge_output"
            ]
            is False
            and output["naive_benchmark_comparison"][
                "serving_role"
            ]
            == "EVALUATION_BENCHMARK_ONLY_NOT_SUBSTITUTED"
            for output in example["targets"].values()
        ),
        "all_twelve_targets_returned": (
            tuple(example["targets"]) == active_engine.target_names
        ),
        "all_numeric_outputs_finite": all(
            math.isfinite(
                float(output["baseline_prediction"])
            )
            and math.isfinite(
                float(output["scenario_prediction"])
            )
            and math.isfinite(
                float(output["incremental_scenario_effect"])
            )
            for output in example["targets"].values()
        ),
        "out_of_range_warns_without_clipping": any(
            warning["warning_type"]
            == "SCENARIO_SHOCK_OUTSIDE_TRAINING_RANGE"
            and warning["field"] == "fed_funds_rate_change_bps"
            for warning in range_result["input_range_warnings"]
        ),
        "strict_out_of_range_policy_rejects": strict_range_rejected,
        "missing_input_rejected": missing_input_rejected,
        "deterministic_predictions": (
            json.dumps(example, sort_keys=True, allow_nan=False)
            == json.dumps(repeated, sort_keys=True, allow_nan=False)
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
        "no_model_binaries_exported": True,
        "no_xgboost_steps_15_16_17_or_application_work": True,
    }
    if not all(validations.values()):
        failed = [
            name for name, passed in validations.items() if not passed
        ]
        raise ScenarioValidationError(
            f"Step 14 acceptance validations failed: {failed}."
        )

    model_gates = _model_gate_audit(active_engine)
    input_ranges = _input_range_table(active_engine)
    summary = {
        "schema_version": "scenario_research_inference_summary_v2",
        "callable_entry_point": (
            "scenario_modeling.src.scenario_inference.predict_scenario"
        ),
        "engine_entry_point": (
            "scenario_modeling.src.scenario_inference."
            "build_research_inference_engine"
        ),
        "feature_count": len(active_engine.feature_names),
        "scenario_shock_count": len(CONTRACT_SHOCK_NAMES),
        "release_flag_count": len(EVENT_FLAG_NAMES),
        "state_feature_count": len(active_engine.state_feature_names),
        "target_count": len(active_engine.target_names),
        "ridge_pipeline_count": len(active_engine.pipelines),
        "ridge_targets": list(active_engine.target_names),
        "low_confidence_promoted_targets": list(
            LOW_CONFIDENCE_PROMOTED_TARGETS
        ),
        "weak_evidence_ridge_targets": list(
            WEAK_EVIDENCE_RIDGE_TARGETS
        ),
        "naive_benchmark_strategies": [
            "naive_zero",
            "naive_training_mean",
            "naive_recent_trailing_21_observation",
        ],
        "naive_benchmark_substitution_count": 0,
        "selected_target_configurations": {
            target: {
                "alpha": active_engine.selected_alphas[target],
                "unit": str(
                    active_engine.assessment.loc[target, "unit"]
                ),
                "step13_evaluation_classification": str(
                    active_engine.assessment.loc[
                        target,
                        "classification",
                    ]
                ),
            }
            for target in active_engine.target_names
        },
        "latest_state_metadata": active_engine.latest_state_metadata,
        "model_training_rows": len(active_engine.training),
        "model_training_first_sample": str(
            active_engine.training["sample_date"].iloc[0]
        ),
        "model_training_last_sample": str(
            active_engine.training["sample_date"].iloc[-1]
        ),
        "locked_final_test": {
            "target_rows_parsed": 0,
            "access_count": 0,
            "target_values_available": False,
        },
        "example_scenario": dict(example["scenario_shocks"]),
        "example_release_flags": dict(example["release_flags"]),
        "example_input_warning_count": len(
            example["input_range_warnings"]
        ),
        "example_prediction_warning_count": len(
            example["prediction_range_warnings"]
        ),
        "validations": validations,
        "completion_gate": "STEP_14_ALL_TARGET_RIDGE_INFERENCE_COMPLETE",
        "scope_boundary": (
            "Corrected Step 14 only: no all-target XGBoost, Steps 15-17, "
            "model binaries, production artifacts, API, Supabase, backend, "
            "frontend, or servers."
        ),
    }
    return {
        "feature_mapping": active_engine.feature_mapping,
        "input_ranges": input_ranges,
        "model_gate_audit": model_gates,
        "example_result": example,
        "neutral_result": neutral,
        "summary": summary,
    }


def persist_inference_artifacts(
    artifacts: Mapping[str, Any],
    output_dir: Path = INFERENCE_OUTPUT_ROOT,
) -> dict[str, str]:
    """Persist research evidence only; never serialize fitted pipelines."""
    paths = {
        "feature_mapping": output_dir / "feature_mapping.csv",
        "input_ranges": output_dir / "input_ranges.csv",
        "model_gate_audit": output_dir / "model_gate_audit.csv",
        "example_result": output_dir / "example_result.json",
        "neutral_result": output_dir / "neutral_result.json",
        "summary": output_dir / "scenario_inference_summary.json",
    }
    csv_texts = {
        name: frame_to_csv_text(artifacts[name])
        for name in (
            "feature_mapping",
            "input_ranges",
            "model_gate_audit",
        )
    }
    json_texts = {
        name: (
            json.dumps(
                _json_safe(artifacts[name]),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        for name in ("example_result", "neutral_result")
    }
    summary = dict(artifacts["summary"])
    content_hashes = {
        **{
            name: hashlib.sha256(text.encode()).hexdigest()
            for name, text in csv_texts.items()
        },
        **{
            name: hashlib.sha256(text.encode()).hexdigest()
            for name, text in json_texts.items()
        },
    }
    summary["output_files"] = {
        name: {
            "path": relative_path(path),
            "sha256": content_hashes.get(name),
        }
        for name, path in paths.items()
    }
    for name, text in csv_texts.items():
        write_text_atomic(paths[name], text)
    for name, text in json_texts.items():
        write_text_atomic(paths[name], text)
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
            "Run corrected all-target Step 14 Ridge research inference."
        )
    )
    parser.add_argument(
        "--scenario-json",
        default=json.dumps(EXAMPLE_SCENARIO),
        help="JSON object containing exactly the eight scenario inputs.",
    )
    parser.add_argument(
        "--out-of-range-policy",
        choices=("warn", "error"),
        default="warn",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print a result without persisting evidence.",
    )
    args = parser.parse_args(argv)
    parsed = json.loads(args.scenario_json)
    if not isinstance(parsed, dict):
        raise ScenarioValidationError(
            "--scenario-json must decode to an object."
        )
    engine = build_research_inference_engine()
    result = engine.predict(
        parsed,
        out_of_range_policy=args.out_of_range_policy,
    )
    artifacts = build_inference_artifacts(
        engine=engine,
        example_scenario=parsed,
    )
    paths = (
        {}
        if args.dry_run
        else persist_inference_artifacts(artifacts)
    )
    print(
        json.dumps(
            {
                "completion_gate": artifacts["summary"][
                    "completion_gate"
                ],
                "state_sample_date": result["context"]["sample_date"],
                "ridge_pipeline_count": len(engine.pipelines),
                "ridge_targets": list(engine.target_names),
                "low_confidence_promoted_targets": list(
                    LOW_CONFIDENCE_PROMOTED_TARGETS
                ),
                "weak_evidence_ridge_targets": list(
                    WEAK_EVIDENCE_RIDGE_TARGETS
                ),
                "naive_benchmark_substitution_count": 0,
                "input_warning_count": len(
                    result["input_range_warnings"]
                ),
                "prediction_warning_count": len(
                    result["prediction_range_warnings"]
                ),
                "locked_final_test_access_count": result[
                    "locked_final_test"
                ]["access_count"],
                "outputs": paths,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
