"""Lean, reloadable Scenario Playground Ridge inference runtime.

This module is intentionally independent from the training and matrix-building
modules.  It loads a trusted Step 17 bundle, validates its integrity, and
serves the frozen 12-target Ridge research workflow without fitting models or
reading repository data.

Joblib artifacts use Python pickle semantics.  Only load bundles obtained from
a trusted source after validating their accompanying SHA-256 manifest.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date
from numbers import Real
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BUNDLE_PAYLOAD_FILENAME = "scenario_model_bundle.joblib"
BUNDLE_MANIFEST_FILENAME = "model_manifest.json"
BUNDLE_CHECKSUMS_FILENAME = "artifact_checksums.json"
RUNTIME_SCHEMA_VERSION = "scenario_model_runtime_v1"
PAYLOAD_SCHEMA_VERSION = "scenario_model_bundle_payload_v1"
EXPLANATION_SCHEMA_VERSION = "scenario_explanation_metadata_v1"
CONFIDENCE_SCHEMA_VERSION = "scenario_confidence_metadata_v1"
RECONCILIATION_TOLERANCE = 1e-9
TOP_DRIVER_COUNT = 3
WARNING_CATEGORIES = (
    "input_range",
    "state_extrapolation",
    "model_performance",
    "scenario_responsiveness",
    "regime",
    "state_freshness",
)
XGBOOST_AGREEMENT_STATUS = "UNAVAILABLE_INTENTIONALLY_POSTPONED"


class BundleValidationError(ValueError):
    """Raised when a model bundle is incomplete, inconsistent, or corrupted."""


class ScenarioValidationError(ValueError):
    """Raised when runtime scenario inputs do not satisfy the frozen contract."""


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _feature_row_hash(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
) -> str:
    values = [
        [name, float(frame.iloc[0][name])]
        for name in feature_names
    ]
    return hashlib.sha256(
        json.dumps(values, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


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
            raise ScenarioValidationError(f"{label}.{name} must be finite.")
        ordered[name] = number
    return ordered


def _ordered_release_flags(
    values: Mapping[str, Any],
    names: Sequence[str],
) -> dict[str, float]:
    if not isinstance(values, Mapping):
        raise ScenarioValidationError("release_flags must be a mapping.")
    required = tuple(names)
    missing = [name for name in required if name not in values]
    extra = [name for name in values if name not in required]
    if missing or extra:
        raise ScenarioValidationError(
            "release_flags keys do not match the five-feature schema; "
            f"missing={missing}; extra={extra}."
        )
    ordered: dict[str, float] = {}
    for name in required:
        value = values[name]
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


def _empirical_percentile(values: np.ndarray, value: float) -> float:
    below = float(np.sum(values < value))
    equal = float(np.sum(values == value))
    return 100.0 * (below + 0.5 * equal) / float(len(values))


def _current_vix_regime(value: float) -> str:
    if value < 20.0:
        return "low_vix_lt_20"
    if value < 30.0:
        return "medium_vix_20_to_30"
    return "high_vix_ge_30"


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
        raise BundleValidationError(
            f"Unknown warning category in bundle runtime: {category}."
        )
    return {
        "target": target,
        "category": category,
        "code": code,
        "severity": severity,
        "message": message,
        "evidence": dict(evidence or {}),
        "causal_effect_claimed": False,
    }


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() == "true"


def _resolve_bundle_directory(bundle_path: str | Path) -> Path:
    path = Path(bundle_path).expanduser().resolve()
    if path.is_file():
        if path.name not in {
            BUNDLE_PAYLOAD_FILENAME,
            BUNDLE_MANIFEST_FILENAME,
            BUNDLE_CHECKSUMS_FILENAME,
        }:
            raise BundleValidationError(
                "bundle_path must be the bundle directory or one of its "
                "primary bundle files."
            )
        return path.parent
    if not path.is_dir():
        raise BundleValidationError(f"Bundle path does not exist: {path}")
    return path


def validate_bundle_hashes(bundle_path: str | Path) -> dict[str, Any]:
    """Validate every file covered by the bundle's SHA-256 checksum index."""
    directory = _resolve_bundle_directory(bundle_path)
    checksum_path = directory / BUNDLE_CHECKSUMS_FILENAME
    if not checksum_path.exists():
        raise BundleValidationError(
            f"Bundle checksum index is missing: {checksum_path}"
        )
    try:
        payload = json.loads(checksum_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleValidationError(
            f"Bundle checksum index is unreadable: {checksum_path}"
        ) from exc
    if payload.get("algorithm") != "sha256":
        raise BundleValidationError("Bundle checksum algorithm must be sha256.")
    hashes = payload.get("files")
    if not isinstance(hashes, dict) or not hashes:
        raise BundleValidationError("Bundle checksum index has no files.")
    checked: dict[str, str] = {}
    for relative_name, expected in hashes.items():
        relative = Path(relative_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise BundleValidationError(
                f"Unsafe relative path in checksum index: {relative_name}"
            )
        path = (directory / relative).resolve()
        try:
            path.relative_to(directory)
        except ValueError as exc:
            raise BundleValidationError(
                f"Checksum path escapes bundle directory: {relative_name}"
            ) from exc
        if not path.is_file():
            raise BundleValidationError(
                f"Hashed bundle file is missing: {relative_name}"
            )
        actual = _sha256_file(path)
        if actual != expected:
            raise BundleValidationError(
                f"SHA-256 mismatch for bundle file: {relative_name}"
            )
        checked[relative_name] = actual
    return {
        "algorithm": "sha256",
        "validated_file_count": len(checked),
        "all_hashes_match": True,
        "files": checked,
    }


@dataclass
class ScenarioModelRuntime:
    """Fitted, inference-only all-target Ridge runtime.

    The object has no training method and retains no target labels or complete
    training DataFrame.  Its feature-only diagnostic reference is bundled only
    to preserve the accepted Step 16 range and neighborhood checks.
    """

    bundle_directory: Path
    manifest: dict[str, Any]
    pipelines: dict[str, Pipeline]
    feature_names: tuple[str, ...]
    state_feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    scenario_input_names: tuple[str, ...]
    event_flag_names: tuple[str, ...]
    scenario_input_mapping: dict[str, str]
    scenario_input_units: dict[str, str]
    release_flag_mapping: dict[str, str]
    target_registry: dict[str, dict[str, Any]]
    feature_mapping: tuple[dict[str, Any], ...]
    default_state: dict[str, float]
    default_state_metadata: dict[str, Any]
    shock_ranges: dict[str, tuple[float, float]]
    state_ranges: dict[str, tuple[float, float]]
    maximum_historical_active_flags: int
    diagnostic_reference: dict[str, Any]
    explanation_configuration: dict[str, Any]
    confidence_configuration: dict[str, Any]
    bundle_metadata: dict[str, Any]

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        bundle_directory: Path,
        manifest: Mapping[str, Any],
    ) -> "ScenarioModelRuntime":
        if payload.get("payload_schema_version") != PAYLOAD_SCHEMA_VERSION:
            raise BundleValidationError(
                "Unsupported Scenario Playground bundle payload schema."
            )
        forbidden = {
            "training",
            "training_frame",
            "target_values",
            "target_labels",
            "training_targets",
        }
        present = sorted(forbidden & set(payload))
        if present:
            raise BundleValidationError(
                f"Lean bundle contains forbidden training fields: {present}."
            )
        engine = cls(
            bundle_directory=bundle_directory,
            manifest=dict(manifest),
            pipelines=dict(payload["pipelines"]),
            feature_names=tuple(payload["feature_names"]),
            state_feature_names=tuple(payload["state_feature_names"]),
            target_names=tuple(payload["target_names"]),
            scenario_input_names=tuple(payload["scenario_input_names"]),
            event_flag_names=tuple(payload["event_flag_names"]),
            scenario_input_mapping=dict(payload["scenario_input_mapping"]),
            scenario_input_units=dict(payload["scenario_input_units"]),
            release_flag_mapping=dict(payload["release_flag_mapping"]),
            target_registry=copy.deepcopy(payload["target_registry"]),
            feature_mapping=tuple(
                copy.deepcopy(payload["feature_mapping"])
            ),
            default_state={
                key: float(value)
                for key, value in payload["default_state"].items()
            },
            default_state_metadata=copy.deepcopy(
                payload["default_state_metadata"]
            ),
            shock_ranges={
                key: (float(value[0]), float(value[1]))
                for key, value in payload["shock_ranges"].items()
            },
            state_ranges={
                key: (float(value[0]), float(value[1]))
                for key, value in payload["state_ranges"].items()
            },
            maximum_historical_active_flags=int(
                payload["maximum_historical_active_flags"]
            ),
            diagnostic_reference=copy.deepcopy(
                payload["diagnostic_reference"]
            ),
            explanation_configuration=copy.deepcopy(
                payload["explanation_configuration"]
            ),
            confidence_configuration=copy.deepcopy(
                payload["confidence_configuration"]
            ),
            bundle_metadata=copy.deepcopy(payload["bundle_metadata"]),
        )
        engine._validate_loaded_runtime()
        return engine

    @property
    def selected_alphas(self) -> dict[str, float]:
        return {
            target: float(self.target_registry[target]["selected_alpha"])
            for target in self.target_names
        }

    @property
    def target_units(self) -> dict[str, str]:
        return {
            target: str(self.target_registry[target]["unit"])
            for target in self.target_names
        }

    def _validate_loaded_runtime(self) -> None:
        if len(self.feature_names) != 58 or len(set(self.feature_names)) != 58:
            raise BundleValidationError(
                "Bundle must contain 58 unique ordered features."
            )
        if (
            len(self.state_feature_names) != 45
            or len(set(self.state_feature_names)) != 45
        ):
            raise BundleValidationError(
                "Bundle must contain 45 unique state features."
            )
        if len(self.scenario_input_names) != 8:
            raise BundleValidationError(
                "Bundle must contain exactly eight scenario inputs."
            )
        if len(self.event_flag_names) != 5:
            raise BundleValidationError(
                "Bundle must contain exactly five release flags."
            )
        if len(self.target_names) != 12:
            raise BundleValidationError(
                "Bundle must contain exactly 12 target models."
            )
        if tuple(self.pipelines) != self.target_names:
            raise BundleValidationError(
                "Pipeline order does not match the frozen target order."
            )
        if set(self.target_registry) != set(self.target_names):
            raise BundleValidationError(
                "Target registry does not match the 12 model targets."
            )
        expected_feature_roles = {
            *self.scenario_input_mapping.values(),
            *self.event_flag_names,
            *self.state_feature_names,
        }
        if expected_feature_roles != set(self.feature_names):
            raise BundleValidationError(
                "Scenario, release-flag, and state mappings do not exactly "
                "cover the frozen 58-feature schema."
            )
        if tuple(
            item["feature_name"] for item in self.feature_mapping
        ) != self.feature_names:
            raise BundleValidationError(
                "Feature metadata order differs from the model schema."
            )
        if set(self.default_state) != set(self.state_feature_names):
            raise BundleValidationError(
                "Bundled default state is not a complete 45-feature state."
            )
        if not all(math.isfinite(value) for value in self.default_state.values()):
            raise BundleValidationError(
                "Bundled default state contains a nonfinite value."
            )
        if len({id(pipeline) for pipeline in self.pipelines.values()}) != 12:
            raise BundleValidationError(
                "The 12 target entries do not contain independent pipelines."
            )
        for target, pipeline in self.pipelines.items():
            if not isinstance(pipeline, Pipeline):
                raise BundleValidationError(
                    f"{target} artifact is not a scikit-learn Pipeline."
                )
            if tuple(pipeline.named_steps) != ("imputer", "scaler", "ridge"):
                raise BundleValidationError(
                    f"{target} pipeline step order changed."
                )
            if not isinstance(
                pipeline.named_steps["imputer"], SimpleImputer
            ):
                raise BundleValidationError(
                    f"{target} imputer type changed."
                )
            if not isinstance(
                pipeline.named_steps["scaler"], StandardScaler
            ):
                raise BundleValidationError(
                    f"{target} scaler type changed."
                )
            ridge = pipeline.named_steps["ridge"]
            if not isinstance(ridge, Ridge):
                raise BundleValidationError(
                    f"{target} estimator is not Ridge."
                )
            expected_alpha = float(
                self.target_registry[target]["selected_alpha"]
            )
            if float(ridge.alpha) != expected_alpha:
                raise BundleValidationError(
                    f"{target} Ridge alpha differs from the registry."
                )
            if int(pipeline.named_steps["imputer"].n_features_in_) != 58:
                raise BundleValidationError(
                    f"{target} imputer was not fitted on 58 features."
                )
            if int(pipeline.named_steps["scaler"].n_features_in_) != 58:
                raise BundleValidationError(
                    f"{target} scaler was not fitted on 58 features."
                )
            feature_names_in = tuple(
                str(item) for item in pipeline.feature_names_in_
            )
            if feature_names_in != self.feature_names:
                raise BundleValidationError(
                    f"{target} fitted feature order changed."
                )
        references = self.diagnostic_reference
        standardized = np.asarray(
            references["standardized_feature_rows"], dtype=float
        )
        sample_dates = tuple(references["sample_dates"])
        nearest = np.asarray(
            references["nearest_neighbor_distances"], dtype=float
        )
        if standardized.shape != (464, 58):
            raise BundleValidationError(
                "Feature-only diagnostic reference must be 464 x 58."
            )
        if len(sample_dates) != 464 or nearest.shape != (464,):
            raise BundleValidationError(
                "Diagnostic dates or neighbor reference has wrong size."
            )
        expected_continuous = {
            *self.scenario_input_mapping.values(),
            *self.state_feature_names,
        }
        if set(references["continuous_feature_values"]) != expected_continuous:
            raise BundleValidationError(
                "Continuous diagnostic references do not cover 53 inputs."
            )
        if set(references["contribution_p99"]) != set(self.target_names):
            raise BundleValidationError(
                "Contribution thresholds do not cover all targets."
            )
        if self.explanation_configuration.get(
            "schema_version"
        ) != EXPLANATION_SCHEMA_VERSION:
            raise BundleValidationError(
                "Frozen explanation schema version changed."
            )
        if self.confidence_configuration.get(
            "schema_version"
        ) != CONFIDENCE_SCHEMA_VERSION:
            raise BundleValidationError(
                "Frozen confidence schema version changed."
            )

    def _resolve_release_flags(
        self,
        scenario: Mapping[str, float],
        release_flags: Mapping[str, Any] | None,
    ) -> tuple[dict[str, float], str]:
        if release_flags is not None:
            return (
                _ordered_release_flags(
                    release_flags,
                    self.event_flag_names,
                ),
                "CALLER_PROVIDED_AND_FIXED_BETWEEN_BASELINE_AND_SCENARIO",
            )
        flags = {name: 0.0 for name in self.event_flag_names}
        for contract_name, flag_name in self.release_flag_mapping.items():
            flags[flag_name] = float(scenario[contract_name] != 0.0)
        return (
            flags,
            (
                "INFERRED_ONCE_FROM_NONZERO_EVENT_SHOCKS_AND_FIXED_"
                "BETWEEN_BASELINE_AND_SCENARIO"
            ),
        )

    def _resolve_state(
        self,
        current_state: Mapping[str, Any] | None,
        *,
        state_sample_date: str | date | None,
        state_cutoff_date: str | date | None,
    ) -> tuple[dict[str, float], str, dict[str, Any]]:
        if current_state is None:
            if state_sample_date is not None or state_cutoff_date is not None:
                raise ScenarioValidationError(
                    "State dates may only be overridden with current_state."
                )
            return (
                dict(self.default_state),
                "BUNDLED_VERSIONED_DEFAULT_STATE",
                copy.deepcopy(self.default_state_metadata),
            )
        state = _ordered_finite_mapping(
            current_state,
            self.state_feature_names,
            label="current_state",
        )
        sample = (
            None
            if state_sample_date is None
            else pd.Timestamp(state_sample_date).date().isoformat()
        )
        cutoff = (
            None
            if state_cutoff_date is None
            else pd.Timestamp(state_cutoff_date).date().isoformat()
        )
        return (
            state,
            "CALLER_PROVIDED_COMPLETE_STATE",
            {
                "sample_date": sample,
                "state_cutoff_date": cutoff,
                "state_cache_contains_targets": False,
                "latest_valid_state_rule": (
                    "Caller supplied all 45 state features. Freshness is "
                    "only evaluated when a state sample date is supplied."
                ),
            },
        )

    def _input_range_warnings(
        self,
        scenario: Mapping[str, float],
        state: Mapping[str, float],
        flags: Mapping[str, float],
    ) -> list[dict[str, Any]]:
        warnings: list[dict[str, Any]] = []
        for contract_name in self.scenario_input_names:
            minimum, maximum = self.shock_ranges[contract_name]
            value = float(scenario[contract_name])
            if value < minimum or value > maximum:
                warnings.append(
                    {
                        "warning_type": "SCENARIO_SHOCK_OUTSIDE_TRAINING_RANGE",
                        "field": contract_name,
                        "historical_feature": self.scenario_input_mapping[
                            contract_name
                        ],
                        "value": value,
                        "training_minimum": minimum,
                        "training_maximum": maximum,
                        "unit": self.scenario_input_units[contract_name],
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
        active_flags = int(sum(flags.values()))
        if active_flags > self.maximum_historical_active_flags:
            warnings.append(
                {
                    "warning_type": (
                        "RELEASE_FLAG_COMBINATION_OUTSIDE_TRAINING_SUPPORT"
                    ),
                    "active_release_flags": active_flags,
                    "historical_maximum": self.maximum_historical_active_flags,
                    "action": "WARN_ONLY_FLAGS_NOT_MODIFIED",
                }
            )
        for contract_name, flag_name in self.release_flag_mapping.items():
            if (
                scenario[contract_name] != 0.0
                and flags[flag_name] == 0.0
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
        flags: Mapping[str, float],
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        baseline_values: dict[str, float] = {}
        scenario_values: dict[str, float] = {}
        for contract_name in self.scenario_input_names:
            feature = self.scenario_input_mapping[contract_name]
            baseline_values[feature] = 0.0
            scenario_values[feature] = float(scenario[contract_name])
        baseline_values.update(flags)
        scenario_values.update(flags)
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
        isolation = {
            "feature_count": len(self.feature_names),
            "feature_order_exact": (
                tuple(baseline.columns) == self.feature_names
                and tuple(scenario_frame.columns) == self.feature_names
            ),
            "changed_features": changed,
            "only_shock_magnitudes_changed": (
                set(changed) <= set(self.scenario_input_mapping.values())
            ),
            "state_identical": bool(
                baseline[list(self.state_feature_names)].equals(
                    scenario_frame[list(self.state_feature_names)]
                )
            ),
            "release_flags_identical": bool(
                baseline[list(self.event_flag_names)].equals(
                    scenario_frame[list(self.event_flag_names)]
                )
            ),
            "baseline_row_sha256": _feature_row_hash(
                baseline, self.feature_names
            ),
            "scenario_row_sha256": _feature_row_hash(
                scenario_frame, self.feature_names
            ),
        }
        for key in (
            "feature_order_exact",
            "only_shock_magnitudes_changed",
            "state_identical",
            "release_flags_identical",
        ):
            if not isolation[key]:
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
        registry = self.target_registry[target]
        oos_minimum, oos_maximum = registry["oos_prediction_range"]
        target_minimum, target_maximum = registry["pretest_target_range"]
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

    def _base_result(
        self,
        scenario: Mapping[str, float],
        state: Mapping[str, float],
        state_source: str,
        state_metadata: Mapping[str, Any],
        flags: Mapping[str, float],
        flag_source: str,
        out_of_range_policy: str,
    ) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
        input_warnings = self._input_range_warnings(
            scenario, state, flags
        )
        shock_warnings = [
            warning
            for warning in input_warnings
            if warning["warning_type"]
            == "SCENARIO_SHOCK_OUTSIDE_TRAINING_RANGE"
        ]
        if out_of_range_policy == "error" and shock_warnings:
            names = [warning["field"] for warning in shock_warnings]
            raise ScenarioValidationError(
                "Scenario shocks outside the model training range: "
                f"{names}. Values were not clipped."
            )
        baseline_row, scenario_row, isolation = self._assemble_pair(
            scenario, state, flags
        )
        outputs: dict[str, dict[str, Any]] = {}
        prediction_warnings: list[dict[str, Any]] = []
        neutral = all(value == 0.0 for value in scenario.values())
        for target in self.target_names:
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
            if neutral:
                if abs(incremental) > 1e-12:
                    raise ScenarioValidationError(
                        f"Neutral scenario changed {target}."
                    )
                incremental = 0.0
            target_warnings = self._prediction_range_warnings(
                target, baseline_prediction, scenario_prediction
            )
            prediction_warnings.extend(target_warnings)
            output = copy.deepcopy(
                self.target_registry[target]["static_output_metadata"]
            )
            output.update(
                {
                    "baseline_prediction": baseline_prediction,
                    "scenario_prediction": scenario_prediction,
                    "incremental_scenario_effect": incremental,
                    "prediction_range_warnings": target_warnings,
                }
            )
            outputs[target] = output
        assets = tuple(self.bundle_metadata["asset_targets"])
        sectors = tuple(self.bundle_metadata["sector_targets"])
        result = {
            "schema_version": "scenario_research_inference_v2",
            "runtime_schema_version": RUNTIME_SCHEMA_VERSION,
            "inference_status": (
                "RESEARCH_ONLY_ALL_TARGET_RIDGE_PREDICTIONS"
            ),
            "research_only": True,
            "production_ready": False,
            "horizon_trading_days": 21,
            "scenario_shocks": dict(scenario),
            "scenario_input_units": dict(self.scenario_input_units),
            "release_flags": dict(flags),
            "release_flag_source": flag_source,
            "release_flag_mapping": dict(self.release_flag_mapping),
            "context": {
                **copy.deepcopy(state_metadata),
                "state_source": state_source,
                "state_feature_count": len(self.state_feature_names),
                "state_features": dict(state),
            },
            "definitions": copy.deepcopy(
                self.bundle_metadata["prediction_definitions"]
            ),
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
                target: outputs[target] for target in assets
            },
            "sector_outputs": {
                target: outputs[target] for target in sectors
            },
            "target_confidence_counts": copy.deepcopy(
                self.bundle_metadata["target_confidence_counts"]
            ),
            "locked_final_test": {
                "target_rows_parsed": 0,
                "access_count": 0,
                "target_values_available": False,
                "runtime_contains_target_values": False,
                "default_state_contains_targets": False,
            },
            "bundle": {
                "bundle_version": self.bundle_metadata["bundle_version"],
                "model_version": self.bundle_metadata["model_version"],
                "data_version": self.bundle_metadata["data_version"],
                "feature_schema_version": self.bundle_metadata[
                    "feature_schema_version"
                ],
                "bundle_directory": str(self.bundle_directory),
                "retraining_performed": False,
                "training_matrix_read": False,
                "target_labels_read": False,
                "trusted_source_only": True,
            },
            "global_warning": (
                "Research only. All 12 outputs are Ridge predictive "
                "associations, not causal effects or production forecasts. "
                "Nine models did not reliably outperform their best naive "
                "evaluation benchmark."
            ),
        }
        return result, baseline_row, scenario_row

    def _driver_text(
        self,
        *,
        feature: str,
        role: str,
        value: float,
        feature_unit: str,
        contribution: float,
        target_unit: str,
        driver_scope: str,
    ) -> str:
        return (
            f"{feature} ({role}, raw {value:.6g} {feature_unit}) has a "
            f"{contribution:+.6g} {target_unit} standardized Ridge "
            f"contribution in the {driver_scope}; this is a predictive "
            "association, not a causal effect."
        )

    def _top_driver_group(
        self,
        frame: pd.DataFrame,
        *,
        target: str,
        target_unit: str,
        driver_kind: str,
        contribution_column: str,
        raw_column: str,
        standardized_column: str,
        positive: bool,
        scope: str,
    ) -> list[dict[str, Any]]:
        subset = frame.loc[
            frame[contribution_column] > 0.0
            if positive
            else frame[contribution_column] < 0.0
        ].copy()
        subset = subset.sort_values(
            [contribution_column, "feature_position"],
            ascending=[not positive, True],
            kind="stable",
        )
        records: list[dict[str, Any]] = []
        for rank, (_, row) in enumerate(
            subset.head(TOP_DRIVER_COUNT).iterrows(),
            1,
        ):
            contribution = float(row[contribution_column])
            raw_value = float(row[raw_column])
            record = {
                "target": target,
                "unit": target_unit,
                "driver_kind": driver_kind,
                "rank": rank,
                "feature_name": str(row["feature_name"]),
                "inference_role": str(row["inference_role"]),
                "raw_value": raw_value,
                "feature_unit": str(row["feature_unit"]),
                "standardized_value": float(row[standardized_column]),
                "contribution": contribution,
                "explanation": self._driver_text(
                    feature=str(row["feature_name"]),
                    role=str(row["inference_role"]),
                    value=raw_value,
                    feature_unit=str(row["feature_unit"]),
                    contribution=contribution,
                    target_unit=target_unit,
                    driver_scope=scope,
                ),
                "causal_effect_claimed": False,
            }
            records.append(record)
        return records

    def _build_explanations(
        self,
        result: Mapping[str, Any],
        baseline: pd.DataFrame,
        scenario: pd.DataFrame,
    ) -> dict[str, Any]:
        targets: dict[str, Any] = {}
        maximum_reconciliation_error = 0.0
        mapping = {
            str(item["feature_name"]): item
            for item in self.feature_mapping
        }
        for target in self.target_names:
            pipeline = self.pipelines[target]
            imputer = pipeline.named_steps["imputer"]
            scaler = pipeline.named_steps["scaler"]
            ridge = pipeline.named_steps["ridge"]
            coefficient = np.asarray(ridge.coef_, dtype=float).reshape(-1)
            threshold = np.asarray(
                self.diagnostic_reference["contribution_p99"][target],
                dtype=float,
            )
            rows: dict[str, pd.DataFrame] = {}
            group_sums: dict[str, dict[str, float]] = {}
            predictions: dict[str, float] = {}
            for row_type, feature_row in (
                ("baseline", baseline),
                ("scenario", scenario),
            ):
                imputed = np.asarray(
                    imputer.transform(feature_row), dtype=float
                )[0]
                standardized = np.asarray(
                    scaler.transform(imputed.reshape(1, -1)),
                    dtype=float,
                )[0]
                contributions = standardized * coefficient
                records: list[dict[str, Any]] = []
                for position, feature in enumerate(self.feature_names):
                    metadata = mapping[feature]
                    records.append(
                        {
                            "feature_position": position,
                            "feature_name": feature,
                            "inference_role": str(
                                metadata["inference_role"]
                            ),
                            "contract_input": str(
                                metadata.get("contract_input", "")
                            ),
                            "feature_unit": str(metadata["unit"]),
                            "raw_value": float(
                                feature_row.iloc[0, position]
                            ),
                            "imputed_value": float(imputed[position]),
                            "scaler_mean": float(scaler.mean_[position]),
                            "scaler_scale": float(scaler.scale_[position]),
                            "standardized_value": float(
                                standardized[position]
                            ),
                            "ridge_coefficient": float(
                                coefficient[position]
                            ),
                            "feature_contribution": float(
                                contributions[position]
                            ),
                            "unusually_large_contribution": bool(
                                abs(contributions[position])
                                > max(float(threshold[position]), 1e-12)
                                + RECONCILIATION_TOLERANCE
                            ),
                        }
                    )
                frame = pd.DataFrame(records)
                rows[row_type] = frame
                group_sums[row_type] = {
                    role: float(
                        frame.loc[
                            frame["inference_role"].eq(role),
                            "feature_contribution",
                        ].sum()
                    )
                    for role in (
                        "current_state",
                        "scenario_shock",
                        "release_flag",
                    )
                }
                prediction = float(pipeline.predict(feature_row)[0])
                predictions[row_type] = prediction
                reconciled = float(ridge.intercept_) + float(
                    frame["feature_contribution"].sum()
                )
                error = abs(reconciled - prediction)
                maximum_reconciliation_error = max(
                    maximum_reconciliation_error, error
                )
                if error > RECONCILIATION_TOLERANCE:
                    raise BundleValidationError(
                        f"Explanation reconciliation failed for {target} "
                        f"{row_type}."
                    )
            baseline_rows = rows["baseline"].set_index("feature_name")
            scenario_rows = rows["scenario"].set_index("feature_name")
            delta_records: list[dict[str, Any]] = []
            for position, feature in enumerate(self.feature_names):
                left = baseline_rows.loc[feature]
                right = scenario_rows.loc[feature]
                delta_records.append(
                    {
                        "feature_position": position,
                        "feature_name": feature,
                        "inference_role": left["inference_role"],
                        "contract_input": left["contract_input"],
                        "feature_unit": left["feature_unit"],
                        "raw_value_change": (
                            float(right["raw_value"])
                            - float(left["raw_value"])
                        ),
                        "standardized_value_change": (
                            float(right["standardized_value"])
                            - float(left["standardized_value"])
                        ),
                        "contribution_change": (
                            float(right["feature_contribution"])
                            - float(left["feature_contribution"])
                        ),
                    }
                )
            deltas = pd.DataFrame(delta_records)
            incremental_groups = {
                role: float(
                    deltas.loc[
                        deltas["inference_role"].eq(role),
                        "contribution_change",
                    ].sum()
                )
                for role in (
                    "current_state",
                    "scenario_shock",
                    "release_flag",
                )
            }
            incremental_reconciled = float(
                deltas["contribution_change"].sum()
            )
            model_incremental = (
                predictions["scenario"] - predictions["baseline"]
            )
            incremental_error = abs(
                incremental_reconciled - model_incremental
            )
            maximum_reconciliation_error = max(
                maximum_reconciliation_error, incremental_error
            )
            if incremental_error > RECONCILIATION_TOLERANCE:
                raise BundleValidationError(
                    f"Incremental explanation failed for {target}."
                )
            target_unit = str(self.target_registry[target]["unit"])
            top_drivers = {
                "baseline_positive": self._top_driver_group(
                    rows["baseline"],
                    target=target,
                    target_unit=target_unit,
                    driver_kind="baseline_positive",
                    contribution_column="feature_contribution",
                    raw_column="raw_value",
                    standardized_column="standardized_value",
                    positive=True,
                    scope="baseline row",
                ),
                "baseline_negative": self._top_driver_group(
                    rows["baseline"],
                    target=target,
                    target_unit=target_unit,
                    driver_kind="baseline_negative",
                    contribution_column="feature_contribution",
                    raw_column="raw_value",
                    standardized_column="standardized_value",
                    positive=False,
                    scope="baseline row",
                ),
                "scenario_positive": self._top_driver_group(
                    rows["scenario"],
                    target=target,
                    target_unit=target_unit,
                    driver_kind="scenario_positive",
                    contribution_column="feature_contribution",
                    raw_column="raw_value",
                    standardized_column="standardized_value",
                    positive=True,
                    scope="scenario row",
                ),
                "scenario_negative": self._top_driver_group(
                    rows["scenario"],
                    target=target,
                    target_unit=target_unit,
                    driver_kind="scenario_negative",
                    contribution_column="feature_contribution",
                    raw_column="raw_value",
                    standardized_column="standardized_value",
                    positive=False,
                    scope="scenario row",
                ),
                "incremental_positive": self._top_driver_group(
                    deltas,
                    target=target,
                    target_unit=target_unit,
                    driver_kind="incremental_positive",
                    contribution_column="contribution_change",
                    raw_column="raw_value_change",
                    standardized_column="standardized_value_change",
                    positive=True,
                    scope="scenario-minus-baseline difference",
                ),
                "incremental_negative": self._top_driver_group(
                    deltas,
                    target=target,
                    target_unit=target_unit,
                    driver_kind="incremental_negative",
                    contribution_column="contribution_change",
                    raw_column="raw_value_change",
                    standardized_column="standardized_value_change",
                    positive=False,
                    scope="scenario-minus-baseline difference",
                ),
            }
            sensitivity = self.diagnostic_reference[
                "sensitivity_summary"
            ][target]
            negligible = sorted(
                str(row["contract_input"])
                for row in sensitivity
                if _as_bool(row["negligible_in_all_states"])
            )
            unstable = sorted(
                str(row["contract_input"])
                for row in sensitivity
                if _as_bool(row["unstable_across_walk_forward_folds"])
            )
            counterintuitive = sorted(
                str(row["contract_input"])
                for row in sensitivity
                if _as_bool(row["counterintuitive_direction_flag"])
            )
            large_features = sorted(
                {
                    str(row["feature_name"])
                    for frame in rows.values()
                    for row in frame.to_dict(orient="records")
                    if _as_bool(row["unusually_large_contribution"])
                }
            )
            output = result["targets"][target]
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
                        "model_intercept": float(ridge.intercept_),
                        "current_state_contribution": group_sums[
                            "baseline"
                        ]["current_state"],
                        "scenario_shock_contribution": group_sums[
                            "baseline"
                        ]["scenario_shock"],
                        "release_flag_contribution": group_sums[
                            "baseline"
                        ]["release_flag"],
                    },
                    "scenario": {
                        "model_intercept": float(ridge.intercept_),
                        "current_state_contribution": group_sums[
                            "scenario"
                        ]["current_state"],
                        "scenario_shock_contribution": group_sums[
                            "scenario"
                        ]["scenario_shock"],
                        "release_flag_contribution": group_sums[
                            "scenario"
                        ]["release_flag"],
                    },
                    "incremental": {
                        "model_intercept_change": 0.0,
                        "current_state_contribution_change": (
                            incremental_groups["current_state"]
                        ),
                        "scenario_shock_contribution_change": (
                            incremental_groups["scenario_shock"]
                        ),
                        "release_flag_contribution_change": (
                            incremental_groups["release_flag"]
                        ),
                    },
                },
                "top_drivers": top_drivers,
                "diagnostics": {
                    "negligible_scenario_inputs": negligible,
                    "unstable_walk_forward_directions": unstable,
                    "counterintuitive_reference_flags": counterintuitive,
                    "unusually_large_contribution_features": large_features,
                    "raw_and_standardized_values_reported_separately": True,
                },
                "warning_codes": copy.deepcopy(
                    output["applicable_warnings"]
                ),
                "known_limitations": copy.deepcopy(
                    output["known_limitations"]
                ),
                "explanation_warning": (
                    "Contributions decompose a standardized Ridge predictive "
                    "association. They do not estimate causal effects."
                ),
                "causal_effect_claimed": False,
            }
        return {
            "schema_version": EXPLANATION_SCHEMA_VERSION,
            "horizon_trading_days": 21,
            "scenario_shocks": copy.deepcopy(result["scenario_shocks"]),
            "release_flags": copy.deepcopy(result["release_flags"]),
            "context": copy.deepcopy(result["context"]),
            "targets": targets,
            "reconciliation": {
                "tolerance": RECONCILIATION_TOLERANCE,
                "maximum_absolute_error": maximum_reconciliation_error,
                "all_targets_passed": True,
            },
            "global_warning": (
                "Research-only Ridge contribution explanations. "
                "Coefficients, contributions, sensitivity slopes, and "
                "direction references are predictive diagnostics rather "
                "than causal financial effects."
            ),
        }

    def _input_diagnostics(
        self,
        result: Mapping[str, Any],
    ) -> pd.DataFrame:
        feature_units = {
            str(item["feature_name"]): str(item["unit"])
            for item in self.feature_mapping
        }
        first_pipeline = self.pipelines[self.target_names[0]]
        scaler = first_pipeline.named_steps["scaler"]
        positions = {
            feature: position
            for position, feature in enumerate(self.feature_names)
        }
        specs: list[tuple[str, str, str, float]] = []
        for contract_input in self.scenario_input_names:
            feature = self.scenario_input_mapping[contract_input]
            specs.append(
                (
                    "scenario_shock",
                    contract_input,
                    feature,
                    float(result["scenario_shocks"][contract_input]),
                )
            )
        for feature in self.state_feature_names:
            specs.append(
                (
                    "current_state",
                    "",
                    feature,
                    float(result["context"]["state_features"][feature]),
                )
            )
        records: list[dict[str, Any]] = []
        raw_references = self.diagnostic_reference[
            "continuous_feature_values"
        ]
        for role, contract_input, feature, value in specs:
            values = np.asarray(raw_references[feature], dtype=float)
            position = positions[feature]
            minimum = float(values.min())
            maximum = float(values.max())
            percentile = _empirical_percentile(values, value)
            outside = value < minimum or value > maximum
            extreme = bool(
                not outside
                and (percentile < 1.0 or percentile > 99.0)
            )
            standardized = (
                value - float(scaler.mean_[position])
            ) / float(scaler.scale_[position])
            records.append(
                {
                    "input_role": role,
                    "contract_input": contract_input,
                    "feature_name": feature,
                    "feature_unit": feature_units[feature],
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
                    "extreme_within_range": extreme,
                    "warning_status": (
                        "OUTSIDE_TRAINING_RANGE"
                        if outside
                        else (
                            "EXTREME_WITHIN_TRAINING_RANGE"
                            if extreme
                            else "WITHIN_TRAINING_RANGE"
                        )
                    ),
                    "clipped": False,
                    "winsorized": False,
                }
            )
        return pd.DataFrame(records)

    def _joint_distance(
        self,
        baseline: pd.DataFrame,
        scenario: pd.DataFrame,
    ) -> dict[str, Any]:
        first_pipeline = self.pipelines[self.target_names[0]]
        imputer = first_pipeline.named_steps["imputer"]
        scaler = first_pipeline.named_steps["scaler"]
        training = np.asarray(
            self.diagnostic_reference["standardized_feature_rows"],
            dtype=float,
        )
        reference_nearest = np.asarray(
            self.diagnostic_reference["nearest_neighbor_distances"],
            dtype=float,
        )
        sample_dates = tuple(
            self.diagnostic_reference["sample_dates"]
        )
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
                "58-feature median-imputed and training-standardized "
                "Euclidean"
            ),
        }
        rows: dict[str, Any] = {}
        for row_name, frame in (
            ("baseline", baseline),
            ("scenario", scenario),
        ):
            transformed = scaler.transform(imputer.transform(frame))
            distances = pairwise_distances(
                transformed, training, metric="euclidean"
            )[0]
            nearest = float(distances.min())
            percentile = _empirical_percentile(
                reference_nearest, nearest
            )
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
            nearest_position = int(np.argmin(distances))
            rows[row_name] = {
                "nearest_training_distance": nearest,
                "distance_percentile_vs_training_neighbors": percentile,
                "distance_status": status,
                "nearest_training_sample_date": str(
                    sample_dates[nearest_position]
                ),
                "included_in_confidence": True,
            }
        return {
            "schema_version": "joint_feature_distance_v1",
            "reference_distribution": reference,
            "rows": rows,
        }

    def _state_freshness(
        self,
        result: Mapping[str, Any],
        diagnostic_as_of_date: str | date | None,
    ) -> dict[str, Any]:
        evaluation_date = (
            date.today()
            if diagnostic_as_of_date is None
            else pd.Timestamp(diagnostic_as_of_date).date()
        )
        metadata = result["context"]
        sample_value = metadata.get("sample_date")
        cutoff_value = metadata.get("state_cutoff_date")
        if sample_value in {None, ""}:
            return {
                "diagnostic_as_of_date": evaluation_date.isoformat(),
                "state_sample_date": None,
                "state_cutoff_date": cutoff_value,
                "calendar_age_days": None,
                "freshness_status": "STATE_DATE_UNAVAILABLE",
                "thresholds_calendar_days": {
                    "fresh_maximum": 7,
                    "aging_maximum": 30,
                    "stale_minimum": 31,
                },
                "included_in_confidence": True,
            }
        state_date = pd.Timestamp(sample_value).date()
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
            "state_cutoff_date": cutoff_value,
            "calendar_age_days": age_days,
            "freshness_status": status,
            "thresholds_calendar_days": {
                "fresh_maximum": 7,
                "aging_maximum": 30,
                "stale_minimum": 31,
            },
            "included_in_confidence": True,
        }

    def _build_confidence(
        self,
        result: Mapping[str, Any],
        baseline: pd.DataFrame,
        scenario: pd.DataFrame,
        *,
        diagnostic_as_of_date: str | date | None,
    ) -> dict[str, Any]:
        inputs = self._input_diagnostics(result)
        joint = self._joint_distance(baseline, scenario)
        freshness = self._state_freshness(
            result, diagnostic_as_of_date
        )
        shock_outside = inputs.loc[
            inputs["input_role"].eq("scenario_shock"),
            "outside_training_range",
        ]
        state_outside = inputs.loc[
            inputs["input_role"].eq("current_state"),
            "outside_training_range",
        ]
        shock_names = sorted(
            inputs.loc[
                inputs["input_role"].eq("scenario_shock")
                & inputs["outside_training_range"],
                "contract_input",
            ].tolist()
        )
        state_names = sorted(
            inputs.loc[
                inputs["input_role"].eq("current_state")
                & inputs["outside_training_range"],
                "feature_name",
            ].tolist()
        )
        distance_status = joint["rows"]["scenario"][
            "distance_status"
        ]
        current_vix = float(
            result["context"]["state_features"]["vix_level_points"]
        )
        current_regime = _current_vix_regime(current_vix)
        targets: dict[str, Any] = {}
        for target in self.target_names:
            output = result["targets"][target]
            registry = self.target_registry[target]
            assessment = registry["assessment"]
            fold_metrics = copy.deepcopy(
                self.diagnostic_reference["fold_metrics"][target]
            )
            regime_metrics = self.diagnostic_reference[
                "regime_metrics"
            ][target]
            if current_regime not in regime_metrics:
                raise BundleValidationError(
                    f"Bundle lacks {current_regime} evidence for {target}."
                )
            similar_regime = copy.deepcopy(
                regime_metrics[current_regime]
            )
            residual = copy.deepcopy(
                self.diagnostic_reference["residual_summary"][target]
            )
            sensitivity = copy.deepcopy(
                self.diagnostic_reference["sensitivity_summary"][target]
            )
            negligible = sorted(
                str(row["contract_input"])
                for row in sensitivity
                if _as_bool(row["negligible_in_all_states"])
            )
            unstable = sorted(
                str(row["contract_input"])
                for row in sensitivity
                if _as_bool(row["unstable_across_walk_forward_folds"])
            )
            counterintuitive = sorted(
                str(row["contract_input"])
                for row in sensitivity
                if _as_bool(row["counterintuitive_direction_flag"])
            )
            warnings = {
                category: [] for category in WARNING_CATEGORIES
            }
            if shock_names:
                warnings["input_range"].append(
                    _warning(
                        target=target,
                        category="input_range",
                        code="SCENARIO_SHOCK_OUTSIDE_TRAINING_RANGE",
                        severity="high",
                        message=(
                            "One or more user shocks are outside the "
                            "authorized pretest training range; values were "
                            "not clipped."
                        ),
                        evidence={"fields": shock_names},
                    )
                )
            if state_names:
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
                        evidence={"fields": state_names},
                    )
                )
            if distance_status != (
                "WITHIN_OBSERVED_COMBINATION_SUPPORT"
            ):
                warnings["state_extrapolation"].append(
                    _warning(
                        target=target,
                        category="state_extrapolation",
                        code=distance_status,
                        severity=(
                            "high"
                            if distance_status
                            in {
                                "OUTSIDE_OBSERVED_NEIGHBORHOOD",
                                "EXTREME_UNFAMILIAR_COMBINATION",
                            }
                            else "medium"
                        ),
                        message=(
                            "The complete scenario row is distant from "
                            "common historical 58-feature combinations."
                        ),
                        evidence=joint["rows"]["scenario"],
                    )
                )
            classification = str(
                output["step13_evaluation_classification"]
            )
            ridge_oos_r2 = float(
                assessment["ridge_oos_r_squared"]
            )
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
                            "Step 13 found weak evidence and did not approve "
                            "this Ridge model beyond experimental research "
                            "use."
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
                            "Step 13 approved only low-confidence "
                            "experimental research inference."
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
                            "Pooled walk-forward out-of-sample R-squared is "
                            "not positive."
                        ),
                        evidence={
                            "out_of_sample_r_squared": ridge_oos_r2
                        },
                    )
                )
            if not best_naive_both:
                warnings["model_performance"].append(
                    _warning(
                        target=target,
                        category="model_performance",
                        code=(
                            "RIDGE_NOT_BETTER_THAN_BEST_NAIVE_ON_BOTH_ERRORS"
                        ),
                        severity="high",
                        message=(
                            "Ridge did not beat the best naive benchmark on "
                            "both MAE and RMSE."
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
                            "One or more final Ridge slopes differ from a "
                            "limited noncausal plausibility reference."
                        ),
                        evidence={"scenario_inputs": counterintuitive},
                    )
                )
            overall_mae = float(assessment["ridge_mae"])
            similar_mae = float(similar_regime["mae"])
            regime_ratio = (
                similar_mae / overall_mae
                if overall_mae > 0.0
                else math.inf
            )
            if regime_ratio > 1.25:
                warnings["regime"].append(
                    _warning(
                        target=target,
                        category="regime",
                        code="CURRENT_VIX_REGIME_ERROR_ELEVATED",
                        severity="medium",
                        message=(
                            "Walk-forward MAE in the current VIX regime is "
                            "more than 25% above pooled Ridge MAE."
                        ),
                        evidence={
                            "current_vix_regime": current_regime,
                            "regime_mae": similar_mae,
                            "overall_mae": overall_mae,
                            "ratio": regime_ratio,
                        },
                    )
                )
            freshness_status = freshness["freshness_status"]
            if freshness_status != "FRESH":
                if freshness_status == "STALE":
                    freshness_code = "STATE_SNAPSHOT_STALE"
                    freshness_severity = "high"
                elif freshness_status == "STATE_DATE_UNAVAILABLE":
                    freshness_code = "STATE_DATE_UNAVAILABLE"
                    freshness_severity = "high"
                elif freshness_status == "STATE_DATE_AFTER_DIAGNOSTIC_DATE":
                    freshness_code = "STATE_DATE_AFTER_DIAGNOSTIC_DATE"
                    freshness_severity = "high"
                else:
                    freshness_code = "STATE_SNAPSHOT_AGING"
                    freshness_severity = "medium"
                warnings["state_freshness"].append(
                    _warning(
                        target=target,
                        category="state_freshness",
                        code=freshness_code,
                        severity=freshness_severity,
                        message=(
                            "The default state snapshot is not current "
                            "relative to the diagnostic date."
                            if freshness_status
                            in {"STALE", "AGING"}
                            else (
                                "The state snapshot date is unavailable or "
                                "not valid relative to the diagnostic date."
                            )
                        ),
                        evidence=freshness,
                    )
                )
            severe_distance = distance_status in {
                "OUTSIDE_OBSERVED_NEIGHBORHOOD",
                "EXTREME_UNFAMILIAR_COMBINATION",
            }
            if bool(shock_outside.any()) or severe_distance:
                overall = "OUTSIDE_HISTORICAL_SUPPORT"
            elif classification == "NAIVE_FALLBACK_REQUIRED":
                overall = "VERY_LOW_RESEARCH_CONFIDENCE"
            elif (
                bool(state_outside.any())
                or distance_status == "UNUSUAL_COMBINATION"
                or freshness_status != "FRESH"
            ):
                overall = "LOW_RESEARCH_CONFIDENCE_WITH_CONTEXT_WARNINGS"
            else:
                overall = "LOW_RESEARCH_CONFIDENCE"
            xgboost = {
                "status": XGBOOST_AGREEMENT_STATUS,
                "available": False,
                "included_in_confidence": False,
                "score": None,
                "penalty_or_bonus": "NONE",
                "reason": (
                    "All-target XGBoost evaluation was intentionally "
                    "postponed. This diagnostic may be added in a later "
                    "model version."
                ),
            }
            flattened = {
                "target": target,
                "unit": output["unit"],
                "selected_alpha": output["selected_alpha"],
                "step13_evaluation_classification": classification,
                "overall_research_confidence": overall,
                "confidence_method": "RULE_BASED_NON_COMPENSATORY",
                "walk_forward_rows": int(
                    assessment["walk_forward_rows"]
                ),
                "walk_forward_mae": overall_mae,
                "walk_forward_rmse": float(assessment["ridge_rmse"]),
                "walk_forward_oos_r_squared": ridge_oos_r2,
                "fold_mae_standard_deviation": float(
                    assessment["fold_mae_standard_deviation"]
                ),
                "worst_fold_mae": float(
                    assessment["worst_fold_mae"]
                ),
                "current_vix_level_points": current_vix,
                "current_vix_regime": current_regime,
                "similar_regime_observations": int(
                    similar_regime["sample_count"]
                ),
                "similar_regime_mae": similar_mae,
                "similar_regime_rmse": float(
                    similar_regime["rmse"]
                ),
                "similar_regime_error_ratio": regime_ratio,
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
            targets[target] = {
                **flattened,
                "input_range": {
                    "outside_scenario_inputs": shock_names,
                    "diagnostic_rows": inputs.loc[
                        inputs["input_role"].eq("scenario_shock")
                    ].to_dict(orient="records"),
                },
                "state_extrapolation": {
                    "outside_state_features": state_names,
                    "joint_distance": joint,
                },
                "model_performance": {
                    "fold_metrics": fold_metrics,
                    "best_naive_comparison": copy.deepcopy(
                        output["naive_benchmark_comparison"]
                    ),
                    "residual_extreme_period": residual,
                },
                "scenario_responsiveness": {
                    "negligible_scenario_inputs": negligible,
                    "unstable_walk_forward_directions": unstable,
                    "counterintuitive_reference_flags": counterintuitive,
                    "sensitivity_rows": sensitivity,
                },
                "regime": {
                    "current_vix_regime": current_regime,
                    "current_regime_metrics": similar_regime,
                },
                "state_freshness": dict(freshness),
                "xgboost_agreement": xgboost,
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
        return {
            "schema_version": CONFIDENCE_SCHEMA_VERSION,
            "diagnostic_as_of_date": freshness[
                "diagnostic_as_of_date"
            ],
            "horizon_trading_days": 21,
            "feature_count": len(self.feature_names),
            "scenario_shocks": copy.deepcopy(result["scenario_shocks"]),
            "release_flags": copy.deepcopy(result["release_flags"]),
            "context": copy.deepcopy(result["context"]),
            "state_freshness": freshness,
            "joint_feature_distance": joint,
            "xgboost_agreement_policy": {
                "status": XGBOOST_AGREEMENT_STATUS,
                "available": False,
                "included_in_confidence": False,
                "score": None,
                "penalty_or_bonus": "NONE",
                "later_version_supported": True,
            },
            "targets": targets,
            "global_warning": (
                "Research confidence is non-compensatory and cannot exceed "
                "the frozen Step 13 target evidence. Diagnostics do not "
                "establish causal effects or production readiness."
            ),
        }

    def predict(
        self,
        scenario_shocks: Mapping[str, Any],
        *,
        current_state: Mapping[str, Any] | None = None,
        release_flags: Mapping[str, Any] | None = None,
        out_of_range_policy: str = "warn",
        state_sample_date: str | date | None = None,
        state_cutoff_date: str | date | None = None,
        diagnostic_as_of_date: str | date | None = None,
    ) -> dict[str, Any]:
        """Return all 12 predictions, explanations, and confidence metadata."""
        if out_of_range_policy not in {"warn", "error"}:
            raise ScenarioValidationError(
                "out_of_range_policy must be 'warn' or 'error'."
            )
        scenario = _ordered_finite_mapping(
            scenario_shocks,
            self.scenario_input_names,
            label="scenario_shocks",
        )
        state, state_source, state_metadata = self._resolve_state(
            current_state,
            state_sample_date=state_sample_date,
            state_cutoff_date=state_cutoff_date,
        )
        flags, flag_source = self._resolve_release_flags(
            scenario, release_flags
        )
        result, baseline, scenario_row = self._base_result(
            scenario,
            state,
            state_source,
            state_metadata,
            flags,
            flag_source,
            out_of_range_policy,
        )
        result["explanations"] = self._build_explanations(
            result, baseline, scenario_row
        )
        result["confidence"] = self._build_confidence(
            result,
            baseline,
            scenario_row,
            diagnostic_as_of_date=diagnostic_as_of_date,
        )
        return _json_safe(result)


def load_scenario_model_bundle(
    bundle_path: str | Path,
    *,
    validate_hashes: bool = True,
) -> ScenarioModelRuntime:
    """Load a trusted Step 17 bundle without fitting or reading model data.

    Joblib relies on pickle and can execute arbitrary code during loading.
    Never load an artifact from an untrusted or unverified source.
    """
    directory = _resolve_bundle_directory(bundle_path)
    if validate_hashes:
        validate_bundle_hashes(directory)
    manifest_path = directory / BUNDLE_MANIFEST_FILENAME
    payload_path = directory / BUNDLE_PAYLOAD_FILENAME
    if not manifest_path.is_file() or not payload_path.is_file():
        raise BundleValidationError(
            "Bundle is missing its manifest or joblib payload."
        )
    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleValidationError("Bundle manifest is unreadable.") from exc
    if manifest.get("runtime_schema_version") != RUNTIME_SCHEMA_VERSION:
        raise BundleValidationError(
            "Bundle manifest has an unsupported runtime schema."
        )
    if not manifest.get("security", {}).get("trusted_source_only"):
        raise BundleValidationError(
            "Bundle manifest does not enforce the trusted-source policy."
        )
    payload = joblib.load(payload_path)
    if not isinstance(payload, Mapping):
        raise BundleValidationError(
            "Joblib bundle payload must be a mapping."
        )
    if payload.get("bundle_metadata", {}).get(
        "bundle_version"
    ) != manifest.get("bundle_version"):
        raise BundleValidationError(
            "Joblib payload and manifest bundle versions differ."
        )
    return ScenarioModelRuntime.from_payload(
        payload,
        bundle_directory=directory,
        manifest=manifest,
    )
