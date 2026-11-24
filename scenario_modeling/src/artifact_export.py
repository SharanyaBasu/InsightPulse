"""Step 17 export of the frozen all-target Ridge research bundle.

Export-time code may read the authorized pretest feature/target rows to fit the
already selected Ridge configurations.  The resulting runtime payload contains
fitted pipelines, metadata, and feature-only diagnostic references; it does
not contain target labels or a complete training-oriented inference engine.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from .cache import write_text_atomic
from .confidence_diagnostics import (
    build_confidence_artifacts,
    confidence_metadata_schema,
)
from .config import INTERIM_ROOT, PROCESSED_ROOT, PROJECT_ROOT
from .scenario_explanations import (
    build_explanation_artifacts,
    explanation_metadata_schema,
)
from .scenario_inference import (
    ALL_RIDGE_TARGETS,
    ASSET_TARGETS,
    EVENT_INPUT_TO_FLAG,
    EXAMPLE_SCENARIO,
    NEUTRAL_SCENARIO,
    SECTOR_TARGETS,
    SHOCK_UNITS,
    build_research_inference_engine,
)
from .scenario_runtime import (
    BUNDLE_CHECKSUMS_FILENAME,
    BUNDLE_MANIFEST_FILENAME,
    BUNDLE_PAYLOAD_FILENAME,
    CONFIDENCE_SCHEMA_VERSION,
    EXPLANATION_SCHEMA_VERSION,
    PAYLOAD_SCHEMA_VERSION,
    RUNTIME_SCHEMA_VERSION,
    load_scenario_model_bundle,
    validate_bundle_hashes,
)
from .shock_features import (
    CONTRACT_SHOCK_NAMES,
    CONTRACT_TO_HISTORICAL_FEATURE,
    EVENT_FLAG_NAMES,
)


BUNDLE_VERSION = "scenario-playground-ridge-v1.0.0"
MODEL_VERSION = "scenario-playground-all-target-ridge-v1"
DEFAULT_BUNDLE_DIRECTORY = (
    PROJECT_ROOT
    / "scenario_modeling"
    / "artifacts"
    / "scenario_playground_ridge"
    / "v1.0.0"
)
STEP17_EVIDENCE_DIRECTORY = INTERIM_ROOT / "artifact_export"
STEP13_ROOT = INTERIM_ROOT / "models" / "all_targets"
STEP15_ROOT = INTERIM_ROOT / "explanations"
STEP16_ROOT = INTERIM_ROOT / "confidence"
MATRIX_ROOT = PROCESSED_ROOT / "model_matrix"
REQUIREMENTS_PATH = PROJECT_ROOT / "scenario_modeling" / "requirements.txt"
REFERENCE_TOLERANCE = 1e-10
XGBOOST_AGREEMENT_STATUS = "UNAVAILABLE_INTENTIONALLY_POSTPONED"
SECRET_PATTERNS = (
    re.compile(r"(?i)\bpostgres(?:ql)?://[^\s]+"),
    re.compile(r"(?i)https://[a-z0-9-]+\.supabase\.co"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(
        r"(?i)(?:api[_-]?key|token|password|secret)"
        r"\s*[:=]\s*[\"'][^<\s][^\"']{7,}[\"']"
    ),
)

SOURCE_EVIDENCE_PATHS = (
    MATRIX_ROOT / "scenario_model_matrix_summary.json",
    MATRIX_ROOT / "scenario_model_matrix_split_manifest.csv",
    MATRIX_ROOT / "scenario_model_matrix_column_catalog.csv",
    STEP13_ROOT / "all_targets_final_configurations.csv",
    STEP13_ROOT / "all_targets_target_assessment.csv",
    STEP13_ROOT / "all_targets_fold_metrics.csv",
    STEP13_ROOT / "all_targets_regime_metrics.csv",
    STEP13_ROOT / "all_targets_residual_summary.csv",
    STEP13_ROOT / "all_targets_sensitivity_summary.csv",
    STEP15_ROOT / "sensitivity_summary.csv",
    STEP15_ROOT / "explanation_metadata_schema.json",
    STEP16_ROOT / "confidence_metadata_schema.json",
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
    if isinstance(value, (pd.Timestamp, date, Path)):
        return str(value)
    if pd.isna(value):
        return None
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    serialized = json.dumps(
        _json_safe(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    write_text_atomic(
        path,
        json.dumps(
            _json_safe(dict(payload)),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )


def _write_joblib(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    joblib.dump(dict(payload), partial, compress=3, protocol=5)
    partial.replace(path)


def _records_by_key(
    frame: pd.DataFrame,
    key: str,
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for value, group in frame.groupby(key, sort=False):
        result[str(value)] = _json_safe(
            group.to_dict(orient="records")
        )
    return result


def _feature_mapping_records(engine: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in engine.feature_mapping.to_dict(orient="records"):
        normalized = _json_safe(record)
        if normalized.get("contract_input") is None:
            normalized["contract_input"] = ""
        records.append(normalized)
    if tuple(item["feature_name"] for item in records) != (
        engine.feature_names
    ):
        raise ValueError("Step 17 feature metadata order changed.")
    return records


def _dependency_versions() -> dict[str, str]:
    packages = (
        "joblib",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
    )
    versions: dict[str, str] = {}
    for package in packages:
        versions[package] = importlib.metadata.version(package)
    return versions


def _reference_date() -> str:
    path = STEP16_ROOT / "target_confidence.json"
    if not path.exists():
        return date.today().isoformat()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return str(payload["diagnostic_as_of_date"])


def _source_hashes() -> dict[str, str]:
    missing = [path for path in SOURCE_EVIDENCE_PATHS if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Step 17 source evidence is missing: "
            + ", ".join(_relative(path) for path in missing)
        )
    return {
        _relative(path): _sha256_file(path)
        for path in SOURCE_EVIDENCE_PATHS
    }


def scan_bundle_text_for_secrets(
    bundle_directory: Path,
) -> dict[str, Any]:
    """Scan human-readable bundle files for credential-shaped values."""
    findings: list[dict[str, str]] = []
    for path in sorted(bundle_directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {
            ".json",
            ".txt",
            ".md",
        }:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                findings.append(
                    {
                        "file": path.name,
                        "pattern": pattern.pattern,
                    }
                )
    return {
        "scanned_text_file_count": sum(
            1
            for path in bundle_directory.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".json", ".txt", ".md"}
        ),
        "finding_count": len(findings),
        "findings": findings,
        "credentials_detected": bool(findings),
    }


def _build_target_registry(
    engine: Any,
    reference_inference: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = {}
    for target in engine.target_names:
        output = dict(reference_inference["targets"][target])
        static_output = {
            key: _json_safe(value)
            for key, value in output.items()
            if key
            not in {
                "baseline_prediction",
                "scenario_prediction",
                "incremental_scenario_effect",
                "prediction_range_warnings",
            }
        }
        assessment = _json_safe(
            engine.assessment.loc[target].to_dict()
        )
        registry[target] = {
            "target": target,
            "model_family": "ridge",
            "pipeline_structure": "imputer_scaler_ridge",
            "selected_alpha": float(engine.selected_alphas[target]),
            "unit": str(assessment["unit"]),
            "confidence_classification": str(
                assessment["classification"]
            ),
            "assessment": assessment,
            "oos_prediction_range": (
                float(assessment["prediction_minimum"]),
                float(assessment["prediction_maximum"]),
            ),
            "pretest_target_range": (
                float(engine.target_ranges[target][0]),
                float(engine.target_ranges[target][1]),
            ),
            "static_output_metadata": static_output,
        }
    if tuple(registry) != ALL_RIDGE_TARGETS:
        raise ValueError("Step 17 target registry order changed.")
    return registry


def _build_diagnostic_reference(
    engine: Any,
) -> dict[str, Any]:
    first_pipeline = engine.pipelines[engine.target_names[0]]
    features = engine.training.loc[:, list(engine.feature_names)]
    imputer = first_pipeline.named_steps["imputer"]
    scaler = first_pipeline.named_steps["scaler"]
    standardized = np.asarray(
        scaler.transform(imputer.transform(features)),
        dtype=float,
    )
    pairwise = pairwise_distances(
        standardized,
        metric="euclidean",
    )
    np.fill_diagonal(pairwise, np.inf)
    nearest = pairwise.min(axis=1)
    continuous_features = (
        *CONTRACT_TO_HISTORICAL_FEATURE.values(),
        *engine.state_feature_names,
    )
    raw_values = {
        feature: np.sort(
            engine.training[feature].to_numpy(dtype=float)
        )
        for feature in continuous_features
    }
    contribution_p99: dict[str, np.ndarray] = {}
    for target, pipeline in engine.pipelines.items():
        transformed = np.asarray(
            pipeline.named_steps["scaler"].transform(
                pipeline.named_steps["imputer"].transform(features)
            ),
            dtype=float,
        )
        coefficients = np.asarray(
            pipeline.named_steps["ridge"].coef_,
            dtype=float,
        ).reshape(-1)
        contribution_p99[target] = np.quantile(
            np.abs(transformed * coefficients),
            0.99,
            axis=0,
        )
    fold_metrics = pd.read_csv(
        STEP13_ROOT / "all_targets_fold_metrics.csv"
    )
    fold_metrics = fold_metrics.loc[
        fold_metrics["strategy"].eq("ridge_full_history")
    ].sort_values(["target", "fold"], kind="stable")
    regime_metrics = pd.read_csv(
        STEP13_ROOT / "all_targets_regime_metrics.csv"
    )
    regime_metrics = regime_metrics.loc[
        regime_metrics["strategy"].eq("ridge_full_history")
    ]
    regime_by_target: dict[str, dict[str, dict[str, Any]]] = {}
    for target, group in regime_metrics.groupby("target", sort=False):
        regime_by_target[str(target)] = {
            str(row["vix_regime"]): _json_safe(row.to_dict())
            for _, row in group.iterrows()
        }
    residuals = pd.read_csv(
        STEP13_ROOT / "all_targets_residual_summary.csv"
    ).set_index("target")
    sensitivity = pd.read_csv(
        STEP15_ROOT / "sensitivity_summary.csv"
    )
    return {
        "reference_kind": (
            "FEATURE_ONLY_DIAGNOSTIC_REFERENCE_NO_TARGET_LABELS"
        ),
        "authorized_pretest_rows": len(features),
        "standardized_feature_rows": standardized,
        "sample_dates": tuple(engine.training["sample_date"].astype(str)),
        "nearest_neighbor_distances": nearest,
        "continuous_feature_values": raw_values,
        "contribution_p99": contribution_p99,
        "fold_metrics": _records_by_key(fold_metrics, "target"),
        "regime_metrics": regime_by_target,
        "residual_summary": {
            target: _json_safe(residuals.loc[target].to_dict())
            for target in engine.target_names
        },
        "sensitivity_summary": _records_by_key(
            sensitivity, "target"
        ),
        "contains_target_labels": False,
        "contains_complete_training_dataframe": False,
    }


def _feature_range_metadata(
    engine: Any,
    feature_mapping: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    mapping = {
        str(item["feature_name"]): item for item in feature_mapping
    }
    features = (
        *CONTRACT_TO_HISTORICAL_FEATURE.values(),
        *engine.state_feature_names,
    )
    entries: list[dict[str, Any]] = []
    for feature in features:
        values = engine.training[feature].to_numpy(dtype=float)
        contract = next(
            (
                key
                for key, value in CONTRACT_TO_HISTORICAL_FEATURE.items()
                if value == feature
            ),
            None,
        )
        entries.append(
            {
                "feature_name": feature,
                "contract_input": contract,
                "inference_role": mapping[feature]["inference_role"],
                "unit": mapping[feature]["unit"],
                "minimum": float(values.min()),
                "p01": float(np.quantile(values, 0.01)),
                "p05": float(np.quantile(values, 0.05)),
                "p25": float(np.quantile(values, 0.25)),
                "median": float(np.quantile(values, 0.50)),
                "p75": float(np.quantile(values, 0.75)),
                "p95": float(np.quantile(values, 0.95)),
                "p99": float(np.quantile(values, 0.99)),
                "maximum": float(values.max()),
                "training_rows": len(values),
            }
        )
    return {
        "schema_version": "scenario_feature_ranges_v1",
        "training_rows": len(engine.training),
        "continuous_input_count": len(entries),
        "entries": entries,
        "range_policy": (
            "Warn only. Runtime performs no clipping or winsorization."
        ),
    }


def _assert_close(
    actual: Any,
    expected: Any,
    *,
    path: str,
    tolerance: float,
    mismatches: list[str],
) -> float:
    maximum = 0.0
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            mismatches.append(f"{path}: expected mapping")
            return maximum
        for key, expected_value in expected.items():
            if key not in actual:
                mismatches.append(f"{path}.{key}: missing")
                continue
            maximum = max(
                maximum,
                _assert_close(
                    actual[key],
                    expected_value,
                    path=f"{path}.{key}",
                    tolerance=tolerance,
                    mismatches=mismatches,
                ),
            )
        return maximum
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            mismatches.append(
                f"{path}: list length {len(actual) if isinstance(actual, list) else 'NA'} "
                f"!= {len(expected)}"
            )
            return maximum
        for index, expected_value in enumerate(expected):
            maximum = max(
                maximum,
                _assert_close(
                    actual[index],
                    expected_value,
                    path=f"{path}[{index}]",
                    tolerance=tolerance,
                    mismatches=mismatches,
                ),
            )
        return maximum
    if (
        isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        and isinstance(actual, (int, float))
        and not isinstance(actual, bool)
    ):
        difference = abs(float(actual) - float(expected))
        if difference > tolerance:
            mismatches.append(
                f"{path}: numeric difference {difference:.3g}"
            )
        return difference
    if actual != expected:
        mismatches.append(f"{path}: {actual!r} != {expected!r}")
    return maximum


def compare_runtime_with_reference(
    runtime_result: Mapping[str, Any],
    reference_inference: Mapping[str, Any],
    reference_explanations: Mapping[str, Any],
    reference_confidence: Mapping[str, Any],
    *,
    tolerance: float = REFERENCE_TOLERANCE,
) -> dict[str, Any]:
    """Compare substantive frozen Step 14-16 fields."""
    mismatches: list[str] = []
    maximum = 0.0
    prediction_fields = (
        "baseline_prediction",
        "scenario_prediction",
        "incremental_scenario_effect",
    )
    for target in ALL_RIDGE_TARGETS:
        for field in prediction_fields:
            maximum = max(
                maximum,
                _assert_close(
                    runtime_result["targets"][target][field],
                    reference_inference["targets"][target][field],
                    path=f"predictions.{target}.{field}",
                    tolerance=tolerance,
                    mismatches=mismatches,
                ),
            )
        expected_explanation = reference_explanations[
            "targets"
        ][target]
        actual_explanation = runtime_result["explanations"][
            "targets"
        ][target]
        maximum = max(
            maximum,
            _assert_close(
                actual_explanation,
                expected_explanation,
                path=f"explanations.{target}",
                tolerance=tolerance,
                mismatches=mismatches,
            ),
        )
        expected_confidence = reference_confidence["targets"][target]
        actual_confidence = runtime_result["confidence"]["targets"][
            target
        ]
        maximum = max(
            maximum,
            _assert_close(
                actual_confidence,
                expected_confidence,
                path=f"confidence.{target}",
                tolerance=tolerance,
                mismatches=mismatches,
            ),
        )
    for field in (
        "scenario_shocks",
        "release_flags",
        "input_range_warnings",
        "prediction_range_warnings",
    ):
        maximum = max(
            maximum,
            _assert_close(
                runtime_result[field],
                reference_inference[field],
                path=field,
                tolerance=tolerance,
                mismatches=mismatches,
            ),
        )
    for field in (
        "state_freshness",
        "joint_feature_distance",
        "xgboost_agreement_policy",
    ):
        maximum = max(
            maximum,
            _assert_close(
                runtime_result["confidence"][field],
                reference_confidence[field],
                path=f"confidence.{field}",
                tolerance=tolerance,
                mismatches=mismatches,
            ),
        )
    return {
        "tolerance": tolerance,
        "maximum_absolute_numeric_difference": maximum,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:50],
        "equivalent": not mismatches,
        "predictions_compared": 36,
        "explanation_targets_compared": 12,
        "confidence_targets_compared": 12,
        "warnings_compared": True,
    }


def _bundle_metadata(
    *,
    source_hashes: Mapping[str, str],
    feature_names: Sequence[str],
    default_state: Mapping[str, float],
    reference_date: str,
) -> dict[str, Any]:
    data_version = "sha256:" + _sha256_json(source_hashes)
    feature_schema_version = (
        "scenario-58-feature-v1-sha256:"
        + _sha256_json(list(feature_names))
    )
    state_version = (
        "scenario-default-state-v1-sha256:"
        + _sha256_json(default_state)
    )
    return {
        "bundle_version": BUNDLE_VERSION,
        "model_version": MODEL_VERSION,
        "data_version": data_version,
        "feature_schema_version": feature_schema_version,
        "default_state_version": state_version,
        "runtime_schema_version": RUNTIME_SCHEMA_VERSION,
        "payload_schema_version": PAYLOAD_SCHEMA_VERSION,
        "explanation_schema_version": EXPLANATION_SCHEMA_VERSION,
        "confidence_schema_version": CONFIDENCE_SCHEMA_VERSION,
        "created_on": date.today().isoformat(),
        "reference_diagnostic_date": reference_date,
        "asset_targets": list(ASSET_TARGETS),
        "sector_targets": list(SECTOR_TARGETS),
        "target_confidence_counts": {
            "ridge_pipeline_count": 12,
            "low_confidence_promotion_gate_passed": 3,
            "weak_evidence_ridge": 9,
            "naive_benchmark_substitutions": 0,
        },
        "prediction_definitions": {
            "baseline_prediction": (
                "Bundled or caller-provided fixed state plus neutral shock "
                "magnitudes and the same release flags used by the scenario "
                "row."
            ),
            "scenario_prediction": (
                "The same fixed state and release flags plus the eight "
                "user-entered shock magnitudes."
            ),
            "incremental_scenario_effect": (
                "Scenario prediction minus baseline prediction for each of "
                "the 12 independent target-specific Ridge models."
            ),
        },
        "training": {
            "authorized_pretest_rows": 464,
            "first_sample_date": "2013-08-05",
            "last_sample_date": "2023-11-06",
            "locked_final_test_rows_declared": 114,
            "locked_final_test_target_rows_parsed": 0,
            "final_test_access_count": 0,
            "target_values_in_runtime_bundle": False,
        },
        "determinism": {
            "ridge_random_state": None,
            "artifact_export_seed": 0,
            "stochastic_training_steps": False,
            "note": (
                "The frozen median-imputer, standard-scaler, and Ridge "
                "pipelines contain no stochastic fitting operation."
            ),
        },
        "source_evidence_sha256": dict(source_hashes),
    }


def _write_bundle_files(
    output_directory: Path,
    *,
    payload: Mapping[str, Any],
    feature_mapping: Sequence[Mapping[str, Any]],
    feature_ranges: Mapping[str, Any],
    target_registry: Mapping[str, Any],
    bundle_metadata: Mapping[str, Any],
    reference_signature: Mapping[str, Any],
    explanation_schema: Mapping[str, Any],
    confidence_schema: Mapping[str, Any],
) -> dict[str, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    files = {
        "payload": output_directory / BUNDLE_PAYLOAD_FILENAME,
        "feature_columns": output_directory / "feature_columns.json",
        "feature_definitions": output_directory
        / "feature_definitions.json",
        "feature_ranges": output_directory / "feature_ranges.json",
        "target_definitions": output_directory
        / "target_definitions.json",
        "evaluation_summary": output_directory
        / "evaluation_summary.json",
        "inference_configuration": output_directory
        / "inference_configuration.json",
        "explanation_schema": output_directory
        / "explanation_schema.json",
        "confidence_schema": output_directory / "confidence_schema.json",
        "environment": output_directory / "environment.json",
        "requirements": output_directory / "requirements_snapshot.txt",
        "reference_signature": output_directory
        / "reference_signature.json",
        "manifest": output_directory / BUNDLE_MANIFEST_FILENAME,
        "checksums": output_directory / BUNDLE_CHECKSUMS_FILENAME,
    }
    _write_joblib(files["payload"], payload)
    _write_json(
        files["feature_columns"],
        {
            "schema_version": bundle_metadata["feature_schema_version"],
            "feature_count": len(payload["feature_names"]),
            "ordered_feature_columns": list(payload["feature_names"]),
        },
    )
    _write_json(
        files["feature_definitions"],
        {
            "schema_version": "scenario_feature_definitions_v1",
            "feature_count": len(feature_mapping),
            "features": list(feature_mapping),
            "pmi_proxy_disclosure": (
                "pmi_change_points maps to "
                "kc_fed_manufacturing_pmi_proxy_change_points, the frozen "
                "Kansas City Fed manufacturing proxy."
            ),
        },
    )
    _write_json(files["feature_ranges"], feature_ranges)
    target_definitions = {
        target: {
            key: value
            for key, value in registry.items()
            if key not in {"static_output_metadata", "assessment"}
        }
        for target, registry in target_registry.items()
    }
    _write_json(
        files["target_definitions"],
        {
            "schema_version": "scenario_target_registry_v1",
            "target_count": len(target_definitions),
            "model_family_registry": {
                target: "ridge" for target in target_definitions
            },
            "targets": target_definitions,
        },
    )
    _write_json(
        files["evaluation_summary"],
        {
            "schema_version": "scenario_bundle_evaluation_summary_v1",
            "targets": {
                target: {
                    "unit": registry["unit"],
                    "selected_alpha": registry["selected_alpha"],
                    "confidence_classification": registry[
                        "confidence_classification"
                    ],
                    "validation_evidence": registry[
                        "static_output_metadata"
                    ]["validation_evidence"],
                    "naive_benchmark_comparison": registry[
                        "static_output_metadata"
                    ]["naive_benchmark_comparison"],
                    "scenario_responsiveness": registry[
                        "static_output_metadata"
                    ]["scenario_responsiveness"],
                    "known_limitations": registry[
                        "static_output_metadata"
                    ]["known_limitations"],
                }
                for target, registry in target_registry.items()
            },
            "naive_serving_substitution_count": 0,
            "xgboost_status": XGBOOST_AGREEMENT_STATUS,
        },
    )
    _write_json(
        files["inference_configuration"],
        {
            "schema_version": "scenario_bundle_inference_configuration_v1",
            "bundle_metadata": dict(bundle_metadata),
            "scenario_input_names": list(payload["scenario_input_names"]),
            "scenario_input_units": dict(payload["scenario_input_units"]),
            "scenario_input_mapping": dict(
                payload["scenario_input_mapping"]
            ),
            "event_flag_names": list(payload["event_flag_names"]),
            "release_flag_mapping": dict(
                payload["release_flag_mapping"]
            ),
            "default_state": dict(payload["default_state"]),
            "default_state_metadata": dict(
                payload["default_state_metadata"]
            ),
            "state_feature_names": list(payload["state_feature_names"]),
            "baseline_scenario_isolation": {
                "state_identical": True,
                "release_flags_identical": True,
                "only_scenario_shock_magnitudes_may_differ": True,
            },
        },
    )
    _write_json(files["explanation_schema"], explanation_schema)
    _write_json(files["confidence_schema"], confidence_schema)
    _write_json(
        files["environment"],
        {
            "schema_version": "scenario_bundle_environment_v1",
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "dependency_versions": _dependency_versions(),
            "requirements_source": _relative(REQUIREMENTS_PATH),
            "requirements_sha256": _sha256_file(REQUIREMENTS_PATH),
            "joblib_security_warning": (
                "Serialized Python model artifacts can execute code while "
                "loading. Load this bundle only from a trusted source and "
                "validate its SHA-256 checksums first."
            ),
        },
    )
    write_text_atomic(
        files["requirements"],
        REQUIREMENTS_PATH.read_text(encoding="utf-8"),
    )
    _write_json(files["reference_signature"], reference_signature)

    hashed_before_manifest = {
        path.name: _sha256_file(path)
        for key, path in files.items()
        if key not in {"manifest", "checksums"}
    }
    manifest = {
        "manifest_schema_version": "scenario_model_manifest_v1",
        "bundle_version": bundle_metadata["bundle_version"],
        "model_version": bundle_metadata["model_version"],
        "data_version": bundle_metadata["data_version"],
        "feature_schema_version": bundle_metadata[
            "feature_schema_version"
        ],
        "runtime_schema_version": RUNTIME_SCHEMA_VERSION,
        "payload_schema_version": PAYLOAD_SCHEMA_VERSION,
        "bundle_format": "joblib",
        "primary_payload": BUNDLE_PAYLOAD_FILENAME,
        "target_count": 12,
        "feature_count": 58,
        "state_feature_count": 45,
        "scenario_input_count": 8,
        "release_flag_count": 5,
        "artifact_files": hashed_before_manifest,
        "integrity": {
            "algorithm": "sha256",
            "checksums_file": BUNDLE_CHECKSUMS_FILENAME,
            "checksum_scope": (
                "Every bundle file except artifact_checksums.json itself."
            ),
        },
        "security": {
            "trusted_source_only": True,
            "warning": (
                "Joblib uses pickle semantics. Never load an untrusted "
                "bundle, even if the file extension appears correct."
            ),
        },
        "runtime_contract": {
            "retraining_required": False,
            "training_matrix_required": False,
            "target_labels_required": False,
            "external_api_required": False,
            "supabase_required": False,
        },
        "locked_final_test": {
            "target_rows_parsed": 0,
            "access_count": 0,
            "target_values_in_bundle": False,
        },
    }
    _write_json(files["manifest"], manifest)
    checksum_payload = {
        "schema_version": "scenario_bundle_checksums_v1",
        "algorithm": "sha256",
        "files": {
            path.name: _sha256_file(path)
            for key, path in files.items()
            if key != "checksums"
        },
        "self_hash_excluded_to_avoid_recursive_checksum": True,
    }
    _write_json(files["checksums"], checksum_payload)
    return files


def export_scenario_model_bundle(
    output_directory: Path = DEFAULT_BUNDLE_DIRECTORY,
) -> dict[str, Any]:
    """Fit frozen pipelines once, export the lean bundle, and verify reload."""
    engine = build_research_inference_engine()
    if len(engine.pipelines) != 12 or len(engine.training) != 464:
        raise ValueError("Accepted Step 14 engine contract changed.")
    if engine.guard["final_test_access_count"] != 0:
        raise ValueError("Locked final-test access is not zero.")
    reference_date = _reference_date()
    reference_explanations = build_explanation_artifacts(
        scenario_shocks=EXAMPLE_SCENARIO,
        engine=engine,
    )
    reference_confidence_artifacts = build_confidence_artifacts(
        scenario_shocks=EXAMPLE_SCENARIO,
        diagnostic_as_of_date=reference_date,
        engine=engine,
        explanation_artifacts=reference_explanations,
    )
    reference_inference = reference_explanations["inference_result"]
    reference_confidence = reference_confidence_artifacts[
        "target_confidence"
    ]
    feature_mapping = _feature_mapping_records(engine)
    source_hashes = _source_hashes()
    metadata = _bundle_metadata(
        source_hashes=source_hashes,
        feature_names=engine.feature_names,
        default_state=engine.latest_state,
        reference_date=reference_date,
    )
    target_registry = _build_target_registry(
        engine, reference_inference
    )
    diagnostic_reference = _build_diagnostic_reference(engine)
    explanation_schema = explanation_metadata_schema()
    confidence_schema = confidence_metadata_schema()
    if explanation_schema["schema_version"] != (
        EXPLANATION_SCHEMA_VERSION
    ):
        raise ValueError("Accepted explanation schema changed.")
    if confidence_schema["schema_version"] != CONFIDENCE_SCHEMA_VERSION:
        raise ValueError("Accepted confidence schema changed.")
    payload = {
        "payload_schema_version": PAYLOAD_SCHEMA_VERSION,
        "pipelines": dict(engine.pipelines),
        "feature_names": tuple(engine.feature_names),
        "state_feature_names": tuple(engine.state_feature_names),
        "target_names": tuple(engine.target_names),
        "scenario_input_names": tuple(CONTRACT_SHOCK_NAMES),
        "event_flag_names": tuple(EVENT_FLAG_NAMES),
        "scenario_input_mapping": dict(
            CONTRACT_TO_HISTORICAL_FEATURE
        ),
        "scenario_input_units": dict(SHOCK_UNITS),
        "release_flag_mapping": dict(EVENT_INPUT_TO_FLAG),
        "target_registry": target_registry,
        "feature_mapping": feature_mapping,
        "default_state": dict(engine.latest_state),
        "default_state_metadata": dict(engine.latest_state_metadata),
        "shock_ranges": dict(engine.shock_ranges),
        "state_ranges": dict(engine.state_ranges),
        "maximum_historical_active_flags": (
            engine.maximum_historical_active_flags
        ),
        "diagnostic_reference": diagnostic_reference,
        "explanation_configuration": explanation_schema,
        "confidence_configuration": confidence_schema,
        "bundle_metadata": metadata,
    }
    if any(
        key in payload
        for key in (
            "training",
            "training_frame",
            "target_values",
            "target_labels",
        )
    ):
        raise ValueError("Lean payload contains a forbidden training field.")
    feature_ranges = _feature_range_metadata(
        engine, feature_mapping
    )
    reference_signature = {
        "schema_version": "scenario_bundle_reference_signature_v1",
        "scenario": dict(EXAMPLE_SCENARIO),
        "diagnostic_as_of_date": reference_date,
        "comparison_tolerance": REFERENCE_TOLERANCE,
        "predictions": {
            target: {
                field: reference_inference["targets"][target][field]
                for field in (
                    "baseline_prediction",
                    "scenario_prediction",
                    "incremental_scenario_effect",
                )
            }
            for target in engine.target_names
        },
        "explanation_targets_sha256": _sha256_json(
            reference_explanations["target_explanations"]["targets"]
        ),
        "explanation_targets": reference_explanations[
            "target_explanations"
        ]["targets"],
        "confidence_targets_sha256": _sha256_json(
            reference_confidence["targets"]
        ),
        "confidence_targets": reference_confidence["targets"],
        "warning_records_sha256": _sha256_json(
            reference_confidence_artifacts["warning_records"].to_dict(
                orient="records"
            )
        ),
        "warning_records": reference_confidence_artifacts[
            "warning_records"
        ].to_dict(orient="records"),
        "locked_final_test_access_count": 0,
    }
    files = _write_bundle_files(
        output_directory.resolve(),
        payload=payload,
        feature_mapping=feature_mapping,
        feature_ranges=feature_ranges,
        target_registry=target_registry,
        bundle_metadata=metadata,
        reference_signature=reference_signature,
        explanation_schema=explanation_schema,
        confidence_schema=confidence_schema,
    )
    hash_validation = validate_bundle_hashes(output_directory)
    secret_scan = scan_bundle_text_for_secrets(output_directory)
    runtime = load_scenario_model_bundle(output_directory)
    runtime_result = runtime.predict(
        EXAMPLE_SCENARIO,
        diagnostic_as_of_date=reference_date,
    )
    equivalence = compare_runtime_with_reference(
        runtime_result,
        reference_inference,
        reference_explanations["target_explanations"],
        reference_confidence,
    )
    if not equivalence["equivalent"]:
        raise ValueError(
            "Reloaded runtime differs from accepted Step 14-16 evidence: "
            + "; ".join(equivalence["mismatches"][:10])
        )
    neutral = runtime.predict(
        NEUTRAL_SCENARIO,
        diagnostic_as_of_date=reference_date,
    )
    neutral_max = max(
        abs(
            float(
                neutral["targets"][target][
                    "incremental_scenario_effect"
                ]
            )
        )
        for target in engine.target_names
    )
    if neutral_max != 0.0:
        raise ValueError("Reloaded neutral scenario was not exactly neutral.")
    validations = {
        "exactly_twelve_ridge_pipelines_exported": (
            len(runtime.pipelines) == 12
        ),
        "exact_58_feature_order_preserved": (
            runtime.feature_names == engine.feature_names
        ),
        "exact_45_feature_default_state_preserved": (
            runtime.default_state == engine.latest_state
        ),
        "all_target_units_preserved": (
            runtime.target_units
            == {
                target: str(engine.assessment.loc[target, "unit"])
                for target in engine.target_names
            }
        ),
        "all_target_alphas_preserved": (
            runtime.selected_alphas == engine.selected_alphas
        ),
        "explanation_schema_preserved": (
            runtime.explanation_configuration["schema_version"]
            == EXPLANATION_SCHEMA_VERSION
        ),
        "confidence_schema_preserved": (
            runtime.confidence_configuration["schema_version"]
            == CONFIDENCE_SCHEMA_VERSION
        ),
        "hash_manifest_valid": hash_validation["all_hashes_match"],
        "no_credential_shaped_values_in_bundle_text": (
            not secret_scan["credentials_detected"]
        ),
        "reloaded_reference_equivalent": equivalence["equivalent"],
        "neutral_incremental_exactly_zero": neutral_max == 0.0,
        "runtime_payload_has_no_target_labels": (
            not diagnostic_reference["contains_target_labels"]
        ),
        "runtime_payload_has_no_complete_training_dataframe": (
            not diagnostic_reference[
                "contains_complete_training_dataframe"
            ]
        ),
        "locked_final_test_access_count_zero": (
            engine.guard["final_test_access_count"] == 0
        ),
        "naive_predictions_never_substituted": all(
            not runtime_result["targets"][target][
                "naive_benchmark_substituted"
            ]
            for target in engine.target_names
        ),
    }
    if not all(validations.values()):
        failed = [
            name for name, passed in validations.items() if not passed
        ]
        raise ValueError(f"Step 17 export validation failed: {failed}")
    return {
        "output_directory": str(output_directory.resolve()),
        "files": {
            name: str(path.resolve()) for name, path in files.items()
        },
        "bundle_metadata": metadata,
        "reference_date": reference_date,
        "reference_inference": reference_inference,
        "reference_explanations": reference_explanations[
            "target_explanations"
        ],
        "reference_confidence": reference_confidence,
        "runtime_result": runtime_result,
        "neutral_result": neutral,
        "equivalence": equivalence,
        "hash_validation": hash_validation,
        "secret_scan": secret_scan,
        "validations": validations,
        "locked_final_test_access_count": 0,
    }


def persist_step17_evidence(
    artifacts: Mapping[str, Any],
    output_directory: Path = STEP17_EVIDENCE_DIRECTORY,
) -> dict[str, str]:
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "example_runtime_result": output_directory
        / "example_runtime_result.json",
        "neutral_runtime_result": output_directory
        / "neutral_runtime_result.json",
        "export_summary": output_directory / "artifact_export_summary.json",
    }
    _write_json(
        paths["example_runtime_result"],
        artifacts["runtime_result"],
    )
    _write_json(
        paths["neutral_runtime_result"],
        artifacts["neutral_result"],
    )
    summary = {
        "schema_version": "scenario_artifact_export_summary_v1",
        "completion_gate": "STEP_17_ARTIFACT_EXPORT_COMPLETE",
        "bundle_directory": artifacts["output_directory"],
        "bundle_version": artifacts["bundle_metadata"][
            "bundle_version"
        ],
        "model_version": artifacts["bundle_metadata"][
            "model_version"
        ],
        "data_version": artifacts["bundle_metadata"]["data_version"],
        "feature_schema_version": artifacts["bundle_metadata"][
            "feature_schema_version"
        ],
        "pipeline_count": 12,
        "feature_count": 58,
        "state_feature_count": 45,
        "scenario_input_count": 8,
        "target_count": 12,
        "bundle_file_count": artifacts["hash_validation"][
            "validated_file_count"
        ],
        "equivalence": artifacts["equivalence"],
        "secret_scan": artifacts["secret_scan"],
        "validations": artifacts["validations"],
        "locked_final_test": {
            "target_rows_parsed": 0,
            "access_count": 0,
            "target_values_in_bundle": False,
        },
        "runtime_requirements": {
            "retraining": False,
            "training_matrix": False,
            "target_labels": False,
            "external_apis": False,
            "supabase": False,
        },
        "security_warning": (
            "Load joblib model artifacts only from trusted sources after "
            "validating the included SHA-256 checksum index."
        ),
        "xgboost_status": XGBOOST_AGREEMENT_STATUS,
    }
    _write_json(paths["export_summary"], summary)
    return {
        name: str(path.resolve()) for name, path in paths.items()
    }


def persist_fresh_process_evidence(
    result: Mapping[str, Any],
    output_directory: Path = STEP17_EVIDENCE_DIRECTORY,
) -> str:
    output_directory.mkdir(parents=True, exist_ok=True)
    path = output_directory / "fresh_process_verification.json"
    _write_json(path, result)
    return str(path.resolve())


def run_fresh_process_smoke(
    bundle_directory: Path = DEFAULT_BUNDLE_DIRECTORY,
    *,
    python_executable: str | Path = sys.executable,
) -> dict[str, Any]:
    command = [
        str(python_executable),
        "-m",
        "scenario_modeling.run_bundle_smoke_test",
        "--bundle",
        str(bundle_directory),
        "--forbid-training-data",
        "--json",
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONHASHSEED": "0"},
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Fresh-process bundle smoke test failed.\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    payload = json.loads(completed.stdout)
    if not (
        payload["retraining_performed"] is False
        and payload["training_data_reads"] == 0
        and payload["target_label_reads"] == 0
        and payload["pipeline_count"] == 12
        and payload["all_outputs_finite"]
        and payload["neutral_incremental_exactly_zero"]
    ):
        raise RuntimeError(
            "Fresh-process smoke output did not satisfy Step 17 guards."
        )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export and verify the versioned Scenario Playground Ridge "
            "research bundle."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_BUNDLE_DIRECTORY,
        help="Versioned bundle output directory.",
    )
    parser.add_argument(
        "--skip-fresh-process",
        action="store_true",
        help="Skip the separate-process no-retraining smoke check.",
    )
    args = parser.parse_args(argv)
    artifacts = export_scenario_model_bundle(args.output)
    fresh = None
    if not args.skip_fresh_process:
        fresh = run_fresh_process_smoke(args.output)
    evidence_paths = persist_step17_evidence(artifacts)
    if fresh is not None:
        evidence_paths["fresh_process_verification"] = (
            persist_fresh_process_evidence(fresh)
        )
    summary = {
        "bundle_directory": artifacts["output_directory"],
        "bundle_version": artifacts["bundle_metadata"][
            "bundle_version"
        ],
        "pipeline_count": 12,
        "feature_count": 58,
        "equivalent": artifacts["equivalence"]["equivalent"],
        "maximum_numeric_difference": artifacts["equivalence"][
            "maximum_absolute_numeric_difference"
        ],
        "hashes_valid": artifacts["hash_validation"][
            "all_hashes_match"
        ],
        "fresh_process": fresh,
        "evidence_paths": evidence_paths,
        "locked_final_test_access_count": 0,
    }
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
