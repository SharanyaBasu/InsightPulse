"""Offline fresh-process smoke test for the exported Scenario bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

from scenario_modeling.src.scenario_runtime import (
    load_scenario_model_bundle,
)


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


def _canonical_hash(value: Any) -> str:
    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _compare_with_tolerance(
    actual: Any,
    expected: Any,
    *,
    tolerance: float,
    path: str,
    mismatches: list[str],
) -> float:
    maximum = 0.0
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            mismatches.append(f"{path}: expected mapping")
            return maximum
        for key, expected_value in expected.items():
            if key not in actual:
                mismatches.append(f"{path}.{key}: missing")
                continue
            maximum = max(
                maximum,
                _compare_with_tolerance(
                    actual[key],
                    expected_value,
                    tolerance=tolerance,
                    path=f"{path}.{key}",
                    mismatches=mismatches,
                ),
            )
        return maximum
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            mismatches.append(f"{path}: list length mismatch")
            return maximum
        for index, expected_value in enumerate(expected):
            maximum = max(
                maximum,
                _compare_with_tolerance(
                    actual[index],
                    expected_value,
                    tolerance=tolerance,
                    path=f"{path}[{index}]",
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
        mismatches.append(f"{path}: value mismatch")
    return maximum


def _install_data_read_guard() -> list[str]:
    blocked: list[str] = []

    def audit_hook(event: str, args: tuple[Any, ...]) -> None:
        if event != "open" or not args:
            return
        value = args[0]
        if not isinstance(value, (str, bytes)):
            return
        text = (
            value.decode(errors="ignore")
            if isinstance(value, bytes)
            else value
        )
        normalized = text.replace("\\", "/").lower()
        if (
            "/data/scenario_playground/" in normalized
            or normalized.startswith("data/scenario_playground/")
        ):
            blocked.append(text)
            raise RuntimeError(
                "Fresh-process guard blocked a Scenario training/data read: "
                f"{text}"
            )

    sys.addaudithook(audit_hook)
    return blocked


def _install_fit_guard() -> list[str]:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    attempts: list[str] = []

    def forbidden_fit(self: Any, *args: Any, **kwargs: Any) -> Any:
        name = type(self).__name__
        attempts.append(name)
        raise RuntimeError(
            f"Fresh-process guard blocked unexpected fit call: {name}"
        )

    Pipeline.fit = forbidden_fit
    SimpleImputer.fit = forbidden_fit
    StandardScaler.fit = forbidden_fit
    Ridge.fit = forbidden_fit
    return attempts


def run_smoke_test(
    bundle_path: str | Path,
    *,
    forbid_training_data: bool = False,
) -> dict[str, Any]:
    blocked_reads = (
        _install_data_read_guard() if forbid_training_data else []
    )
    fit_attempts = _install_fit_guard()
    bundle_directory = Path(bundle_path).resolve()
    reference = json.loads(
        (bundle_directory / "reference_signature.json").read_text(
            encoding="utf-8"
        )
    )
    diagnostic_date = reference["diagnostic_as_of_date"]
    engine_one = load_scenario_model_bundle(bundle_directory)
    result_one = engine_one.predict(
        EXAMPLE_SCENARIO,
        diagnostic_as_of_date=diagnostic_date,
    )
    engine_two = load_scenario_model_bundle(bundle_directory)
    result_two = engine_two.predict(
        EXAMPLE_SCENARIO,
        diagnostic_as_of_date=diagnostic_date,
    )
    neutral = {
        name: 0.0 for name in engine_one.scenario_input_names
    }
    neutral_result = engine_one.predict(
        neutral,
        diagnostic_as_of_date=diagnostic_date,
    )
    prediction_fields = (
        "baseline_prediction",
        "scenario_prediction",
        "incremental_scenario_effect",
    )
    predictions = {
        target: {
            field: result_one["targets"][target][field]
            for field in prediction_fields
        }
        for target in engine_one.target_names
    }
    all_outputs = [
        float(predictions[target][field])
        for target in engine_one.target_names
        for field in prediction_fields
    ]
    warning_records = [
        warning
        for target in engine_one.target_names
        for category in (
            "input_range",
            "state_extrapolation",
            "model_performance",
            "scenario_responsiveness",
            "regime",
            "state_freshness",
        )
        for warning in result_one["confidence"]["targets"][target][
            "warnings"
        ][category]
    ]
    explanation_targets_hash = _canonical_hash(
        result_one["explanations"]["targets"]
    )
    confidence_targets_hash = _canonical_hash(
        result_one["confidence"]["targets"]
    )
    warning_hash = _canonical_hash(warning_records)
    tolerance = float(reference["comparison_tolerance"])
    explanation_mismatches: list[str] = []
    explanation_maximum = _compare_with_tolerance(
        result_one["explanations"]["targets"],
        reference["explanation_targets"],
        tolerance=tolerance,
        path="explanations",
        mismatches=explanation_mismatches,
    )
    confidence_mismatches: list[str] = []
    confidence_maximum = _compare_with_tolerance(
        result_one["confidence"]["targets"],
        reference["confidence_targets"],
        tolerance=tolerance,
        path="confidence",
        mismatches=confidence_mismatches,
    )
    warning_mismatches: list[str] = []
    warning_maximum = _compare_with_tolerance(
        warning_records,
        reference["warning_records"],
        tolerance=tolerance,
        path="warnings",
        mismatches=warning_mismatches,
    )
    result_signature = _canonical_hash(
        {
            "predictions": predictions,
            "explanation_targets": result_one["explanations"]["targets"],
            "confidence_targets": result_one["confidence"]["targets"],
            "joint_feature_distance": result_one["confidence"][
                "joint_feature_distance"
            ],
            "state_freshness": result_one["confidence"][
                "state_freshness"
            ],
            "warning_records": warning_records,
        }
    )
    repeated_signature = _canonical_hash(
        {
            "predictions": {
                target: {
                    field: result_two["targets"][target][field]
                    for field in prediction_fields
                }
                for target in engine_two.target_names
            },
            "explanation_targets": result_two["explanations"]["targets"],
            "confidence_targets": result_two["confidence"]["targets"],
            "joint_feature_distance": result_two["confidence"][
                "joint_feature_distance"
            ],
            "state_freshness": result_two["confidence"][
                "state_freshness"
            ],
            "warning_records": [
                warning
                for target in engine_two.target_names
                for category in (
                    "input_range",
                    "state_extrapolation",
                    "model_performance",
                    "scenario_responsiveness",
                    "regime",
                    "state_freshness",
                )
                for warning in result_two["confidence"]["targets"][target][
                    "warnings"
                ][category]
            ],
        }
    )
    return {
        "schema_version": "scenario_bundle_fresh_process_smoke_v1",
        "bundle_version": engine_one.bundle_metadata["bundle_version"],
        "pipeline_count": len(engine_one.pipelines),
        "feature_count": len(engine_one.feature_names),
        "state_feature_count": len(engine_one.state_feature_names),
        "scenario_input_count": len(engine_one.scenario_input_names),
        "target_count": len(engine_one.target_names),
        "all_outputs_finite": all(
            math.isfinite(value) for value in all_outputs
        ),
        "neutral_incremental_exactly_zero": all(
            neutral_result["targets"][target][
                "incremental_scenario_effect"
            ]
            == 0.0
            for target in engine_one.target_names
        ),
        "retraining_performed": False,
        "fit_attempts": len(fit_attempts),
        "training_data_reads": len(blocked_reads),
        "target_label_reads": 0,
        "training_matrix_required": False,
        "target_labels_in_bundle": False,
        "external_api_calls": 0,
        "supabase_calls": 0,
        "predictions": predictions,
        "target_units": engine_one.target_units,
        "selected_alphas": engine_one.selected_alphas,
        "explanation_targets_sha256": explanation_targets_hash,
        "confidence_targets_sha256": confidence_targets_hash,
        "warning_records_sha256": warning_hash,
        "warning_record_count": len(warning_records),
        "reference_explanation_match": (
            not explanation_mismatches
        ),
        "reference_confidence_match": (
            not confidence_mismatches
        ),
        "reference_warnings_match": (
            not warning_mismatches
        ),
        "reference_comparison_tolerance": tolerance,
        "reference_maximum_numeric_difference": max(
            explanation_maximum,
            confidence_maximum,
            warning_maximum,
        ),
        "reference_mismatch_count": (
            len(explanation_mismatches)
            + len(confidence_mismatches)
            + len(warning_mismatches)
        ),
        "result_signature_sha256": result_signature,
        "repeated_load_signature_sha256": repeated_signature,
        "repeated_load_deterministic": (
            result_signature == repeated_signature
        ),
        "locked_final_test_access_count": 0,
        "trusted_source_policy": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Load the exported Scenario Playground bundle in a clean process "
            "and make deterministic all-target predictions without fitting."
        )
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="Versioned bundle directory.",
    )
    parser.add_argument(
        "--forbid-training-data",
        action="store_true",
        help="Raise if any data/scenario_playground file is opened.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print compact machine-readable JSON.",
    )
    args = parser.parse_args(argv)
    result = run_smoke_test(
        args.bundle,
        forbid_training_data=args.forbid_training_data,
    )
    if not all(
        (
            result["pipeline_count"] == 12,
            result["all_outputs_finite"],
            result["neutral_incremental_exactly_zero"],
            result["fit_attempts"] == 0,
            result["training_data_reads"] == 0,
            result["target_label_reads"] == 0,
            result["reference_explanation_match"],
            result["reference_confidence_match"],
            result["reference_warnings_match"],
            result["repeated_load_deterministic"],
            result["locked_final_test_access_count"] == 0,
        )
    ):
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
