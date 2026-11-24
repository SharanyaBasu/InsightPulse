"""Matched, training-only scenario diagnostics with verified fold Ridge models.

No loaders, network access, or import-time execution. Callers supply eligible
training observations, validation features, cached Ridge evidence, and an
already fitted XGBoost model. These are predictive associations, not effects
identified by a causal experiment.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .all_target_models import TARGET_NAMES, _representative_states, _target_unit
from .model_matrix import FINAL_FEATURE_NAMES
from .ridge_baselines import make_ridge_pipeline
from .scenario_inference import SHOCK_UNITS
from .shock_features import (
    CONTRACT_SHOCK_NAMES, CONTRACT_TO_HISTORICAL_FEATURE, EVENT_FLAG_NAMES,
    HISTORICAL_SHOCK_NAMES,
)


NEAR_FLAT_IQR_FRACTION = 0.02
ABSOLUTE_TOLERANCE = 1e-10
RELATIVE_TOLERANCE = 1e-11
DIRECTION_EPSILON = 1e-12
STATE_QUANTILES = {"low_vix": 0.10, "median_vix": 0.50, "high_vix": 0.90}
SENSITIVITY_TABLES = (
    "sensitivity_detail", "sensitivity_summary", "ridge_reconstruction_audit",
    "sensitivity_flags",
)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _schema(features: Sequence[str]) -> list[str]:
    if tuple(features) != FINAL_FEATURE_NAMES:
        raise ValueError("Matched sensitivity requires the exact frozen 58-feature order.")
    return list(features)


def _input_hash(row: pd.Series, features: Sequence[str]) -> str:
    values = [float(row[name]).hex() for name in features]
    return hashlib.sha256(_json(values).encode("utf-8")).hexdigest()


def select_representative_states(
    training: pd.DataFrame, features: Sequence[str] = FINAL_FEATURE_NAMES,
) -> list[tuple[str, pd.Series]]:
    """Reuse Step 12/13's real-row 10th/50th/90th VIX percentile convention."""
    _schema(features)
    if training.empty or not np.isfinite(training["vix_level_points"].to_numpy(dtype=float)).all():
        raise ValueError("Representative states require finite eligible-training VIX.")
    # Resetting the index makes first-occurrence ties stable even with duplicate
    # external index labels. This does not synthesize or reorder observations.
    return _representative_states(training.reset_index(drop=True))


def construct_shock_grid(values: pd.Series) -> dict[str, Any]:
    """Select up to five distinct observed values by evenly spaced support ranks."""
    numeric = np.asarray(values, dtype=float)
    supported = np.unique(numeric[np.isfinite(numeric)])
    count = len(supported)
    if count <= 5:
        selected = supported.copy()
    else:
        selected = supported[np.rint(np.linspace(0, count - 1, 5)).astype(int)]
        if np.any(supported == 0.0) and not np.any(selected == 0.0):
            # Preserve both observed endpoints and exactly five distinct values.
            interior = 1 + int(np.argmin(np.abs(selected[1:-1])))
            selected[interior] = 0.0
            selected.sort()
    warnings = []
    if count < 5:
        warnings.append("SPARSE_OBSERVED_SUPPORT")
    if not np.any(supported == 0.0):
        warnings.append("NEUTRAL_ZERO_NOT_OBSERVED")
    if not np.isfinite(numeric).all():
        warnings.append("NONFINITE_TRAINING_SUPPORT_EXCLUDED")
    return {
        "grid_values": selected.tolist(), "supported_distinct_values": count,
        "training_min": float(supported[0]) if count else None,
        "training_max": float(supported[-1]) if count else None,
        "sparse_grid": count < 5, "zero_observed": bool(np.any(supported == 0.0)),
        "support_status": "AVAILABLE" if count else "UNAVAILABLE",
        "support_warnings": warnings,
    }


def training_target_scale(values: pd.Series) -> dict[str, Any]:
    numeric = np.asarray(values, dtype=float)
    if not len(numeric):
        return {"training_target_iqr": None, "iqr_status": "UNAVAILABLE", "training_target_range": None}
    if not np.isfinite(numeric).all():
        return {"training_target_iqr": None, "iqr_status": "NONFINITE_TRAINING_TARGET", "training_target_range": None}
    with np.errstate(over="ignore", invalid="ignore"):
        q25, q75 = np.quantile(numeric, [0.25, 0.75])
        iqr = float(q75 - q25)
        value_range = float(np.max(numeric) - np.min(numeric))
    if not np.isfinite(iqr):
        return {"training_target_iqr": None, "iqr_status": "NONFINITE_IQR", "training_target_range": value_range if np.isfinite(value_range) else None}
    return {
        "training_target_iqr": iqr, "iqr_status": "AVAILABLE" if iqr > 0 else "ZERO_IQR",
        "training_target_range": value_range if np.isfinite(value_range) else None,
    }


def sensitivity_metadata() -> dict[str, Any]:
    return {
        "state_selection": "Real eligible-training row nearest VIX quantiles; first occurrence resolves ties.",
        "state_quantiles": dict(STATE_QUANTILES),
        "shock_grid": "Up to five distinct observed values at evenly spaced sorted-support ranks; retain endpoints and observed zero.",
        "unsupported_zero": "Neutral baseline remains zero; an unobserved zero is not added to the observed shock grid.",
        "feature_schema": list(FINAL_FEATURE_NAMES),
        "shock_units": dict(SHOCK_UNITS),
        "contract_to_historical_feature": dict(CONTRACT_TO_HISTORICAL_FEATURE),
        "pmi_disclosure": "PMI Change uses the Kansas City Fed manufacturing proxy, not national ISM PMI.",
        "target_units": {target: _target_unit(target) for target in TARGET_NAMES},
        "near_flat_iqr_fraction": NEAR_FLAT_IQR_FRACTION,
        "iqr_handling": "Training-target IQR only; normalized span and near-flat are null when zero, nonfinite or unavailable.",
        "ridge_reconstruction_tolerance": {"absolute": ABSOLUTE_TOLERANCE, "relative": RELATIVE_TOLERANCE},
        "numerical_checks": "Finite predictions; identical neutral row yields exactly zero increment; repeated identical batches must match exactly.",
        "direction_epsilon": DIRECTION_EPSILON,
        "discontinuity_warning": "Reuse Step 13 envelope max(training target range, 10 return percentage points or 50 yield bps); flag span above envelope or adjacent jump above half envelope.",
        "promotion_policy": "Diagnostics only. No promotion criteria added; unresolved Ridge reconstruction mismatches block promotion approval.",
        "interpretation": "Predictive associations, not causal economic effects. Piecewise constant and nonmonotonic responses are permitted.",
    }


def matched_response_tables(
    training: pd.DataFrame, features: Sequence[str], *, target: str, fold: int,
    predictors: Mapping[str, Callable[[pd.DataFrame], np.ndarray]],
) -> dict[str, pd.DataFrame]:
    """Supply identical neutral/scenario batches to passed fitted predictors."""
    columns = _schema(features)
    if target not in TARGET_NAMES or not predictors or not set(predictors) <= {"ridge", "xgboost"}:
        raise ValueError("Unknown target or model family.")
    scale = training_target_scale(training[target])
    iqr = scale["training_target_iqr"]
    normalizable = scale["iqr_status"] == "AVAILABLE"
    envelope = max(50.0 if target == "ten_year_yield_bps" else 10.0, scale["training_target_range"] or 0.0)
    details, summaries = [], []
    for state_name, state in select_representative_states(training, columns):
        neutral = state[columns].copy()
        neutral.loc[list(HISTORICAL_SHOCK_NAMES)] = 0.0
        frozen = [name for name in columns if name not in HISTORICAL_SHOCK_NAMES]
        flags_json = _json({name: float(neutral[name]) for name in EVENT_FLAG_NAMES})
        for shock in CONTRACT_SHOCK_NAMES:
            feature = CONTRACT_TO_HISTORICAL_FEATURE[shock]
            grid = construct_shock_grid(training[feature])
            grid_values = grid["grid_values"]
            evaluation_values = sorted(set([0.0, *grid_values]))
            batch = pd.DataFrame([neutral.copy() for _ in evaluation_values], columns=columns)
            batch[feature] = evaluation_values
            if not batch[frozen].eq(neutral[frozen], axis="columns").all().all():
                raise AssertionError("State or release flags changed during a shock sweep.")
            neutral_index = evaluation_values.index(0.0)
            common = {
                "target": target, "unit": _target_unit(target), "fold": fold,
                "state": state_name, "state_sample_date": str(state["sample_date"]),
                "state_vix": float(state["vix_level_points"]), "state_quantile": STATE_QUANTILES[state_name],
                "state_selection": "nearest_training_vix_quantile_first_tie",
                "shock": shock, "historical_feature": feature, "shock_unit": SHOCK_UNITS[shock],
                "grid_values_json": _json(grid_values),
                **{key: value for key, value in grid.items() if key not in {"grid_values", "support_warnings"}},
                "release_flags_json": flags_json, "neutral_input_sha256": _input_hash(neutral, columns),
                **scale,
            }
            input_hashes = [_input_hash(row, columns) for _, row in batch.iterrows()]
            for family, predict in predictors.items():
                first = np.asarray(predict(batch.copy(deep=True)), dtype=float).reshape(-1)
                repeated = np.asarray(predict(batch.copy(deep=True)), dtype=float).reshape(-1)
                if len(first) != len(batch) or len(repeated) != len(batch):
                    raise ValueError("Sensitivity predictor returned the wrong row count.")
                predictions_finite = bool(np.isfinite(first).all() and np.isfinite(repeated).all())
                repeatable = bool(np.array_equal(first, repeated, equal_nan=True))
                baseline = float(first[neutral_index]) if np.isfinite(first[neutral_index]) else None
                grid_indices = [evaluation_values.index(value) for value in grid_values]
                responses = first[grid_indices]
                with np.errstate(over="ignore", invalid="ignore"):
                    raw_span = float(np.ptp(responses)) if predictions_finite and len(responses) else None
                    raw_changes = np.diff(responses) if predictions_finite else np.array([])
                    raw_increments = first - first[neutral_index]
                    repeat_difference = float(np.max(np.abs(first - repeated))) if predictions_finite else None
                arithmetic_finite = bool((raw_span is None or np.isfinite(raw_span)) and np.isfinite(raw_changes).all() and np.isfinite(raw_increments).all())
                valid = predictions_finite and arithmetic_finite
                span = raw_span if valid else None
                changes = raw_changes if valid else np.array([])
                signs = np.sign(changes[np.abs(changes) > DIRECTION_EPSILON])
                turns = int(np.sum(signs[1:] != signs[:-1]))
                max_jump = float(np.max(np.abs(changes))) if len(changes) else (0.0 if span is not None else None)
                endpoint = float(responses[-1] - responses[0]) if span is not None else None
                near_flat = bool(span <= NEAR_FLAT_IQR_FRACTION * iqr) if normalizable and span is not None else None
                warnings = list(grid["support_warnings"])
                if not normalizable:
                    warnings.append("IQR_NORMALIZATION_UNAVAILABLE")
                if not predictions_finite:
                    warnings.append("NONFINITE_PREDICTION")
                elif not arithmetic_finite:
                    warnings.append("NONFINITE_RESPONSE_ARITHMETIC")
                if not repeatable:
                    warnings.append("NONREPEATABLE_PREDICTION")
                if span is not None and span > envelope:
                    warnings.append("RESPONSE_SPAN_EXCEEDS_STEP13_ENVELOPE")
                if max_jump is not None and max_jump > envelope / 2:
                    warnings.append("SUSPICIOUS_ADJACENT_JUMP")
                for index, (value, prediction_index) in enumerate(zip(grid_values, grid_indices, strict=True)):
                    scenario = float(first[prediction_index]) if np.isfinite(first[prediction_index]) else None
                    increment = scenario - baseline if scenario is not None and baseline is not None else None
                    if increment is not None and not np.isfinite(increment):
                        increment = None
                    details.append({
                        **common, "model_family": family, "grid_index": index, "grid_value": value,
                        "input_sha256": input_hashes[prediction_index],
                        "baseline_prediction": baseline, "scenario_prediction": scenario,
                        "incremental_effect": increment, "numerically_valid": valid,
                        "repeatable": repeatable,
                    })
                summaries.append({
                    **common, "model_family": family, "baseline_prediction": baseline,
                    "neutral_incremental_effect": 0.0 if baseline is not None else None,
                    "raw_response_range": span,
                    "normalized_response_range": span / iqr if normalizable and span is not None else None,
                    "near_flat_threshold": NEAR_FLAT_IQR_FRACTION * iqr if normalizable else None,
                    "near_flat": near_flat, "direction_changes": turns,
                    "adjacent_changes_json": _json(changes.tolist()), "maximum_adjacent_jump": max_jump,
                    "endpoint_change": endpoint, "step13_warning_envelope": envelope,
                    "numerically_valid": valid, "repeatable": repeatable,
                    "maximum_repeat_difference": repeat_difference if repeat_difference is not None and np.isfinite(repeat_difference) else None,
                    "warnings_json": _json(warnings),
                })
    return {"sensitivity_detail": pd.DataFrame(details), "sensitivity_summary": pd.DataFrame(summaries)}


def reconstruct_verified_ridge(
    training: pd.DataFrame, validation: pd.DataFrame, features: Sequence[str], *,
    target: str, fold: int, fit_record: Mapping[str, Any], saved_predictions: pd.DataFrame,
) -> tuple[Any | None, dict[str, Any]]:
    """Reconstruct one original-alpha Ridge and gate it on saved predictions."""
    columns = _schema(features)
    unit = _target_unit(target)
    alpha = float(fit_record["selected_alpha"])
    if target not in TARGET_NAMES or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Invalid original Ridge target or alpha.")
    for frame in (training, validation):
        if frame.empty or frame["sample_date"].duplicated().any() or not frame["sample_date"].is_monotonic_increasing:
            raise ValueError("Ridge reconstruction requires unique chronological rows.")
    boundary = str(validation["target_baseline_date"].iloc[0])
    if not (pd.to_datetime(training["target_end_date"]) < pd.Timestamp(boundary)).all():
        raise ValueError("Ridge reconstruction violates the target-end-date purge.")
    if not set(training["sample_date"]).isdisjoint(validation["sample_date"]):
        raise ValueError("Ridge reconstruction training and validation overlap.")
    expected = {
        "target": target, "fold": fold, "unit": unit, "outer_training_rows": len(training),
        "fit_first_sample": training["sample_date"].iloc[0], "fit_last_sample": training["sample_date"].iloc[-1],
    }
    if any(fit_record.get(key) != value for key, value in expected.items()):
        raise ValueError("Ridge reconstruction differs from the original fitting scope.")
    saved = saved_predictions.loc[
        saved_predictions["target"].eq(target) & saved_predictions["fold"].eq(fold)
        & saved_predictions["strategy"].eq("ridge_full_history")
    ].copy()
    if len(saved) != len(validation) or saved["sample_date"].duplicated().any() or set(saved["sample_date"]) != set(validation["sample_date"]):
        raise ValueError("Incomplete authoritative Ridge validation predictions.")
    saved = saved.set_index("sample_date").loc[validation["sample_date"]]
    if not saved["unit"].eq(unit).all():
        raise ValueError("Authoritative Ridge units differ.")
    for name in ("target_baseline_date", "target_end_date"):
        if saved[name].tolist() != validation[name].tolist():
            raise ValueError("Authoritative Ridge validation boundaries differ.")
    pipeline = make_ridge_pipeline(alpha)
    pipeline.fit(training[columns], training[target])
    observed = np.asarray(pipeline.predict(validation[columns]), dtype=float)
    reference = saved["prediction"].to_numpy(dtype=float)
    finite = bool(np.isfinite(observed).all() and np.isfinite(reference).all())
    difference = np.abs(observed - reference)
    passed = bool(finite and np.all(difference <= ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * np.abs(reference)))
    audit = {
        **expected, "selected_alpha": alpha, "comparison_count": len(reference),
        "training_last_target_end": str(training["target_end_date"].max()),
        "validation_first_baseline": boundary,
        "maximum_absolute_difference": float(difference.max()) if finite else None,
        "absolute_tolerance": ABSOLUTE_TOLERANCE, "relative_tolerance": RELATIVE_TOLERANCE,
        "passed": passed, "status": "PASS" if passed else "FAIL",
        "reason": "" if passed else ("NONFINITE_PREDICTIONS" if not finite else "SAVED_PREDICTION_MISMATCH"),
        "promotion_approval_blocked": not passed,
    }
    return (pipeline if passed else None), audit


def diagnose_target_fold(
    training: pd.DataFrame, validation: pd.DataFrame, features: Sequence[str], *,
    target: str, fold: int, xgboost_fit: Any, fit_record: Mapping[str, Any],
    saved_predictions: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Reuse fitted XGBoost and admit reconstructed Ridge only after verification."""
    if xgboost_fit.target != target or xgboost_fit.unit != _target_unit(target) or xgboost_fit.outer_fold != fold or tuple(xgboost_fit.feature_names) != tuple(features):
        raise ValueError("XGBoost model identity does not match the diagnostic fold.")
    ridge, audit = reconstruct_verified_ridge(
        training, validation, features, target=target, fold=fold,
        fit_record=fit_record, saved_predictions=saved_predictions,
    )
    predictors = {"xgboost": xgboost_fit.predict}
    if ridge is not None:
        predictors["ridge"] = ridge.predict
    tables = matched_response_tables(training, features, target=target, fold=fold, predictors=predictors)
    tables["ridge_reconstruction_audit"] = pd.DataFrame([audit])
    return tables


def responsiveness_flags(summary: pd.DataFrame) -> pd.DataFrame:
    """Record evidence-backed warnings without applying model-promotion rules."""
    rows = []
    for (family, target), group in summary.groupby(["model_family", "target"], sort=True):
        evidence = []
        for shock, shock_rows in group.groupby("shock", sort=True):
            near = shock_rows["near_flat"].eq(True)
            evidence.append({"shock": shock, "near_flat_checks": int(near.sum()), "checks": len(shock_rows),
                             "all_near_flat": bool(near.all()),
                             "max_raw_range": float(shock_rows["raw_response_range"].max()) if shock_rows["raw_response_range"].notna().any() else None})
        flat = sum(item["all_near_flat"] for item in evidence)
        fold_flat = []
        for fold, fold_rows in group.groupby("fold", sort=True):
            all_flat_shocks = [shock for shock, checks in fold_rows.groupby("shock", sort=True)
                               if checks["near_flat"].eq(True).all()]
            fold_flat.append({"fold": int(fold), "uniformly_near_flat_shocks": all_flat_shocks,
                              "count": len(all_flat_shocks)})
        reversals = []
        for dimension, keys in (("across_states", ["fold", "shock"]), ("across_folds", ["state", "shock"])):
            for key, subset in group.groupby(keys, sort=True):
                active = subset.loc[subset["near_flat"].eq(False) & subset["numerically_valid"].eq(True)]
                values = active["endpoint_change"].dropna()
                if (values > DIRECTION_EPSILON).any() and (values < -DIRECTION_EPSILON).any():
                    reversals.append({"dimension": dimension, "key": list(key), "responses": active[["fold", "state", "shock", "endpoint_change", "raw_response_range", "normalized_response_range"]].to_dict("records")})
        warnings = []
        if flat:
            warnings.append("ALL_EIGHT_SHOCKS_NEAR_FLAT" if flat == 8 else "LIMITED_SHOCK_RESPONSIVENESS")
        if any(item["count"] == 8 for item in fold_flat):
            warnings.append("ALL_EIGHT_SHOCKS_NEAR_FLAT_IN_FOLD")
        if reversals:
            warnings.append("NONFLAT_RESPONSE_SIGN_INSTABILITY")
        inherited = sorted({warning for value in group["warnings_json"] for warning in json.loads(value)})
        warnings.extend(inherited)
        rows.append({
            "model_family": family, "target": target, "unit": _target_unit(target),
            "checks": len(group), "uniformly_near_flat_shocks": flat,
            "shock_evidence_json": _json(evidence), "sign_instability_json": _json(reversals),
            "fold_near_flat_evidence_json": _json(fold_flat),
            "sign_instability_groups": len(reversals),
            "nonfinite_checks": int((~group["numerically_valid"]).sum()),
            "nonrepeatable_checks": int((~group["repeatable"]).sum()),
            "warnings_json": _json(sorted(set(warnings))),
        })
    return pd.DataFrame(rows)


def combine_sensitivity_results(results: Sequence[dict[str, pd.DataFrame]]) -> dict[str, Any]:
    tables = {name: pd.concat([result[name] for result in results], ignore_index=True)
              for name in SENSITIVITY_TABLES if name != "sensitivity_flags"}
    tables["sensitivity_flags"] = responsiveness_flags(tables["sensitivity_summary"])
    metadata = sensitivity_metadata()
    metadata["promotion_approval_blocked_by_ridge_mismatch"] = bool((~tables["ridge_reconstruction_audit"]["passed"]).any())
    tables["sensitivity_metadata"] = metadata
    return tables


def validate_sensitivity_artifacts(artifacts: Mapping[str, Any]) -> None:
    """Check complete optional experiment diagnostics before publication."""
    audit = artifacts["ridge_reconstruction_audit"]
    expected = {(fold, target) for fold in range(1, 6) for target in TARGET_NAMES}
    if len(audit) != 60 or audit.duplicated(["fold", "target"]).any() or set(audit[["fold", "target"]].itertuples(index=False, name=None)) != expected:
        raise ValueError("Sensitivity requires all 60 Ridge verification records.")
    summary, detail = artifacts["sensitivity_summary"], artifacts["sensitivity_detail"]
    keys = ["model_family", "target", "fold", "state", "shock"]
    if summary.duplicated(keys).any() or detail.duplicated([*keys, "grid_index"]).any():
        raise ValueError("Duplicate sensitivity keys.")
    expected_summary = set()
    for record in audit.itertuples(index=False):
        families = ("ridge", "xgboost") if record.passed else ("xgboost",)
        expected_summary.update((family, record.target, record.fold, state, shock)
                                for family in families for state in STATE_QUANTILES for shock in CONTRACT_SHOCK_NAMES)
    if set(summary[keys].itertuples(index=False, name=None)) != expected_summary:
        raise ValueError("Sensitivity model membership differs from reconstruction gates.")
    if not summary["unit"].eq(summary["target"].map(_target_unit)).all():
        raise ValueError("Sensitivity units changed.")
    if not summary["shock_unit"].eq(summary["shock"].map(SHOCK_UNITS)).all():
        raise ValueError("Shock units changed.")
    groups = dict(tuple(detail.groupby(keys, sort=False))) if len(detail) else {}
    if not set(groups) <= expected_summary:
        raise ValueError("Unexpected detailed sensitivity keys.")
    for _, row in summary.iterrows():
        values = json.loads(row["grid_values_json"])
        part = groups.get(tuple(row[key] for key in keys), detail.iloc[:0])
        if len(values) > 5 or part["grid_value"].tolist() != values:
            raise ValueError("Detailed grids differ from their summary.")
        if row["numerically_valid"] and (not np.isfinite(part[["baseline_prediction", "scenario_prediction", "incremental_effect"]].to_numpy(dtype=float)).all() or row["neutral_incremental_effect"] != 0):
            raise ValueError("Invalid neutral or finite-response evidence.")
        if len(part):
            with np.errstate(over="ignore", invalid="ignore"):
                expected_increments = part["scenario_prediction"].to_numpy(dtype=float) - part["baseline_prediction"].to_numpy(dtype=float)
            expected_increments[~np.isfinite(expected_increments)] = np.nan
            if not np.allclose(expected_increments, part["incremental_effect"].to_numpy(dtype=float), rtol=0, atol=0, equal_nan=True):
                raise ValueError("Incremental effects differ from scenario minus baseline.")
        zero = part.loc[part["grid_value"].eq(0)]
        if row["numerically_valid"] and not zero["incremental_effect"].eq(0).all():
            raise ValueError("Nonzero neutral incremental effect.")
    matched_keys = ["target", "fold", "state", "shock", "grid_index"]
    for name in ("input_sha256", "release_flags_json", "neutral_input_sha256", "grid_values_json"):
        if not detail.groupby(matched_keys)[name].nunique().le(1).all():
            raise ValueError("Ridge/XGBoost did not receive identical scenario inputs.")
    metadata = artifacts["summary"]["sensitivity_metadata"]
    if metadata["feature_schema"] != list(FINAL_FEATURE_NAMES) or metadata["promotion_approval_blocked_by_ridge_mismatch"] != bool((~audit["passed"]).any()):
        raise ValueError("Sensitivity metadata or mismatch blocker is inconsistent.")
