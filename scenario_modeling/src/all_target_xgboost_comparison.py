"""Build comparison evidence using frozen folds and cached Ridge predictions.

Importing this module performs no data access or training. Historical execution
requires an explicit call to build_all_target_comparison_artifacts. Persistence
is a separate operation and never serializes fitted models.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .all_target_models import (
    OUTPUT_ROOT as RIDGE_EVIDENCE_ROOT,
    TARGET_NAMES,
    _aggregate_metrics,
    _json_safe,
    _target_unit,
    build_walk_forward_folds,
    load_all_pretest_rows,
)
from .config import INTERIM_ROOT, PROJECT_ROOT, PROCESSED_ROOT, RAW_ROOT
from .evaluation import regression_metrics
from .feature_sources import frame_to_csv_text
from .model_matrix import FINAL_FEATURE_NAMES
from .ridge_baselines import MATRIX_PATH, NAIVE_STRATEGIES, SPLIT_MANIFEST_PATH
from .walk_forward_validation import _inner_split, fit_target_xgboost
from .xgboost_prototype import ARCHITECTURES, SEED


OUTPUT_ROOT = INTERIM_ROOT / "models" / "all_target_xgboost_comparison"
APPROVED_PRETEST_SHA256 = "7f14589043fb7ae6d93a172595c477593e6ef5e8bf0f8f87cdde43c94db2e008"
FINGERPRINT_FORMAT = "insightpulse-retained-pretest-floathex-v1"
ABSOLUTE_TOLERANCE = 1e-10
RELATIVE_TOLERANCE = 1e-11
RIDGE_STRATEGY = "ridge_full_history"
XGBOOST_STRATEGY = "xgboost_full_history"
SAVED_STRATEGIES = (*NAIVE_STRATEGIES, RIDGE_STRATEGY)
STRATEGIES = (*SAVED_STRATEGIES, XGBOOST_STRATEGY)
KEY_COLUMNS = ("fold", "sample_date", "target", "strategy")
MATCH_COLUMNS = ("fold", "sample_date", "target")
SAVED_COLUMNS = (
    "fold", "sample_date", "target_baseline_date", "target_end_date", "target",
    "unit", "strategy", "actual", "prediction", "residual", "absolute_error",
    "vix_level_points",
)
TRAINING_COUNTS = (160, 222, 280, 342, 401)
TABLE_ROWS = {
    "authoritative_predictions": 14400,
    "xgboost_predictions": 3600,
    "predictions": 18000,
    "candidate_audit": 180,
    "xgboost_fit_audit": 60,
    "ridge_fit_audit": 60,
    "fold_metrics": 300,
    "pooled_metrics": 60,
    "family_comparison": 12,
    "naive_comparison": 72,
    "regime_metrics": 180,
    "generalization": 120,
    "fold_manifest": 5,
}
DEPENDENCE_NOTE = (
    "The 300 validation sample identifiers are distinct. Their 21-session "
    "forward-label windows can overlap within and across folds, so pooled "
    "errors are descriptive and observations are not assumed independent."
)


@dataclass
class SavedComparisonEvidence:
    """Verified cached evidence, retaining the original prediction columns."""

    predictions: pd.DataFrame
    ridge_fit_audit: pd.DataFrame
    fold_manifest: pd.DataFrame
    hashes: dict[str, str]
    summary_sha256: str


def _sha256(path: Path) -> str:
    if path.resolve() == MATRIX_PATH.resolve():
        raise ValueError("Full canonical matrix hashing is forbidden.")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def authorized_pretest_fingerprint(
    frame: pd.DataFrame,
    features: Sequence[str] = FINAL_FEATURE_NAMES,
    targets: Sequence[str] = TARGET_NAMES,
) -> str:
    """Reproduce Task 1's ordered, UTF-8, compact JSON float.hex fingerprint."""
    columns = ["sample_date", *features, *targets]
    rows = [
        [str(row[0]), *[float(value).hex() for value in row[1:]]]
        for row in frame[columns].itertuples(index=False, name=None)
    ]
    content = json.dumps(
        {"format": FINGERPRINT_FORMAT, "columns": columns, "rows": rows},
        ensure_ascii=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def load_authoritative_evidence(directory: Path = RIDGE_EVIDENCE_ROOT) -> SavedComparisonEvidence:
    """Verify saved output checksums before loading the three required tables."""
    directory = Path(directory).resolve()
    summary_path = directory / "all_targets_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    outputs = summary["output_files"]
    required = {"fold_predictions", "fit_audit", "fold_manifest"}
    if not required <= set(outputs):
        raise ValueError("Required cached Ridge evidence is missing.")
    verified: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for name, record in outputs.items():
        if not record.get("sha256"):
            continue
        path = (PROJECT_ROOT / record["path"]).resolve()
        if not path.is_relative_to(directory) or path.suffix != ".csv":
            raise ValueError("Cached evidence paths must stay inside their CSV evidence directory.")
        digest = _sha256(path)
        if digest != record["sha256"]:
            raise ValueError(f"Cached evidence checksum mismatch: {name}")
        verified[name], hashes[name] = path, digest
    if not required <= set(verified):
        raise ValueError("Required cached Ridge evidence lacks checksums.")
    return SavedComparisonEvidence(
        predictions=pd.read_csv(verified["fold_predictions"]),
        ridge_fit_audit=pd.read_csv(verified["fit_audit"]),
        fold_manifest=pd.read_csv(verified["fold_manifest"]),
        hashes=hashes,
        summary_sha256=_sha256(summary_path),
    )


def _validation_reference(frame: pd.DataFrame, folds: Sequence) -> pd.DataFrame:
    if len(folds) != 5:
        raise ValueError("Exactly five frozen outer folds are required.")
    reference = []
    samples: set[str] = set()
    indexed = frame.set_index("sample_date", drop=False)
    for fold, (training, validation) in enumerate(folds, 1):
        if len(training) != TRAINING_COUNTS[fold - 1] or len(validation) != 60:
            raise ValueError("Frozen outer-fold observation counts changed.")
        dates = validation["sample_date"].tolist()
        if len(set(dates)) != 60 or samples.intersection(dates):
            raise ValueError("Outer-validation sample identifiers must be disjoint.")
        samples.update(dates)
        if not set(training["sample_date"]) <= set(indexed.index):
            raise ValueError("Outer training contains an unauthorized sample.")
        if set(training["sample_date"]).intersection(dates):
            raise ValueError("Outer training and validation samples overlap.")
        if not (pd.to_datetime(training["target_end_date"]) < pd.Timestamp(validation["target_baseline_date"].iloc[0])).all():
            raise ValueError("Outer training labels overlap validation.")
        current = indexed.loc[dates].copy()
        for column in ("target_baseline_date", "target_end_date"):
            if current[column].tolist() != validation[column].tolist():
                raise ValueError("Fold timing differs from authorized pretest timing.")
        long = current.melt(
            id_vars=["sample_date", "target_baseline_date", "target_end_date", "vix_level_points"],
            value_vars=list(TARGET_NAMES), var_name="target", value_name="actual",
        )
        long["fold"] = fold
        long["unit"] = long["target"].map(_target_unit)
        reference.append(long)
    return pd.concat(reference, ignore_index=True)


def _align_values(observed: pd.DataFrame, reference: pd.DataFrame) -> None:
    joined = observed.merge(
        reference, on=list(MATCH_COLUMNS), how="left", validate="many_to_one",
        indicator=True, suffixes=("", "_reference"),
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError("Prediction sample membership does not match authorized data.")
    for column in ("unit", "target_baseline_date", "target_end_date"):
        if not joined[column].eq(joined[f"{column}_reference"]).all():
            raise ValueError(f"Prediction {column} differs from authoritative data.")
    for column in ("actual", "vix_level_points"):
        if not np.isclose(
            joined[column], joined[f"{column}_reference"],
            atol=ABSOLUTE_TOLERANCE, rtol=RELATIVE_TOLERANCE,
        ).all():
            raise ValueError(f"Prediction {column} differs from authoritative data.")


def validate_authoritative_predictions(
    frame: pd.DataFrame, folds: Sequence, predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Reject any missing, duplicated, unexpected or mismatched cached record."""
    if tuple(predictions.columns) != SAVED_COLUMNS:
        raise ValueError("Cached prediction column names or order changed.")
    reference = _validation_reference(frame, folds)
    expected = {
        (*key, strategy)
        for key in reference[list(MATCH_COLUMNS)].itertuples(index=False, name=None)
        for strategy in SAVED_STRATEGIES
    }
    observed = set(predictions[list(KEY_COLUMNS)].itertuples(index=False, name=None))
    if len(predictions) != 14400 or predictions.duplicated(list(KEY_COLUMNS)).any() or observed != expected:
        raise ValueError("Cached predictions contain duplicate, missing or unexpected keys.")
    if not np.isfinite(predictions[["actual", "prediction", "residual", "absolute_error", "vix_level_points"]].to_numpy(dtype=float)).all():
        raise ValueError("Cached prediction values must be finite.")
    _align_values(predictions, reference)
    return reference


def _validate_ridge_audit(audit: pd.DataFrame, folds: Sequence) -> None:
    required = {"fold", "target", "unit", "selected_alpha", "outer_training_rows", "fit_first_sample", "fit_last_sample"}
    if not required <= set(audit.columns):
        raise ValueError("Historical fold-specific Ridge configurations are incomplete.")
    expected = {(fold, target) for fold in range(1, 6) for target in TARGET_NAMES}
    if len(audit) != 60 or audit.duplicated(["fold", "target"]).any() or set(audit[["fold", "target"]].itertuples(index=False, name=None)) != expected:
        raise ValueError("Historical Ridge configuration keys changed.")
    for row in audit.itertuples(index=False):
        training = folds[int(row.fold) - 1][0]
        if row.unit != _target_unit(row.target) or row.outer_training_rows != len(training):
            raise ValueError("Historical Ridge fitting scope or units changed.")
        if row.fit_first_sample != training["sample_date"].iloc[0] or row.fit_last_sample != training["sample_date"].iloc[-1]:
            raise ValueError("Historical Ridge fitting boundaries changed.")
        if not np.isfinite(row.selected_alpha) or row.selected_alpha <= 0:
            raise ValueError("Historical Ridge alpha is invalid.")


def _annotate_metrics(metrics: dict[str, Any], group: pd.DataFrame) -> dict[str, Any]:
    result = dict(metrics)
    result["metric_status"] = "AVAILABLE"
    result["oos_r_squared_undefined_reason"] = ""
    result["prediction_actual_correlation_undefined_reason"] = ""
    if group["actual"].nunique() == 1:
        result["oos_r_squared"] = None
        result["oos_r_squared_undefined_reason"] = "CONSTANT_ACTUAL_VALUES"
        result["prediction_actual_correlation"] = None
        result["prediction_actual_correlation_undefined_reason"] = "CONSTANT_ACTUAL_VALUES"
    elif group["prediction"].nunique() == 1:
        result["prediction_actual_correlation"] = None
        result["prediction_actual_correlation_undefined_reason"] = "CONSTANT_PREDICTIONS"
    return result


def _metrics(group: pd.DataFrame, minimum_samples: int = 2) -> dict[str, Any]:
    if len(group) < minimum_samples:
        reason = f"INSUFFICIENT_SAMPLES_REQUIRES_{minimum_samples}"
        return {
            "sample_count": len(group), "metric_status": reason,
            "mae": None, "rmse": None, "oos_r_squared": None,
            "directional_accuracy": None, "prediction_actual_correlation": None,
            "oos_r_squared_undefined_reason": reason,
            "prediction_actual_correlation_undefined_reason": reason,
        }
    return _annotate_metrics(regression_metrics(group["actual"], group["prediction"]), group)


def _improvement(model_error: float, reference_error: float) -> float | None:
    return 100.0 * (reference_error - model_error) / reference_error if reference_error != 0 else None


def comparison_metric_tables(
    predictions: pd.DataFrame,
    ridge_audit: pd.DataFrame,
    xgboost_audit: pd.DataFrame,
    *,
    minimum_regime_samples: int = 20,
) -> dict[str, pd.DataFrame]:
    """Apply established target-specific metrics to aligned model families."""
    if minimum_regime_samples < 2:
        raise ValueError("Regime metrics require a minimum of at least two samples.")
    fold_rows = []
    for (target, strategy, fold), group in predictions.groupby(["target", "strategy", "fold"], sort=True):
        fold_rows.append({"target": target, "unit": _target_unit(target), "strategy": strategy, "fold": fold, **_metrics(group)})
    fold_metrics = pd.DataFrame(fold_rows)
    pooled = _aggregate_metrics(predictions, fold_metrics).drop(columns="folds_won_by_mae")
    for index, row in pooled.iterrows():
        group = predictions.loc[predictions["target"].eq(row["target"]) & predictions["strategy"].eq(row["strategy"])]
        annotated = _annotate_metrics(row.to_dict(), group)
        for column, value in annotated.items():
            pooled.loc[index, column] = value
    family_rows, naive_rows, generalization_rows, regime_rows = [], [], [], []
    for target in TARGET_NAMES:
        target_pooled = pooled.loc[pooled["target"].eq(target)].set_index("strategy")
        target_folds = fold_metrics.loc[fold_metrics["target"].eq(target)].pivot(index="fold", columns="strategy", values="mae")
        ridge, boosted = target_pooled.loc[RIDGE_STRATEGY], target_pooled.loc[XGBOOST_STRATEGY]
        family_rows.append({
            "target": target, "unit": _target_unit(target),
            "ridge_mae": ridge["mae"], "xgboost_mae": boosted["mae"],
            "ridge_rmse": ridge["rmse"], "xgboost_rmse": boosted["rmse"],
            "xgboost_mae_improvement_pct": _improvement(boosted["mae"], ridge["mae"]),
            "xgboost_rmse_improvement_pct": _improvement(boosted["rmse"], ridge["rmse"]),
            "improvement_undefined_reason": "ZERO_RIDGE_ERROR" if ridge["mae"] == 0 or ridge["rmse"] == 0 else "",
            "ridge_mae_fold_wins": int((target_folds[RIDGE_STRATEGY] < target_folds[XGBOOST_STRATEGY]).sum()),
            "xgboost_mae_fold_wins": int((target_folds[XGBOOST_STRATEGY] < target_folds[RIDGE_STRATEGY]).sum()),
            "mae_fold_ties": int((target_folds[XGBOOST_STRATEGY] == target_folds[RIDGE_STRATEGY]).sum()),
            "ridge_worst_fold_mae": ridge["worst_fold_mae"],
            "xgboost_worst_fold_mae": boosted["worst_fold_mae"],
            "ridge_fold_mae_std": ridge["fold_mae_std"],
            "xgboost_fold_mae_std": boosted["fold_mae_std"],
        })
        for strategy in (RIDGE_STRATEGY, XGBOOST_STRATEGY):
            model = target_pooled.loc[strategy]
            for benchmark in NAIVE_STRATEGIES:
                naive = target_pooled.loc[benchmark]
                naive_rows.append({
                    "target": target, "unit": _target_unit(target), "strategy": strategy,
                    "benchmark": benchmark, "model_mae": model["mae"], "benchmark_mae": naive["mae"],
                    "model_rmse": model["rmse"], "benchmark_rmse": naive["rmse"],
                    "mae_improvement_pct": _improvement(model["mae"], naive["mae"]),
                    "rmse_improvement_pct": _improvement(model["rmse"], naive["rmse"]),
                    "improvement_undefined_reason": "ZERO_BENCHMARK_ERROR" if naive["mae"] == 0 or naive["rmse"] == 0 else "",
                    "mae_fold_wins": int((target_folds[strategy] < target_folds[benchmark]).sum()),
                })
            fit_table = ridge_audit if strategy == RIDGE_STRATEGY else xgboost_audit
            for fold in range(1, 6):
                fit = fit_table.loc[fit_table["target"].eq(target) & fit_table["fold"].eq(fold)].iloc[0]
                outer = fold_metrics.loc[fold_metrics["target"].eq(target) & fold_metrics["strategy"].eq(strategy) & fold_metrics["fold"].eq(fold)].iloc[0]
                train_mae, train_rmse = fit.get("training_mae"), fit.get("training_rmse")
                available = train_mae is not None and train_rmse is not None and np.isfinite([train_mae, train_rmse]).all()
                generalization_rows.append({
                    "target": target, "unit": _target_unit(target), "strategy": strategy, "fold": fold,
                    "training_metric_status": "AVAILABLE" if available else "UNAVAILABLE_IN_SAVED_EVIDENCE",
                    "training_mae": train_mae if available else None,
                    "training_rmse": train_rmse if available else None,
                    "inner_validation_mae": fit.get("inner_validation_mae"),
                    "outer_validation_mae": outer["mae"], "outer_validation_rmse": outer["rmse"],
                    "outer_minus_training_mae": outer["mae"] - train_mae if available else None,
                    "outer_minus_training_rmse": outer["rmse"] - train_rmse if available else None,
                })
        for strategy, group in predictions.loc[predictions["target"].eq(target)].groupby("strategy", sort=True):
            for name, mask in (
                ("low_vix_lt_20", group["vix_level_points"] < 20),
                ("medium_vix_20_to_30", group["vix_level_points"].ge(20) & group["vix_level_points"].lt(30)),
                ("high_vix_ge_30", group["vix_level_points"] >= 30),
            ):
                regime_rows.append({"target": target, "unit": _target_unit(target), "strategy": strategy, "vix_regime": name, **_metrics(group.loc[mask], minimum_regime_samples)})
    return {
        "fold_metrics": fold_metrics, "pooled_metrics": pooled,
        "family_comparison": pd.DataFrame(family_rows), "naive_comparison": pd.DataFrame(naive_rows),
        "generalization": pd.DataFrame(generalization_rows), "regime_metrics": pd.DataFrame(regime_rows),
    }


def _source_identity() -> dict[str, Any]:
    def git(*arguments: str) -> str:
        return subprocess.run(["git", *arguments], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    paths = (
        "scenario_modeling/src/all_target_xgboost_comparison.py",
        "scenario_modeling/src/walk_forward_validation.py",
        "scenario_modeling/src/xgboost_prototype.py",
        "scenario_modeling/src/all_target_models.py",
        "scenario_modeling/src/ridge_baselines.py",
        "scenario_modeling/src/evaluation.py",
        "scenario_modeling/src/model_matrix.py",
        "scenario_modeling/requirements.txt",
    )
    return {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"), "source_files_sha256": {name: _sha256(PROJECT_ROOT / name) for name in paths}}


def _reproducibility_metadata(frame, features, evidence, fold_manifest, guard, minimum_regime_samples):
    observed = authorized_pretest_fingerprint(frame, features, TARGET_NAMES)
    if observed != APPROVED_PRETEST_SHA256:
        raise ValueError("Authorized pretest fingerprint differs from the approved Task 1 identity.")
    return {
        "schema_version": "scenario_ridge_xgboost_comparison_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_count": 58, "target_count": 12, "fold_count": 5,
        "target_horizon_sessions": 21, "validation_rows_per_fold": 60,
        "strategies": list(STRATEGIES),
        "source_identity": _source_identity(),
        "approved_pretest_sha256": APPROVED_PRETEST_SHA256,
        "observed_pretest_sha256": observed,
        "pretest_serialization": {"format": FINGERPRINT_FORMAT, "columns": ["sample_date", *features, *TARGET_NAMES], "rows": 464, "encoding": "UTF-8", "numeric_representation": "Python float.hex binary64 strings", "json_separators": [",", ":"], "trailing_newline": False},
        "historical_identity_limitation": "The Task 1 fingerprint identifies current authorized inputs; historical full-input byte identity remains unproven.",
        "feature_schema_sha256": hashlib.sha256(json.dumps(list(features), separators=(",", ":")).encode("utf-8")).hexdigest(),
        "split_manifest_sha256": _sha256(SPLIT_MANIFEST_PATH),
        "outer_fold_manifest_sha256": hashlib.sha256(frame_to_csv_text(fold_manifest).encode("utf-8")).hexdigest(),
        "cached_ridge_output_sha256": evidence.hashes,
        "cached_ridge_summary_sha256": evidence.summary_sha256,
        "software_versions": {"python": sys.version.split()[0], **{name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "joblib")}},
        "architectures": [asdict(item) for item in ARCHITECTURES],
        "execution_settings": {"seed": SEED, "device": "cpu", "nthread": 1, "max_bin": 128, "objective": "reg:squarederror", "eval_metric": "mae", "tree_method": "hist", "candidate_practical_tie_fraction": 0.01},
        "matching_tolerance": {"absolute": ABSOLUTE_TOLERANCE, "relative": RELATIVE_TOLERANCE},
        "minimum_regime_samples": minimum_regime_samples,
        "fold_win_rule": "Strict per-fold MAE comparison; exact equality is recorded as a tie.",
        "validation_dependence": DEPENDENCE_NOTE,
        "matrix_guard": dict(guard),
        "ridge_refitted": False,
        "scope": "Predictive comparison infrastructure; sensitivity and promotion decisions are separate tasks.",
    }


def build_all_target_comparison_artifacts(
    *, evidence_dir: Path = RIDGE_EVIDENCE_ROOT, minimum_regime_samples: int = 20,
    include_sensitivity: bool = False,
) -> dict[str, Any]:
    """Explicitly run the frozen comparison and return evidence without writing it."""
    if minimum_regime_samples < 2:
        raise ValueError("Regime metrics require a minimum of at least two samples.")
    frame, features, targets, guard = load_all_pretest_rows()
    if tuple(features) != FINAL_FEATURE_NAMES or tuple(targets) != TARGET_NAMES:
        raise ValueError("The frozen feature or target order changed.")
    if len(frame) != 464 or frame["sample_date"].duplicated().any() or not frame["sample_date"].is_monotonic_increasing:
        raise ValueError("Expected 464 unique chronological retained observations.")
    if not np.isfinite(frame[[*features, *targets]].to_numpy(dtype=float)).all():
        raise ValueError("Authorized pretest values must be complete and finite.")
    if guard.get("matrix_rows_declared") != 584 or guard.get("locked_final_test_target_rows_parsed") != 0 or guard.get("locked_final_test_rows_declared") != 114:
        raise ValueError("The protected final-test loading contract changed.")
    fold_manifest, folds = build_walk_forward_folds(frame)
    evidence = load_authoritative_evidence(evidence_dir)
    if not fold_manifest.fillna("").astype(str).equals(evidence.fold_manifest.fillna("").astype(str)):
        raise ValueError("Step 13 cached folds differ from the frozen folds.")
    reference = validate_authoritative_predictions(frame, folds, evidence.predictions)
    _validate_ridge_audit(evidence.ridge_fit_audit, folds)
    summary = _reproducibility_metadata(frame, features, evidence, fold_manifest, guard, minimum_regime_samples)
    prediction_rows, fit_rows, candidate_rows = [], [], []
    sensitivity_results = []
    if include_sensitivity:
        from .matched_scenario_sensitivity import diagnose_target_fold
    for fold, (training, validation) in enumerate(folds, 1):
        for target in targets:
            training_columns = ["sample_date", "target_baseline_date", "target_end_date", *features, target]
            fitted = fit_target_xgboost(
                training[training_columns].copy(), features, target=target,
                outer_validation_baseline=validation["target_baseline_date"].iloc[0], outer_fold=fold,
            )
            if fitted.target != target or fitted.unit != _target_unit(target) or fitted.feature_names != tuple(features) or fitted.outer_fold != fold:
                raise ValueError("The target fitter returned inconsistent target/schema metadata.")
            if fitted.outer_training_rows != len(training):
                raise ValueError("The target fitter returned an inconsistent training count.")
            prediction = np.asarray(fitted.predict(validation[list(features)]), dtype=float).reshape(-1)
            if len(prediction) != 60 or not np.isfinite(prediction).all():
                raise ValueError("XGBoost validation predictions must contain 60 finite values.")
            selected = fitted.selected_candidate
            training_prediction = fitted.predict(training[list(features)])
            training_metrics = regression_metrics(training[target], training_prediction)
            if include_sensitivity:
                original_fit = evidence.ridge_fit_audit.loc[
                    evidence.ridge_fit_audit["target"].eq(target)
                    & evidence.ridge_fit_audit["fold"].eq(fold)
                ].iloc[0].to_dict()
                sensitivity_results.append(diagnose_target_fold(
                    training[training_columns].copy(),
                    validation[["sample_date", "target_baseline_date", "target_end_date", *features]].copy(),
                    features, target=target, fold=fold, xgboost_fit=fitted,
                    fit_record=original_fit, saved_predictions=evidence.predictions,
                ))
            fit_record = {
                "fold": fold, "target": target, "unit": fitted.unit,
                "selected_architecture": selected.architecture.name,
                "simplicity_rank": selected.architecture.complexity_rank,
                "selected_boosting_rounds": selected.boosting_rounds,
                "inner_validation_mae": selected.inner_mae,
                "practical_tie_mae_limit": fitted.practical_tie_mae_limit,
                "inner_training_rows": fitted.inner_training_rows,
                "inner_validation_rows": fitted.inner_validation_rows,
                "inner_purge_rows": fitted.inner_purge_rows,
                "outer_training_rows": fitted.outer_training_rows,
                "outer_purge_rows": int(fold_manifest.loc[fold_manifest["fold"].eq(fold), "outer_purge_rows"].iloc[0]),
                "outer_validation_rows": len(validation),
                "outer_validation_first_sample": validation["sample_date"].iloc[0],
                "outer_validation_last_sample": validation["sample_date"].iloc[-1],
                "outer_validation_last_target_end": validation["target_end_date"].iloc[-1],
                "seed": fitted.seed, **fitted.boundaries,
                "training_mae": training_metrics["mae"], "training_rmse": training_metrics["rmse"],
                "refit_parameters_json": json.dumps(fitted.refit_parameters, sort_keys=True),
            }
            fit_rows.append(fit_record)
            for candidate in fitted.candidates:
                candidate_rows.append({
                    **{name: value for name, value in fit_record.items() if name not in {"training_mae", "training_rmse", "refit_parameters_json", "inner_validation_mae"}},
                    "architecture": candidate.architecture.name,
                    "simplicity_rank": candidate.architecture.complexity_rank,
                    "inner_mae": candidate.inner_mae,
                    "best_iteration": candidate.best_iteration,
                    "boosting_rounds": candidate.boosting_rounds,
                    "selected": candidate.architecture.name == selected.architecture.name,
                    "within_practical_tie": candidate.inner_mae <= fitted.practical_tie_mae_limit,
                    "architecture_json": json.dumps(asdict(candidate.architecture), sort_keys=True),
                    "parameters_json": json.dumps(candidate.parameters, sort_keys=True),
                })
            for row, value in zip(validation.itertuples(index=False), prediction, strict=True):
                actual = float(getattr(row, target))
                prediction_rows.append({
                    "fold": fold, "sample_date": row.sample_date,
                    "target_baseline_date": row.target_baseline_date, "target_end_date": row.target_end_date,
                    "target": target, "unit": fitted.unit, "strategy": XGBOOST_STRATEGY,
                    "actual": actual, "prediction": float(value), "residual": actual - float(value),
                    "absolute_error": abs(actual - float(value)), "vix_level_points": row.vix_level_points,
                    "model_family": "xgboost", "selected_architecture": selected.architecture.name,
                    "selected_boosting_rounds": selected.boosting_rounds,
                })
    xgboost_predictions = pd.DataFrame(prediction_rows)
    _align_values(xgboost_predictions, reference)
    authoritative = evidence.predictions.copy(deep=True)
    existing = authoritative.copy(deep=True)
    existing["model_family"] = np.where(existing["strategy"].eq(RIDGE_STRATEGY), "ridge", "naive")
    predictions = pd.concat([existing, xgboost_predictions], ignore_index=True).sort_values(list(KEY_COLUMNS), kind="stable").reset_index(drop=True)
    xgboost_audit = pd.DataFrame(fit_rows)
    artifacts = {
        "authoritative_predictions": authoritative,
        "xgboost_predictions": xgboost_predictions,
        "predictions": predictions,
        "candidate_audit": pd.DataFrame(candidate_rows),
        "xgboost_fit_audit": xgboost_audit,
        "ridge_fit_audit": evidence.ridge_fit_audit.copy(deep=True),
        "fold_manifest": fold_manifest.copy(deep=True),
        **comparison_metric_tables(predictions, evidence.ridge_fit_audit, xgboost_audit, minimum_regime_samples=minimum_regime_samples),
        "summary": summary,
    }
    if include_sensitivity:
        from .matched_scenario_sensitivity import combine_sensitivity_results

        diagnostics = combine_sensitivity_results(sensitivity_results)
        summary["sensitivity_metadata"] = diagnostics.pop("sensitivity_metadata")
        summary["ridge_refitted"] = True
        summary["ridge_reconstruction_for_sensitivity_only"] = True
        summary["ridge_evaluation_predictions_reused"] = True
        summary["scope"] = "Predictive comparison with optional matched noncausal sensitivity diagnostics; no promotion decisions."
        source = "scenario_modeling/src/matched_scenario_sensitivity.py"
        summary["source_identity"]["source_files_sha256"][source] = _sha256(PROJECT_ROOT / source)
        artifacts.update(diagnostics)
    validate_comparison_artifacts(artifacts)
    return artifacts


def validate_comparison_artifacts(artifacts: dict[str, Any]) -> None:
    """Validate complete comparison shapes and preserved benchmark records."""
    from .matched_scenario_sensitivity import SENSITIVITY_TABLES

    optional = set(SENSITIVITY_TABLES) if "sensitivity_detail" in artifacts else set()
    if set(artifacts) != {*TABLE_ROWS, "summary", *optional}:
        raise ValueError("Comparison artifact names are incomplete or unexpected.")
    for name, expected in TABLE_ROWS.items():
        if not isinstance(artifacts[name], pd.DataFrame) or len(artifacts[name]) != expected:
            raise ValueError(f"Unexpected comparison shape: {name}")
    combined = artifacts["predictions"]
    for name, strategies, expected_rows in (
        ("authoritative_predictions", SAVED_STRATEGIES, 14400),
        ("xgboost_predictions", (XGBOOST_STRATEGY,), 3600),
        ("predictions", STRATEGIES, 18000),
    ):
        table = artifacts[name]
        if table.duplicated(list(KEY_COLUMNS)).any() or set(table["target"]) != set(TARGET_NAMES) or set(table["strategy"]) != set(strategies) or set(table["fold"]) != set(range(1, 6)):
            raise ValueError(f"Unexpected prediction keys: {name}")
        if not table["unit"].eq(table["target"].map(_target_unit)).all():
            raise ValueError("Comparison target units changed.")
        if not table.groupby(["target", "strategy", "fold"]).size().eq(60).all() or len(table) != expected_rows:
            raise ValueError(f"Unexpected prediction membership: {name}")
        if not np.isfinite(table[["actual", "prediction"]].to_numpy(dtype=float)).all():
            raise ValueError("Comparison predictions must be finite.")
    authoritative = artifacts["authoritative_predictions"]
    reference = authoritative.drop_duplicates(list(MATCH_COLUMNS))
    _align_values(artifacts["xgboost_predictions"], reference)
    preserved = combined.loc[combined["strategy"].isin(SAVED_STRATEGIES), list(SAVED_COLUMNS)].sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
    original = authoritative.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
    try:
        # Concatenation can promote integer-valued context columns to floats.
        # The cached table retains its dtypes; combined values must stay exact.
        pd.testing.assert_frame_equal(preserved, original, check_dtype=False, check_exact=True)
    except AssertionError as error:
        raise ValueError("Authoritative Ridge/naive records were modified.") from error
    candidates = artifacts["candidate_audit"]
    if candidates.duplicated(["fold", "target", "architecture"]).any() or not candidates.groupby(["fold", "target"])["selected"].sum().eq(1).all():
        raise ValueError("Candidate-selection audit is inconsistent.")
    expected_fits = {(fold, target) for fold in range(1, 6) for target in TARGET_NAMES}
    for name in ("ridge_fit_audit", "xgboost_fit_audit"):
        audit = artifacts[name]
        if audit.duplicated(["fold", "target"]).any() or set(audit[["fold", "target"]].itertuples(index=False, name=None)) != expected_fits:
            raise ValueError(f"Unexpected fitting audit keys: {name}")
    for _, group in candidates.groupby(["fold", "target"]):
        if set(group["architecture"]) != {item.name for item in ARCHITECTURES} or len(group) != 3:
            raise ValueError("The frozen candidate menu changed.")
    if optional:
        from .matched_scenario_sensitivity import validate_sensitivity_artifacts

        validate_sensitivity_artifacts(artifacts)


def persist_comparison_artifacts(
    artifacts: dict[str, Any], *, output_dir: Path = OUTPUT_ROOT,
) -> dict[str, str]:
    """Publish a new checksummed run with exclusive directory ownership."""
    validate_comparison_artifacts(artifacts)
    destination = Path(output_dir).resolve()
    protected = (
        PROJECT_ROOT / "scenario_modeling", PROJECT_ROOT / "backend", PROJECT_ROOT / "frontend",
        RAW_ROOT, PROCESSED_ROOT, RIDGE_EVIDENCE_ROOT,
        INTERIM_ROOT / "models" / "ridge", INTERIM_ROOT / "models" / "walk_forward",
        INTERIM_ROOT / "models" / "xgboost_prototype",
    )
    if any(destination.is_relative_to(path.resolve()) for path in protected):
        raise ValueError("Comparison output cannot enter protected source, data or artifact directories.")
    if destination.is_relative_to(PROJECT_ROOT.resolve()) and not destination.is_relative_to(OUTPUT_ROOT.resolve()):
        raise ValueError("Repository comparison outputs must stay inside the dedicated comparison directory.")
    if destination.exists():
        raise FileExistsError(f"Comparison output already exists: {destination}")
    table_names = [name for name, value in artifacts.items() if isinstance(value, pd.DataFrame)]
    contents = {f"comparison_{name}.csv": frame_to_csv_text(artifacts[name]).encode("utf-8") for name in table_names}
    summary = dict(artifacts["summary"])
    summary["table_rows"] = {name: len(artifacts[name]) for name in table_names}
    summary["output_files"] = {name: {"sha256": hashlib.sha256(content).hexdigest()} for name, content in contents.items()}
    contents["comparison_summary.json"] = (json.dumps(_json_safe(summary), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    checksums = {name: hashlib.sha256(content).hexdigest() for name, content in contents.items()}
    contents["comparison_checksums.json"] = (json.dumps({"algorithm": "sha256", "files": checksums}, indent=2, sort_keys=True) + "\n").encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    published: list[Path] = []
    with tempfile.TemporaryDirectory(prefix=".comparison-", dir=destination.parent) as staging_name:
        staging = Path(staging_name)
        for name, content in contents.items():
            with (staging / name).open("xb") as handle:
                handle.write(content)
        destination.mkdir(exist_ok=False)
        try:
            for name in contents:
                target = destination / name
                (staging / name).rename(target)
                published.append(target)
        except BaseException:
            for path in published:
                path.unlink()
            if not any(destination.iterdir()):
                destination.rmdir()
            raise
    return {name: str(destination / name) for name in contents}
