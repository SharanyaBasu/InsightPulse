"""Train and evaluate Step 10 naive and Ridge baselines.

Only ``development_train`` rows fit preprocessing and Ridge. Validation rows
produce the first out-of-sample predictions and select Ridge alpha. Locked
final-test target rows are never parsed from the model-matrix CSV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .cache import write_text_atomic
from .config import INTERIM_ROOT, RAW_ROOT
from .evaluation import metrics_by_calendar_year, regression_metrics
from .feature_sources import frame_to_csv_text, relative_path
from .model_matrix import MATRIX_OUTPUT_ROOT
from .sampling import TARGET_SPECS, TargetSpec


MATRIX_PATH = MATRIX_OUTPUT_ROOT / "scenario_model_matrix_raw.csv"
MATRIX_SUMMARY_PATH = (
    MATRIX_OUTPUT_ROOT / "scenario_model_matrix_summary.json"
)
SPLIT_MANIFEST_PATH = (
    MATRIX_OUTPUT_ROOT / "scenario_model_matrix_split_manifest.csv"
)
LEAKAGE_SUMMARY_PATH = (
    INTERIM_ROOT / "audits" / "leakage" / "leakage_audit_summary.json"
)
RIDGE_OUTPUT_ROOT = INTERIM_ROOT / "models" / "ridge"

RIDGE_ALPHAS = (
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0,
    10000.0,
    100000.0,
    1000000.0,
)
NAIVE_STRATEGIES = (
    "naive_zero",
    "naive_training_mean",
    "naive_recent_trailing_21_observation",
)
PREDICTION_STRATEGIES = NAIVE_STRATEGIES + ("ridge",)
TRAILING_OBSERVATIONS = 21


@dataclass(frozen=True)
class ModelingData:
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    development: pd.DataFrame
    validation: pd.DataFrame
    development_manifest: pd.DataFrame
    validation_manifest: pd.DataFrame
    guard: dict[str, Any]
    matrix_summary: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required Step 8-9 summary is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (pd.Timestamp, date)):
        return str(value)
    return value


def _read_unlocked_matrix_prefix(
    matrix_path: Path,
    *,
    row_limit: int,
    numeric_columns: Sequence[str],
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Parse exactly the unlocked chronological prefix, never locked rows."""
    if not matrix_path.exists():
        raise FileNotFoundError(f"Approved model matrix is missing: {matrix_path}")
    with matrix_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Approved model matrix has no CSV header.")
        columns = tuple(reader.fieldnames)
        records = list(itertools.islice(reader, row_limit))
    if len(records) != row_limit:
        raise ValueError(
            f"Expected {row_limit} unlocked rows, parsed {len(records)}."
        )
    frame = pd.DataFrame.from_records(records, columns=columns)
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame, columns


def load_development_and_validation(
    *,
    matrix_path: Path = MATRIX_PATH,
    matrix_summary_path: Path = MATRIX_SUMMARY_PATH,
    split_manifest_path: Path = SPLIT_MANIFEST_PATH,
    leakage_summary_path: Path = LEAKAGE_SUMMARY_PATH,
) -> ModelingData:
    """Load only the chronological matrix prefix before the locked final test."""
    matrix_summary = _read_json(matrix_summary_path)
    leakage_summary = _read_json(leakage_summary_path)
    if leakage_summary.get("training_approval") != (
        "APPROVED_FOR_STEP_10_RIDGE_TRAINING"
    ):
        raise ValueError("The Step 9 training gate is not approved.")
    if int(matrix_summary.get("final_row_count", -1)) != 584:
        raise ValueError("Step 10 requires the approved 584-row matrix.")
    if int(matrix_summary.get("final_feature_count", -1)) != 58:
        raise ValueError("Step 10 requires the approved 58-feature schema.")

    feature_names = tuple(matrix_summary["feature_names"])
    target_names = tuple(matrix_summary["target_names"])
    manifest = pd.read_csv(split_manifest_path)
    if manifest["sample_date"].duplicated().any():
        raise ValueError("Split manifest contains duplicate sample dates.")
    if not manifest["sample_date"].is_monotonic_increasing:
        raise ValueError("Split manifest is not chronological.")

    locked = manifest[manifest["modeling_role"] == "final_test_locked"].copy()
    unlocked = manifest[manifest["modeling_role"] != "final_test_locked"].copy()
    development_manifest = manifest[
        manifest["modeling_role"] == "development_train"
    ].copy()
    validation_manifest = manifest[
        manifest["modeling_role"] == "validation"
    ].copy()
    purged = manifest[
        manifest["modeling_role"].isin(
            [
                "development_boundary_purge",
                "validation_boundary_purge",
            ]
        )
    ].copy()
    if len(locked) != 114:
        raise ValueError(f"Expected 114 locked rows, found {len(locked)}.")
    if len(unlocked) != 470:
        raise ValueError(f"Expected 470 pre-test rows, found {len(unlocked)}.")
    if len(development_manifest) != 354 or len(validation_manifest) != 110:
        raise ValueError("Frozen development/validation counts changed.")
    if len(purged) != 6:
        raise ValueError("Target-end-date purge count changed.")
    if not (
        locked["target_access_policy"] == "LOCKED_UNTIL_FINAL_EVALUATION"
    ).all():
        raise ValueError("One or more final-test rows are not locked.")

    matrix, columns = _read_unlocked_matrix_prefix(
        matrix_path,
        row_limit=len(unlocked),
        numeric_columns=[*feature_names, *target_names],
    )
    expected_columns = tuple(matrix_summary["matrix_schema"])
    if columns != expected_columns:
        raise ValueError("Approved matrix schema changed.")
    if matrix["sample_date"].tolist() != unlocked["sample_date"].tolist():
        raise ValueError(
            "The chronological unlocked matrix prefix does not match the split "
            "manifest. Locked rows may not be interleaved."
        )
    if set(matrix["sample_date"]) & set(locked["sample_date"]):
        raise ValueError("A locked final-test row was parsed.")

    indexed = matrix.set_index("sample_date", drop=False)
    development = indexed.loc[
        development_manifest["sample_date"].tolist()
    ].reset_index(drop=True)
    validation = indexed.loc[
        validation_manifest["sample_date"].tolist()
    ].reset_index(drop=True)
    if set(development["sample_date"]) & set(validation["sample_date"]):
        raise ValueError("Development and validation samples overlap.")

    value_frame = pd.concat(
        [
            development[[*feature_names, *target_names]],
            validation[[*feature_names, *target_names]],
        ],
        ignore_index=True,
    )
    if value_frame.isna().any().any():
        raise ValueError("Unlocked modeling rows contain missing values.")
    if not np.isfinite(value_frame.to_numpy(dtype=float)).all():
        raise ValueError("Unlocked modeling rows contain non-finite values.")

    first_validation_baseline = pd.Timestamp(
        validation_manifest["target_baseline_date"].min()
    )
    if not (
        pd.to_datetime(development_manifest["target_end_date"])
        < first_validation_baseline
    ).all():
        raise ValueError(
            "A development label crosses the validation boundary."
        )

    guard = {
        "matrix_rows_declared": int(matrix_summary["final_row_count"]),
        "matrix_rows_parsed": int(len(matrix)),
        "development_rows_used_for_fit": int(len(development)),
        "validation_rows_used_for_prediction": int(len(validation)),
        "boundary_purge_rows_parsed_but_not_used": int(len(purged)),
        "locked_final_test_rows_declared": int(len(locked)),
        "locked_final_test_target_rows_parsed": 0,
        "last_parsed_sample_date": str(matrix["sample_date"].iloc[-1]),
        "first_locked_sample_date": str(locked["sample_date"].iloc[0]),
        "final_test_target_values_available_to_step_10": False,
        "reader_rule": (
            "Parse exactly the first 470 chronological pre-final-test CSV rows; "
            "do not parse the remaining 114 locked matrix rows."
        ),
    }
    return ModelingData(
        feature_names=feature_names,
        target_names=target_names,
        development=development,
        validation=validation,
        development_manifest=development_manifest.reset_index(drop=True),
        validation_manifest=validation_manifest.reset_index(drop=True),
        guard=guard,
        matrix_summary=matrix_summary,
    )


def make_ridge_pipeline(alpha: float) -> Pipeline:
    if alpha <= 0:
        raise ValueError("Ridge alpha must be positive.")
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True,
                ),
            ),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "ridge",
                Ridge(
                    alpha=float(alpha),
                    fit_intercept=True,
                    solver="auto",
                    tol=1e-4,
                ),
            ),
        ]
    )


def _bounded_yahoo_series(
    spec: TargetSpec,
    *,
    maximum_date: date,
    raw_root: Path,
) -> tuple[pd.Series, Path]:
    path = raw_root / spec.relative_path
    records: list[tuple[date, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or spec.value_field not in reader.fieldnames:
            raise ValueError(
                f"{path} does not contain {spec.value_field!r}."
            )
        previous_date: date | None = None
        for row in reader:
            current_date = date.fromisoformat(row["Date"])
            if previous_date is not None and current_date <= previous_date:
                raise ValueError(f"{path} is not strictly chronological.")
            previous_date = current_date
            if current_date > maximum_date:
                break
            raw_value = row.get(spec.value_field, "")
            if raw_value in {"", "nan", "NaN", "null", "None"}:
                continue
            value = float(raw_value)
            if math.isfinite(value):
                records.append((current_date, value))
    if not records:
        raise ValueError(f"No bounded source values loaded for {spec.target}.")
    series = pd.Series(
        [value for _, value in records],
        index=pd.Index([day for day, _ in records], name="source_date"),
        name=spec.target,
        dtype=float,
    )
    return series, path


def _json_scalar(line: str) -> str:
    token = line.split(":", 1)[1].strip().rstrip(",")
    value = json.loads(token)
    if not isinstance(value, str):
        raise ValueError(f"Expected a JSON string scalar: {line!r}")
    return value


def _bounded_fred_series(
    spec: TargetSpec,
    *,
    maximum_date: date,
    raw_root: Path,
) -> tuple[pd.Series, Path]:
    """Stream the ordered FRED observation objects and stop before the lock."""
    path = raw_root / spec.relative_path
    records: list[tuple[date, float]] = []
    current_date: date | None = None
    previous_date: date | None = None
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith('"date":'):
                current_date = date.fromisoformat(_json_scalar(line))
                if previous_date is not None and current_date <= previous_date:
                    raise ValueError(f"{path} observations are not chronological.")
                previous_date = current_date
                if current_date > maximum_date:
                    break
            elif current_date is not None and line.startswith('"value":'):
                raw_value = _json_scalar(line)
                if raw_value == ".":
                    continue
                value = float(raw_value)
                if math.isfinite(value):
                    records.append((current_date, value))
    if not records:
        raise ValueError(f"No bounded source values loaded for {spec.target}.")
    series = pd.Series(
        [value for _, value in records],
        index=pd.Index([day for day, _ in records], name="source_date"),
        name=spec.target,
        dtype=float,
    )
    return series, path


def _bounded_target_series(
    spec: TargetSpec,
    *,
    maximum_date: date,
    raw_root: Path,
) -> tuple[pd.Series, Path]:
    if spec.file_format == "yahoo_csv":
        return _bounded_yahoo_series(
            spec, maximum_date=maximum_date, raw_root=raw_root
        )
    if spec.file_format == "fred_json":
        return _bounded_fred_series(
            spec, maximum_date=maximum_date, raw_root=raw_root
        )
    raise ValueError(
        f"Unsupported target source format for trailing baseline: "
        f"{spec.file_format}"
    )


def build_trailing_baseline_audit(
    data: ModelingData,
    *,
    raw_root: Path = RAW_ROOT,
) -> pd.DataFrame:
    """Build target-consistent, known-at-baseline 21-observation momentum."""
    validation_manifest = data.validation_manifest.set_index("sample_date")
    maximum_baseline = max(
        pd.to_datetime(
            data.validation_manifest["target_baseline_date"]
        ).dt.date
    )
    rows: list[dict[str, Any]] = []
    for spec in TARGET_SPECS:
        series, source_path = _bounded_target_series(
            spec,
            maximum_date=maximum_baseline,
            raw_root=raw_root,
        )
        positions = {
            source_date: position
            for position, source_date in enumerate(series.index)
        }
        for sample_date in data.validation["sample_date"]:
            baseline_date = pd.Timestamp(
                validation_manifest.loc[
                    sample_date, "target_baseline_date"
                ]
            ).date()
            if baseline_date not in positions:
                raise ValueError(
                    f"{spec.target} has no exact value on {baseline_date}."
                )
            current_position = positions[baseline_date]
            previous_position = current_position - TRAILING_OBSERVATIONS
            if previous_position < 0:
                raise ValueError(
                    f"{spec.target} lacks {TRAILING_OBSERVATIONS} prior "
                    f"observations at {baseline_date}."
                )
            previous_date = series.index[previous_position]
            current_value = float(series.iloc[current_position])
            previous_value = float(series.iloc[previous_position])
            if spec.target == "ten_year_yield_bps":
                prediction = 100.0 * (current_value - previous_value)
                unit = "basis_points"
                formula = "100 * (current_yield_pct - prior_yield_pct)"
            else:
                if previous_value == 0:
                    raise ValueError(
                        f"{spec.target} trailing baseline has a zero denominator."
                    )
                prediction = 100.0 * (
                    current_value / previous_value - 1.0
                )
                unit = "percent"
                formula = "100 * (current_value / prior_value - 1)"
            rows.append(
                {
                    "sample_date": sample_date,
                    "target_baseline_date": baseline_date.isoformat(),
                    "target": spec.target,
                    "source": spec.source,
                    "source_path": relative_path(source_path),
                    "source_field": spec.value_field,
                    "value_policy": spec.value_policy,
                    "prior_source_date": previous_date.isoformat(),
                    "current_source_date": baseline_date.isoformat(),
                    "trailing_source_observations": TRAILING_OBSERVATIONS,
                    "prior_value": previous_value,
                    "current_value": current_value,
                    "prediction": float(prediction),
                    "unit": unit,
                    "formula": formula,
                    "maximum_source_date_parsed": series.index.max().isoformat(),
                    "bounded_reader_cutoff": maximum_baseline.isoformat(),
                    "point_in_time_pass": bool(
                        previous_date < baseline_date <= maximum_baseline
                        and series.index.max() <= maximum_baseline
                    ),
                }
            )
    audit = pd.DataFrame(rows)
    audit = audit.sort_values(["sample_date", "target"]).reset_index(drop=True)
    expected_rows = len(data.validation) * len(data.target_names)
    if len(audit) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} trailing predictions, found {len(audit)}."
        )
    if not audit["point_in_time_pass"].all():
        raise ValueError("A recent-trailing baseline uses future information.")
    return audit


def _target_unit(target: str) -> str:
    return "basis_points" if target == "ten_year_yield_bps" else "percent"


def _preprocessing_audit_rows(
    *,
    target: str,
    pipeline: Pipeline,
    development_features: pd.DataFrame,
    validation_features: pd.DataFrame,
    feature_names: Sequence[str],
) -> list[dict[str, Any]]:
    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]
    rows = []
    for position, feature in enumerate(feature_names):
        development_values = development_features[feature].to_numpy(
            dtype=float
        )
        validation_values = validation_features[feature].to_numpy(dtype=float)
        train_median = float(np.median(development_values))
        train_mean = float(np.mean(development_values))
        rows.append(
            {
                "target": target,
                "feature": feature,
                "fit_role": "development_train",
                "fit_row_count": int(len(development_features)),
                "validation_row_count_not_fit": int(
                    len(validation_features)
                ),
                "imputer_strategy": "median",
                "imputer_statistic": float(imputer.statistics_[position]),
                "development_raw_median": train_median,
                "imputer_matches_development_median": bool(
                    np.isclose(
                        imputer.statistics_[position],
                        train_median,
                        rtol=1e-12,
                        atol=1e-12,
                    )
                ),
                "scaler_mean": float(scaler.mean_[position]),
                "development_raw_mean": train_mean,
                "validation_raw_mean_not_fit": float(
                    np.mean(validation_values)
                ),
                "scaler_mean_matches_development_mean": bool(
                    np.isclose(
                        scaler.mean_[position],
                        train_mean,
                        rtol=1e-12,
                        atol=1e-12,
                    )
                ),
                "scaler_scale": float(scaler.scale_[position]),
                "validation_excluded_from_preprocessing_fit": True,
                "locked_final_test_excluded_from_preprocessing_fit": True,
            }
        )
    return rows


def _strategy_metric_rows(
    predictions: pd.DataFrame,
    *,
    target_names: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    prediction_columns = {
        "naive_zero": "naive_zero_prediction",
        "naive_training_mean": "naive_training_mean_prediction",
        "naive_recent_trailing_21_observation": (
            "naive_recent_trailing_21_observation_prediction"
        ),
        "ridge": "ridge_prediction",
    }
    for target in target_names:
        subset = predictions[predictions["target"] == target]
        for strategy, column in prediction_columns.items():
            rows.append(
                {
                    "target": target,
                    "unit": _target_unit(target),
                    "strategy": strategy,
                    "evaluation_role": "validation",
                    **regression_metrics(subset["actual"], subset[column]),
                }
            )
    return pd.DataFrame(rows)


def _comparison_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target, group in metrics.groupby("target", sort=False):
        ridge = group[group["strategy"] == "ridge"].iloc[0]
        naive = group[group["strategy"].isin(NAIVE_STRATEGIES)].copy()
        best_mae = naive.sort_values(["mae", "rmse", "strategy"]).iloc[0]
        best_rmse = naive.sort_values(["rmse", "mae", "strategy"]).iloc[0]
        ranked = group.sort_values(["mae", "rmse", "strategy"]).reset_index(
            drop=True
        )
        rows.append(
            {
                "target": target,
                "unit": ridge["unit"],
                "ridge_mae": float(ridge["mae"]),
                "best_naive_mae": float(best_mae["mae"]),
                "best_naive_by_mae": str(best_mae["strategy"]),
                "ridge_mae_improvement_pct": float(
                    100.0
                    * (best_mae["mae"] - ridge["mae"])
                    / best_mae["mae"]
                ),
                "ridge_beats_best_naive_mae": bool(
                    ridge["mae"] < best_mae["mae"]
                ),
                "ridge_rmse": float(ridge["rmse"]),
                "best_naive_rmse": float(best_rmse["rmse"]),
                "best_naive_by_rmse": str(best_rmse["strategy"]),
                "ridge_rmse_improvement_pct": float(
                    100.0
                    * (best_rmse["rmse"] - ridge["rmse"])
                    / best_rmse["rmse"]
                ),
                "ridge_beats_best_naive_rmse": bool(
                    ridge["rmse"] < best_rmse["rmse"]
                ),
                "ridge_oos_r_squared": float(ridge["oos_r_squared"]),
                "ridge_prediction_actual_correlation": float(
                    ridge["prediction_actual_correlation"]
                ),
                "ridge_directional_accuracy": float(
                    ridge["directional_accuracy"]
                ),
                "ridge_mae_rank_among_four": int(
                    ranked.index[ranked["strategy"] == "ridge"][0] + 1
                ),
            }
        )
    return pd.DataFrame(rows)


def build_ridge_artifacts(
    *,
    alphas: Sequence[float] = RIDGE_ALPHAS,
    raw_root: Path = RAW_ROOT,
) -> dict[str, Any]:
    """Fit naive baselines and one selected Ridge pipeline per target."""
    data = load_development_and_validation()
    if tuple(alphas) != RIDGE_ALPHAS:
        if not alphas or any(float(alpha) <= 0 for alpha in alphas):
            raise ValueError("The Ridge alpha grid must contain positive values.")
    X_development = data.development.loc[:, list(data.feature_names)]
    X_validation = data.validation.loc[:, list(data.feature_names)]
    trailing_audit = build_trailing_baseline_audit(data, raw_root=raw_root)
    trailing_lookup = trailing_audit.set_index(["sample_date", "target"])[
        "prediction"
    ]

    alpha_search_rows: list[dict[str, Any]] = []
    configuration_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    preprocessing_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    validation_manifest = data.validation_manifest.set_index("sample_date")
    selected_alphas: dict[str, float] = {}
    for target in data.target_names:
        y_development = data.development[target].to_numpy(dtype=float)
        y_validation = data.validation[target].to_numpy(dtype=float)
        candidate_records: list[dict[str, Any]] = []
        for alpha in alphas:
            candidate = make_ridge_pipeline(float(alpha))
            candidate.fit(X_development, y_development)
            candidate_prediction = candidate.predict(X_validation)
            candidate_metrics = regression_metrics(
                y_validation, candidate_prediction
            )
            record = {
                "target": target,
                "unit": _target_unit(target),
                "alpha": float(alpha),
                "selection_role": "validation",
                "selection_primary_metric": "mae",
                **candidate_metrics,
            }
            alpha_search_rows.append(record)
            candidate_records.append(record)

        selected = sorted(
            candidate_records,
            key=lambda record: (
                record["mae"],
                record["rmse"],
                record["alpha"],
            ),
        )[0]
        selected_alpha = float(selected["alpha"])
        selected_alphas[target] = selected_alpha
        pipeline = make_ridge_pipeline(selected_alpha)
        pipeline.fit(X_development, y_development)
        ridge_prediction = pipeline.predict(X_validation)
        if not np.isfinite(ridge_prediction).all():
            raise ValueError(f"{target} Ridge produced non-finite predictions.")

        training_mean = float(np.mean(y_development))
        for row_position, sample_date in enumerate(
            data.validation["sample_date"]
        ):
            manifest_row = validation_manifest.loc[sample_date]
            actual = float(y_validation[row_position])
            ridge_value = float(ridge_prediction[row_position])
            prediction_rows.append(
                {
                    "sample_date": sample_date,
                    "target_baseline_date": manifest_row[
                        "target_baseline_date"
                    ],
                    "target_end_date": manifest_row["target_end_date"],
                    "target": target,
                    "unit": _target_unit(target),
                    "actual": actual,
                    "naive_zero_prediction": 0.0,
                    "naive_training_mean_prediction": training_mean,
                    "naive_recent_trailing_21_observation_prediction": float(
                        trailing_lookup.loc[(sample_date, target)]
                    ),
                    "ridge_prediction": ridge_value,
                    "ridge_residual": actual - ridge_value,
                    "selected_alpha": selected_alpha,
                    "fit_role": "development_train",
                    "evaluation_role": "validation",
                }
            )

        ridge_step = pipeline.named_steps["ridge"]
        coefficients = np.asarray(ridge_step.coef_, dtype=float)
        coefficient_order = np.argsort(-np.abs(coefficients))
        ranks = np.empty_like(coefficient_order)
        ranks[coefficient_order] = np.arange(1, len(coefficients) + 1)
        for position, feature in enumerate(data.feature_names):
            coefficient_rows.append(
                {
                    "target": target,
                    "unit": _target_unit(target),
                    "feature": feature,
                    "standardized_coefficient": float(
                        coefficients[position]
                    ),
                    "absolute_coefficient": float(
                        abs(coefficients[position])
                    ),
                    "absolute_coefficient_rank": int(ranks[position]),
                    "selected_alpha": selected_alpha,
                }
            )
        preprocessing_rows.extend(
            _preprocessing_audit_rows(
                target=target,
                pipeline=pipeline,
                development_features=X_development,
                validation_features=X_validation,
                feature_names=data.feature_names,
            )
        )
        configuration_rows.append(
            {
                "target": target,
                "unit": _target_unit(target),
                "selected_alpha": selected_alpha,
                "selection_role": "validation",
                "selection_primary_metric": "mae",
                "selection_tiebreakers": "rmse_then_smaller_alpha",
                "development_fit_rows": int(len(X_development)),
                "validation_selection_rows": int(len(X_validation)),
                "feature_count": len(data.feature_names),
                "pipeline_steps": (
                    "SimpleImputer(median) -> StandardScaler -> Ridge"
                ),
                "ridge_solver": "auto",
                "ridge_fit_intercept": True,
                "ridge_tolerance": 1e-4,
                "intercept": float(ridge_step.intercept_),
                "coefficient_l1_norm": float(np.abs(coefficients).sum()),
                "coefficient_l2_norm": float(np.linalg.norm(coefficients)),
                "maximum_absolute_coefficient": float(
                    np.abs(coefficients).max()
                ),
                "validation_mae": float(selected["mae"]),
                "validation_rmse": float(selected["rmse"]),
                "validation_oos_r_squared": float(
                    selected["oos_r_squared"]
                ),
                "validation_prediction_actual_correlation": float(
                    selected["prediction_actual_correlation"]
                ),
                "validation_directional_accuracy": float(
                    selected["directional_accuracy"]
                ),
            }
        )

    predictions = pd.DataFrame(prediction_rows).sort_values(
        ["sample_date", "target"]
    ).reset_index(drop=True)
    alpha_search = pd.DataFrame(alpha_search_rows).sort_values(
        ["target", "alpha"]
    ).reset_index(drop=True)
    selected_configurations = pd.DataFrame(configuration_rows)
    selected_configurations["target_order"] = (
        selected_configurations["target"]
        .map({name: position for position, name in enumerate(data.target_names)})
    )
    selected_configurations = selected_configurations.sort_values(
        "target_order"
    ).drop(columns="target_order").reset_index(drop=True)
    coefficients = pd.DataFrame(coefficient_rows).sort_values(
        ["target", "absolute_coefficient_rank"]
    ).reset_index(drop=True)
    preprocessing_audit = pd.DataFrame(preprocessing_rows).sort_values(
        ["target", "feature"]
    ).reset_index(drop=True)

    metrics = _strategy_metric_rows(
        predictions, target_names=data.target_names
    )
    target_order = {name: position for position, name in enumerate(data.target_names)}
    strategy_order = {
        name: position for position, name in enumerate(PREDICTION_STRATEGIES)
    }
    metrics["_target_order"] = metrics["target"].map(target_order)
    metrics["_strategy_order"] = metrics["strategy"].map(strategy_order)
    metrics = metrics.sort_values(
        ["_target_order", "_strategy_order"]
    ).drop(columns=["_target_order", "_strategy_order"]).reset_index(drop=True)

    long_predictions = predictions.melt(
        id_vars=[
            "sample_date",
            "target",
            "unit",
            "actual",
        ],
        value_vars=[
            "naive_zero_prediction",
            "naive_training_mean_prediction",
            "naive_recent_trailing_21_observation_prediction",
            "ridge_prediction",
        ],
        var_name="strategy",
        value_name="prediction",
    )
    long_predictions["strategy"] = long_predictions["strategy"].map(
        {
            "naive_zero_prediction": "naive_zero",
            "naive_training_mean_prediction": "naive_training_mean",
            "naive_recent_trailing_21_observation_prediction": (
                "naive_recent_trailing_21_observation"
            ),
            "ridge_prediction": "ridge",
        }
    )
    metrics_by_year = metrics_by_calendar_year(long_predictions)
    comparison = _comparison_table(metrics)

    prediction_value_columns = [
        "naive_zero_prediction",
        "naive_training_mean_prediction",
        "naive_recent_trailing_21_observation_prediction",
        "ridge_prediction",
    ]
    selected_ridge_metrics = metrics[metrics["strategy"] == "ridge"]
    validation_dates = pd.to_datetime(predictions["sample_date"])
    expected_prediction_rows = len(data.validation) * len(data.target_names)
    validations = {
        "approved_584_row_58_feature_matrix_used": bool(
            data.guard["matrix_rows_declared"] == 584
            and len(data.feature_names) == 58
        ),
        "only_pre_final_test_matrix_prefix_parsed": bool(
            data.guard["matrix_rows_parsed"] == 470
            and data.guard["locked_final_test_target_rows_parsed"] == 0
        ),
        "development_fit_count_is_354": len(data.development) == 354,
        "validation_prediction_count_is_110": len(data.validation) == 110,
        "six_boundary_purge_rows_not_used_for_fit_or_evaluation": bool(
            data.guard["boundary_purge_rows_parsed_but_not_used"] == 6
            and len(data.development) + len(data.validation) == 464
        ),
        "all_58_features_used_in_every_ridge": bool(
            (selected_configurations["feature_count"] == 58).all()
        ),
        "all_12_targets_trained": bool(
            set(selected_configurations["target"]) == set(data.target_names)
            and len(selected_configurations) == 12
        ),
        "all_required_naive_baselines_evaluated": bool(
            set(metrics["strategy"]) == set(PREDICTION_STRATEGIES)
            and len(metrics) == 48
        ),
        "trailing_baseline_is_point_in_time": bool(
            len(trailing_audit) == expected_prediction_rows
            and trailing_audit["point_in_time_pass"].all()
        ),
        "all_alpha_candidates_evaluated_per_target": bool(
            len(alpha_search) == len(data.target_names) * len(alphas)
            and (
                alpha_search.groupby("target")["alpha"].nunique()
                == len(alphas)
            ).all()
        ),
        "all_validation_predictions_present_and_finite": bool(
            len(predictions) == expected_prediction_rows
            and predictions[prediction_value_columns].notna().all().all()
            and np.isfinite(
                predictions[prediction_value_columns].to_numpy(dtype=float)
            ).all()
        ),
        "predictions_are_validation_only": bool(
            predictions["evaluation_role"].eq("validation").all()
            and validation_dates.min().date()
            == pd.Timestamp(
                data.validation["sample_date"].iloc[0]
            ).date()
            and validation_dates.max().date()
            == pd.Timestamp(
                data.validation["sample_date"].iloc[-1]
            ).date()
        ),
        "preprocessing_is_inside_every_pipeline": bool(
            selected_configurations["pipeline_steps"].eq(
                "SimpleImputer(median) -> StandardScaler -> Ridge"
            ).all()
        ),
        "imputer_fit_matches_development_only": bool(
            preprocessing_audit[
                "imputer_matches_development_median"
            ].all()
        ),
        "scaler_fit_matches_development_only": bool(
            preprocessing_audit[
                "scaler_mean_matches_development_mean"
            ].all()
        ),
        "validation_and_locked_test_excluded_from_preprocessing_fit": bool(
            preprocessing_audit[
                "validation_excluded_from_preprocessing_fit"
            ].all()
            and preprocessing_audit[
                "locked_final_test_excluded_from_preprocessing_fit"
            ].all()
        ),
        "ridge_predictions_have_nonzero_distribution": bool(
            (
                selected_ridge_metrics[
                    "prediction_standard_deviation"
                ]
                > 0
            ).all()
        ),
        "all_required_metrics_and_years_present": bool(
            len(metrics_by_year) == 144
            and set(metrics_by_year["calendar_year"]) == {2021, 2022, 2023}
        ),
        "coefficient_and_preprocessing_audits_cover_all_models": bool(
            len(coefficients) == 12 * 58
            and len(preprocessing_audit) == 12 * 58
        ),
        "target_end_date_purge_remains_in_force": bool(
            (
                pd.to_datetime(
                    data.development_manifest["target_end_date"]
                )
                < pd.Timestamp(
                    data.validation_manifest[
                        "target_baseline_date"
                    ].min()
                )
            ).all()
        ),
        "no_model_objects_or_final_production_artifacts_exported": True,
        "no_step_11_xgboost_walk_forward_or_application_work": True,
    }
    if not all(validations.values()):
        failed = [name for name, passed in validations.items() if not passed]
        raise ValueError(f"Step 10 Ridge validation failed: {failed}")

    source_evidence = [
        {
            "path": relative_path(MATRIX_PATH),
            "upstream_recorded_sha256": data.matrix_summary["output_files"][
                "matrix"
            ]["sha256"],
            "values_parsed": (
                "First 470 pre-final-test rows only; 114 locked rows not parsed."
            ),
        },
        {
            "path": relative_path(SPLIT_MANIFEST_PATH),
            "upstream_recorded_sha256": data.matrix_summary["output_files"][
                "split_manifest"
            ]["sha256"],
            "values_parsed": "All rows; split metadata contains no target values.",
        },
    ]
    for spec in TARGET_SPECS:
        matching = trailing_audit[trailing_audit["target"] == spec.target]
        source_evidence.append(
            {
                "path": str(matching["source_path"].iloc[0]),
                "upstream_recorded_sha256": next(
                    (
                        item["sha256"]
                        for item in data.matrix_summary["source_manifest"]
                        if item["path"]
                        == str(matching["source_path"].iloc[0])
                    ),
                    None,
                ),
                "values_parsed": (
                    "Chronological source observations only through "
                    f"{matching['maximum_source_date_parsed'].max()}."
                ),
            }
        )

    summary = {
        "schema_version": "scenario_ridge_baselines_v1",
        "construction_mode": "offline_approved_matrix_only",
        "software_versions": {
            "python": (
                f"{sys.version_info.major}.{sys.version_info.minor}."
                f"{sys.version_info.micro}"
            ),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "matrix_guard": data.guard,
        "feature_count": len(data.feature_names),
        "target_count": len(data.target_names),
        "feature_names": list(data.feature_names),
        "target_names": list(data.target_names),
        "development_fit_rows": len(data.development),
        "development_first_sample": str(
            data.development["sample_date"].iloc[0]
        ),
        "development_last_sample": str(
            data.development["sample_date"].iloc[-1]
        ),
        "validation_rows": len(data.validation),
        "validation_first_sample": str(
            data.validation["sample_date"].iloc[0]
        ),
        "validation_last_sample": str(
            data.validation["sample_date"].iloc[-1]
        ),
        "alpha_grid": [float(alpha) for alpha in alphas],
        "alpha_selection": {
            "role": "validation",
            "primary_metric": "mae",
            "tiebreakers": ["rmse", "smaller_alpha"],
            "note": (
                "These are validation-selection metrics, not final-test "
                "performance. Later evaluation must use the locked holdout."
            ),
        },
        "pipeline": {
            "steps": [
                "SimpleImputer(strategy='median')",
                "StandardScaler()",
                "Ridge(fit_intercept=True, solver='auto')",
            ],
            "fit_role": "development_train",
            "fit_rows": len(data.development),
            "validation_used_in_fit": False,
            "locked_final_test_used_in_fit_or_selection": False,
        },
        "naive_baselines": {
            "naive_zero": "Predict 0 in the target's output unit.",
            "naive_training_mean": (
                "Predict the development-training target mean."
            ),
            "naive_recent_trailing_21_observation": (
                "Predict the same target instrument's movement over its prior "
                "21 nonmissing source observations ending at target_baseline_date."
            ),
        },
        "metric_definitions": {
            "mae": "Mean absolute error in target units.",
            "rmse": "Root mean squared error in target units.",
            "oos_r_squared": (
                "1 - validation SSE / validation mean-baseline SST."
            ),
            "prediction_actual_correlation": (
                "Pearson correlation; undefined for constant predictions."
            ),
            "directional_accuracy": (
                "Exact sign agreement; neutral prediction matches only a "
                "zero realized target."
            ),
            "performance_by_time_period": "Validation calendar years.",
        },
        "selected_alphas": selected_alphas,
        "selected_configurations": _json_safe(
            selected_configurations.to_dict("records")
        ),
        "validation_metrics": _json_safe(metrics.to_dict("records")),
        "ridge_naive_comparison": _json_safe(
            comparison.to_dict("records")
        ),
        "ridge_beats_best_naive_mae_count": int(
            comparison["ridge_beats_best_naive_mae"].sum()
        ),
        "ridge_beats_best_naive_rmse_count": int(
            comparison["ridge_beats_best_naive_rmse"].sum()
        ),
        "validations": validations,
        "source_evidence": source_evidence,
        "limitations": [
            (
                "Alpha is selected on the single validation period, so its "
                "reported metric is a selection result rather than a final "
                "unbiased performance estimate."
            ),
            (
                "Event-aware labels overlap within development and validation; "
                "the frozen boundary purge is preserved, but walk-forward "
                "evaluation remains necessary in its later step."
            ),
            (
                "The recent-trailing baseline uses 21 nonmissing observations "
                "from each target source, not 21 event-aware rows."
            ),
            (
                "The PMI regional proxy and CL=F continuous-futures limitations "
                "remain inherited from the approved matrix."
            ),
        ],
        "training_gate": (
            "STEP_10_RIDGE_COMPLETE_READY_FOR_STEP_11_XGBOOST_PROTOTYPE"
        ),
        "scope_boundary": (
            "Step 10 only: no final-test access, Step 11, XGBoost, walk-forward "
            "validation, production artifact export, Supabase, backend, "
            "frontend, or server work."
        ),
    }
    return {
        "predictions": predictions,
        "metrics": metrics,
        "metrics_by_year": metrics_by_year,
        "alpha_search": alpha_search,
        "selected_configurations": selected_configurations,
        "coefficients": coefficients,
        "preprocessing_audit": preprocessing_audit,
        "trailing_baseline_audit": trailing_audit,
        "comparison": comparison,
        "summary": summary,
    }


def persist_ridge_artifacts(
    artifacts: dict[str, Any],
    *,
    output_dir: Path = RIDGE_OUTPUT_ROOT,
) -> dict[str, str]:
    paths = {
        "predictions": output_dir / "ridge_validation_predictions.csv",
        "metrics": output_dir / "ridge_validation_metrics.csv",
        "metrics_by_year": (
            output_dir / "ridge_validation_metrics_by_year.csv"
        ),
        "alpha_search": output_dir / "ridge_alpha_search.csv",
        "selected_configurations": (
            output_dir / "ridge_selected_configurations.csv"
        ),
        "coefficients": output_dir / "ridge_standardized_coefficients.csv",
        "preprocessing_audit": (
            output_dir / "ridge_preprocessing_fit_audit.csv"
        ),
        "trailing_baseline_audit": (
            output_dir / "ridge_trailing_baseline_audit.csv"
        ),
        "comparison": output_dir / "ridge_naive_comparison.csv",
        "summary": output_dir / "ridge_baseline_summary.json",
    }
    dataframe_names = tuple(name for name in paths if name != "summary")
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
        json.dumps(
            _json_safe(summary),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
    )
    return {name: str(path.resolve()) for name, path in paths.items()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train and validate all 12 Scenario Playground naive and Ridge "
            "baselines without parsing the locked final-test target rows."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Train and validate in memory without writing Step 10 outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RIDGE_OUTPUT_ROOT,
        help=(
            "Output directory (default: "
            "data/scenario_playground/interim/models/ridge)."
        ),
    )
    args = parser.parse_args(argv)
    artifacts = build_ridge_artifacts()
    outputs = (
        {}
        if args.dry_run
        else persist_ridge_artifacts(
            artifacts, output_dir=args.output_dir
        )
    )
    summary = artifacts["summary"]
    print(
        json.dumps(
            {
                "development_fit_rows": summary["development_fit_rows"],
                "validation_rows": summary["validation_rows"],
                "feature_count": summary["feature_count"],
                "target_count": summary["target_count"],
                "ridge_beats_best_naive_mae_count": summary[
                    "ridge_beats_best_naive_mae_count"
                ],
                "ridge_beats_best_naive_rmse_count": summary[
                    "ridge_beats_best_naive_rmse_count"
                ],
                "locked_final_test_target_rows_parsed": summary[
                    "matrix_guard"
                ]["locked_final_test_target_rows_parsed"],
                "training_gate": summary["training_gate"],
                "outputs": outputs,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
