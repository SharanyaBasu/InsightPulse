"""Step 13 purged full-history Ridge evaluation for all 12 targets.

The module reuses the frozen Step 12 fold design without importing or fitting
XGBoost. It reads only the pre-final-test matrix prefix and does not export
production model binaries.
"""

from __future__ import annotations

import argparse
import hashlib
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
from sklearn.pipeline import Pipeline

from .cache import write_text_atomic
from .config import INTERIM_ROOT, RAW_ROOT
from .evaluation import regression_metrics
from .feature_sources import frame_to_csv_text, relative_path
from .ridge_baselines import (
    NAIVE_STRATEGIES,
    RIDGE_ALPHAS,
    TRAILING_OBSERVATIONS,
    _bounded_target_series,
    load_development_and_validation,
    make_ridge_pipeline,
)
from .sampling import TARGET_SPECS, TargetSpec
from .shock_features import CONTRACT_TO_HISTORICAL_FEATURE


FOLD_COUNT = 5
VALIDATION_ROWS_PER_FOLD = 60
INITIAL_TRAIN_ROWS = 164
INNER_VALIDATION_FRACTION = 0.20
RIDGE_STRATEGY = "ridge_full_history"
ALL_STRATEGIES = (*NAIVE_STRATEGIES, RIDGE_STRATEGY)
TARGET_NAMES = tuple(spec.target for spec in TARGET_SPECS)
SCENARIO_INPUTS = tuple(CONTRACT_TO_HISTORICAL_FEATURE)
STEP12_FOLD_MANIFEST_PATH = (
    INTERIM_ROOT / "models" / "walk_forward" / "sp500_fold_manifest.csv"
)
OUTPUT_ROOT = INTERIM_ROOT / "models" / "all_targets"
CLASSIFICATIONS = (
    "APPROVED_FOR_EXPERIMENTAL_INFERENCE",
    "APPROVED_WITH_LOW_CONFIDENCE",
    "NAIVE_FALLBACK_REQUIRED",
    "NOT_APPROVED",
)


@dataclass
class FittedRidge:
    target: str
    pipeline: Pipeline
    training: pd.DataFrame
    selected_alpha: float
    inner_training_rows: int
    inner_validation_rows: int
    inner_purge_rows: int
    inner_validation_mae: float
    inner_validation_rmse: float
    training_mae: float
    training_rmse: float
    alpha_search: list[dict[str, Any]]


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
        return str(value)
    return value


def _target_unit(target: str) -> str:
    return "basis_points" if target == "ten_year_yield_bps" else "percent"


def _meaningful_response_threshold(target: str) -> float:
    return 1.0 if target == "ten_year_yield_bps" else 0.05


def load_all_pretest_rows() -> tuple[
    pd.DataFrame,
    tuple[str, ...],
    tuple[str, ...],
    dict[str, Any],
]:
    """Load the same 464 retained pre-final-test rows used by Step 12."""
    data = load_development_and_validation()
    frame = pd.concat(
        [data.development, data.validation],
        ignore_index=True,
    )
    manifest = pd.concat(
        [data.development_manifest, data.validation_manifest],
        ignore_index=True,
    )
    frame = frame.merge(
        manifest[
            ["sample_date", "target_baseline_date", "target_end_date"]
        ],
        on="sample_date",
        how="left",
        validate="one_to_one",
    )
    frame = frame.sort_values("sample_date", kind="stable").reset_index(
        drop=True
    )
    if len(frame) != 464:
        raise ValueError(f"Expected 464 retained pre-test rows, found {len(frame)}.")
    if frame["sample_date"].duplicated().any():
        raise ValueError("Pre-test frame contains duplicate sample dates.")
    if tuple(data.target_names) != TARGET_NAMES:
        raise ValueError("Frozen target names or order changed.")
    if len(data.feature_names) != 58:
        raise ValueError("Frozen feature count changed.")
    return (
        frame,
        tuple(data.feature_names),
        tuple(data.target_names),
        dict(data.guard),
    )


def build_walk_forward_folds(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[tuple[pd.DataFrame, pd.DataFrame]]]:
    """Recreate the frozen Step 12 expanding folds and target-end purges."""
    rows: list[dict[str, Any]] = []
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for fold_index in range(FOLD_COUNT):
        validation_start = (
            INITIAL_TRAIN_ROWS + fold_index * VALIDATION_ROWS_PER_FOLD
        )
        validation_end = validation_start + VALIDATION_ROWS_PER_FOLD
        validation = frame.iloc[validation_start:validation_end].copy()
        candidates = frame.iloc[:validation_start].copy()
        first_baseline = pd.Timestamp(
            validation["target_baseline_date"].iloc[0]
        )
        retain = (
            pd.to_datetime(candidates["target_end_date"]) < first_baseline
        )
        training = candidates.loc[retain].copy()
        purged = candidates.loc[~retain].copy()
        if len(validation) != VALIDATION_ROWS_PER_FOLD:
            raise ValueError("A Step 13 outer validation fold is incomplete.")
        if len(training) < 140:
            raise ValueError("A Step 13 outer training fold is too small.")
        if not (
            pd.to_datetime(training["target_end_date"]) < first_baseline
        ).all():
            raise ValueError("An outer training label overlaps validation.")
        rows.append(
            {
                "fold": fold_index + 1,
                "training_first_sample": training["sample_date"].iloc[0],
                "training_last_sample": training["sample_date"].iloc[-1],
                "training_rows": len(training),
                "outer_purge_rows": len(purged),
                "purge_first_sample": (
                    purged["sample_date"].iloc[0] if len(purged) else None
                ),
                "purge_last_sample": (
                    purged["sample_date"].iloc[-1] if len(purged) else None
                ),
                "validation_first_sample": (
                    validation["sample_date"].iloc[0]
                ),
                "validation_last_sample": (
                    validation["sample_date"].iloc[-1]
                ),
                "validation_first_baseline": (
                    validation["target_baseline_date"].iloc[0]
                ),
                "validation_last_target_end": (
                    validation["target_end_date"].iloc[-1]
                ),
                "validation_rows": len(validation),
                "purge_rule": (
                    "training target_end_date < first validation "
                    "target_baseline_date"
                ),
            }
        )
        folds.append((training, validation))
    manifest = pd.DataFrame(rows)
    if not STEP12_FOLD_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Step 12 fold manifest is missing: {STEP12_FOLD_MANIFEST_PATH}"
        )
    frozen = pd.read_csv(STEP12_FOLD_MANIFEST_PATH)
    comparable = [
        "fold",
        "training_first_sample",
        "training_last_sample",
        "training_rows",
        "outer_purge_rows",
        "validation_first_sample",
        "validation_last_sample",
        "validation_first_baseline",
        "validation_last_target_end",
        "validation_rows",
        "purge_rule",
    ]
    if not manifest[comparable].astype(str).equals(
        frozen[comparable].astype(str)
    ):
        raise ValueError("Step 13 folds differ from the frozen Step 12 folds.")
    return manifest, folds


def _inner_split(
    training: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    split = max(
        1,
        int(
            math.floor(
                len(training) * (1.0 - INNER_VALIDATION_FRACTION)
            )
        ),
    )
    inner_validation = training.iloc[split:].copy()
    candidates = training.iloc[:split].copy()
    first_baseline = pd.Timestamp(
        inner_validation["target_baseline_date"].iloc[0]
    )
    retain = (
        pd.to_datetime(candidates["target_end_date"]) < first_baseline
    )
    inner_training = candidates.loc[retain].copy()
    purge_count = int((~retain).sum())
    if len(inner_training) < 75 or len(inner_validation) < 20:
        raise ValueError("A purged inner split is too small.")
    if purge_count < 1:
        raise ValueError("A purged inner split removed no overlapping label.")
    return inner_training, inner_validation, purge_count


def _fit_ridge(
    training: pd.DataFrame,
    feature_names: Sequence[str],
    target: str,
    *,
    evaluation_scope: str,
    fold: int | None,
) -> FittedRidge:
    inner_training, inner_validation, purge_count = _inner_split(training)
    search_rows: list[dict[str, Any]] = []
    for alpha in RIDGE_ALPHAS:
        pipeline = make_ridge_pipeline(float(alpha))
        pipeline.fit(
            inner_training[list(feature_names)],
            inner_training[target],
        )
        inner_prediction = pipeline.predict(
            inner_validation[list(feature_names)]
        )
        metrics = regression_metrics(
            inner_validation[target],
            inner_prediction,
        )
        search_rows.append(
            {
                "evaluation_scope": evaluation_scope,
                "fold": fold,
                "target": target,
                "alpha": float(alpha),
                "inner_training_rows": len(inner_training),
                "inner_validation_rows": len(inner_validation),
                "inner_purge_rows": purge_count,
                "inner_validation_mae": metrics["mae"],
                "inner_validation_rmse": metrics["rmse"],
            }
        )
    selected = sorted(
        search_rows,
        key=lambda row: (
            row["inner_validation_mae"],
            row["inner_validation_rmse"],
            row["alpha"],
        ),
    )[0]
    for row in search_rows:
        row["selected"] = row["alpha"] == selected["alpha"]
    pipeline = make_ridge_pipeline(float(selected["alpha"]))
    pipeline.fit(training[list(feature_names)], training[target])
    training_metrics = regression_metrics(
        training[target],
        pipeline.predict(training[list(feature_names)]),
    )
    return FittedRidge(
        target=target,
        pipeline=pipeline,
        training=training,
        selected_alpha=float(selected["alpha"]),
        inner_training_rows=len(inner_training),
        inner_validation_rows=len(inner_validation),
        inner_purge_rows=purge_count,
        inner_validation_mae=float(selected["inner_validation_mae"]),
        inner_validation_rmse=float(selected["inner_validation_rmse"]),
        training_mae=float(training_metrics["mae"]),
        training_rmse=float(training_metrics["rmse"]),
        alpha_search=search_rows,
    )


def _coefficient_rows(
    fitted: FittedRidge,
    feature_names: Sequence[str],
    *,
    evaluation_scope: str,
    fold: int | None,
) -> list[dict[str, Any]]:
    ridge = fitted.pipeline.named_steps["ridge"]
    scaler = fitted.pipeline.named_steps["scaler"]
    standardized = np.asarray(ridge.coef_, dtype=float)
    scales = np.asarray(scaler.scale_, dtype=float)
    raw_unit = standardized / scales
    absolute = np.abs(standardized)
    total = float(absolute.sum())
    normalized = absolute / total if total > 0 else np.zeros_like(absolute)
    ranks = pd.Series(absolute).rank(
        method="first",
        ascending=False,
    ).astype(int)
    feature_to_contract = {
        feature: contract
        for contract, feature in CONTRACT_TO_HISTORICAL_FEATURE.items()
    }
    return [
        {
            "evaluation_scope": evaluation_scope,
            "fold": fold,
            "target": fitted.target,
            "feature": feature,
            "is_scenario_input": feature in feature_to_contract,
            "contract_input": feature_to_contract.get(feature, ""),
            "standardized_coefficient": float(standardized[position]),
            "raw_unit_coefficient": float(raw_unit[position]),
            "absolute_standardized_coefficient": float(absolute[position]),
            "normalized_absolute_importance": float(normalized[position]),
            "importance_rank": int(ranks.iloc[position]),
            "selected_alpha": fitted.selected_alpha,
        }
        for position, feature in enumerate(feature_names)
    ]


def _representative_states(
    training: pd.DataFrame,
) -> list[tuple[str, pd.Series]]:
    states: list[tuple[str, pd.Series]] = []
    vix = training["vix_level_points"]
    for name, quantile in (
        ("low_vix", 0.10),
        ("median_vix", 0.50),
        ("high_vix", 0.90),
    ):
        target_value = float(vix.quantile(quantile))
        index = (vix - target_value).abs().idxmin()
        states.append((name, training.loc[index].copy()))
    return states


def _sensitivity_rows(
    *,
    fold: int,
    fitted: FittedRidge,
    feature_names: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    threshold = _meaningful_response_threshold(fitted.target)
    sensible_floor = (
        50.0 if fitted.target == "ten_year_yield_bps" else 10.0
    )
    observed_target_range = float(
        fitted.training[fitted.target].max()
        - fitted.training[fitted.target].min()
    )
    maximum_sensible_span = max(
        sensible_floor,
        observed_target_range,
    )
    for state_name, state in _representative_states(fitted.training):
        reference = state.copy()
        for feature in CONTRACT_TO_HISTORICAL_FEATURE.values():
            reference[feature] = 0.0
        neutral_prediction = float(
            fitted.pipeline.predict(
                pd.DataFrame([reference])[list(feature_names)]
            )[0]
        )
        for contract_input, feature in (
            CONTRACT_TO_HISTORICAL_FEATURE.items()
        ):
            values = fitted.training[feature].to_numpy(dtype=float)
            grid = np.unique(
                np.append(
                    np.linspace(
                        float(values.min()),
                        float(values.max()),
                        5,
                    ),
                    0.0,
                )
            )
            scenarios = pd.DataFrame(
                [reference.copy() for _ in grid]
            )
            scenarios.loc[:, feature] = grid
            predictions = fitted.pipeline.predict(
                scenarios[list(feature_names)]
            )
            differences = np.diff(predictions.astype(float))
            nonzero_signs = np.sign(
                differences[np.abs(differences) > 1e-12]
            )
            direction_changes = (
                int(np.sum(nonzero_signs[1:] != nonzero_signs[:-1]))
                if len(nonzero_signs) > 1
                else 0
            )
            span = float(predictions.max() - predictions.min())
            end_to_end = float(predictions[-1] - predictions[0])
            maximum_jump = (
                float(np.max(np.abs(differences)))
                if len(differences)
                else 0.0
            )
            if abs(end_to_end) < 1e-12:
                response_direction = "flat"
            elif end_to_end > 0:
                response_direction = "positive"
            else:
                response_direction = "negative"
            rows.append(
                {
                    "fold": fold,
                    "target": fitted.target,
                    "unit": _target_unit(fitted.target),
                    "representative_state": state_name,
                    "contract_input": contract_input,
                    "model_feature": feature,
                    "training_minimum": float(values.min()),
                    "training_maximum": float(values.max()),
                    "neutral_prediction": neutral_prediction,
                    "lower_value_prediction": float(predictions[0]),
                    "upper_value_prediction": float(predictions[-1]),
                    "minimum_prediction": float(predictions.min()),
                    "maximum_prediction": float(predictions.max()),
                    "prediction_span": span,
                    "end_to_end_effect": end_to_end,
                    "response_direction": response_direction,
                    "maximum_adjacent_jump": maximum_jump,
                    "direction_change_count": direction_changes,
                    "monotonic_over_grid": direction_changes == 0,
                    "meaningful_response_threshold": threshold,
                    "meaningful_response": span >= threshold,
                    "maximum_sensible_span": maximum_sensible_span,
                    "finite_and_sensible": bool(
                        np.isfinite(predictions).all()
                        and span <= maximum_sensible_span
                        and maximum_jump <= maximum_sensible_span / 2.0
                    ),
                }
            )
    return rows


def _trailing_baseline(
    folds: Sequence[tuple[pd.DataFrame, pd.DataFrame]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold_number, (_, validation) in enumerate(folds, 1):
        maximum_baseline = max(
            pd.to_datetime(validation["target_baseline_date"]).dt.date
        )
        for spec in TARGET_SPECS:
            series, source_path = _bounded_target_series(
                spec,
                maximum_date=maximum_baseline,
                raw_root=RAW_ROOT,
            )
            positions = {
                source_date: position
                for position, source_date in enumerate(series.index)
            }
            for row in validation.itertuples(index=False):
                baseline_date = pd.Timestamp(
                    row.target_baseline_date
                ).date()
                if baseline_date not in positions:
                    raise ValueError(
                        f"{spec.target} lacks an exact value on "
                        f"{baseline_date}."
                    )
                current_position = positions[baseline_date]
                prior_position = current_position - TRAILING_OBSERVATIONS
                if prior_position < 0:
                    raise ValueError(
                        f"{spec.target} lacks {TRAILING_OBSERVATIONS} "
                        f"prior observations."
                    )
                previous_date = series.index[prior_position]
                current_value = float(series.iloc[current_position])
                previous_value = float(series.iloc[prior_position])
                if spec.target == "ten_year_yield_bps":
                    prediction = 100.0 * (
                        current_value - previous_value
                    )
                    formula = (
                        "100 * (current_yield_pct - prior_yield_pct)"
                    )
                else:
                    if previous_value == 0:
                        raise ValueError(
                            f"{spec.target} trailing denominator is zero."
                        )
                    prediction = 100.0 * (
                        current_value / previous_value - 1.0
                    )
                    formula = (
                        "100 * (current_value / prior_value - 1)"
                    )
                rows.append(
                    {
                        "fold": fold_number,
                        "sample_date": row.sample_date,
                        "target_baseline_date": (
                            baseline_date.isoformat()
                        ),
                        "target": spec.target,
                        "source": spec.source,
                        "source_path": relative_path(source_path),
                        "source_field": spec.value_field,
                        "value_policy": spec.value_policy,
                        "prior_source_date": previous_date.isoformat(),
                        "current_source_date": baseline_date.isoformat(),
                        "trailing_source_observations": (
                            TRAILING_OBSERVATIONS
                        ),
                        "prior_value": previous_value,
                        "current_value": current_value,
                        "prediction": float(prediction),
                        "unit": _target_unit(spec.target),
                        "formula": formula,
                        "maximum_source_date_parsed": (
                            series.index.max().isoformat()
                        ),
                        "bounded_reader_cutoff": (
                            maximum_baseline.isoformat()
                        ),
                        "point_in_time_pass": bool(
                            previous_date < baseline_date
                            and series.index.max() <= maximum_baseline
                        ),
                    }
                )
    audit = pd.DataFrame(rows)
    expected = (
        FOLD_COUNT
        * VALIDATION_ROWS_PER_FOLD
        * len(TARGET_NAMES)
    )
    if len(audit) != expected:
        raise ValueError(
            f"Expected {expected} trailing audit rows, found {len(audit)}."
        )
    if not audit["point_in_time_pass"].all():
        raise ValueError("A trailing baseline is not point in time.")
    return audit.sort_values(
        ["target", "fold", "sample_date"],
        kind="stable",
    ).reset_index(drop=True)


def _aggregate_metrics(
    predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (target, strategy), group in predictions.groupby(
        ["target", "strategy"],
        sort=True,
    ):
        per_fold = fold_metrics.loc[
            fold_metrics["target"].eq(target)
            & fold_metrics["strategy"].eq(strategy)
        ]
        fold_wins = 0
        for fold in range(1, FOLD_COUNT + 1):
            own_mae = float(
                per_fold.loc[per_fold["fold"].eq(fold), "mae"].iloc[0]
            )
            minimum = float(
                fold_metrics.loc[
                    fold_metrics["target"].eq(target)
                    & fold_metrics["fold"].eq(fold),
                    "mae",
                ].min()
            )
            fold_wins += int(np.isclose(own_mae, minimum))
        rows.append(
            {
                "target": target,
                "unit": _target_unit(target),
                "strategy": strategy,
                **regression_metrics(
                    group["actual"],
                    group["prediction"],
                ),
                "fold_mae_mean": float(per_fold["mae"].mean()),
                "fold_mae_std": float(per_fold["mae"].std(ddof=1)),
                "worst_fold_mae": float(per_fold["mae"].max()),
                "best_fold_mae": float(per_fold["mae"].min()),
                "folds_won_by_mae": fold_wins,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["target", "mae", "strategy"],
        kind="stable",
    ).reset_index(drop=True)


def _regime_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    bins = [-np.inf, 20.0, 30.0, np.inf]
    labels = [
        "low_vix_lt_20",
        "medium_vix_20_to_30",
        "high_vix_ge_30",
    ]
    for (target, strategy), group in predictions.groupby(
        ["target", "strategy"],
        sort=True,
    ):
        regimes = pd.cut(
            group["vix_level_points"],
            bins,
            labels=labels,
            right=False,
        )
        for label in labels:
            subset = group.loc[regimes.eq(label)]
            rows.append(
                {
                    "target": target,
                    "unit": _target_unit(target),
                    "strategy": strategy,
                    "vix_regime": label,
                    **regression_metrics(
                        subset["actual"],
                        subset["prediction"],
                    ),
                }
            )
    return pd.DataFrame(rows)


def _coefficient_stability(coefficients: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (target, feature), group in coefficients.groupby(
        ["target", "feature"],
        sort=True,
    ):
        values = group["standardized_coefficient"].to_numpy(dtype=float)
        signs = np.sign(values[np.abs(values) > 1e-12])
        positive = int((signs > 0).sum())
        negative = int((signs < 0).sum())
        if positive > negative:
            dominant = "positive"
        elif negative > positive:
            dominant = "negative"
        else:
            dominant = "mixed"
        agreement = (
            float(max(positive, negative) / len(signs))
            if len(signs)
            else 1.0
        )
        first = group.iloc[0]
        rows.append(
            {
                "target": target,
                "feature": feature,
                "is_scenario_input": bool(first["is_scenario_input"]),
                "contract_input": first["contract_input"],
                "fold_count": len(group),
                "coefficient_mean": float(values.mean()),
                "coefficient_standard_deviation": float(
                    values.std(ddof=1)
                ),
                "mean_absolute_coefficient": float(np.abs(values).mean()),
                "minimum_coefficient": float(values.min()),
                "maximum_coefficient": float(values.max()),
                "positive_fold_count": positive,
                "negative_fold_count": negative,
                "dominant_sign": dominant,
                "dominant_sign_agreement": agreement,
                "sign_stable_all_folds": bool(
                    positive == len(values) or negative == len(values)
                ),
                "mean_importance_rank": float(
                    group["importance_rank"].mean()
                ),
                "mean_normalized_absolute_importance": float(
                    group["normalized_absolute_importance"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _sensitivity_summary(sensitivity: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (target, contract_input), group in sensitivity.groupby(
        ["target", "contract_input"],
        sort=True,
    ):
        meaningful = group.loc[group["meaningful_response"]]
        directions = sorted(
            set(meaningful["response_direction"]) - {"flat"}
        )
        rows.append(
            {
                "target": target,
                "unit": _target_unit(target),
                "contract_input": contract_input,
                "fold_state_checks": len(group),
                "meaningful_checks": int(
                    group["meaningful_response"].sum()
                ),
                "folds_with_meaningful_response": int(
                    meaningful["fold"].nunique()
                ),
                "latest_fold_meaningful": bool(
                    (
                        meaningful["fold"] == FOLD_COUNT
                    ).any()
                ),
                "median_prediction_span": float(
                    group["prediction_span"].median()
                ),
                "maximum_prediction_span": float(
                    group["prediction_span"].max()
                ),
                "meaningful_directions_observed": (
                    "|".join(directions) if directions else "none"
                ),
                "direction_consistent_when_meaningful": bool(
                    len(directions) <= 1
                ),
                "positive_meaningful_checks": int(
                    meaningful["response_direction"].eq("positive").sum()
                ),
                "negative_meaningful_checks": int(
                    meaningful["response_direction"].eq("negative").sum()
                ),
                "all_monotonic": bool(
                    group["monotonic_over_grid"].all()
                ),
                "all_sensible": bool(
                    group["finite_and_sensible"].all()
                ),
                "meaningful_response_threshold": (
                    _meaningful_response_threshold(target)
                ),
            }
        )
    return pd.DataFrame(rows)


def _residual_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    ridge = predictions.loc[
        predictions["strategy"].eq(RIDGE_STRATEGY)
    ].copy()
    rows: list[dict[str, Any]] = []
    for target, group in ridge.groupby("target", sort=True):
        residual = group["residual"].to_numpy(dtype=float)
        absolute = np.abs(residual)
        worst_position = int(np.argmax(absolute))
        worst = group.iloc[worst_position]
        quantiles = np.quantile(residual, [0.05, 0.50, 0.95])
        rows.append(
            {
                "target": target,
                "unit": _target_unit(target),
                "sample_count": len(group),
                "residual_mean": float(residual.mean()),
                "residual_standard_deviation": float(
                    residual.std(ddof=1)
                ),
                "residual_p05": float(quantiles[0]),
                "residual_median": float(quantiles[1]),
                "residual_p95": float(quantiles[2]),
                "maximum_absolute_error": float(absolute.max()),
                "worst_sample_date": worst["sample_date"],
                "worst_actual": float(worst["actual"]),
                "worst_prediction": float(worst["prediction"]),
            }
        )
    return pd.DataFrame(rows)


def _naive_comparison(
    aggregate: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    regime_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target in TARGET_NAMES:
        target_metrics = aggregate.loc[
            aggregate["target"].eq(target)
        ]
        ridge = target_metrics.loc[
            target_metrics["strategy"].eq(RIDGE_STRATEGY)
        ].iloc[0]
        naive = target_metrics.loc[
            target_metrics["strategy"].isin(NAIVE_STRATEGIES)
        ]
        best_mae = naive.sort_values(
            ["mae", "rmse", "strategy"],
            kind="stable",
        ).iloc[0]
        best_rmse = naive.sort_values(
            ["rmse", "mae", "strategy"],
            kind="stable",
        ).iloc[0]
        fold_wins = 0
        for fold in range(1, FOLD_COUNT + 1):
            fold_rows = fold_metrics.loc[
                fold_metrics["target"].eq(target)
                & fold_metrics["fold"].eq(fold)
            ]
            ridge_mae = float(
                fold_rows.loc[
                    fold_rows["strategy"].eq(RIDGE_STRATEGY),
                    "mae",
                ].iloc[0]
            )
            best_naive_fold = float(
                fold_rows.loc[
                    fold_rows["strategy"].isin(NAIVE_STRATEGIES),
                    "mae",
                ].min()
            )
            fold_wins += int(ridge_mae < best_naive_fold)
        regime_wins = 0
        for regime in regime_metrics["vix_regime"].unique():
            regime_rows = regime_metrics.loc[
                regime_metrics["target"].eq(target)
                & regime_metrics["vix_regime"].eq(regime)
            ]
            ridge_mae = float(
                regime_rows.loc[
                    regime_rows["strategy"].eq(RIDGE_STRATEGY),
                    "mae",
                ].iloc[0]
            )
            best_naive_regime = float(
                regime_rows.loc[
                    regime_rows["strategy"].isin(NAIVE_STRATEGIES),
                    "mae",
                ].min()
            )
            regime_wins += int(ridge_mae < best_naive_regime)
        rows.append(
            {
                "target": target,
                "unit": _target_unit(target),
                "ridge_mae": float(ridge["mae"]),
                "ridge_rmse": float(ridge["rmse"]),
                "ridge_oos_r_squared": float(ridge["oos_r_squared"]),
                "ridge_directional_accuracy": float(
                    ridge["directional_accuracy"]
                ),
                "best_naive_mae_strategy": best_mae["strategy"],
                "best_naive_mae": float(best_mae["mae"]),
                "best_naive_rmse_strategy": best_rmse["strategy"],
                "best_naive_rmse": float(best_rmse["rmse"]),
                "ridge_mae_improvement_vs_best_naive_pct": float(
                    100.0
                    * (best_mae["mae"] - ridge["mae"])
                    / best_mae["mae"]
                ),
                "ridge_rmse_improvement_vs_best_naive_pct": float(
                    100.0
                    * (best_rmse["rmse"] - ridge["rmse"])
                    / best_rmse["rmse"]
                ),
                "ridge_beats_best_naive_mae": bool(
                    ridge["mae"] < best_mae["mae"]
                ),
                "ridge_beats_best_naive_rmse": bool(
                    ridge["rmse"] < best_rmse["rmse"]
                ),
                "ridge_fold_mae_wins_vs_best_naive": fold_wins,
                "ridge_vix_regime_mae_wins_vs_best_naive": regime_wins,
            }
        )
    return pd.DataFrame(rows)


def _known_limitations(
    *,
    target: str,
    ridge_metrics: pd.Series,
    comparison: pd.Series,
    responsive_any: int,
    responsive_latest: int,
    consistent: int,
    final_alpha: float,
) -> str:
    limitations = [
        "final test not accessed",
        "predictive associations are not causal effects",
        "PMI input uses the disclosed Kansas City Fed manufacturing proxy",
    ]
    if ridge_metrics["oos_r_squared"] <= 0:
        limitations.append("nonpositive pooled out-of-sample R-squared")
    if not (
        comparison["ridge_beats_best_naive_mae"]
        and comparison["ridge_beats_best_naive_rmse"]
    ):
        limitations.append("does not beat the best naive on both MAE and RMSE")
    if responsive_any < 8:
        limitations.append(
            f"only {responsive_any} of 8 shocks are meaningful in any fold"
        )
    if responsive_latest < 4:
        limitations.append(
            f"only {responsive_latest} of 8 shocks are meaningful in fold 5"
        )
    if consistent < 6:
        limitations.append(
            f"only {consistent} responsive shocks have stable signs"
        )
    if final_alpha >= 100000.0:
        limitations.append("final pre-test fit requires very strong shrinkage")
    if target == "oil":
        limitations.append(
            "target uses a continuous CL=F futures series with roll effects"
        )
    return " | ".join(limitations)


def _target_assessment(
    *,
    aggregate: pd.DataFrame,
    naive_comparison: pd.DataFrame,
    sensitivity_summary: pd.DataFrame,
    coefficient_stability: pd.DataFrame,
    final_configurations: pd.DataFrame,
    fit_audit: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in TARGET_SPECS:
        target = spec.target
        ridge = aggregate.loc[
            aggregate["target"].eq(target)
            & aggregate["strategy"].eq(RIDGE_STRATEGY)
        ].iloc[0]
        comparison = naive_comparison.loc[
            naive_comparison["target"].eq(target)
        ].iloc[0]
        sensitivity = sensitivity_summary.loc[
            sensitivity_summary["target"].eq(target)
        ]
        scenario_coefficients = coefficient_stability.loc[
            coefficient_stability["target"].eq(target)
            & coefficient_stability["is_scenario_input"]
        ]
        final_configuration = final_configurations.loc[
            final_configurations["target"].eq(target)
        ].iloc[0]
        target_fits = fit_audit.loc[fit_audit["target"].eq(target)]

        responsive_any = int(
            (sensitivity["folds_with_meaningful_response"] > 0).sum()
        )
        responsive_three_plus = int(
            (sensitivity["folds_with_meaningful_response"] >= 3).sum()
        )
        responsive_all = int(
            (sensitivity["folds_with_meaningful_response"] == FOLD_COUNT).sum()
        )
        responsive_latest = int(
            sensitivity["latest_fold_meaningful"].sum()
        )
        consistent = int(
            (
                (sensitivity["folds_with_meaningful_response"] > 0)
                & sensitivity["direction_consistent_when_meaningful"]
            ).sum()
        )
        coefficient_sign_stable = int(
            scenario_coefficients["sign_stable_all_folds"].sum()
        )
        all_sensible = bool(sensitivity["all_sensible"].all())

        actual_abs_max = max(
            abs(float(ridge["actual_mean"]))
            + 6.0 * float(ridge["actual_standard_deviation"]),
            1e-12,
        )
        prediction_abs_max = max(
            abs(float(ridge["prediction_minimum"])),
            abs(float(ridge["prediction_maximum"])),
        )
        prediction_std_ratio = float(
            ridge["prediction_standard_deviation"]
            / ridge["actual_standard_deviation"]
        )
        prediction_abs_bound_ratio = float(
            prediction_abs_max / actual_abs_max
        )
        pathological = bool(
            not all_sensible
            or prediction_std_ratio > 2.0
            or prediction_abs_bound_ratio > 2.0
        )

        predictive_usable = bool(
            ridge["mae"] <= comparison["best_naive_mae"] * 1.02
            and ridge["rmse"] <= comparison["best_naive_rmse"] * 1.02
            and comparison["ridge_fold_mae_wins_vs_best_naive"] >= 2
        )
        predictive_strong = bool(
            comparison["ridge_beats_best_naive_mae"]
            and comparison["ridge_beats_best_naive_rmse"]
            and ridge["oos_r_squared"] > 0
            and comparison["ridge_fold_mae_wins_vs_best_naive"] >= 3
        )
        scenario_broad = bool(
            responsive_any == 8
            and responsive_three_plus >= 6
            and responsive_latest >= 4
            and consistent >= 6
        )

        if pathological or responsive_any < 4:
            classification = "NOT_APPROVED"
            fallback = "NO_SCENARIO_INFERENCE"
            classification_reason = (
                "Predictions or scenario responsiveness fail the minimum "
                "suitability gate."
            )
        elif not predictive_usable:
            classification = "NAIVE_FALLBACK_REQUIRED"
            fallback = str(comparison["best_naive_mae_strategy"])
            classification_reason = (
                "Ridge does not remain within 2% of the best naive on both "
                "errors with at least two fold wins."
            )
        elif predictive_strong and scenario_broad:
            classification = "APPROVED_FOR_EXPERIMENTAL_INFERENCE"
            fallback = ""
            classification_reason = (
                "Repeated predictive and broad scenario-response gates pass."
            )
        else:
            classification = "APPROVED_WITH_LOW_CONFIDENCE"
            fallback = ""
            classification_reason = (
                "Minimum predictive and scenario gates pass, but one or more "
                "strong experimental-approval gates fail."
            )

        rows.append(
            {
                "target": target,
                "model_name": f"{target}_model",
                "unit": _target_unit(target),
                "source": spec.source,
                "source_field": spec.value_field,
                "value_policy": spec.value_policy,
                "model_family": "ridge",
                "history_strategy": "full_history",
                "final_pretest_alpha": float(
                    final_configuration["selected_alpha"]
                ),
                "fold_selected_alphas_json": json.dumps(
                    [
                        float(value)
                        for value in target_fits.sort_values("fold")[
                            "selected_alpha"
                        ]
                    ],
                    separators=(",", ":"),
                ),
                "walk_forward_rows": int(ridge["sample_count"]),
                "ridge_mae": float(ridge["mae"]),
                "ridge_rmse": float(ridge["rmse"]),
                "ridge_oos_r_squared": float(ridge["oos_r_squared"]),
                "ridge_prediction_actual_correlation": float(
                    ridge["prediction_actual_correlation"]
                ),
                "ridge_directional_accuracy": float(
                    ridge["directional_accuracy"]
                ),
                "fold_mae_standard_deviation": float(
                    ridge["fold_mae_std"]
                ),
                "worst_fold_mae": float(ridge["worst_fold_mae"]),
                "best_naive_mae_strategy": (
                    comparison["best_naive_mae_strategy"]
                ),
                "best_naive_mae": float(comparison["best_naive_mae"]),
                "best_naive_rmse_strategy": (
                    comparison["best_naive_rmse_strategy"]
                ),
                "best_naive_rmse": float(comparison["best_naive_rmse"]),
                "ridge_mae_improvement_vs_best_naive_pct": float(
                    comparison[
                        "ridge_mae_improvement_vs_best_naive_pct"
                    ]
                ),
                "ridge_rmse_improvement_vs_best_naive_pct": float(
                    comparison[
                        "ridge_rmse_improvement_vs_best_naive_pct"
                    ]
                ),
                "ridge_fold_mae_wins_vs_best_naive": int(
                    comparison["ridge_fold_mae_wins_vs_best_naive"]
                ),
                "ridge_vix_regime_mae_wins_vs_best_naive": int(
                    comparison[
                        "ridge_vix_regime_mae_wins_vs_best_naive"
                    ]
                ),
                "prediction_minimum": float(ridge["prediction_minimum"]),
                "prediction_maximum": float(ridge["prediction_maximum"]),
                "prediction_standard_deviation_ratio": (
                    prediction_std_ratio
                ),
                "prediction_pathological": pathological,
                "responsive_inputs_any_fold": responsive_any,
                "responsive_inputs_three_or_more_folds": (
                    responsive_three_plus
                ),
                "responsive_inputs_all_folds": responsive_all,
                "responsive_inputs_latest_fold": responsive_latest,
                "directionally_consistent_responsive_inputs": consistent,
                "scenario_coefficient_sign_stable_all_folds": (
                    coefficient_sign_stable
                ),
                "all_sensitivity_outputs_sensible": all_sensible,
                "predictive_usable_gate": predictive_usable,
                "predictive_strong_gate": predictive_strong,
                "scenario_broad_gate": scenario_broad,
                "classification": classification,
                "classification_reason": classification_reason,
                "fallback_strategy": fallback,
                "suitable_for_scenario_inference": classification in {
                    "APPROVED_FOR_EXPERIMENTAL_INFERENCE",
                    "APPROVED_WITH_LOW_CONFIDENCE",
                },
                "production_ready": False,
                "test_metrics_status": (
                    "NOT_COMPUTED_LOCKED_FINAL_TEST_NOT_AUTHORIZED"
                ),
                "known_limitations": _known_limitations(
                    target=target,
                    ridge_metrics=ridge,
                    comparison=comparison,
                    responsive_any=responsive_any,
                    responsive_latest=responsive_latest,
                    consistent=consistent,
                    final_alpha=float(
                        final_configuration["selected_alpha"]
                    ),
                ),
            }
        )
    return pd.DataFrame(rows)


def build_all_target_artifacts() -> dict[str, Any]:
    """Build all Step 13 research outputs without final-test access."""
    frame, feature_names, target_names, guard = load_all_pretest_rows()
    fold_manifest, folds = build_walk_forward_folds(frame)
    trailing_audit = _trailing_baseline(folds)
    trailing_lookup = trailing_audit.set_index(
        ["fold", "sample_date", "target"]
    )["prediction"]

    prediction_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    alpha_search_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []

    for fold_number, (training, validation) in enumerate(folds, 1):
        for target in target_names:
            fitted = _fit_ridge(
                training,
                feature_names,
                target,
                evaluation_scope="outer_fold_inner_selection",
                fold=fold_number,
            )
            alpha_search_rows.extend(fitted.alpha_search)
            coefficient_rows.extend(
                _coefficient_rows(
                    fitted,
                    feature_names,
                    evaluation_scope="outer_fold",
                    fold=fold_number,
                )
            )
            sensitivity_rows.extend(
                _sensitivity_rows(
                    fold=fold_number,
                    fitted=fitted,
                    feature_names=feature_names,
                )
            )
            fit_rows.append(
                {
                    "fold": fold_number,
                    "target": target,
                    "unit": _target_unit(target),
                    "model_family": "ridge",
                    "history_strategy": "full_history",
                    "outer_training_rows": len(training),
                    "fit_first_sample": training["sample_date"].iloc[0],
                    "fit_last_sample": training["sample_date"].iloc[-1],
                    "inner_training_rows": fitted.inner_training_rows,
                    "inner_validation_rows": fitted.inner_validation_rows,
                    "inner_purge_rows": fitted.inner_purge_rows,
                    "selected_alpha": fitted.selected_alpha,
                    "training_mae": fitted.training_mae,
                    "training_rmse": fitted.training_rmse,
                    "inner_validation_mae": (
                        fitted.inner_validation_mae
                    ),
                    "inner_validation_rmse": (
                        fitted.inner_validation_rmse
                    ),
                    "preprocessing_fit_scope": (
                        "current purged outer-fold training rows only"
                    ),
                    "locked_final_test_excluded": True,
                }
            )
            strategies = {
                "naive_zero": np.zeros(len(validation), dtype=float),
                "naive_training_mean": np.repeat(
                    float(training[target].mean()),
                    len(validation),
                ),
                "naive_recent_trailing_21_observation": np.asarray(
                    [
                        float(
                            trailing_lookup.loc[
                                (
                                    fold_number,
                                    row.sample_date,
                                    target,
                                )
                            ]
                        )
                        for row in validation.itertuples(index=False)
                    ]
                ),
                RIDGE_STRATEGY: fitted.pipeline.predict(
                    validation[list(feature_names)]
                ),
            }
            for strategy, predicted in strategies.items():
                metrics = regression_metrics(
                    validation[target],
                    predicted,
                )
                fold_metric_rows.append(
                    {
                        "fold": fold_number,
                        "target": target,
                        "unit": _target_unit(target),
                        "strategy": strategy,
                        **metrics,
                    }
                )
                for position, row in enumerate(
                    validation.itertuples(index=False)
                ):
                    actual = float(getattr(row, target))
                    prediction = float(predicted[position])
                    prediction_rows.append(
                        {
                            "fold": fold_number,
                            "sample_date": row.sample_date,
                            "target_baseline_date": (
                                row.target_baseline_date
                            ),
                            "target_end_date": row.target_end_date,
                            "target": target,
                            "unit": _target_unit(target),
                            "strategy": strategy,
                            "actual": actual,
                            "prediction": prediction,
                            "residual": actual - prediction,
                            "absolute_error": abs(actual - prediction),
                            "vix_level_points": float(
                                row.vix_level_points
                            ),
                        }
                    )

    predictions = pd.DataFrame(prediction_rows)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    fit_audit = pd.DataFrame(fit_rows)
    alpha_search = pd.DataFrame(alpha_search_rows)
    fold_coefficients = pd.DataFrame(coefficient_rows)
    sensitivity = pd.DataFrame(sensitivity_rows)

    aggregate = _aggregate_metrics(predictions, fold_metrics)
    regimes = _regime_metrics(predictions)
    coefficient_stability = _coefficient_stability(fold_coefficients)
    sensitivity_summary = _sensitivity_summary(sensitivity)
    residuals = _residual_summary(predictions)
    naive_comparison = _naive_comparison(
        aggregate,
        fold_metrics,
        regimes,
    )

    final_configuration_rows: list[dict[str, Any]] = []
    final_coefficient_rows: list[dict[str, Any]] = []
    for target in target_names:
        final_fitted = _fit_ridge(
            frame,
            feature_names,
            target,
            evaluation_scope="final_pretest_inner_selection",
            fold=None,
        )
        alpha_search = pd.concat(
            [
                alpha_search,
                pd.DataFrame(final_fitted.alpha_search),
            ],
            ignore_index=True,
        )
        final_coefficient_rows.extend(
            _coefficient_rows(
                final_fitted,
                feature_names,
                evaluation_scope="final_pretest_refit",
                fold=None,
            )
        )
        final_configuration_rows.append(
            {
                "target": target,
                "model_name": f"{target}_model",
                "unit": _target_unit(target),
                "model_family": "ridge",
                "history_strategy": "full_history",
                "selected_alpha": final_fitted.selected_alpha,
                "training_rows": len(frame),
                "training_first_sample": frame["sample_date"].iloc[0],
                "training_last_sample": frame["sample_date"].iloc[-1],
                "training_last_target_end": frame["target_end_date"].iloc[-1],
                "inner_training_rows": (
                    final_fitted.inner_training_rows
                ),
                "inner_validation_rows": (
                    final_fitted.inner_validation_rows
                ),
                "inner_purge_rows": final_fitted.inner_purge_rows,
                "inner_validation_mae": (
                    final_fitted.inner_validation_mae
                ),
                "inner_validation_rmse": (
                    final_fitted.inner_validation_rmse
                ),
                "full_pretest_training_mae": (
                    final_fitted.training_mae
                ),
                "full_pretest_training_rmse": (
                    final_fitted.training_rmse
                ),
                "configuration_status": (
                    "FROZEN_FROM_PRETEST_DATA_FOR_STEP_14_RESEARCH"
                ),
                "model_binary_exported": False,
                "locked_final_test_used": False,
            }
        )
    final_configurations = pd.DataFrame(final_configuration_rows)
    final_coefficients = pd.DataFrame(final_coefficient_rows)
    assessment = _target_assessment(
        aggregate=aggregate,
        naive_comparison=naive_comparison,
        sensitivity_summary=sensitivity_summary,
        coefficient_stability=coefficient_stability,
        final_configurations=final_configurations,
        fit_audit=fit_audit,
    )

    classification_counts = {
        classification: int(
            assessment["classification"].eq(classification).sum()
        )
        for classification in CLASSIFICATIONS
    }
    pathological_count = int(
        assessment["prediction_pathological"].sum()
    )
    validations = {
        "all_twelve_targets_present": (
            tuple(assessment["target"]) == TARGET_NAMES
        ),
        "ridge_only_full_history": bool(
            assessment["model_family"].eq("ridge").all()
            and assessment["history_strategy"].eq("full_history").all()
        ),
        "five_frozen_step_12_folds_reused": len(fold_manifest) == FOLD_COUNT,
        "all_outer_purges_nonempty": bool(
            (fold_manifest["outer_purge_rows"] > 0).all()
        ),
        "all_inner_purges_nonempty": bool(
            (fit_audit["inner_purge_rows"] > 0).all()
        ),
        "sixty_fold_target_fits": len(fit_audit) == 12 * FOLD_COUNT,
        "all_four_strategies_present": (
            set(predictions["strategy"]) == set(ALL_STRATEGIES)
        ),
        "three_hundred_predictions_per_target_strategy": bool(
            predictions.groupby(["target", "strategy"])
            .size()
            .eq(FOLD_COUNT * VALIDATION_ROWS_PER_FOLD)
            .all()
        ),
        "all_predictions_finite": bool(
            np.isfinite(
                predictions[["actual", "prediction", "residual"]]
            ).all().all()
        ),
        "all_trailing_baselines_point_in_time": bool(
            trailing_audit["point_in_time_pass"].all()
        ),
        "all_58_coefficients_present_per_fold_target": (
            len(fold_coefficients) == 12 * FOLD_COUNT * 58
        ),
        "all_eight_shocks_checked_per_target": (
            len(sensitivity_summary) == 12 * 8
        ),
        "all_sensitivity_outputs_finite_and_sensible": bool(
            sensitivity["finite_and_sensible"].all()
        ),
        "all_targets_classified": bool(
            assessment["classification"].isin(CLASSIFICATIONS).all()
        ),
        "no_pathological_ridge_prediction_sets": pathological_count == 0,
        "all_final_pretest_configurations_frozen": (
            len(final_configurations) == 12
            and final_configurations["model_binary_exported"].eq(False).all()
        ),
        "event_aware_matrix_and_58_features_preserved": (
            len(feature_names) == 58
        ),
        "locked_final_test_target_rows_parsed_zero": (
            guard["locked_final_test_target_rows_parsed"] == 0
        ),
        "final_test_not_accessed": True,
        "no_production_artifacts_exported": True,
        "no_step_14_or_application_work": True,
    }
    if not all(validations.values()):
        failures = [
            name for name, passed in validations.items() if not passed
        ]
        non_sensible_targets = sorted(
            sensitivity.loc[
                ~sensitivity["finite_and_sensible"],
                "target",
            ].unique()
        )
        non_sensible_checks = sensitivity.loc[
            ~sensitivity["finite_and_sensible"],
            [
                "target",
                "fold",
                "contract_input",
                "training_minimum",
                "training_maximum",
                "prediction_span",
                "maximum_adjacent_jump",
                "maximum_sensible_span",
            ],
        ].drop_duplicates().to_dict(orient="records")
        pathological_targets = assessment.loc[
            assessment["prediction_pathological"],
            [
                "target",
                "prediction_standard_deviation_ratio",
                "prediction_minimum",
                "prediction_maximum",
            ],
        ].to_dict(orient="records")
        raise ValueError(
            "Step 13 validation failed: "
            f"{failures}; non_sensible_targets={non_sensible_targets}; "
            f"non_sensible_checks={non_sensible_checks}; "
            f"pathological_targets={pathological_targets}"
        )

    summary = {
        "schema_version": "scenario_all_target_ridge_v1",
        "model_family": "ridge",
        "history_strategy": "full_history",
        "sampling_design": "EVENT_AWARE",
        "target_horizon_sessions": 21,
        "matrix_rows_declared": guard["matrix_rows_declared"],
        "pretest_rows_evaluated": len(frame),
        "feature_count": len(feature_names),
        "target_count": len(target_names),
        "fold_count": FOLD_COUNT,
        "validation_rows_per_fold": VALIDATION_ROWS_PER_FOLD,
        "walk_forward_rows_per_target": (
            FOLD_COUNT * VALIDATION_ROWS_PER_FOLD
        ),
        "matrix_guard": {
            **guard,
            "final_test_target_values_available_to_step_13": False,
            "final_test_access_count": 0,
        },
        "final_test_decision": (
            "NOT_ACCESSED: Step 13 lists test metrics but does not explicitly "
            "authorize the one-time locked-test opening required by the user."
        ),
        "classification_rules": {
            "minimum_scenario_gate": (
                "At least four of eight inputs have a meaningful response in "
                "one or more folds; all sensitivity outputs must be sensible."
            ),
            "predictive_usable_gate": (
                "Ridge MAE and RMSE are each within 2% of the corresponding "
                "best naive metric and Ridge beats the best naive MAE in at "
                "least two folds."
            ),
            "strong_predictive_gate": (
                "Ridge beats best naive MAE and RMSE, pooled OOS R-squared is "
                "positive, and Ridge wins at least three folds."
            ),
            "broad_scenario_gate": (
                "All eight inputs respond in at least one fold; at least six "
                "respond in three folds; at least four respond in fold 5; and "
                "at least six responsive inputs have stable directions."
            ),
        },
        "meaningful_response_thresholds": {
            "price_return_targets_percentage_points": 0.05,
            "ten_year_yield_basis_points": 1.0,
        },
        "classification_counts": classification_counts,
        "approved_experimental_targets": assessment.loc[
            assessment["classification"].eq(
                "APPROVED_FOR_EXPERIMENTAL_INFERENCE"
            ),
            "target",
        ].tolist(),
        "approved_low_confidence_targets": assessment.loc[
            assessment["classification"].eq(
                "APPROVED_WITH_LOW_CONFIDENCE"
            ),
            "target",
        ].tolist(),
        "naive_fallback_targets": assessment.loc[
            assessment["classification"].eq(
                "NAIVE_FALLBACK_REQUIRED"
            ),
            "target",
        ].tolist(),
        "not_approved_targets": assessment.loc[
            assessment["classification"].eq("NOT_APPROVED"),
            "target",
        ].tolist(),
        "target_classifications": {
            row.target: row.classification
            for row in assessment.itertuples()
        },
        "model_set_ready_for_step_14": True,
        "step_14_readiness": (
            "READY_FOR_STEP_14_RESEARCH_INFERENCE_WITH_PER_TARGET_GATES; "
            "not approved for production and no final-test estimate exists."
        ),
        "software_versions": {
            "python": (
                f"{sys.version_info.major}.{sys.version_info.minor}."
                f"{sys.version_info.micro}"
            ),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "validations": validations,
        "training_gate": (
            "STEP_13_ALL_TARGET_RIDGE_EVALUATION_COMPLETE_"
            "READY_FOR_STEP_14_WITH_TARGET_GATES"
        ),
        "scope_boundary": (
            "Step 13 only: no final-test access, Step 14, XGBoost, production "
            "artifact export, Supabase, backend, frontend, or servers."
        ),
    }
    return {
        "fold_manifest": fold_manifest,
        "trailing_baseline_audit": trailing_audit,
        "alpha_search": alpha_search,
        "fit_audit": fit_audit,
        "fold_predictions": predictions,
        "fold_metrics": fold_metrics,
        "aggregate_metrics": aggregate,
        "naive_comparison": naive_comparison,
        "regime_metrics": regimes,
        "residual_summary": residuals,
        "fold_coefficients": fold_coefficients,
        "coefficient_stability": coefficient_stability,
        "sensitivity": sensitivity,
        "sensitivity_summary": sensitivity_summary,
        "final_configurations": final_configurations,
        "final_coefficients": final_coefficients,
        "target_assessment": assessment,
        "summary": summary,
    }


def persist_all_target_artifacts(
    artifacts: dict[str, Any],
    output_dir: Path = OUTPUT_ROOT,
) -> dict[str, str]:
    paths = {
        name: output_dir / f"all_targets_{name}.csv"
        for name in artifacts
        if name != "summary"
    }
    paths["summary"] = output_dir / "all_targets_summary.json"
    texts = {
        name: frame_to_csv_text(artifacts[name])
        for name in paths
        if name != "summary"
    }
    summary = dict(artifacts["summary"])
    summary["output_files"] = {
        name: {
            "path": relative_path(path),
            "sha256": (
                hashlib.sha256(texts[name].encode()).hexdigest()
                if name in texts
                else None
            ),
        }
        for name, path in paths.items()
    }
    for name, text in texts.items():
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
        name: str(path.resolve())
        for name, path in paths.items()
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run Step 13 purged full-history Ridge evaluation for all targets."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and validate all outputs without persisting CSV/JSON.",
    )
    args = parser.parse_args(argv)
    artifacts = build_all_target_artifacts()
    paths = (
        {}
        if args.dry_run
        else persist_all_target_artifacts(artifacts)
    )
    summary = artifacts["summary"]
    print(
        json.dumps(
            {
                "targets": summary["target_count"],
                "folds": summary["fold_count"],
                "classification_counts": (
                    summary["classification_counts"]
                ),
                "final_test_access_count": summary["matrix_guard"][
                    "final_test_access_count"
                ],
                "gate": summary["training_gate"],
                "outputs": paths,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
