"""Train and diagnose the Step 11 S&P 500 XGBoost prototype.

Only ``development_train`` rows update model parameters. The frozen
``validation`` rows are used for early stopping and configuration selection.
Locked final-test target rows are never parsed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import xgboost as xgb

from .cache import write_text_atomic
from .config import INTERIM_ROOT
from .evaluation import metrics_by_calendar_year, regression_metrics
from .feature_sources import frame_to_csv_text, relative_path
from .ridge_baselines import (
    MATRIX_PATH,
    RIDGE_OUTPUT_ROOT,
    load_development_and_validation,
)


TARGET = "sp500"
TARGET_UNIT = "percent"
SEED = 20260725
RECENT_WINDOW_YEARS = 5
XGBOOST_OUTPUT_ROOT = INTERIM_ROOT / "models" / "xgboost_prototype"
RIDGE_PREDICTIONS_PATH = (
    RIDGE_OUTPUT_ROOT / "ridge_validation_predictions.csv"
)
RIDGE_METRICS_PATH = RIDGE_OUTPUT_ROOT / "ridge_validation_metrics.csv"

SCENARIO_INPUT_FEATURES = {
    "fed_funds_rate_change_bps": "fed_funds_rate_change_bps",
    "cpi_surprise_pct": "cpi_surprise_pct",
    "oil_price_change_pct": "oil_price_change_pct",
    "gdp_growth_surprise_pct": "gdp_growth_surprise_pct",
    "unemployment_change_pct": "unemployment_change_pct",
    "pmi_change_points": (
        "kc_fed_manufacturing_pmi_proxy_change_points"
    ),
    "dxy_change_pct": "dxy_change_pct",
    "vix_change_points": "vix_change_points",
}


@dataclass(frozen=True)
class Architecture:
    name: str
    complexity_rank: int
    max_depth: int
    learning_rate: float
    min_child_weight: float
    subsample: float
    colsample_bytree: float
    reg_alpha: float
    reg_lambda: float
    gamma: float
    max_rounds: int
    early_stopping_rounds: int


ARCHITECTURES = (
    Architecture(
        name="regularized_stumps",
        complexity_rank=1,
        max_depth=1,
        learning_rate=0.03,
        min_child_weight=8.0,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_alpha=0.10,
        reg_lambda=10.0,
        gamma=0.00,
        max_rounds=500,
        early_stopping_rounds=40,
    ),
    Architecture(
        name="regularized_depth2",
        complexity_rank=2,
        max_depth=2,
        learning_rate=0.025,
        min_child_weight=8.0,
        subsample=0.80,
        colsample_bytree=0.70,
        reg_alpha=0.25,
        reg_lambda=15.0,
        gamma=0.05,
        max_rounds=600,
        early_stopping_rounds=40,
    ),
    Architecture(
        name="strongly_regularized_depth2",
        complexity_rank=3,
        max_depth=2,
        learning_rate=0.02,
        min_child_weight=15.0,
        subsample=0.70,
        colsample_bytree=0.65,
        reg_alpha=0.75,
        reg_lambda=30.0,
        gamma=0.10,
        max_rounds=700,
        early_stopping_rounds=50,
    ),
)
TRAINING_WINDOWS = ("full_development", "recent_5y")
EXPECTED_CANDIDATE_COUNT = len(ARCHITECTURES) * len(TRAINING_WINDOWS)


@dataclass
class CandidateResult:
    candidate_id: str
    architecture: Architecture
    training_window: str
    training_frame: pd.DataFrame
    booster: xgb.Booster
    development_prediction: np.ndarray
    validation_prediction: np.ndarray
    development_metrics: dict[str, Any]
    validation_metrics: dict[str, Any]
    evaluation_history: dict[str, dict[str, list[float]]]
    best_iteration: int
    best_score: float


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (pd.Timestamp, date)):
        return str(value)
    return value


def _training_frame(
    development: pd.DataFrame,
    training_window: str,
) -> tuple[pd.DataFrame, str]:
    dates = pd.to_datetime(development["sample_date"])
    if training_window == "full_development":
        return development.copy(), str(dates.min().date())
    if training_window == "recent_5y":
        cutoff = dates.max() - pd.DateOffset(years=RECENT_WINDOW_YEARS)
        frame = development.loc[dates >= cutoff].copy()
        if len(frame) < 200:
            raise ValueError(
                "The fixed recent-history window has fewer than 200 rows."
            )
        return frame, str(cutoff.date())
    raise ValueError(f"Unknown training window: {training_window}")


def _xgboost_parameters(
    architecture: Architecture,
    *,
    base_score: float,
) -> dict[str, Any]:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "booster": "gbtree",
        "tree_method": "hist",
        "device": "cpu",
        "max_bin": 128,
        "max_depth": architecture.max_depth,
        "eta": architecture.learning_rate,
        "min_child_weight": architecture.min_child_weight,
        "subsample": architecture.subsample,
        "colsample_bytree": architecture.colsample_bytree,
        "alpha": architecture.reg_alpha,
        "lambda": architecture.reg_lambda,
        "gamma": architecture.gamma,
        "base_score": float(base_score),
        "seed": SEED,
        "nthread": 1,
        "verbosity": 0,
        "validate_parameters": True,
    }


def _dmatrix(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    labels: np.ndarray | None = None,
) -> xgb.DMatrix:
    return xgb.DMatrix(
        frame.loc[:, list(feature_names)].to_numpy(dtype=float),
        label=labels,
        feature_names=list(feature_names),
        missing=np.nan,
    )


def _fit_candidate(
    *,
    development: pd.DataFrame,
    validation: pd.DataFrame,
    feature_names: Sequence[str],
    architecture: Architecture,
    training_window: str,
) -> CandidateResult:
    training_frame, _ = _training_frame(development, training_window)
    y_training = training_frame[TARGET].to_numpy(dtype=float)
    y_validation = validation[TARGET].to_numpy(dtype=float)
    dtraining = _dmatrix(
        training_frame,
        feature_names,
        labels=y_training,
    )
    dvalidation = _dmatrix(
        validation,
        feature_names,
        labels=y_validation,
    )
    evaluation_history: dict[str, dict[str, list[float]]] = {}
    booster = xgb.train(
        _xgboost_parameters(
            architecture,
            base_score=float(np.mean(y_training)),
        ),
        dtraining,
        num_boost_round=architecture.max_rounds,
        evals=[(dtraining, "development"), (dvalidation, "validation")],
        early_stopping_rounds=architecture.early_stopping_rounds,
        evals_result=evaluation_history,
        verbose_eval=False,
    )
    best_iteration = int(booster.best_iteration)
    iteration_range = (0, best_iteration + 1)
    development_prediction = booster.predict(
        dtraining,
        iteration_range=iteration_range,
    )
    validation_prediction = booster.predict(
        dvalidation,
        iteration_range=iteration_range,
    )
    candidate_id = f"{training_window}__{architecture.name}"
    return CandidateResult(
        candidate_id=candidate_id,
        architecture=architecture,
        training_window=training_window,
        training_frame=training_frame,
        booster=booster,
        development_prediction=development_prediction,
        validation_prediction=validation_prediction,
        development_metrics=regression_metrics(
            y_training,
            development_prediction,
        ),
        validation_metrics=regression_metrics(
            y_validation,
            validation_prediction,
        ),
        evaluation_history=evaluation_history,
        best_iteration=best_iteration,
        best_score=float(booster.best_score),
    )


def _candidate_tables(
    results: Sequence[CandidateResult],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    learning_rows: list[dict[str, Any]] = []
    for result in results:
        architecture = result.architecture
        dates = pd.to_datetime(result.training_frame["sample_date"])
        validation_mae = float(result.validation_metrics["mae"])
        development_mae = float(result.development_metrics["mae"])
        summary_rows.append(
            {
                "candidate_id": result.candidate_id,
                "training_window": result.training_window,
                "architecture": architecture.name,
                "complexity_rank": architecture.complexity_rank,
                "training_rows": len(result.training_frame),
                "training_first_sample": str(dates.min().date()),
                "training_last_sample": str(dates.max().date()),
                "validation_rows": int(
                    result.validation_metrics["sample_count"]
                ),
                "max_depth": architecture.max_depth,
                "learning_rate": architecture.learning_rate,
                "min_child_weight": architecture.min_child_weight,
                "subsample": architecture.subsample,
                "colsample_bytree": architecture.colsample_bytree,
                "reg_alpha": architecture.reg_alpha,
                "reg_lambda": architecture.reg_lambda,
                "gamma": architecture.gamma,
                "max_rounds": architecture.max_rounds,
                "early_stopping_rounds": (
                    architecture.early_stopping_rounds
                ),
                "best_iteration_zero_based": result.best_iteration,
                "trees_used": result.best_iteration + 1,
                "rounds_evaluated": len(
                    result.evaluation_history["validation"]["mae"]
                ),
                "stopped_early": len(
                    result.evaluation_history["validation"]["mae"]
                )
                < architecture.max_rounds,
                "development_mae": development_mae,
                "validation_mae": validation_mae,
                "mae_generalization_gap": (
                    validation_mae - development_mae
                ),
                "mae_validation_to_development_ratio": (
                    validation_mae / development_mae
                    if development_mae > 0
                    else math.nan
                ),
                "development_rmse": result.development_metrics["rmse"],
                "validation_rmse": result.validation_metrics["rmse"],
                "rmse_generalization_gap": (
                    result.validation_metrics["rmse"]
                    - result.development_metrics["rmse"]
                ),
                "validation_oos_r_squared": (
                    result.validation_metrics["oos_r_squared"]
                ),
                "validation_prediction_actual_correlation": (
                    result.validation_metrics[
                        "prediction_actual_correlation"
                    ]
                ),
                "validation_directional_accuracy": (
                    result.validation_metrics["directional_accuracy"]
                ),
                "validation_prediction_minimum": (
                    result.validation_metrics["prediction_minimum"]
                ),
                "validation_prediction_maximum": (
                    result.validation_metrics["prediction_maximum"]
                ),
                "validation_prediction_standard_deviation": (
                    result.validation_metrics[
                        "prediction_standard_deviation"
                    ]
                ),
            }
        )
        for evaluation_role, metrics in (
            ("development_in_sample", result.development_metrics),
            ("validation_selection", result.validation_metrics),
        ):
            metric_rows.append(
                {
                    "candidate_id": result.candidate_id,
                    "training_window": result.training_window,
                    "architecture": architecture.name,
                    "evaluation_role": evaluation_role,
                    "target": TARGET,
                    "unit": TARGET_UNIT,
                    **metrics,
                }
            )
        development_curve = result.evaluation_history["development"]["mae"]
        validation_curve = result.evaluation_history["validation"]["mae"]
        for index, (development_mae_round, validation_mae_round) in enumerate(
            zip(development_curve, validation_curve, strict=True)
        ):
            learning_rows.append(
                {
                    "candidate_id": result.candidate_id,
                    "training_window": result.training_window,
                    "architecture": architecture.name,
                    "boosting_round": index + 1,
                    "development_mae": float(development_mae_round),
                    "validation_mae": float(validation_mae_round),
                    "is_best_iteration": index == result.best_iteration,
                }
            )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["validation_mae", "validation_rmse", "complexity_rank", "candidate_id"],
        kind="stable",
    ).reset_index(drop=True)
    summary["validation_mae_rank"] = np.arange(1, len(summary) + 1)
    metrics = pd.DataFrame(metric_rows).sort_values(
        ["candidate_id", "evaluation_role"],
        kind="stable",
    ).reset_index(drop=True)
    learning = pd.DataFrame(learning_rows).sort_values(
        ["candidate_id", "boosting_round"],
        kind="stable",
    ).reset_index(drop=True)
    return summary, metrics, learning


def _fixed_baseline_predictions(
    validation: pd.DataFrame,
) -> pd.DataFrame:
    if not RIDGE_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Step 10 predictions are missing: {RIDGE_PREDICTIONS_PATH}"
        )
    frame = pd.read_csv(RIDGE_PREDICTIONS_PATH)
    frame = frame.loc[frame["target"].eq(TARGET)].copy()
    if len(frame) != len(validation):
        raise ValueError("The fixed S&P Ridge prediction count changed.")
    if frame["sample_date"].tolist() != validation["sample_date"].tolist():
        raise ValueError("Step 10 S&P predictions do not match validation.")
    if not np.allclose(
        frame["actual"].to_numpy(dtype=float),
        validation[TARGET].to_numpy(dtype=float),
        rtol=0,
        atol=1e-10,
    ):
        raise ValueError("Step 10 and approved-matrix S&P targets differ.")
    return frame


def _comparison_tables(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    strategy_columns = {
        "naive_zero": "naive_zero_prediction",
        "naive_training_mean": "naive_training_mean_prediction",
        "naive_recent_trailing_21_observation": (
            "naive_recent_trailing_21_observation_prediction"
        ),
        "ridge": "ridge_prediction",
        "xgboost": "xgboost_prediction",
    }
    rows: list[dict[str, Any]] = []
    long_frames: list[pd.DataFrame] = []
    for strategy, column in strategy_columns.items():
        rows.append(
            {
                "target": TARGET,
                "unit": TARGET_UNIT,
                "strategy": strategy,
                "evaluation_role": "validation",
                **regression_metrics(
                    predictions["actual"],
                    predictions[column],
                ),
            }
        )
        long_frames.append(
            pd.DataFrame(
                {
                    "sample_date": predictions["sample_date"],
                    "target": TARGET,
                    "strategy": strategy,
                    "actual": predictions["actual"],
                    "prediction": predictions[column],
                }
            )
        )
    metrics = pd.DataFrame(rows)
    by_year = metrics_by_calendar_year(
        pd.concat(long_frames, ignore_index=True)
    )
    return metrics, by_year


def _feature_importance(
    result: CandidateResult,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    gain = result.booster.get_score(importance_type="gain")
    total_gain = result.booster.get_score(importance_type="total_gain")
    weight = result.booster.get_score(importance_type="weight")
    cover = result.booster.get_score(importance_type="cover")
    rows = [
        {
            "feature": feature,
            "gain": float(gain.get(feature, 0.0)),
            "total_gain": float(total_gain.get(feature, 0.0)),
            "split_count": int(weight.get(feature, 0.0)),
            "cover": float(cover.get(feature, 0.0)),
            "is_scenario_input": feature
            in set(SCENARIO_INPUT_FEATURES.values()),
        }
        for feature in feature_names
    ]
    frame = pd.DataFrame(rows)
    total = float(frame["total_gain"].sum())
    frame["normalized_total_gain"] = (
        frame["total_gain"] / total if total > 0 else 0.0
    )
    frame = frame.sort_values(
        ["total_gain", "split_count", "feature"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    frame["importance_rank"] = np.arange(1, len(frame) + 1)
    frame.insert(0, "candidate_id", result.candidate_id)
    return frame


def _residuals_over_time(
    predictions: pd.DataFrame,
    validation: pd.DataFrame,
    candidate_id: str,
) -> pd.DataFrame:
    residuals = predictions[
        ["sample_date", "target_baseline_date", "target_end_date", "actual"]
    ].copy()
    residuals.insert(0, "candidate_id", candidate_id)
    residuals["prediction"] = predictions["xgboost_prediction"]
    residuals["residual"] = residuals["actual"] - residuals["prediction"]
    residuals["absolute_error"] = residuals["residual"].abs()
    residuals["squared_error"] = residuals["residual"] ** 2
    residuals["rolling_20_sample_mae"] = (
        residuals["absolute_error"].rolling(20, min_periods=1).mean()
    )
    residuals["calendar_year"] = pd.to_datetime(
        residuals["sample_date"]
    ).dt.year
    residuals["vix_level_points"] = validation[
        "vix_level_points"
    ].to_numpy(dtype=float)
    return residuals


def _vix_regime_metrics(
    residuals: pd.DataFrame,
) -> pd.DataFrame:
    bins = [-np.inf, 20.0, 30.0, np.inf]
    labels = ["low_vix_lt_20", "medium_vix_20_to_30", "high_vix_ge_30"]
    regimes = pd.cut(
        residuals["vix_level_points"],
        bins=bins,
        labels=labels,
        right=False,
    )
    rows: list[dict[str, Any]] = []
    for label in labels:
        subset = residuals.loc[regimes.eq(label)]
        if subset.empty:
            raise ValueError(f"No validation rows in VIX regime {label}.")
        rows.append(
            {
                "vix_regime": label,
                "evaluation_role": "validation",
                "lower_bound_inclusive": (
                    None if label == labels[0] else bins[labels.index(label)]
                ),
                "upper_bound_exclusive": (
                    None
                    if label == labels[-1]
                    else bins[labels.index(label) + 1]
                ),
                **regression_metrics(
                    subset["actual"],
                    subset["prediction"],
                ),
            }
        )
    return pd.DataFrame(rows)


def _crisis_period_metrics(
    selected: CandidateResult,
    validation: pd.DataFrame,
) -> pd.DataFrame:
    development = selected.training_frame.copy()
    development["prediction"] = selected.development_prediction
    validation_frame = validation.copy()
    validation_frame["prediction"] = selected.validation_prediction
    periods = (
        (
            "covid_market_shock",
            "development_in_sample",
            development,
            "2020-02-19",
            "2020-06-30",
        ),
        (
            "inflation_tightening_2022",
            "validation_out_of_sample",
            validation_frame,
            "2022-01-01",
            "2022-12-31",
        ),
        (
            "regional_bank_stress_2023",
            "validation_out_of_sample",
            validation_frame,
            "2023-03-08",
            "2023-05-05",
        ),
    )
    rows: list[dict[str, Any]] = []
    for name, role, frame, start, end in periods:
        dates = pd.to_datetime(frame["sample_date"])
        subset = frame.loc[
            dates.between(pd.Timestamp(start), pd.Timestamp(end))
        ]
        if subset.empty:
            raise ValueError(f"No rows found for stress period {name}.")
        rows.append(
            {
                "period": name,
                "evaluation_role": role,
                "period_start": start,
                "period_end": end,
                "interpretation": (
                    "diagnostic only; in-sample"
                    if role == "development_in_sample"
                    else "out-of-sample validation diagnostic"
                ),
                **regression_metrics(
                    subset[TARGET],
                    subset["prediction"],
                ),
            }
        )
    return pd.DataFrame(rows)


def _sensitivity_tables(
    selected: CandidateResult,
    feature_names: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    training = selected.training_frame
    reference = training.loc[:, list(feature_names)].median(axis=0)
    for feature in SCENARIO_INPUT_FEATURES.values():
        reference.loc[feature] = 0.0
    reference_frame = pd.DataFrame(
        [reference.to_dict()],
        columns=list(feature_names),
    )
    neutral_prediction = float(
        selected.booster.predict(
            _dmatrix(reference_frame, feature_names),
            iteration_range=(0, selected.best_iteration + 1),
        )[0]
    )
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for contract_name, feature in SCENARIO_INPUT_FEATURES.items():
        values = training[feature].to_numpy(dtype=float)
        grid = np.linspace(float(values.min()), float(values.max()), 9)
        grid = np.unique(np.append(grid, 0.0))
        scenario_frame = pd.DataFrame(
            np.repeat(
                reference_frame.to_numpy(dtype=float),
                len(grid),
                axis=0,
            ),
            columns=list(feature_names),
        )
        scenario_frame.loc[:, feature] = grid
        predictions = selected.booster.predict(
            _dmatrix(scenario_frame, feature_names),
            iteration_range=(0, selected.best_iteration + 1),
        )
        for grid_index, (input_value, prediction) in enumerate(
            zip(grid, predictions, strict=True)
        ):
            rows.append(
                {
                    "candidate_id": selected.candidate_id,
                    "contract_input": contract_name,
                    "model_feature": feature,
                    "grid_index": grid_index,
                    "input_value": float(input_value),
                    "training_minimum": float(values.min()),
                    "training_maximum": float(values.max()),
                    "neutral_prediction": neutral_prediction,
                    "scenario_prediction": float(prediction),
                    "incremental_effect": (
                        float(prediction) - neutral_prediction
                    ),
                    "within_selected_training_range": bool(
                        values.min() <= input_value <= values.max()
                    ),
                }
            )
        differences = np.diff(predictions.astype(float))
        nonzero_signs = np.sign(differences[np.abs(differences) > 1e-12])
        sign_changes = (
            int(np.sum(nonzero_signs[1:] != nonzero_signs[:-1]))
            if len(nonzero_signs) > 1
            else 0
        )
        summary_rows.append(
            {
                "candidate_id": selected.candidate_id,
                "contract_input": contract_name,
                "model_feature": feature,
                "grid_point_count": len(grid),
                "training_minimum": float(values.min()),
                "training_maximum": float(values.max()),
                "minimum_prediction": float(np.min(predictions)),
                "maximum_prediction": float(np.max(predictions)),
                "prediction_span": float(
                    np.max(predictions) - np.min(predictions)
                ),
                "maximum_adjacent_jump": (
                    float(np.max(np.abs(differences)))
                    if len(differences)
                    else 0.0
                ),
                "direction_change_count": sign_changes,
                "monotonic_over_grid": sign_changes == 0,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summary_rows)


def build_xgboost_prototype_artifacts() -> dict[str, Any]:
    """Build all Step 11 S&P-only prototype outputs in memory."""
    data = load_development_and_validation()
    if data.matrix_summary.get("sampling_design") != "EVENT_AWARE":
        raise ValueError("Step 11 requires the frozen event-aware matrix.")
    results = [
        _fit_candidate(
            development=data.development,
            validation=data.validation,
            feature_names=data.feature_names,
            architecture=architecture,
            training_window=training_window,
        )
        for training_window in TRAINING_WINDOWS
        for architecture in ARCHITECTURES
    ]
    candidate_summary, candidate_metrics, learning_curves = (
        _candidate_tables(results)
    )
    selected_id = str(candidate_summary.iloc[0]["candidate_id"])
    selected = next(
        result for result in results if result.candidate_id == selected_id
    )
    repeat = _fit_candidate(
        development=data.development,
        validation=data.validation,
        feature_names=data.feature_names,
        architecture=selected.architecture,
        training_window=selected.training_window,
    )
    deterministic_max_difference = float(
        np.max(
            np.abs(
                repeat.validation_prediction
                - selected.validation_prediction
            )
        )
    )

    fixed = _fixed_baseline_predictions(data.validation)
    predictions = fixed[
        [
            "sample_date",
            "target_baseline_date",
            "target_end_date",
            "target",
            "unit",
            "actual",
            "naive_zero_prediction",
            "naive_training_mean_prediction",
            "naive_recent_trailing_21_observation_prediction",
            "ridge_prediction",
        ]
    ].copy()
    predictions["xgboost_prediction"] = selected.validation_prediction
    predictions["xgboost_residual"] = (
        predictions["actual"] - predictions["xgboost_prediction"]
    )
    predictions["selected_candidate_id"] = selected_id
    predictions["fit_role"] = "development_train"
    predictions["evaluation_role"] = "validation"

    comparison, comparison_by_year = _comparison_tables(predictions)
    feature_importance = _feature_importance(
        selected,
        data.feature_names,
    )
    residuals = _residuals_over_time(
        predictions,
        data.validation,
        selected_id,
    )
    vix_regimes = _vix_regime_metrics(residuals)
    crisis_periods = _crisis_period_metrics(selected, data.validation)
    sensitivity, sensitivity_summary = _sensitivity_tables(
        selected,
        data.feature_names,
    )

    fixed_metrics = pd.read_csv(RIDGE_METRICS_PATH)
    fixed_metrics = fixed_metrics.loc[fixed_metrics["target"].eq(TARGET)]
    baseline_names = {
        "naive_zero",
        "naive_training_mean",
        "naive_recent_trailing_21_observation",
        "ridge",
    }
    baseline_match = True
    for strategy in baseline_names:
        expected = fixed_metrics.loc[
            fixed_metrics["strategy"].eq(strategy)
        ].iloc[0]
        observed = comparison.loc[
            comparison["strategy"].eq(strategy)
        ].iloc[0]
        baseline_match &= bool(
            np.isclose(expected["mae"], observed["mae"], atol=1e-10)
            and np.isclose(expected["rmse"], observed["rmse"], atol=1e-10)
        )

    xgboost_metrics = comparison.loc[
        comparison["strategy"].eq("xgboost")
    ].iloc[0]
    ridge_metrics = comparison.loc[
        comparison["strategy"].eq("ridge")
    ].iloc[0]
    naive = comparison.loc[
        comparison["strategy"].str.startswith("naive_")
    ]
    best_naive_mae = float(naive["mae"].min())
    best_naive_rmse = float(naive["rmse"].min())
    selected_row = candidate_summary.loc[
        candidate_summary["candidate_id"].eq(selected_id)
    ].iloc[0]
    selected_overfit_flag = bool(
        selected_row["mae_validation_to_development_ratio"] > 1.5
        and selected_row["mae_generalization_gap"] > 1.0
    )
    validation_actual_abs_max = float(
        predictions["actual"].abs().max()
    )
    validation_prediction_abs_max = float(
        predictions["xgboost_prediction"].abs().max()
    )

    validations = {
        "event_aware_sampling_preserved": (
            data.matrix_summary.get("sampling_design") == "EVENT_AWARE"
        ),
        "approved_584_row_58_feature_matrix_used": (
            data.guard["matrix_rows_declared"] == 584
            and len(data.feature_names) == 58
        ),
        "development_fit_count_is_354": len(data.development) == 354,
        "validation_prediction_count_is_110": (
            len(data.validation) == 110 and len(predictions) == 110
        ),
        "six_boundary_purge_rows_not_used": (
            data.guard["boundary_purge_rows_parsed_but_not_used"] == 6
        ),
        "target_end_date_purge_remains_in_force": bool(
            (
                pd.to_datetime(
                    data.development_manifest["target_end_date"]
                )
                < pd.Timestamp(
                    data.validation_manifest["target_baseline_date"].min()
                )
            ).all()
        ),
        "only_sp500_target_trained": TARGET == "sp500",
        "candidate_count_is_six": len(results) == EXPECTED_CANDIDATE_COUNT,
        "full_and_recent_windows_evaluated": set(
            candidate_summary["training_window"]
        )
        == set(TRAINING_WINDOWS),
        "all_candidates_shallow_regularized_and_bounded": bool(
            (candidate_summary["max_depth"] <= 2).all()
            and (candidate_summary["subsample"] < 1).all()
            and (candidate_summary["colsample_bytree"] < 1).all()
            and (candidate_summary["reg_lambda"] > 0).all()
            and (candidate_summary["trees_used"] > 0).all()
            and (
                candidate_summary["trees_used"]
                <= candidate_summary["max_rounds"]
            ).all()
        ),
        "all_candidate_predictions_finite": bool(
            candidate_metrics[
                [
                    "mae",
                    "rmse",
                    "prediction_minimum",
                    "prediction_maximum",
                ]
            ]
            .map(np.isfinite)
            .all()
            .all()
        ),
        "selected_predictions_are_finite_and_bounded": bool(
            np.isfinite(predictions["xgboost_prediction"]).all()
            and validation_prediction_abs_max
            <= max(10.0, 2.0 * validation_actual_abs_max)
        ),
        "selected_training_is_deterministic": (
            deterministic_max_difference <= 1e-12
        ),
        "fixed_ridge_and_naive_metrics_reproduced": baseline_match,
        "feature_importance_covers_all_58_features": (
            len(feature_importance) == 58
            and set(feature_importance["feature"])
            == set(data.feature_names)
        ),
        "all_eight_scenario_inputs_have_sensitivity_checks": (
            set(sensitivity_summary["contract_input"])
            == set(SCENARIO_INPUT_FEATURES)
        ),
        "vix_regimes_cover_all_validation_rows": (
            int(vix_regimes["sample_count"].sum()) == 110
        ),
        "required_stress_periods_are_present": (
            len(crisis_periods) == 3
        ),
        "locked_final_test_target_rows_parsed_is_zero": (
            data.guard["locked_final_test_target_rows_parsed"] == 0
        ),
        "locked_final_test_unavailable": not data.guard[
            "final_test_target_values_available_to_step_10"
        ],
        "no_walk_forward_or_other_target_training": True,
        "no_production_model_artifact_exported": True,
    }
    if not all(validations.values()):
        failures = [name for name, passed in validations.items() if not passed]
        raise ValueError(f"Step 11 validation failed: {failures}")

    xgboost_beats_ridge_mae = bool(
        xgboost_metrics["mae"] < ridge_metrics["mae"]
    )
    xgboost_beats_ridge_rmse = bool(
        xgboost_metrics["rmse"] < ridge_metrics["rmse"]
    )
    summary = {
        "schema_version": "scenario_xgboost_prototype_v1",
        "construction_mode": "offline_approved_matrix_only",
        "target": TARGET,
        "target_unit": TARGET_UNIT,
        "sampling_decision": (
            "EVENT_AWARE is frozen; weekly sampling was not reopened."
        ),
        "matrix_path": relative_path(MATRIX_PATH),
        "matrix_guard": {
            **data.guard,
            "final_test_target_values_available_to_step_11": False,
        },
        "feature_count": len(data.feature_names),
        "development_rows": len(data.development),
        "validation_rows": len(data.validation),
        "candidate_count": len(results),
        "candidate_design": {
            "architectures": [asdict(item) for item in ARCHITECTURES],
            "training_windows": list(TRAINING_WINDOWS),
            "recent_window_years": RECENT_WINDOW_YEARS,
            "selection_primary_metric": "validation_mae",
            "selection_tiebreakers": [
                "validation_rmse",
                "lower_complexity_rank",
                "candidate_id",
            ],
            "early_stopping_metric": "validation_mae",
            "seed": SEED,
            "nthread": 1,
        },
        "selected_candidate": _json_safe(selected_row.to_dict()),
        "selected_overfit_flag": selected_overfit_flag,
        "deterministic_repeat_max_prediction_difference": (
            deterministic_max_difference
        ),
        "comparison": {
            "xgboost": _json_safe(xgboost_metrics.to_dict()),
            "ridge": _json_safe(ridge_metrics.to_dict()),
            "best_naive_mae": best_naive_mae,
            "best_naive_rmse": best_naive_rmse,
            "xgboost_beats_ridge_mae": xgboost_beats_ridge_mae,
            "xgboost_beats_ridge_rmse": xgboost_beats_ridge_rmse,
            "xgboost_beats_best_naive_mae": bool(
                xgboost_metrics["mae"] < best_naive_mae
            ),
            "xgboost_beats_best_naive_rmse": bool(
                xgboost_metrics["rmse"] < best_naive_rmse
            ),
        },
        "diagnostic_definitions": {
            "overfit_flag": (
                "validation/development MAE ratio > 1.5 and absolute MAE "
                "gap > 1 percentage point"
            ),
            "vix_regimes": [
                "low: VIX < 20",
                "medium: 20 <= VIX < 30",
                "high: VIX >= 30",
            ],
            "sensitivity_reference": (
                "Selected-window median state with all eight scenario-shock "
                "features set to neutral zero; one shock varied at a time "
                "within its selected-training min/max."
            ),
        },
        "top_10_features_by_total_gain": _json_safe(
            feature_importance.head(10).to_dict("records")
        ),
        "scenario_sensitivity_summary": _json_safe(
            sensitivity_summary.to_dict("records")
        ),
        "software_versions": {
            "python": (
                f"{sys.version_info.major}.{sys.version_info.minor}."
                f"{sys.version_info.micro}"
            ),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "xgboost": xgb.__version__,
        },
        "validations": validations,
        "limitations": [
            (
                "Validation is used both for early stopping and candidate "
                "selection; Step 12 walk-forward evidence is still required."
            ),
            (
                "Event-aware 21-session labels overlap within each split; "
                "the frozen boundary purge is preserved, while repeated fold "
                "purges belong to Step 12."
            ),
            (
                "COVID stress metrics are in-sample because the frozen "
                "validation period begins in May 2021."
            ),
            (
                "Sensitivity curves vary one feature at a time and do not "
                "prove a causal economic relationship."
            ),
            (
                "The PMI regional proxy and continuous CL=F limitations "
                "remain inherited from the approved matrix."
            ),
        ],
        "prototype_decision": (
            "Carry both Ridge and the selected shallow XGBoost candidate into "
            "Step 12 walk-forward validation. Do not promote either model "
            "from this single validation period."
        ),
        "training_gate": (
            "STEP_11_XGBOOST_PROTOTYPE_COMPLETE_READY_FOR_STEP_12_"
            "WALK_FORWARD_VALIDATION"
        ),
        "scope_boundary": (
            "Step 11 only: no Step 12, walk-forward validation, other target "
            "training, final-test target access, production export, Supabase, "
            "backend, frontend, or server work."
        ),
    }
    return {
        "candidate_summary": candidate_summary,
        "candidate_metrics": candidate_metrics,
        "learning_curves": learning_curves,
        "validation_predictions": predictions,
        "comparison_metrics": comparison,
        "comparison_metrics_by_year": comparison_by_year,
        "feature_importance": feature_importance,
        "residuals_over_time": residuals,
        "vix_regime_metrics": vix_regimes,
        "crisis_period_metrics": crisis_periods,
        "scenario_sensitivity": sensitivity,
        "scenario_sensitivity_summary": sensitivity_summary,
        "summary": summary,
    }


def persist_xgboost_prototype_artifacts(
    artifacts: dict[str, Any],
    *,
    output_dir: Path = XGBOOST_OUTPUT_ROOT,
) -> dict[str, str]:
    paths = {
        "candidate_summary": output_dir / "sp500_candidate_summary.csv",
        "candidate_metrics": output_dir / "sp500_candidate_metrics.csv",
        "learning_curves": output_dir / "sp500_learning_curves.csv",
        "validation_predictions": (
            output_dir / "sp500_validation_predictions.csv"
        ),
        "comparison_metrics": (
            output_dir / "sp500_comparison_metrics.csv"
        ),
        "comparison_metrics_by_year": (
            output_dir / "sp500_comparison_metrics_by_year.csv"
        ),
        "feature_importance": (
            output_dir / "sp500_feature_importance.csv"
        ),
        "residuals_over_time": (
            output_dir / "sp500_residuals_over_time.csv"
        ),
        "vix_regime_metrics": (
            output_dir / "sp500_vix_regime_metrics.csv"
        ),
        "crisis_period_metrics": (
            output_dir / "sp500_crisis_period_metrics.csv"
        ),
        "scenario_sensitivity": (
            output_dir / "sp500_scenario_sensitivity.csv"
        ),
        "scenario_sensitivity_summary": (
            output_dir / "sp500_scenario_sensitivity_summary.csv"
        ),
        "summary": output_dir / "sp500_xgboost_prototype_summary.json",
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
            "Train and diagnose the Step 11 S&P 500 shallow XGBoost "
            "prototype without parsing locked final-test target rows."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build all outputs in memory without writing local artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=XGBOOST_OUTPUT_ROOT,
        help=f"Output directory (default: {XGBOOST_OUTPUT_ROOT}).",
    )
    args = parser.parse_args(argv)
    artifacts = build_xgboost_prototype_artifacts()
    paths = (
        {}
        if args.dry_run
        else persist_xgboost_prototype_artifacts(
            artifacts,
            output_dir=args.output_dir,
        )
    )
    summary = artifacts["summary"]
    selected = summary["selected_candidate"]
    print(
        json.dumps(
            {
                "target": TARGET,
                "candidate_count": summary["candidate_count"],
                "selected_candidate_id": selected["candidate_id"],
                "selected_training_rows": selected["training_rows"],
                "trees_used": selected["trees_used"],
                "validation_mae": selected["validation_mae"],
                "validation_rmse": selected["validation_rmse"],
                "ridge_validation_mae": summary["comparison"]["ridge"]["mae"],
                "locked_final_test_target_rows_parsed": summary[
                    "matrix_guard"
                ]["locked_final_test_target_rows_parsed"],
                "outputs": paths,
                "training_gate": summary["training_gate"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
