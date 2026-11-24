"""Deterministic regression metrics for Scenario Playground model stages."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd


def regression_metrics(
    actual: Sequence[float] | pd.Series | np.ndarray,
    predicted: Sequence[float] | pd.Series | np.ndarray,
) -> dict[str, Any]:
    """Calculate the frozen Step 10 out-of-sample metric set.

    Directional accuracy is exact sign agreement. A zero prediction is a
    neutral direction and is correct only when the realized target is zero.
    Correlation is undefined for a constant prediction and is returned as NaN.
    """
    y_true = np.asarray(actual, dtype=float).reshape(-1)
    y_pred = np.asarray(predicted, dtype=float).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Actual and predicted shapes differ: {y_true.shape} vs {y_pred.shape}"
        )
    if len(y_true) == 0:
        raise ValueError("Cannot evaluate an empty prediction set.")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Regression metrics require finite values.")

    residual = y_true - y_pred
    absolute_error = np.abs(residual)
    squared_error = residual**2
    mae = float(absolute_error.mean())
    rmse = float(np.sqrt(squared_error.mean()))
    centered = y_true - y_true.mean()
    denominator = float(np.dot(centered, centered))
    r_squared = (
        float(1.0 - np.dot(residual, residual) / denominator)
        if denominator > 0
        else math.nan
    )
    if np.std(y_true, ddof=0) > 0 and np.std(y_pred, ddof=0) > 0:
        correlation = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        correlation = math.nan
    directional_accuracy = float(
        (np.sign(y_true) == np.sign(y_pred)).mean()
    )
    quantiles = np.quantile(
        y_pred, [0.05, 0.25, 0.50, 0.75, 0.95]
    )
    return {
        "sample_count": int(len(y_true)),
        "mae": mae,
        "rmse": rmse,
        "oos_r_squared": r_squared,
        "prediction_actual_correlation": correlation,
        "directional_accuracy": directional_accuracy,
        "actual_mean": float(y_true.mean()),
        "actual_standard_deviation": float(y_true.std(ddof=1)),
        "prediction_mean": float(y_pred.mean()),
        "prediction_standard_deviation": float(y_pred.std(ddof=1)),
        "prediction_minimum": float(y_pred.min()),
        "prediction_p05": float(quantiles[0]),
        "prediction_p25": float(quantiles[1]),
        "prediction_median": float(quantiles[2]),
        "prediction_p75": float(quantiles[3]),
        "prediction_p95": float(quantiles[4]),
        "prediction_maximum": float(y_pred.max()),
        "prediction_negative_count": int((y_pred < 0).sum()),
        "prediction_zero_count": int((np.abs(y_pred) <= 1e-12).sum()),
        "prediction_positive_count": int((y_pred > 0).sum()),
    }


def metrics_by_calendar_year(
    predictions: pd.DataFrame,
    *,
    target_column: str = "target",
    strategy_column: str = "strategy",
    date_column: str = "sample_date",
    actual_column: str = "actual",
    prediction_column: str = "prediction",
) -> pd.DataFrame:
    """Calculate the same metrics for each target, strategy, and year."""
    required = {
        target_column,
        strategy_column,
        date_column,
        actual_column,
        prediction_column,
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction table is missing columns: {missing}")
    frame = predictions.copy()
    frame["calendar_year"] = pd.to_datetime(frame[date_column]).dt.year
    rows: list[dict[str, Any]] = []
    for (target, strategy, year), group in frame.groupby(
        [target_column, strategy_column, "calendar_year"],
        sort=True,
    ):
        rows.append(
            {
                "target": str(target),
                "strategy": str(strategy),
                "calendar_year": int(year),
                **regression_metrics(
                    group[actual_column], group[prediction_column]
                ),
            }
        )
    return pd.DataFrame(rows)
