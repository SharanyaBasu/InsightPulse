from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scenario_modeling.src.evaluation import regression_metrics
from scenario_modeling.src.feature_sources import frame_to_csv_text
from scenario_modeling.src.ridge_baselines import (
    NAIVE_STRATEGIES,
    PREDICTION_STRATEGIES,
    RIDGE_ALPHAS,
    _read_unlocked_matrix_prefix,
    build_ridge_artifacts,
    load_development_and_validation,
    make_ridge_pipeline,
)


class EvaluationUnitTests(unittest.TestCase):
    def test_regression_metrics_match_manual_values(self) -> None:
        actual = np.array([1.0, -2.0, 3.0])
        predicted = np.array([0.0, -1.0, 2.0])
        metrics = regression_metrics(actual, predicted)
        self.assertAlmostEqual(metrics["mae"], 1.0)
        self.assertAlmostEqual(metrics["rmse"], 1.0)
        self.assertAlmostEqual(metrics["directional_accuracy"], 2.0 / 3.0)
        self.assertEqual(metrics["sample_count"], 3)

    def test_constant_prediction_correlation_is_undefined(self) -> None:
        metrics = regression_metrics([1.0, -1.0], [0.0, 0.0])
        self.assertTrue(np.isnan(metrics["prediction_actual_correlation"]))
        self.assertEqual(metrics["directional_accuracy"], 0.0)

    def test_ridge_pipeline_contains_all_preprocessing(self) -> None:
        pipeline = make_ridge_pipeline(10.0)
        self.assertEqual(
            tuple(pipeline.named_steps),
            ("imputer", "scaler", "ridge"),
        )
        self.assertEqual(
            pipeline.named_steps["imputer"].strategy, "median"
        )
        self.assertTrue(pipeline.named_steps["scaler"].with_mean)
        self.assertEqual(pipeline.named_steps["ridge"].alpha, 10.0)

    def test_prefix_reader_does_not_parse_locked_sentinel_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.csv"
            path.write_text(
                "sample_date,feature,target\n"
                "2020-01-01,1.0,2.0\n"
                "2020-01-02,3.0,4.0\n"
                "LOCKED,not-a-number,not-a-number\n",
                encoding="utf-8",
            )
            frame, columns = _read_unlocked_matrix_prefix(
                path,
                row_limit=2,
                numeric_columns=["feature", "target"],
            )
        self.assertEqual(frame.shape, (2, 3))
        self.assertEqual(columns, ("sample_date", "feature", "target"))
        self.assertNotIn("LOCKED", set(frame["sample_date"]))


class RidgeBaselineAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = load_development_and_validation()
        cls.artifacts = build_ridge_artifacts()

    def test_frozen_matrix_split_and_lock_guard_are_preserved(self) -> None:
        self.assertEqual(len(self.data.feature_names), 58)
        self.assertEqual(len(self.data.target_names), 12)
        self.assertEqual(len(self.data.development), 354)
        self.assertEqual(len(self.data.validation), 110)
        self.assertEqual(self.data.guard["matrix_rows_parsed"], 470)
        self.assertEqual(
            self.data.guard["locked_final_test_rows_declared"], 114
        )
        self.assertEqual(
            self.data.guard["locked_final_test_target_rows_parsed"], 0
        )
        self.assertFalse(
            self.data.guard[
                "final_test_target_values_available_to_step_10"
            ]
        )

    def test_all_targets_baselines_and_predictions_are_present(self) -> None:
        predictions = self.artifacts["predictions"]
        metrics = self.artifacts["metrics"]
        self.assertEqual(len(predictions), 110 * 12)
        self.assertEqual(len(metrics), 12 * 4)
        self.assertEqual(set(metrics["strategy"]), set(PREDICTION_STRATEGIES))
        self.assertEqual(
            set(metrics[metrics["strategy"] != "ridge"]["strategy"]),
            set(NAIVE_STRATEGIES),
        )
        self.assertTrue(
            predictions["evaluation_role"].eq("validation").all()
        )
        self.assertTrue(
            predictions[
                [
                    "actual",
                    "naive_zero_prediction",
                    "naive_training_mean_prediction",
                    "naive_recent_trailing_21_observation_prediction",
                    "ridge_prediction",
                ]
            ]
            .map(np.isfinite)
            .all()
            .all()
        )

    def test_alpha_search_and_selected_configurations_are_complete(self) -> None:
        search = self.artifacts["alpha_search"]
        selected = self.artifacts["selected_configurations"]
        self.assertEqual(len(search), 12 * len(RIDGE_ALPHAS))
        self.assertEqual(len(selected), 12)
        self.assertTrue(set(selected["selected_alpha"]) <= set(RIDGE_ALPHAS))
        self.assertTrue((selected["development_fit_rows"] == 354).all())
        self.assertTrue((selected["validation_selection_rows"] == 110).all())
        self.assertTrue((selected["feature_count"] == 58).all())

    def test_preprocessing_statistics_are_development_only(self) -> None:
        audit = self.artifacts["preprocessing_audit"]
        self.assertEqual(len(audit), 12 * 58)
        self.assertTrue(
            audit["imputer_matches_development_median"].all()
        )
        self.assertTrue(
            audit["scaler_mean_matches_development_mean"].all()
        )
        self.assertTrue(
            audit["validation_excluded_from_preprocessing_fit"].all()
        )
        self.assertTrue(
            audit[
                "locked_final_test_excluded_from_preprocessing_fit"
            ].all()
        )
        self.assertTrue((audit["fit_row_count"] == 354).all())

    def test_recent_trailing_baseline_is_target_specific_and_point_in_time(
        self,
    ) -> None:
        audit = self.artifacts["trailing_baseline_audit"]
        self.assertEqual(len(audit), 110 * 12)
        self.assertTrue(audit["point_in_time_pass"].all())
        self.assertTrue(
            (
                pd.to_datetime(audit["prior_source_date"])
                < pd.to_datetime(audit["current_source_date"])
            ).all()
        )
        self.assertTrue(
            (
                pd.to_datetime(audit["current_source_date"])
                <= pd.to_datetime(audit["target_baseline_date"])
            ).all()
        )
        self.assertTrue(
            (audit["trailing_source_observations"] == 21).all()
        )

    def test_all_required_metrics_and_year_periods_exist(self) -> None:
        metrics = self.artifacts["metrics"]
        by_year = self.artifacts["metrics_by_year"]
        required = {
            "mae",
            "rmse",
            "oos_r_squared",
            "prediction_actual_correlation",
            "directional_accuracy",
            "prediction_mean",
            "prediction_standard_deviation",
            "prediction_minimum",
            "prediction_maximum",
        }
        self.assertTrue(required <= set(metrics.columns))
        self.assertEqual(len(by_year), 12 * 4 * 3)
        self.assertEqual(set(by_year["calendar_year"]), {2021, 2022, 2023})

    def test_coefficients_and_naive_comparisons_are_complete(self) -> None:
        coefficients = self.artifacts["coefficients"]
        comparison = self.artifacts["comparison"]
        self.assertEqual(len(coefficients), 12 * 58)
        self.assertTrue(
            coefficients["standardized_coefficient"].map(np.isfinite).all()
        )
        self.assertEqual(len(comparison), 12)
        self.assertTrue(
            set(comparison["best_naive_by_mae"]) <= set(NAIVE_STRATEGIES)
        )
        self.assertTrue(
            set(comparison["best_naive_by_rmse"]) <= set(NAIVE_STRATEGIES)
        )

    def test_all_step_10_acceptance_validations_pass(self) -> None:
        validations = self.artifacts["summary"]["validations"]
        self.assertTrue(all(validations.values()))
        self.assertEqual(
            self.artifacts["summary"]["training_gate"],
            "STEP_10_RIDGE_COMPLETE_READY_FOR_STEP_11_XGBOOST_PROTOTYPE",
        )

    def test_output_serialization_is_repeatable(self) -> None:
        predictions = self.artifacts["predictions"]
        self.assertEqual(
            frame_to_csv_text(predictions),
            frame_to_csv_text(predictions.copy()),
        )


if __name__ == "__main__":
    unittest.main()
