from __future__ import annotations

import unittest

import numpy as np

from scenario_modeling.src.feature_sources import frame_to_csv_text
from scenario_modeling.src.xgboost_prototype import (
    ARCHITECTURES,
    EXPECTED_CANDIDATE_COUNT,
    SCENARIO_INPUT_FEATURES,
    TRAINING_WINDOWS,
    build_xgboost_prototype_artifacts,
)


class XGBoostPrototypeAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifacts = build_xgboost_prototype_artifacts()
        cls.summary = cls.artifacts["summary"]

    def test_candidate_set_is_small_shallow_and_regularized(self) -> None:
        candidates = self.artifacts["candidate_summary"]
        self.assertEqual(len(candidates), EXPECTED_CANDIDATE_COUNT)
        self.assertEqual(
            set(candidates["training_window"]),
            set(TRAINING_WINDOWS),
        )
        self.assertEqual(
            set(candidates["architecture"]),
            {item.name for item in ARCHITECTURES},
        )
        self.assertTrue((candidates["max_depth"] <= 2).all())
        self.assertTrue((candidates["subsample"] < 1).all())
        self.assertTrue((candidates["colsample_bytree"] < 1).all())
        self.assertTrue((candidates["reg_lambda"] > 0).all())
        self.assertTrue(candidates["stopped_early"].all())

    def test_frozen_matrix_split_purge_and_lock_are_preserved(self) -> None:
        guard = self.summary["matrix_guard"]
        self.assertEqual(guard["matrix_rows_declared"], 584)
        self.assertEqual(guard["matrix_rows_parsed"], 470)
        self.assertEqual(self.summary["feature_count"], 58)
        self.assertEqual(self.summary["development_rows"], 354)
        self.assertEqual(self.summary["validation_rows"], 110)
        self.assertEqual(
            guard["boundary_purge_rows_parsed_but_not_used"], 6
        )
        self.assertEqual(guard["locked_final_test_rows_declared"], 114)
        self.assertEqual(
            guard["locked_final_test_target_rows_parsed"], 0
        )
        self.assertFalse(
            guard["final_test_target_values_available_to_step_11"]
        )
        self.assertIn("EVENT_AWARE", self.summary["sampling_decision"])

    def test_only_sp500_validation_predictions_are_produced(self) -> None:
        predictions = self.artifacts["validation_predictions"]
        self.assertEqual(len(predictions), 110)
        self.assertEqual(set(predictions["target"]), {"sp500"})
        self.assertTrue(predictions["fit_role"].eq("development_train").all())
        self.assertTrue(predictions["evaluation_role"].eq("validation").all())
        self.assertTrue(
            predictions[
                [
                    "actual",
                    "ridge_prediction",
                    "xgboost_prediction",
                    "xgboost_residual",
                ]
            ]
            .map(np.isfinite)
            .all()
            .all()
        )

    def test_selected_candidate_and_metrics_are_reproducible(self) -> None:
        selected = self.summary["selected_candidate"]
        self.assertEqual(
            selected["candidate_id"],
            "recent_5y__regularized_stumps",
        )
        self.assertEqual(selected["training_rows"], 234)
        self.assertEqual(selected["max_depth"], 1)
        self.assertEqual(selected["trees_used"], 82)
        self.assertAlmostEqual(
            selected["validation_mae"],
            3.5836675657126333,
            places=10,
        )
        self.assertEqual(
            self.summary[
                "deterministic_repeat_max_prediction_difference"
            ],
            0.0,
        )

    def test_xgboost_is_compared_with_fixed_step_10_baselines(self) -> None:
        metrics = self.artifacts["comparison_metrics"]
        self.assertEqual(
            set(metrics["strategy"]),
            {
                "naive_zero",
                "naive_training_mean",
                "naive_recent_trailing_21_observation",
                "ridge",
                "xgboost",
            },
        )
        self.assertEqual(len(metrics), 5)
        comparison = self.summary["comparison"]
        self.assertTrue(comparison["xgboost_beats_ridge_mae"])
        self.assertTrue(comparison["xgboost_beats_ridge_rmse"])
        self.assertTrue(comparison["xgboost_beats_best_naive_mae"])
        self.assertTrue(comparison["xgboost_beats_best_naive_rmse"])

    def test_required_diagnostic_outputs_are_complete(self) -> None:
        importance = self.artifacts["feature_importance"]
        residuals = self.artifacts["residuals_over_time"]
        regimes = self.artifacts["vix_regime_metrics"]
        crises = self.artifacts["crisis_period_metrics"]
        by_year = self.artifacts["comparison_metrics_by_year"]
        self.assertEqual(len(importance), 58)
        self.assertEqual(len(residuals), 110)
        self.assertEqual(int(regimes["sample_count"].sum()), 110)
        self.assertEqual(len(regimes), 3)
        self.assertEqual(len(crises), 3)
        self.assertEqual(set(by_year["calendar_year"]), {2021, 2022, 2023})
        self.assertEqual(set(by_year["strategy"]), set(
            self.artifacts["comparison_metrics"]["strategy"]
        ))

    def test_all_eight_scenario_inputs_have_sensitivity_results(self) -> None:
        detail = self.artifacts["scenario_sensitivity"]
        summary = self.artifacts["scenario_sensitivity_summary"]
        self.assertEqual(
            set(summary["contract_input"]),
            set(SCENARIO_INPUT_FEATURES),
        )
        self.assertEqual(
            set(detail["contract_input"]),
            set(SCENARIO_INPUT_FEATURES),
        )
        self.assertTrue(detail["within_selected_training_range"].all())
        self.assertTrue(
            detail[
                [
                    "neutral_prediction",
                    "scenario_prediction",
                    "incremental_effect",
                ]
            ]
            .map(np.isfinite)
            .all()
            .all()
        )

    def test_all_step_11_validation_guards_pass(self) -> None:
        self.assertTrue(all(self.summary["validations"].values()))
        self.assertEqual(
            self.summary["training_gate"],
            (
                "STEP_11_XGBOOST_PROTOTYPE_COMPLETE_READY_FOR_STEP_12_"
                "WALK_FORWARD_VALIDATION"
            ),
        )

    def test_output_serialization_is_repeatable(self) -> None:
        predictions = self.artifacts["validation_predictions"]
        self.assertEqual(
            frame_to_csv_text(predictions),
            frame_to_csv_text(predictions.copy()),
        )


if __name__ == "__main__":
    unittest.main()
