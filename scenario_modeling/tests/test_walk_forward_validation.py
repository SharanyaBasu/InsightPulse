from __future__ import annotations

import unittest

import numpy as np

from scenario_modeling.src.feature_sources import frame_to_csv_text
from scenario_modeling.src.walk_forward_validation import (
    FOLD_COUNT,
    MODEL_VARIANTS,
    NAIVE_VARIANTS,
    VALIDATION_ROWS_PER_FOLD,
    build_walk_forward_artifacts,
)
from scenario_modeling.src.xgboost_prototype import (
    SCENARIO_INPUT_FEATURES,
)


class WalkForwardValidationAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifacts = build_walk_forward_artifacts()
        cls.summary = cls.artifacts["summary"]

    def test_five_expanding_folds_have_target_end_date_purges(self) -> None:
        folds = self.artifacts["fold_manifest"]
        self.assertEqual(len(folds), FOLD_COUNT)
        self.assertEqual(
            folds["training_rows"].tolist(),
            [160, 222, 280, 342, 401],
        )
        self.assertEqual(
            folds["outer_purge_rows"].tolist(),
            [4, 2, 4, 2, 3],
        )
        self.assertTrue(
            folds["validation_rows"].eq(
                VALIDATION_ROWS_PER_FOLD
            ).all()
        )
        self.assertTrue(
            folds["purge_rule"]
            .str.contains("target_end_date")
            .all()
        )

    def test_all_history_and_model_variants_use_fold_local_fits(self) -> None:
        audit = self.artifacts["fit_audit"]
        self.assertEqual(len(audit), FOLD_COUNT * len(MODEL_VARIANTS))
        self.assertEqual(set(audit["variant"]), set(MODEL_VARIANTS))
        self.assertTrue((audit["inner_purge_rows"] > 0).all())
        self.assertTrue((audit["inner_training_rows"] >= 75).all())
        self.assertTrue((audit["inner_validation_rows"] >= 20).all())
        self.assertTrue(
            audit[
                [
                    "inner_validation_mae",
                    "inner_validation_rmse",
                    "training_mae",
                    "training_rmse",
                ]
            ]
            .map(np.isfinite)
            .all()
            .all()
        )

    def test_each_variant_has_three_hundred_out_of_sample_rows(self) -> None:
        predictions = self.artifacts["fold_predictions"]
        expected = set(MODEL_VARIANTS) | set(NAIVE_VARIANTS)
        self.assertEqual(set(predictions["variant"]), expected)
        self.assertTrue(
            predictions.groupby("variant")
            .size()
            .eq(FOLD_COUNT * VALIDATION_ROWS_PER_FOLD)
            .all()
        )
        self.assertTrue(
            predictions[["actual", "prediction", "residual"]]
            .map(np.isfinite)
            .all()
            .all()
        )

    def test_repeated_results_select_full_history_ridge(self) -> None:
        self.assertEqual(self.summary["approved_model_family"], "ridge")
        self.assertEqual(
            self.summary["approved_variant"],
            "ridge_full_history",
        )
        self.assertEqual(
            self.summary["approved_history_window_strategy"],
            "full_history",
        )
        self.assertEqual(
            self.summary["models_proceeding_to_step_13"],
            ["ridge"],
        )
        ridge = self.summary["approved_ridge_variant"]
        xgboost = self.summary["evaluated_xgboost_variant"]
        self.assertAlmostEqual(ridge["mae"], 3.575126099770335, places=10)
        self.assertAlmostEqual(
            xgboost["mae"],
            3.6315224358034794,
            places=10,
        )
        self.assertLess(ridge["worst_fold_mae"], xgboost["worst_fold_mae"])
        self.assertLess(ridge["fold_mae_std"], xgboost["fold_mae_std"])

    def test_recent_history_is_only_a_small_mae_win(self) -> None:
        recent = self.summary["lowest_mae_ridge_variant"]
        approved = self.summary["approved_ridge_variant"]
        self.assertEqual(recent["variant"], "ridge_recent_5y")
        relative_difference = (
            approved["mae"] / recent["mae"] - 1.0
        )
        self.assertGreater(relative_difference, 0.0)
        self.assertLess(relative_difference, 0.01)
        self.assertGreater(
            recent["worst_fold_mae"],
            approved["worst_fold_mae"],
        )
        self.assertGreater(
            recent["fold_mae_std"],
            approved["fold_mae_std"],
        )

    def test_scenario_sensitivity_checks_all_eight_inputs(self) -> None:
        sensitivity = self.artifacts["sensitivity_summary"]
        self.assertEqual(
            set(sensitivity["contract_input"]),
            set(SCENARIO_INPUT_FEATURES),
        )
        self.assertEqual(
            len(sensitivity),
            len(MODEL_VARIANTS) * len(SCENARIO_INPUT_FEATURES),
        )
        ridge = sensitivity.loc[
            sensitivity["variant"].eq("ridge_full_history")
        ]
        xgboost = sensitivity.loc[
            sensitivity["variant"].eq("xgboost_full_history")
        ]
        self.assertEqual(
            set(
                ridge.loc[
                    ridge["folds_with_meaningful_response"] > 0,
                    "contract_input",
                ]
            ),
            set(SCENARIO_INPUT_FEATURES),
        )
        self.assertEqual(
            set(
                xgboost.loc[
                    xgboost["folds_with_meaningful_response"] > 0,
                    "contract_input",
                ]
            ),
            {"vix_change_points"},
        )
        self.assertTrue(
            self.artifacts["sensitivity"]["finite_and_sensible"].all()
        )

    def test_locked_final_test_is_not_parsed(self) -> None:
        guard = self.summary["matrix_guard"]
        self.assertEqual(self.summary["matrix_rows_declared"], 584)
        self.assertEqual(self.summary["pretest_rows_evaluated"], 464)
        self.assertEqual(self.summary["feature_count"], 58)
        self.assertEqual(guard["locked_final_test_rows_declared"], 114)
        self.assertEqual(guard["locked_final_test_target_rows_parsed"], 0)
        self.assertFalse(
            guard["final_test_target_values_available_to_step_12"]
        )

    def test_repeated_build_is_deterministic(self) -> None:
        repeated = build_walk_forward_artifacts()
        first = self.artifacts["fold_predictions"]
        second = repeated["fold_predictions"]
        self.assertEqual(
            frame_to_csv_text(first),
            frame_to_csv_text(second),
        )

    def test_all_acceptance_guards_pass(self) -> None:
        self.assertTrue(all(self.summary["validations"].values()))
        self.assertFalse(
            self.summary["step_11_xgboost_promotion_confirmed"]
        )
        self.assertTrue(
            self.summary[
                "xgboost_continues_to_ignore_most_scenario_inputs"
            ]
        )
        self.assertTrue(
            self.summary["scenario_inference_readiness"].startswith(
                "NOT_APPROVED_FOR_PRODUCTION_INFERENCE"
            )
        )
        self.assertEqual(
            self.summary["training_gate"],
            (
                "STEP_12_WALK_FORWARD_COMPLETE_READY_FOR_"
                "STEP_13_ALL_TARGETS"
            ),
        )


if __name__ == "__main__":
    unittest.main()
