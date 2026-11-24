from __future__ import annotations

import unittest

import numpy as np

from scenario_modeling.src.all_target_models import (
    ALL_STRATEGIES,
    CLASSIFICATIONS,
    FOLD_COUNT,
    RIDGE_ALPHAS,
    RIDGE_STRATEGY,
    SCENARIO_INPUTS,
    TARGET_NAMES,
    VALIDATION_ROWS_PER_FOLD,
    build_all_target_artifacts,
)
from scenario_modeling.src.feature_sources import frame_to_csv_text


class AllTargetModelsAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifacts = build_all_target_artifacts()
        cls.summary = cls.artifacts["summary"]

    def test_frozen_walk_forward_folds_and_purges_are_reused(self) -> None:
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

    def test_all_targets_use_fold_local_full_history_ridge(self) -> None:
        audit = self.artifacts["fit_audit"]
        self.assertEqual(len(audit), len(TARGET_NAMES) * FOLD_COUNT)
        self.assertEqual(set(audit["target"]), set(TARGET_NAMES))
        self.assertTrue(audit["model_family"].eq("ridge").all())
        self.assertTrue(
            audit["history_strategy"].eq("full_history").all()
        )
        self.assertTrue((audit["inner_purge_rows"] > 0).all())
        self.assertTrue(
            audit["locked_final_test_excluded"].all()
        )
        self.assertTrue(
            set(audit["selected_alpha"]) <= set(RIDGE_ALPHAS)
        )

    def test_each_target_and_strategy_has_three_hundred_oos_rows(
        self,
    ) -> None:
        predictions = self.artifacts["fold_predictions"]
        self.assertEqual(set(predictions["strategy"]), set(ALL_STRATEGIES))
        self.assertEqual(set(predictions["target"]), set(TARGET_NAMES))
        self.assertTrue(
            predictions.groupby(["target", "strategy"])
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

    def test_naive_comparisons_and_ridge_metrics_are_complete(self) -> None:
        comparison = self.artifacts["naive_comparison"]
        aggregate = self.artifacts["aggregate_metrics"]
        self.assertEqual(len(comparison), len(TARGET_NAMES))
        self.assertEqual(
            len(aggregate),
            len(TARGET_NAMES) * len(ALL_STRATEGIES),
        )
        ridge = aggregate.loc[
            aggregate["strategy"].eq(RIDGE_STRATEGY)
        ]
        self.assertEqual(len(ridge), len(TARGET_NAMES))
        self.assertTrue(
            ridge[
                [
                    "mae",
                    "rmse",
                    "oos_r_squared",
                    "directional_accuracy",
                ]
            ]
            .map(np.isfinite)
            .all()
            .all()
        )

    def test_coefficients_and_all_eight_shocks_are_audited(self) -> None:
        coefficients = self.artifacts["fold_coefficients"]
        stability = self.artifacts["coefficient_stability"]
        sensitivity = self.artifacts["sensitivity"]
        sensitivity_summary = self.artifacts["sensitivity_summary"]
        self.assertEqual(
            len(coefficients),
            len(TARGET_NAMES) * FOLD_COUNT * 58,
        )
        self.assertEqual(len(stability), len(TARGET_NAMES) * 58)
        self.assertEqual(
            len(sensitivity),
            len(TARGET_NAMES) * FOLD_COUNT * 3 * len(SCENARIO_INPUTS),
        )
        self.assertEqual(
            len(sensitivity_summary),
            len(TARGET_NAMES) * len(SCENARIO_INPUTS),
        )
        self.assertEqual(
            set(sensitivity_summary["contract_input"]),
            set(SCENARIO_INPUTS),
        )
        self.assertTrue(sensitivity["finite_and_sensible"].all())

    def test_target_classifications_and_fallbacks_are_evidence_based(
        self,
    ) -> None:
        assessment = self.artifacts["target_assessment"].set_index(
            "target"
        )
        expected_low_confidence = {
            "sp500",
            "energy",
            "consumer_discretionary",
        }
        expected_fallback = set(TARGET_NAMES) - expected_low_confidence
        self.assertEqual(
            set(
                assessment.index[
                    assessment["classification"].eq(
                        "APPROVED_WITH_LOW_CONFIDENCE"
                    )
                ]
            ),
            expected_low_confidence,
        )
        self.assertEqual(
            set(
                assessment.index[
                    assessment["classification"].eq(
                        "NAIVE_FALLBACK_REQUIRED"
                    )
                ]
            ),
            expected_fallback,
        )
        self.assertTrue(
            assessment["classification"].isin(CLASSIFICATIONS).all()
        )
        self.assertFalse(assessment["prediction_pathological"].any())
        self.assertTrue(
            assessment.loc[
                list(expected_fallback),
                "fallback_strategy",
            ].ne("").all()
        )

    def test_selected_pretest_configurations_export_no_models(self) -> None:
        configurations = self.artifacts["final_configurations"]
        self.assertEqual(len(configurations), len(TARGET_NAMES))
        self.assertTrue(
            set(configurations["selected_alpha"]) <= set(RIDGE_ALPHAS)
        )
        self.assertTrue(configurations["training_rows"].eq(464).all())
        self.assertFalse(configurations["model_binary_exported"].any())
        self.assertFalse(configurations["locked_final_test_used"].any())

    def test_locked_final_test_is_never_parsed_or_evaluated(self) -> None:
        guard = self.summary["matrix_guard"]
        self.assertEqual(self.summary["matrix_rows_declared"], 584)
        self.assertEqual(self.summary["pretest_rows_evaluated"], 464)
        self.assertEqual(self.summary["feature_count"], 58)
        self.assertEqual(guard["locked_final_test_rows_declared"], 114)
        self.assertEqual(guard["locked_final_test_target_rows_parsed"], 0)
        self.assertEqual(guard["final_test_access_count"], 0)
        self.assertFalse(
            guard["final_test_target_values_available_to_step_13"]
        )
        self.assertTrue(
            self.artifacts["target_assessment"][
                "test_metrics_status"
            ]
            .eq("NOT_COMPUTED_LOCKED_FINAL_TEST_NOT_AUTHORIZED")
            .all()
        )

    def test_expected_reference_metrics_are_reproducible(self) -> None:
        assessment = self.artifacts["target_assessment"].set_index(
            "target"
        )
        self.assertAlmostEqual(
            assessment.loc["sp500", "ridge_mae"],
            3.575126099770335,
            places=6,
        )
        self.assertAlmostEqual(
            assessment.loc["oil", "ridge_mae"],
            14.94125029344819,
            places=6,
        )
        self.assertAlmostEqual(
            assessment.loc["energy", "ridge_mae"],
            7.211095746335221,
            places=6,
        )
        self.assertAlmostEqual(
            assessment.loc[
                "consumer_discretionary",
                "ridge_directional_accuracy",
            ],
            0.6166666666666667,
            places=6,
        )

    def test_all_acceptance_guards_and_serialization_pass(self) -> None:
        self.assertTrue(all(self.summary["validations"].values()))
        self.assertEqual(
            self.summary["training_gate"],
            (
                "STEP_13_ALL_TARGET_RIDGE_EVALUATION_COMPLETE_"
                "READY_FOR_STEP_14_WITH_TARGET_GATES"
            ),
        )
        predictions = self.artifacts["fold_predictions"]
        self.assertEqual(
            frame_to_csv_text(predictions),
            frame_to_csv_text(predictions.copy()),
        )


if __name__ == "__main__":
    unittest.main()
