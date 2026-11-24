from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scenario_modeling.src.scenario_explanations import (
    EXPLANATION_SCHEMA_VERSION,
    RECONCILIATION_TOLERANCE,
    build_explanation_artifacts,
    persist_explanation_artifacts,
)
from scenario_modeling.src.scenario_inference import (
    ALL_RIDGE_TARGETS,
    CONTRACT_SHOCK_NAMES,
    build_research_inference_engine,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ScenarioExplanationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = build_research_inference_engine()
        cls.artifacts = build_explanation_artifacts(engine=cls.engine)

    def test_all_targets_and_feature_contributions_are_complete(
        self,
    ) -> None:
        contributions = self.artifacts["feature_contributions"]
        deltas = self.artifacts["contribution_deltas"]
        self.assertEqual(tuple(self.engine.target_names), ALL_RIDGE_TARGETS)
        self.assertEqual(len(contributions), 12 * 2 * 58)
        self.assertEqual(len(deltas), 12 * 58)
        self.assertEqual(
            set(contributions["row_type"]),
            {"baseline", "scenario"},
        )
        for target in ALL_RIDGE_TARGETS:
            target_rows = contributions.loc[
                contributions["target"].eq(target)
            ]
            for row_type in ("baseline", "scenario"):
                row = target_rows.loc[
                    target_rows["row_type"].eq(row_type)
                ]
                self.assertEqual(
                    tuple(row["feature_name"]),
                    self.engine.feature_names,
                )

    def test_exact_ridge_contribution_reconciliation(self) -> None:
        reconciliation = self.artifacts["prediction_reconciliation"]
        self.assertEqual(len(reconciliation), 12 * 3)
        self.assertTrue(reconciliation["reconciliation_passed"].all())
        self.assertLessEqual(
            float(
                reconciliation[
                    "absolute_reconciliation_error"
                ].max()
            ),
            RECONCILIATION_TOLERANCE,
        )
        numeric = reconciliation[
            [
                "model_intercept",
                "current_state_contribution",
                "scenario_shock_contribution",
                "release_flag_contribution",
                "model_prediction",
                "reconciled_prediction",
            ]
        ].to_numpy(dtype=float)
        self.assertTrue(np.isfinite(numeric).all())

    def test_incremental_contribution_changes_are_isolated(self) -> None:
        reconciliation = self.artifacts["prediction_reconciliation"]
        incremental = reconciliation.loc[
            reconciliation["row_type"].eq("incremental_difference")
        ]
        self.assertTrue(
            incremental[
                [
                    "model_intercept",
                    "current_state_contribution",
                    "release_flag_contribution",
                ]
            ]
            .abs()
            .le(RECONCILIATION_TOLERANCE)
            .all()
            .all()
        )
        self.assertTrue(
            np.allclose(
                incremental["scenario_shock_contribution"],
                incremental["model_prediction"],
                rtol=0.0,
                atol=RECONCILIATION_TOLERANCE,
            )
        )

    def test_raw_standardized_and_contribution_values_are_distinct(
        self,
    ) -> None:
        contributions = self.artifacts["feature_contributions"]
        required = {
            "raw_value",
            "imputed_value",
            "scaler_mean",
            "scaler_scale",
            "standardized_value",
            "ridge_coefficient",
            "feature_contribution",
            "raw_zero_but_standardized_nonzero",
            "unusually_large_contribution",
        }
        self.assertTrue(required <= set(contributions.columns))
        self.assertGreater(
            int(
                contributions[
                    "raw_zero_but_standardized_nonzero"
                ].sum()
            ),
            0,
        )

    def test_human_readable_drivers_exist_for_all_targets(self) -> None:
        payload = self.artifacts["target_explanations"]
        self.assertEqual(
            payload["schema_version"],
            EXPLANATION_SCHEMA_VERSION,
        )
        self.assertEqual(tuple(payload["targets"]), ALL_RIDGE_TARGETS)
        for target, target_payload in payload["targets"].items():
            self.assertEqual(target_payload["model_family"], "ridge")
            self.assertFalse(target_payload["causal_effect_claimed"])
            self.assertIn(
                "predictive association",
                target_payload["explanation_warning"],
            )
            self.assertTrue(target_payload["top_drivers"])
            explanations = [
                item["explanation"]
                for group in target_payload["top_drivers"].values()
                for item in group
            ]
            self.assertTrue(explanations)
            self.assertTrue(
                all("not a causal effect" in text for text in explanations)
            )

    def test_all_shocks_targets_and_vix_states_are_sensitivity_tested(
        self,
    ) -> None:
        curves = self.artifacts["sensitivity_curves"]
        state_summary = self.artifacts["sensitivity_state_summary"]
        summary = self.artifacts["sensitivity_summary"]
        self.assertEqual(len(summary), 12 * 8)
        self.assertEqual(len(state_summary), 12 * 8 * 3)
        self.assertEqual(
            set(summary["contract_input"]),
            set(CONTRACT_SHOCK_NAMES),
        )
        self.assertEqual(
            set(state_summary["representative_state"]),
            {"low_vix", "median_vix", "high_vix"},
        )
        self.assertTrue(summary["all_finite"].all())
        self.assertTrue(summary["all_monotonic"].all())
        self.assertFalse(summary["any_discontinuity"].any())
        self.assertTrue(curves["state_fixed"].all())
        self.assertTrue(curves["release_flags_fixed"].all())
        grouped = curves.groupby(
            ["target", "contract_input", "representative_state"]
        )
        self.assertTrue((grouped.size() >= 7).all())

    def test_response_quality_flags_are_explicit(self) -> None:
        summary = self.artifacts["sensitivity_summary"]
        required = {
            "negligible_in_all_states",
            "counterintuitive_direction_flag",
            "unstable_across_walk_forward_folds",
            "direction_reference",
            "step13_folds_with_meaningful_response",
        }
        self.assertTrue(required <= set(summary.columns))
        self.assertEqual(
            len(summary.loc[summary["target"].eq("sp500")]),
            8,
        )

    def test_step14_invariants_and_final_test_lock_are_preserved(
        self,
    ) -> None:
        validations = self.artifacts["summary"]["validations"]
        self.assertTrue(all(validations.values()))
        self.assertEqual(
            self.engine.guard["locked_final_test_target_rows_parsed"],
            0,
        )
        self.assertEqual(self.engine.guard["final_test_access_count"], 0)
        self.assertTrue(
            validations["neutral_incremental_predictions_zero"]
        )
        self.assertTrue(
            validations[
                "repeated_explanatory_inference_deterministic"
            ]
        )
        self.assertTrue(
            validations["naive_benchmarks_not_substituted"]
        )

    def test_schema_hashes_and_dependency_free_svg_persistence(
        self,
    ) -> None:
        temporary_root = PROJECT_ROOT / "tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            dir=temporary_root,
        ) as directory:
            output_dir = Path(directory)
            paths = persist_explanation_artifacts(
                self.artifacts,
                output_dir=output_dir,
            )
            summary_path = Path(paths["summary"])
            summary = json.loads(
                summary_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                summary["completion_gate"],
                "STEP_15_EXPLANATIONS_AND_SENSITIVITY_COMPLETE",
            )
            for metadata in summary["output_files"].values():
                if metadata["sha256"] is None:
                    continue
                path = PROJECT_ROOT / metadata["path"]
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    metadata["sha256"],
                )
            self.assertEqual(len(summary["sensitivity_figures"]), 12)
            for metadata in summary["sensitivity_figures"].values():
                path = PROJECT_ROOT / metadata["path"]
                self.assertEqual(path.suffix, ".svg")
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    metadata["sha256"],
                )
                self.assertIn(
                    "<svg",
                    path.read_text(encoding="utf-8"),
                )
            self.assertFalse(
                any(
                    Path(path).suffix
                    in {".joblib", ".pkl", ".pickle", ".bin"}
                    for path in paths.values()
                )
            )


if __name__ == "__main__":
    unittest.main()
