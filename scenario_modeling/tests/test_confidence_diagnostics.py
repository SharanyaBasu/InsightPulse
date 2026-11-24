from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scenario_modeling.src.confidence_diagnostics import (
    CONFIDENCE_SCHEMA_VERSION,
    STEP13_FOLD_METRICS_PATH,
    WARNING_CATEGORIES,
    XGBOOST_AGREEMENT_STATUS,
    build_confidence_artifacts,
    persist_confidence_artifacts,
)
from scenario_modeling.src.scenario_explanations import (
    build_explanation_artifacts,
)
from scenario_modeling.src.scenario_inference import (
    ALL_RIDGE_TARGETS,
    build_research_inference_engine,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfidenceDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = build_research_inference_engine()
        cls.explanations = build_explanation_artifacts(
            engine=cls.engine
        )
        cls.artifacts = build_confidence_artifacts(
            diagnostic_as_of_date="2026-07-26",
            engine=cls.engine,
            explanation_artifacts=cls.explanations,
        )

    def test_all_shock_and_state_inputs_have_range_diagnostics(
        self,
    ) -> None:
        diagnostics = self.artifacts["input_diagnostics"]
        self.assertEqual(len(diagnostics), 53)
        self.assertEqual(
            int(diagnostics["input_role"].eq("scenario_shock").sum()),
            8,
        )
        self.assertEqual(
            int(diagnostics["input_role"].eq("current_state").sum()),
            45,
        )
        numeric = diagnostics[
            [
                "value",
                "historical_minimum",
                "historical_maximum",
                "empirical_percentile",
                "standardized_distance_from_training_mean",
            ]
        ].to_numpy(dtype=float)
        self.assertTrue(np.isfinite(numeric).all())
        self.assertTrue(
            diagnostics["empirical_percentile"].between(0, 100).all()
        )

    def test_joint_historical_distance_is_finite_and_structured(
        self,
    ) -> None:
        diagnostic = self.artifacts["joint_feature_distance"]
        self.assertEqual(
            diagnostic["reference_distribution"]["training_rows"],
            464,
        )
        self.assertEqual(
            diagnostic["reference_distribution"]["feature_count"],
            58,
        )
        for row_name in ("baseline", "scenario"):
            row = diagnostic["rows"][row_name]
            self.assertTrue(
                np.isfinite(row["nearest_training_distance"])
            )
            self.assertTrue(
                0
                <= row[
                    "distance_percentile_vs_training_neighbors"
                ]
                <= 100
            )

    def test_all_targets_have_structured_noncompensatory_confidence(
        self,
    ) -> None:
        diagnostics = self.artifacts[
            "target_confidence_diagnostics"
        ]
        payload = self.artifacts["target_confidence"]
        self.assertEqual(tuple(diagnostics["target"]), ALL_RIDGE_TARGETS)
        self.assertEqual(
            payload["schema_version"],
            CONFIDENCE_SCHEMA_VERSION,
        )
        self.assertEqual(tuple(payload["targets"]), ALL_RIDGE_TARGETS)
        for target, target_payload in payload["targets"].items():
            self.assertEqual(
                set(target_payload["warnings"]),
                set(WARNING_CATEGORIES),
            )
            self.assertEqual(
                target_payload["confidence_method"],
                "RULE_BASED_NON_COMPENSATORY",
            )
            self.assertFalse(target_payload["production_ready"])
            self.assertFalse(target_payload["causal_effect_claimed"])
            self.assertFalse(
                target_payload["naive_benchmark_substituted"]
            )

    def test_xgboost_agreement_is_unavailable_and_excluded(self) -> None:
        diagnostics = self.artifacts[
            "target_confidence_diagnostics"
        ]
        self.assertTrue(
            diagnostics["xgboost_agreement_status"].eq(
                XGBOOST_AGREEMENT_STATUS
            ).all()
        )
        self.assertFalse(
            diagnostics["xgboost_agreement_included"].any()
        )
        self.assertTrue(
            diagnostics["xgboost_agreement_score"].isna().all()
        )
        for target_payload in self.artifacts[
            "target_confidence"
        ]["targets"].values():
            agreement = target_payload["xgboost_agreement"]
            self.assertFalse(agreement["available"])
            self.assertFalse(agreement["included_in_confidence"])
            self.assertIsNone(agreement["score"])
            self.assertEqual(agreement["penalty_or_bonus"], "NONE")

    def test_walk_forward_and_regime_evidence_match_step13(self) -> None:
        diagnostics = self.artifacts[
            "target_confidence_diagnostics"
        ].set_index("target")
        source = pd.read_csv(STEP13_FOLD_METRICS_PATH)
        source = source.loc[
            source["strategy"].eq("ridge_full_history")
        ]
        for target in ALL_RIDGE_TARGETS:
            target_source = source.loc[source["target"].eq(target)]
            self.assertEqual(len(target_source), 5)
            self.assertAlmostEqual(
                diagnostics.loc[target, "walk_forward_mae"],
                self.engine.assessment.loc[target, "ridge_mae"],
                places=12,
            )
            self.assertAlmostEqual(
                diagnostics.loc[target, "walk_forward_rmse"],
                self.engine.assessment.loc[target, "ridge_rmse"],
                places=12,
            )
            self.assertGreater(
                int(
                    diagnostics.loc[
                        target, "similar_regime_observations"
                    ]
                ),
                0,
            )

    def test_step13_confidence_gates_are_not_promoted(self) -> None:
        diagnostics = self.artifacts[
            "target_confidence_diagnostics"
        ]
        low = diagnostics.loc[
            diagnostics[
                "step13_evaluation_classification"
            ].eq("APPROVED_WITH_LOW_CONFIDENCE")
        ]
        weak = diagnostics.loc[
            diagnostics[
                "step13_evaluation_classification"
            ].eq("NAIVE_FALLBACK_REQUIRED")
        ]
        self.assertEqual(len(low), 3)
        self.assertEqual(len(weak), 9)
        self.assertTrue(
            low["overall_research_confidence"].str.startswith(
                "LOW_RESEARCH_CONFIDENCE"
            ).all()
        )
        self.assertTrue(
            weak["overall_research_confidence"].eq(
                "VERY_LOW_RESEARCH_CONFIDENCE"
            ).all()
        )
        self.assertFalse(
            diagnostics["overall_research_confidence"]
            .str.contains("HIGH")
            .any()
        )

    def test_state_freshness_is_explicit_and_reproducible(self) -> None:
        freshness = self.artifacts["state_freshness"]
        self.assertEqual(
            freshness["diagnostic_as_of_date"],
            "2026-07-26",
        )
        self.assertEqual(freshness["state_sample_date"], "2026-06-11")
        self.assertEqual(freshness["calendar_age_days"], 45)
        self.assertEqual(freshness["freshness_status"], "STALE")
        diagnostics = self.artifacts[
            "target_confidence_diagnostics"
        ]
        self.assertTrue(
            diagnostics["state_freshness_warning_count"].gt(0).all()
        )

    def test_determinism_schema_and_final_test_guards_pass(self) -> None:
        validations = self.artifacts["summary"]["validations"]
        self.assertTrue(all(validations.values()))
        self.assertTrue(validations["diagnostics_deterministic"])
        self.assertTrue(validations["metadata_schema_valid"])
        self.assertEqual(
            self.engine.guard["locked_final_test_target_rows_parsed"],
            0,
        )
        self.assertEqual(self.engine.guard["final_test_access_count"], 0)

    def test_persisted_hashes_validate_without_model_artifacts(
        self,
    ) -> None:
        temporary_root = PROJECT_ROOT / "tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            dir=temporary_root,
        ) as directory:
            paths = persist_confidence_artifacts(
                self.artifacts,
                output_dir=Path(directory),
            )
            summary = json.loads(
                Path(paths["summary"]).read_text(encoding="utf-8")
            )
            self.assertEqual(
                summary["completion_gate"],
                "STEP_16_CONFIDENCE_DIAGNOSTICS_COMPLETE",
            )
            for metadata in summary["output_files"].values():
                if metadata["sha256"] is None:
                    continue
                path = PROJECT_ROOT / metadata["path"]
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    metadata["sha256"],
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
