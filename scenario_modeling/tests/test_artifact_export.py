from __future__ import annotations

import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import joblib

from scenario_modeling.src.artifact_export import (
    EXAMPLE_SCENARIO,
    NEUTRAL_SCENARIO,
    export_scenario_model_bundle,
    run_fresh_process_smoke,
    scan_bundle_text_for_secrets,
)
from scenario_modeling.src.scenario_runtime import (
    BUNDLE_CHECKSUMS_FILENAME,
    BUNDLE_MANIFEST_FILENAME,
    BUNDLE_PAYLOAD_FILENAME,
    BundleValidationError,
    ScenarioValidationError,
    load_scenario_model_bundle,
    validate_bundle_hashes,
)


EXPECTED_ALPHAS = {
    "sp500": 10_000.0,
    "nasdaq": 10_000.0,
    "ten_year_yield_bps": 1_000_000.0,
    "dxy": 100.0,
    "gold": 10.0,
    "oil": 1_000_000.0,
    "technology": 1_000_000.0,
    "energy": 10_000.0,
    "financials": 100_000.0,
    "utilities": 1_000.0,
    "healthcare": 100_000.0,
    "consumer_discretionary": 10_000.0,
}


class ArtifactExportAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.bundle = Path(cls.temporary.name) / "bundle"
        cls.artifacts = export_scenario_model_bundle(cls.bundle)
        cls.engine = load_scenario_model_bundle(cls.bundle)
        cls.example = cls.engine.predict(
            EXAMPLE_SCENARIO,
            diagnostic_as_of_date=cls.artifacts["reference_date"],
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_bundle_contains_required_versioned_files(self) -> None:
        required = {
            BUNDLE_PAYLOAD_FILENAME,
            BUNDLE_MANIFEST_FILENAME,
            BUNDLE_CHECKSUMS_FILENAME,
            "feature_columns.json",
            "feature_definitions.json",
            "feature_ranges.json",
            "target_definitions.json",
            "evaluation_summary.json",
            "inference_configuration.json",
            "explanation_schema.json",
            "confidence_schema.json",
            "environment.json",
            "requirements_snapshot.txt",
            "reference_signature.json",
        }
        self.assertTrue(required <= {path.name for path in self.bundle.iterdir()})
        manifest = json.loads(
            (self.bundle / BUNDLE_MANIFEST_FILENAME).read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            manifest["bundle_version"],
            "scenario-playground-ridge-v1.0.0",
        )
        self.assertFalse(
            manifest["runtime_contract"]["retraining_required"]
        )
        self.assertFalse(
            manifest["runtime_contract"]["training_matrix_required"]
        )

    def test_exactly_twelve_fitted_independent_ridge_pipelines(self) -> None:
        self.assertEqual(len(self.engine.pipelines), 12)
        self.assertEqual(
            tuple(self.engine.pipelines),
            tuple(EXPECTED_ALPHAS),
        )
        self.assertEqual(
            len({id(value) for value in self.engine.pipelines.values()}),
            12,
        )
        for target, pipeline in self.engine.pipelines.items():
            self.assertEqual(
                tuple(pipeline.named_steps),
                ("imputer", "scaler", "ridge"),
            )
            self.assertEqual(
                float(pipeline.named_steps["ridge"].alpha),
                EXPECTED_ALPHAS[target],
            )

    def test_exact_feature_order_state_and_public_contract(self) -> None:
        self.assertEqual(len(self.engine.feature_names), 58)
        self.assertEqual(len(self.engine.state_feature_names), 45)
        self.assertEqual(len(self.engine.scenario_input_names), 8)
        self.assertEqual(len(self.engine.event_flag_names), 5)
        self.assertEqual(
            self.engine.scenario_input_mapping["pmi_change_points"],
            "kc_fed_manufacturing_pmi_proxy_change_points",
        )
        self.assertEqual(
            tuple(
                self.engine.pipelines["sp500"].feature_names_in_
            ),
            self.engine.feature_names,
        )

    def test_target_alphas_units_and_ridge_only_registry(self) -> None:
        self.assertEqual(self.engine.selected_alphas, EXPECTED_ALPHAS)
        self.assertEqual(
            self.engine.target_units["ten_year_yield_bps"],
            "basis_points",
        )
        for target in self.engine.target_names:
            if target != "ten_year_yield_bps":
                self.assertEqual(
                    self.engine.target_units[target],
                    "percent",
                )
            self.assertEqual(
                self.engine.target_registry[target]["model_family"],
                "ridge",
            )

    def test_all_predictions_are_finite_numeric_and_no_naive_substitution(
        self,
    ) -> None:
        for target in self.engine.target_names:
            output = self.example["targets"][target]
            for field in (
                "baseline_prediction",
                "scenario_prediction",
                "incremental_scenario_effect",
            ):
                self.assertIsInstance(output[field], float)
                self.assertTrue(math.isfinite(output[field]))
            self.assertFalse(output["naive_benchmark_substituted"])
            self.assertEqual(output["model_family"], "ridge")

    def test_neutral_scenario_is_exactly_zero_incremental(self) -> None:
        result = self.engine.predict(
            NEUTRAL_SCENARIO,
            diagnostic_as_of_date=self.artifacts["reference_date"],
        )
        for target in self.engine.target_names:
            output = result["targets"][target]
            self.assertEqual(
                output["baseline_prediction"],
                output["scenario_prediction"],
            )
            self.assertEqual(
                output["incremental_scenario_effect"],
                0.0,
            )

    def test_state_and_release_flags_are_isolated(self) -> None:
        isolation = self.example["feature_assembly"]
        self.assertTrue(isolation["feature_order_exact"])
        self.assertTrue(isolation["only_shock_magnitudes_changed"])
        self.assertTrue(isolation["state_identical"])
        self.assertTrue(isolation["release_flags_identical"])
        self.assertEqual(
            self.example["context"]["sample_date"],
            "2026-06-11",
        )
        self.assertEqual(
            self.example["context"]["state_cutoff_date"],
            "2026-06-09",
        )

    def test_explanations_reconcile_and_schemas_are_frozen(self) -> None:
        explanations = self.example["explanations"]
        self.assertEqual(
            explanations["schema_version"],
            "scenario_explanation_metadata_v1",
        )
        self.assertTrue(
            explanations["reconciliation"]["all_targets_passed"]
        )
        self.assertLessEqual(
            explanations["reconciliation"][
                "maximum_absolute_error"
            ],
            explanations["reconciliation"]["tolerance"],
        )
        self.assertEqual(
            tuple(explanations["targets"]),
            self.engine.target_names,
        )

    def test_confidence_warnings_and_postponed_xgboost_are_preserved(
        self,
    ) -> None:
        confidence = self.example["confidence"]
        self.assertEqual(
            confidence["schema_version"],
            "scenario_confidence_metadata_v1",
        )
        self.assertEqual(
            confidence["xgboost_agreement_policy"]["status"],
            "UNAVAILABLE_INTENTIONALLY_POSTPONED",
        )
        self.assertEqual(
            confidence["state_freshness"]["state_sample_date"],
            "2026-06-11",
        )
        for target in self.engine.target_names:
            record = confidence["targets"][target]
            self.assertIn(
                record["overall_research_confidence"],
                {
                    "LOW_RESEARCH_CONFIDENCE",
                    "LOW_RESEARCH_CONFIDENCE_WITH_CONTEXT_WARNINGS",
                    "VERY_LOW_RESEARCH_CONFIDENCE",
                    "OUTSIDE_HISTORICAL_SUPPORT",
                },
            )
            self.assertFalse(record["naive_benchmark_substituted"])

    def test_explicit_complete_state_and_dates_are_supported(self) -> None:
        state = dict(self.engine.default_state)
        flags = {name: 0 for name in self.engine.event_flag_names}
        result = self.engine.predict(
            NEUTRAL_SCENARIO,
            current_state=state,
            release_flags=flags,
            state_sample_date="2026-07-20",
            state_cutoff_date="2026-07-20",
            diagnostic_as_of_date="2026-07-26",
        )
        self.assertEqual(
            result["context"]["state_source"],
            "CALLER_PROVIDED_COMPLETE_STATE",
        )
        self.assertEqual(
            result["confidence"]["state_freshness"][
                "freshness_status"
            ],
            "FRESH",
        )

    def test_invalid_scenario_inputs_are_rejected(self) -> None:
        missing = dict(EXAMPLE_SCENARIO)
        missing.pop("vix_change_points")
        with self.assertRaises(ScenarioValidationError):
            self.engine.predict(missing)
        extra = dict(EXAMPLE_SCENARIO)
        extra["unexpected"] = 1.0
        with self.assertRaises(ScenarioValidationError):
            self.engine.predict(extra)
        nonfinite = dict(EXAMPLE_SCENARIO)
        nonfinite["vix_change_points"] = math.inf
        with self.assertRaises(ScenarioValidationError):
            self.engine.predict(nonfinite)

    def test_payload_has_no_training_frame_or_target_labels(self) -> None:
        payload = joblib.load(self.bundle / BUNDLE_PAYLOAD_FILENAME)
        for forbidden in (
            "training",
            "training_frame",
            "target_values",
            "target_labels",
            "training_targets",
        ):
            self.assertNotIn(forbidden, payload)
        diagnostic = payload["diagnostic_reference"]
        self.assertFalse(diagnostic["contains_target_labels"])
        self.assertFalse(
            diagnostic["contains_complete_training_dataframe"]
        )

    def test_hashes_validate_and_detect_tampering(self) -> None:
        validation = validate_bundle_hashes(self.bundle)
        self.assertTrue(validation["all_hashes_match"])
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "bundle"
            shutil.copytree(self.bundle, copied)
            path = copied / "feature_columns.json"
            path.write_text(
                path.read_text(encoding="utf-8") + " ",
                encoding="utf-8",
            )
            with self.assertRaises(BundleValidationError):
                load_scenario_model_bundle(copied)

    def test_bundle_text_contains_no_credential_shaped_values(self) -> None:
        result = scan_bundle_text_for_secrets(self.bundle)
        self.assertFalse(result["credentials_detected"])
        self.assertEqual(result["finding_count"], 0)

    def test_repeated_loads_are_deterministic(self) -> None:
        first = load_scenario_model_bundle(self.bundle).predict(
            EXAMPLE_SCENARIO,
            diagnostic_as_of_date=self.artifacts["reference_date"],
        )
        second = load_scenario_model_bundle(self.bundle).predict(
            EXAMPLE_SCENARIO,
            diagnostic_as_of_date=self.artifacts["reference_date"],
        )
        self.assertEqual(first, second)

    def test_fresh_process_performs_no_fit_or_data_read(self) -> None:
        result = run_fresh_process_smoke(self.bundle)
        self.assertEqual(result["fit_attempts"], 0)
        self.assertEqual(result["training_data_reads"], 0)
        self.assertEqual(result["target_label_reads"], 0)
        self.assertTrue(result["reference_explanation_match"])
        self.assertTrue(result["reference_confidence_match"])
        self.assertTrue(result["reference_warnings_match"])
        self.assertTrue(result["repeated_load_deterministic"])
        self.assertEqual(result["locked_final_test_access_count"], 0)


if __name__ == "__main__":
    unittest.main()
