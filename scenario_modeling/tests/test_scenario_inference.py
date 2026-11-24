from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scenario_modeling.src.config import PROJECT_ROOT
from scenario_modeling.src.scenario_inference import (
    ALL_RIDGE_TARGETS,
    ASSET_TARGETS,
    EVENT_INPUT_TO_FLAG,
    EXAMPLE_SCENARIO,
    EXPECTED_TARGET_CONFIGURATIONS,
    LOW_CONFIDENCE_PROMOTED_TARGETS,
    NEUTRAL_SCENARIO,
    SECTOR_TARGETS,
    WEAK_EVIDENCE_RIDGE_TARGETS,
    ScenarioValidationError,
    build_inference_artifacts,
    build_research_inference_engine,
    persist_inference_artifacts,
    predict_scenario,
)
from scenario_modeling.src.shock_features import (
    CONTRACT_SHOCK_NAMES,
    CONTRACT_TO_HISTORICAL_FEATURE,
    EVENT_FLAG_NAMES,
)


class ScenarioInferenceAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = build_research_inference_engine()
        cls.example = cls.engine.predict(EXAMPLE_SCENARIO)

    def test_exact_contract_mapping_and_feature_schema(self) -> None:
        self.assertEqual(len(CONTRACT_SHOCK_NAMES), 8)
        self.assertEqual(len(EVENT_FLAG_NAMES), 5)
        self.assertEqual(len(self.engine.feature_names), 58)
        self.assertEqual(len(self.engine.state_feature_names), 45)
        self.assertEqual(
            CONTRACT_TO_HISTORICAL_FEATURE["pmi_change_points"],
            "kc_fed_manufacturing_pmi_proxy_change_points",
        )
        self.assertEqual(
            EVENT_INPUT_TO_FLAG,
            {
                "fed_funds_rate_change_bps": "fomc_release_flag",
                "cpi_surprise_pct": "cpi_release_flag",
                "unemployment_change_pct": (
                    "unemployment_release_flag"
                ),
                "gdp_growth_surprise_pct": "gdp_release_flag",
                "pmi_change_points": "pmi_release_flag",
            },
        )

    def test_exactly_twelve_independent_ridge_pipelines_are_fitted(
        self,
    ) -> None:
        self.assertEqual(
            tuple(self.engine.pipelines),
            ALL_RIDGE_TARGETS,
        )
        self.assertEqual(len(self.engine.pipelines), 12)
        self.assertEqual(
            len({id(value) for value in self.engine.pipelines.values()}),
            12,
        )
        self.assertEqual(
            len(
                {
                    id(value.named_steps["ridge"])
                    for value in self.engine.pipelines.values()
                }
            ),
            12,
        )

    def test_each_pipeline_uses_frozen_alpha_and_training_only_preprocessing(
        self,
    ) -> None:
        expected_alphas = {
            target: float(configuration["alpha"])
            for target, configuration
            in EXPECTED_TARGET_CONFIGURATIONS.items()
        }
        self.assertEqual(self.engine.selected_alphas, expected_alphas)
        for target, pipeline in self.engine.pipelines.items():
            self.assertEqual(
                tuple(pipeline.named_steps),
                ("imputer", "scaler", "ridge"),
            )
            self.assertEqual(
                pipeline.named_steps["ridge"].alpha,
                expected_alphas[target],
            )
            self.assertEqual(
                pipeline.named_steps["imputer"].n_features_in_,
                58,
            )
            self.assertEqual(
                pipeline.named_steps["scaler"].n_features_in_,
                58,
            )
            seen = np.asarray(
                pipeline.named_steps["scaler"].n_samples_seen_
            )
            self.assertEqual(int(seen.max()), 464)

    def test_bounded_pretest_training_and_final_test_guard(self) -> None:
        self.assertEqual(len(self.engine.training), 464)
        self.assertEqual(
            self.engine.training["sample_date"].iloc[0],
            "2013-08-05",
        )
        self.assertEqual(
            self.engine.training["sample_date"].iloc[-1],
            "2023-11-06",
        )
        self.assertEqual(
            self.engine.guard["locked_final_test_rows_declared"],
            114,
        )
        self.assertEqual(
            self.engine.guard[
                "locked_final_test_target_rows_parsed"
            ],
            0,
        )
        self.assertEqual(
            self.engine.guard["final_test_access_count"],
            0,
        )

    def test_latest_context_uses_feature_only_step_7_cache(self) -> None:
        metadata = self.engine.latest_state_metadata
        self.assertEqual(metadata["sample_date"], "2026-06-11")
        self.assertEqual(metadata["state_cutoff_date"], "2026-06-09")
        self.assertFalse(metadata["state_cache_contains_targets"])
        self.assertEqual(len(self.engine.latest_state), 45)
        self.assertTrue(
            all(
                math.isfinite(value)
                for value in self.engine.latest_state.values()
            )
        )

    def test_feature_mapping_preserves_order_roles_and_exclusion(
        self,
    ) -> None:
        mapping = self.engine.feature_mapping
        self.assertEqual(len(mapping), 58)
        self.assertEqual(
            tuple(mapping["feature_name"]),
            self.engine.feature_names,
        )
        self.assertEqual(
            mapping["inference_role"].value_counts().to_dict(),
            {
                "current_state": 45,
                "scenario_shock": 8,
                "release_flag": 5,
            },
        )
        self.assertNotIn(
            "oil_volatility_20d_annualized_pct",
            set(mapping["feature_name"]),
        )

    def test_default_release_flags_are_derived_once_and_fixed(self) -> None:
        self.assertEqual(
            self.example["release_flags"],
            {
                "fomc_release_flag": 0.0,
                "cpi_release_flag": 1.0,
                "unemployment_release_flag": 0.0,
                "gdp_release_flag": 0.0,
                "pmi_release_flag": 0.0,
            },
        )
        isolation = self.example["feature_assembly"]
        self.assertTrue(isolation["state_identical"])
        self.assertTrue(isolation["release_flags_identical"])
        self.assertTrue(isolation["only_shock_magnitudes_changed"])
        self.assertEqual(
            set(isolation["changed_features"]),
            {
                "cpi_surprise_pct",
                "oil_price_change_pct",
                "dxy_change_pct",
                "vix_change_points",
            },
        )

    def test_explicit_flags_support_zero_surprise_release_context(
        self,
    ) -> None:
        flags = {name: 0 for name in EVENT_FLAG_NAMES}
        flags["cpi_release_flag"] = 1
        result = self.engine.predict(
            NEUTRAL_SCENARIO,
            release_flags=flags,
        )
        self.assertEqual(result["release_flags"]["cpi_release_flag"], 1.0)
        self.assertTrue(
            result["release_flag_source"].startswith("CALLER_PROVIDED")
        )
        for target in ALL_RIDGE_TARGETS:
            output = result["targets"][target]
            self.assertEqual(
                output["baseline_prediction"],
                output["scenario_prediction"],
            )
            self.assertEqual(
                output["incremental_scenario_effect"],
                0.0,
            )

    def test_all_twelve_outputs_are_numeric_ridge_predictions(
        self,
    ) -> None:
        expected = {
            "sp500": (0.622544914652, 0.559085836704, -0.063459077948),
            "nasdaq": (0.872260735715, 0.797249671981, -0.075011063734),
            "ten_year_yield_bps": (
                0.476885019571,
                0.480230481793,
                0.003345462222,
            ),
            "dxy": (-0.195015039764, 0.406943556194, 0.601958595957),
            "gold": (2.265317821541, 2.153837477316, -0.111480344225),
            "oil": (1.072738714851, 1.075137396311, 0.002398681460),
            "technology": (
                1.813203627325,
                1.812471300162,
                -0.000732327163,
            ),
            "energy": (0.093057884873, 0.026581854733, -0.066476030140),
            "financials": (
                0.828518460821,
                0.822247687719,
                -0.006270773103,
            ),
            "utilities": (
                1.249663240043,
                0.997987785731,
                -0.251675454312,
            ),
            "healthcare": (
                0.981351463455,
                0.978166276696,
                -0.003185186759,
            ),
            "consumer_discretionary": (
                0.674845255874,
                0.587243490153,
                -0.087601765721,
            ),
        }
        self.assertEqual(tuple(self.example["targets"]), ALL_RIDGE_TARGETS)
        for target, values in expected.items():
            output = self.example["targets"][target]
            self.assertTrue(output["inference_supported"])
            self.assertEqual(output["model_family"], "ridge")
            self.assertEqual(output["primary_output_strategy"], "ridge")
            self.assertFalse(output["naive_benchmark_substituted"])
            self.assertFalse(output["causal_effect_claimed"])
            self.assertAlmostEqual(
                output["baseline_prediction"],
                values[0],
                places=9,
            )
            self.assertAlmostEqual(
                output["scenario_prediction"],
                values[1],
                places=9,
            )
            self.assertAlmostEqual(
                output["incremental_scenario_effect"],
                values[2],
                places=9,
            )

    def test_target_units_alphas_and_confidence_match_step13(self) -> None:
        for target, expected in EXPECTED_TARGET_CONFIGURATIONS.items():
            output = self.example["targets"][target]
            self.assertEqual(output["unit"], expected["unit"])
            self.assertEqual(
                output["selected_alpha"],
                float(expected["alpha"]),
            )
            self.assertEqual(
                output["confidence_classification"],
                expected["classification"],
            )
            assessment = self.engine.assessment.loc[target]
            evidence = output["validation_evidence"]
            self.assertEqual(evidence["mae"], assessment["ridge_mae"])
            self.assertEqual(evidence["rmse"], assessment["ridge_rmse"])
            self.assertEqual(
                evidence["out_of_sample_r_squared"],
                assessment["ridge_oos_r_squared"],
            )
            self.assertEqual(
                evidence["directional_accuracy"],
                assessment["ridge_directional_accuracy"],
            )

    def test_weak_models_are_callable_with_explicit_warnings(self) -> None:
        self.assertEqual(len(LOW_CONFIDENCE_PROMOTED_TARGETS), 3)
        self.assertEqual(len(WEAK_EVIDENCE_RIDGE_TARGETS), 9)
        for target in WEAK_EVIDENCE_RIDGE_TARGETS:
            output = self.example["targets"][target]
            self.assertEqual(
                output["inference_status"],
                "WEAK_EVIDENCE_RIDGE_RESEARCH_INFERENCE",
            )
            self.assertEqual(
                output["confidence_level"],
                "WEAK_EVIDENCE_EXPERIMENTAL",
            )
            self.assertIn(
                "WEAK_EVIDENCE_EXPERIMENTAL_RIDGE",
                output["applicable_warnings"],
            )
            self.assertIn(
                "did not reliably outperform",
                output["warning"],
            )
        for target in LOW_CONFIDENCE_PROMOTED_TARGETS:
            output = self.example["targets"][target]
            self.assertEqual(
                output["inference_status"],
                "LOW_CONFIDENCE_RIDGE_RESEARCH_INFERENCE",
            )

    def test_naive_strategies_are_comparison_benchmarks_only(self) -> None:
        for output in self.example["targets"].values():
            comparison = output["naive_benchmark_comparison"]
            self.assertEqual(
                comparison["available_evaluation_strategies"],
                [
                    "naive_zero",
                    "naive_training_mean",
                    "naive_recent_trailing_21_observation",
                ],
            )
            self.assertEqual(
                comparison["serving_role"],
                "EVALUATION_BENCHMARK_ONLY_NOT_SUBSTITUTED",
            )
            self.assertFalse(
                comparison["substituted_for_ridge_output"]
            )
            self.assertNotIn("fallback_strategy", output)

    def test_neutral_scenario_has_zero_effect_for_all_targets(
        self,
    ) -> None:
        result = self.engine.predict(NEUTRAL_SCENARIO)
        self.assertEqual(sum(result["release_flags"].values()), 0.0)
        for target in ALL_RIDGE_TARGETS:
            output = result["targets"][target]
            self.assertEqual(
                output["baseline_prediction"],
                output["scenario_prediction"],
            )
            self.assertEqual(
                output["incremental_scenario_effect"],
                0.0,
            )
            self.assertTrue(
                math.isfinite(output["baseline_prediction"])
            )

    def test_input_name_type_and_finiteness_validation(self) -> None:
        missing = dict(EXAMPLE_SCENARIO)
        missing.pop("vix_change_points")
        with self.assertRaises(ScenarioValidationError):
            self.engine.predict(missing)
        extra = {**EXAMPLE_SCENARIO, "unknown": 1.0}
        with self.assertRaises(ScenarioValidationError):
            self.engine.predict(extra)
        boolean = dict(EXAMPLE_SCENARIO)
        boolean["vix_change_points"] = True
        with self.assertRaises(ScenarioValidationError):
            self.engine.predict(boolean)
        nonfinite = dict(EXAMPLE_SCENARIO)
        nonfinite["vix_change_points"] = float("nan")
        with self.assertRaises(ScenarioValidationError):
            self.engine.predict(nonfinite)

    def test_custom_state_and_release_flag_validation(self) -> None:
        incomplete_state = dict(self.engine.latest_state)
        incomplete_state.pop(next(iter(incomplete_state)))
        with self.assertRaises(ScenarioValidationError):
            self.engine.predict(
                EXAMPLE_SCENARIO,
                current_state=incomplete_state,
            )
        incomplete_flags = {name: 0 for name in EVENT_FLAG_NAMES}
        incomplete_flags.pop("pmi_release_flag")
        with self.assertRaises(ScenarioValidationError):
            self.engine.predict(
                EXAMPLE_SCENARIO,
                release_flags=incomplete_flags,
            )
        invalid_flags = {name: 0 for name in EVENT_FLAG_NAMES}
        invalid_flags["cpi_release_flag"] = 2
        with self.assertRaises(ScenarioValidationError):
            self.engine.predict(
                EXAMPLE_SCENARIO,
                release_flags=invalid_flags,
            )

    def test_out_of_range_warns_or_errors_without_clipping(self) -> None:
        scenario = dict(EXAMPLE_SCENARIO)
        scenario["fed_funds_rate_change_bps"] = 76.0
        result = self.engine.predict(scenario)
        self.assertEqual(
            result["scenario_shocks"][
                "fed_funds_rate_change_bps"
            ],
            76.0,
        )
        self.assertFalse(
            result["input_range_policy"]["clipping_performed"]
        )
        self.assertTrue(
            any(
                warning["field"] == "fed_funds_rate_change_bps"
                and warning["warning_type"]
                == "SCENARIO_SHOCK_OUTSIDE_TRAINING_RANGE"
                for warning in result["input_range_warnings"]
            )
        )
        with self.assertRaises(ScenarioValidationError):
            self.engine.predict(
                scenario,
                out_of_range_policy="error",
            )

    def test_output_grouping_and_determinism(self) -> None:
        repeated = predict_scenario(
            EXAMPLE_SCENARIO,
            engine=self.engine,
        )
        self.assertEqual(
            json.dumps(self.example, sort_keys=True),
            json.dumps(repeated, sort_keys=True),
        )
        self.assertEqual(
            tuple(self.example["asset_outputs"]),
            ASSET_TARGETS,
        )
        self.assertEqual(
            tuple(self.example["sector_outputs"]),
            SECTOR_TARGETS,
        )
        self.assertEqual(len(self.example["targets"]), 12)
        second_engine = build_research_inference_engine()
        rebuilt = second_engine.predict(EXAMPLE_SCENARIO)
        self.assertEqual(
            json.dumps(self.example, sort_keys=True),
            json.dumps(rebuilt, sort_keys=True),
        )

    def test_artifact_acceptance_checks_and_persistence(self) -> None:
        artifacts = build_inference_artifacts(engine=self.engine)
        self.assertTrue(
            all(artifacts["summary"]["validations"].values())
        )
        self.assertEqual(len(artifacts["feature_mapping"]), 58)
        self.assertEqual(len(artifacts["input_ranges"]), 53)
        self.assertEqual(len(artifacts["model_gate_audit"]), 12)
        audit = artifacts["model_gate_audit"]
        self.assertTrue(audit["ridge_pipeline_fitted"].all())
        self.assertFalse(audit["naive_benchmark_substituted"].any())
        self.assertEqual(
            artifacts["summary"]["completion_gate"],
            "STEP_14_ALL_TARGET_RIDGE_INFERENCE_COMPLETE",
        )
        self.assertEqual(
            artifacts["summary"]["ridge_pipeline_count"],
            12,
        )
        temporary_root = PROJECT_ROOT / "tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            dir=temporary_root,
        ) as directory:
            paths = persist_inference_artifacts(
                artifacts,
                output_dir=Path(directory),
            )
            self.assertEqual(len(paths), 6)
            self.assertFalse(
                any(
                    Path(path).suffix
                    in {".joblib", ".pkl", ".pickle", ".bin"}
                    for path in paths.values()
                )
            )
            example = json.loads(
                Path(paths["example_result"]).read_text(
                    encoding="utf-8"
                )
            )
            for output in example["targets"].values():
                self.assertEqual(output["model_family"], "ridge")
                self.assertTrue(
                    math.isfinite(output["baseline_prediction"])
                )
                self.assertTrue(
                    math.isfinite(output["scenario_prediction"])
                )
                self.assertTrue(
                    math.isfinite(
                        output["incremental_scenario_effect"]
                    )
                )


if __name__ == "__main__":
    unittest.main()
