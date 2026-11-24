"""Task 4 synthetic-only tests. No historical loader or artifact is needed."""

from __future__ import annotations

import hashlib
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from scenario_modeling.src import all_target_xgboost_comparison as comparison
from scenario_modeling.src import matched_scenario_sensitivity as sensitivity
from scenario_modeling.src import walk_forward_validation as workflow
from scenario_modeling.src.model_matrix import FINAL_FEATURE_NAMES
from scenario_modeling.src.ridge_baselines import make_ridge_pipeline
from scenario_modeling.src.shock_features import (
    CONTRACT_SHOCK_NAMES, CONTRACT_TO_HISTORICAL_FEATURE, EVENT_FLAG_NAMES,
    HISTORICAL_SHOCK_NAMES,
)
from scenario_modeling.tests.test_all_target_xgboost_comparison import (
    SyntheticFit, synthetic_evidence, synthetic_pretest, write_synthetic_evidence,
)


def diagnostic_fixture():
    frame = synthetic_pretest()
    training = frame.iloc[:160].copy()
    training["vix_level_points"] = np.linspace(10, 40, len(training))
    for name in HISTORICAL_SHOCK_NAMES:
        training[name] = np.resize([-2.0, -1.0, 0.0, 1.0, 2.0], len(training))
    training["sp500"] = np.linspace(-5, 5, len(training))
    training["ten_year_yield_bps"] = 10 * training["sp500"]
    validation = frame.iloc[200:220][["sample_date", "target_baseline_date", "target_end_date", *FINAL_FEATURE_NAMES]].copy()
    record = {
        "target": "sp500", "fold": 1, "unit": "percent", "selected_alpha": 10.0,
        "outer_training_rows": len(training), "fit_first_sample": training["sample_date"].iloc[0],
        "fit_last_sample": training["sample_date"].iloc[-1],
    }
    return training, validation, record


def saved_ridge_fixture(training, validation, record):
    pipeline = make_ridge_pipeline(record["selected_alpha"])
    pipeline.fit(training[list(FINAL_FEATURE_NAMES)], training[record["target"]])
    saved = validation[["sample_date", "target_baseline_date", "target_end_date"]].copy()
    saved["target"], saved["fold"], saved["unit"] = record["target"], record["fold"], record["unit"]
    saved["strategy"] = "ridge_full_history"
    saved["prediction"] = pipeline.predict(validation[list(FINAL_FEATURE_NAMES)])
    return saved, pipeline


class MatchedSensitivityTests(unittest.TestCase):
    def setUp(self):
        self.training, self.validation, self.record = diagnostic_fixture()

    def tables(self, *, training=None, predictors=None, target="sp500"):
        if predictors is None:
            predictors = {"ridge": lambda x: x[HISTORICAL_SHOCK_NAMES[0]].to_numpy() * 0.1,
                          "xgboost": lambda x: np.where(x[HISTORICAL_SHOCK_NAMES[0]] > 0, 0.5, 0.0)}
        return sensitivity.matched_response_tables(
            self.training if training is None else training, FINAL_FEATURE_NAMES,
            target=target, fold=1, predictors=predictors,
        )

    def test_reuses_tenth_median_ninetieth_training_vix(self):
        states = sensitivity.select_representative_states(self.training)
        for (name, state), quantile in zip(states, (0.10, 0.50, 0.90), strict=True):
            index = (self.training.vix_level_points - self.training.vix_level_points.quantile(quantile)).abs().idxmin()
            self.assertEqual(state.sample_date, self.training.loc[index, "sample_date"])
            self.assertEqual(sensitivity.STATE_QUANTILES[name], quantile)

    def test_states_are_training_rows_and_exclude_validation(self):
        for _, state in sensitivity.select_representative_states(self.training):
            self.assertIn(state.sample_date, set(self.training.sample_date))
            self.assertNotIn(state.sample_date, set(self.validation.sample_date))

    def test_duplicate_index_and_ties_choose_first_real_row(self):
        training = self.training.copy()
        training.index = np.zeros(len(training), dtype=int)
        training["vix_level_points"] = 20.0
        self.assertTrue(all(row.sample_date == training.sample_date.iloc[0] for _, row in sensitivity.select_representative_states(training)))

    def test_nonfinite_vix_rejected(self):
        self.training.loc[self.training.index[0], "vix_level_points"] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            sensitivity.select_representative_states(self.training)

    def test_grid_is_observed_deterministic_five_points_and_zero(self):
        values = pd.Series([-8, -7, -6, -5, -4, -3, -2, -1, 0, 2, 4, 6, 8, 10])
        first = sensitivity.construct_shock_grid(values)
        self.assertEqual(first, sensitivity.construct_shock_grid(values.sample(frac=1, random_state=1)))
        self.assertEqual(len(first["grid_values"]), 5)
        self.assertIn(0.0, first["grid_values"])
        self.assertTrue(set(first["grid_values"]) <= set(values))
        self.assertEqual(first["grid_values"][0], -8)
        self.assertEqual(first["grid_values"][-1], 10)

    def test_sparse_grid_preserves_only_distinct_support(self):
        grid = sensitivity.construct_shock_grid(pd.Series([0, 0, 25, 25]))
        self.assertEqual(grid["grid_values"], [0.0, 25.0])
        self.assertTrue(grid["sparse_grid"])
        self.assertEqual(grid["supported_distinct_values"], 2)

    def test_unobserved_zero_not_added_to_positive_grid(self):
        grid = sensitivity.construct_shock_grid(pd.Series([1, 2, 3, 4, 5, 6]))
        self.assertNotIn(0.0, grid["grid_values"])
        self.assertIn("NEUTRAL_ZERO_NOT_OBSERVED", grid["support_warnings"])

    def test_nonfinite_support_explicit_and_empty_grid_allowed(self):
        grid = sensitivity.construct_shock_grid(pd.Series([np.nan, np.inf]))
        self.assertEqual(grid["grid_values"], [])
        self.assertEqual(grid["support_status"], "UNAVAILABLE")
        self.assertIsNone(grid["training_min"])

    def test_exact_58_feature_order_required(self):
        with self.assertRaisesRegex(ValueError, "58-feature"):
            sensitivity.matched_response_tables(self.training, FINAL_FEATURE_NAMES[::-1], target="sp500", fold=1, predictors={"ridge": Mock()})

    def test_both_models_receive_identical_features_states_flags_and_grids(self):
        batches = {name: [] for name in ("ridge", "xgboost")}
        def predictor(name):
            def predict(frame):
                batches[name].append(frame.copy())
                return frame[HISTORICAL_SHOCK_NAMES[0]].to_numpy(dtype=float)
            return predict
        self.tables(predictors={name: predictor(name) for name in batches})
        self.assertEqual(len(batches["ridge"]), 48)
        for left, right in zip(batches["ridge"], batches["xgboost"], strict=True):
            pd.testing.assert_frame_equal(left, right)
            self.assertEqual(tuple(left.columns), FINAL_FEATURE_NAMES)
            for flag in EVENT_FLAG_NAMES:
                self.assertEqual(left[flag].nunique(), 1)
            varying = left[list(HISTORICAL_SHOCK_NAMES)].nunique()
            self.assertEqual(int((varying > 1).sum()), 1)
            for name in left.columns.difference(HISTORICAL_SHOCK_NAMES):
                self.assertEqual(left[name].nunique(), 1)

    def test_source_states_flags_and_training_remain_unchanged(self):
        original = self.training.copy(deep=True)
        tables = self.tables()
        pd.testing.assert_frame_equal(original, self.training)
        for state, row in sensitivity.select_representative_states(self.training):
            part = tables["sensitivity_summary"].query("state == @state")
            expected = {name: float(row[name]) for name in EVENT_FLAG_NAMES}
            self.assertTrue(all(json.loads(value) == expected for value in part.release_flags_json))

    def test_neutral_effect_exactly_zero_and_increment_identity(self):
        tables = self.tables()
        detail = tables["sensitivity_detail"]
        self.assertTrue(detail.loc[detail.grid_value.eq(0), "incremental_effect"].eq(0).all())
        np.testing.assert_array_equal(detail.scenario_prediction - detail.baseline_prediction, detail.incremental_effect)
        self.assertTrue(tables["sensitivity_summary"].neutral_incremental_effect.eq(0).all())

    def test_return_and_yield_units_preserved(self):
        self.assertEqual(set(self.tables()["sensitivity_detail"].unit), {"percent"})
        self.assertEqual(set(self.tables(target="ten_year_yield_bps")["sensitivity_detail"].unit), {"basis_points"})

    def test_all_eight_units_and_pmi_proxy_mapping_recorded(self):
        detail = self.tables()["sensitivity_detail"]
        self.assertEqual(set(detail.shock), set(CONTRACT_SHOCK_NAMES))
        pmi = detail.loc[detail.shock.eq("pmi_change_points")]
        self.assertTrue(pmi.historical_feature.eq("kc_fed_manufacturing_pmi_proxy_change_points").all())
        self.assertEqual(sensitivity.sensitivity_metadata()["shock_units"], sensitivity.SHOCK_UNITS)

    def test_iqr_is_training_only(self):
        summary = self.tables()["sensitivity_summary"]
        self.assertTrue(summary.training_target_iqr.eq(5.0).all())
        self.assertNotIn("actual", summary.columns)

    def test_near_flat_uses_two_percent_iqr(self):
        for coefficient, expected in ((0.01, True), (0.1, False)):
            tables = self.tables(predictors={"ridge": lambda x: x[HISTORICAL_SHOCK_NAMES[0]].to_numpy() * coefficient})
            row = tables["sensitivity_summary"].iloc[0]
            self.assertEqual(row.near_flat, expected)
            self.assertAlmostEqual(row.near_flat_threshold, 0.1)
            self.assertAlmostEqual(row.normalized_response_range, row.raw_response_range / 5.0)

    def test_zero_iqr_null_normalization_and_near_flat(self):
        self.training["sp500"] = 1.0
        summary = self.tables()["sensitivity_summary"]
        self.assertTrue(summary.normalized_response_range.isna().all())
        self.assertTrue(summary.near_flat.isna().all())
        self.assertTrue(summary.iqr_status.eq("ZERO_IQR").all())

    def test_nonfinite_and_unavailable_iqr_explicit(self):
        self.assertEqual(sensitivity.training_target_scale(pd.Series([], dtype=float))["iqr_status"], "UNAVAILABLE")
        self.training.loc[0, "sp500"] = np.nan
        summary = self.tables()["sensitivity_summary"]
        self.assertTrue(summary.near_flat.isna().all())
        self.assertTrue(summary.iqr_status.eq("NONFINITE_TRAINING_TARGET").all())

    def test_overflowed_iqr_is_explicitly_unavailable(self):
        scale = sensitivity.training_target_scale(pd.Series([-1e308, -1e308, 1e308, 1e308]))
        self.assertEqual(scale["iqr_status"], "NONFINITE_IQR")
        self.assertIsNone(scale["training_target_iqr"])
        self.assertIsNone(scale["training_target_range"])

    def test_unobserved_neutral_remains_separate_from_supported_grid(self):
        self.training[HISTORICAL_SHOCK_NAMES[0]] = np.resize([1, 2, 3, 4, 5], len(self.training))
        tables = self.tables()
        part = tables["sensitivity_detail"].loc[lambda x: x.shock.eq(CONTRACT_SHOCK_NAMES[0])]
        self.assertFalse(part.grid_value.eq(0).any())
        summary = tables["sensitivity_summary"].loc[lambda x: x.shock.eq(CONTRACT_SHOCK_NAMES[0])]
        self.assertTrue(summary.baseline_prediction.eq(0).all())
        self.assertTrue(summary.neutral_incremental_effect.eq(0).all())

    def test_piecewise_constant_model_is_valid_and_repeatable(self):
        summary = self.tables()["sensitivity_summary"]
        xgb = summary.loc[summary.model_family.eq("xgboost")]
        self.assertTrue(xgb.numerically_valid.all() and xgb.repeatable.all())
        row = xgb.loc[xgb.shock.eq(CONTRACT_SHOCK_NAMES[0])].iloc[0]
        self.assertEqual(json.loads(row.adjacent_changes_json), [0.0, 0.0, 0.5, 0.0])
        self.assertEqual(row.direction_changes, 0)

    def test_nonlinear_nonmonotonic_response_and_signed_changes(self):
        summary = self.tables(predictors={"xgboost": lambda x: x[HISTORICAL_SHOCK_NAMES[0]].to_numpy() ** 2})["sensitivity_summary"]
        row = summary.iloc[0]
        self.assertEqual(json.loads(row.adjacent_changes_json), [-3.0, -1.0, 1.0, 3.0])
        self.assertEqual(row.direction_changes, 1)
        self.assertTrue(row.numerically_valid)
        self.assertEqual(row.maximum_adjacent_jump, 3.0)

    def test_extreme_jump_warns_with_existing_step13_envelope(self):
        summary = self.tables(predictors={"xgboost": lambda x: np.where(x[HISTORICAL_SHOCK_NAMES[0]] > 0, 100.0, 0.0)})["sensitivity_summary"]
        row = summary.iloc[0]
        self.assertEqual(row.step13_warning_envelope, 10.0)
        self.assertIn("SUSPICIOUS_ADJACENT_JUMP", json.loads(row.warnings_json))
        self.assertEqual(row.maximum_adjacent_jump, 100.0)

    def test_nonfinite_predictions_flagged_without_invalid_json(self):
        tables = self.tables(predictors={"xgboost": lambda x: np.full(len(x), np.nan)})
        self.assertFalse(tables["sensitivity_summary"].numerically_valid.any())
        self.assertTrue(tables["sensitivity_summary"].raw_response_range.isna().all())
        self.assertIn("NONFINITE_PREDICTION", json.loads(tables["sensitivity_summary"].iloc[0].warnings_json))

    def test_finite_prediction_arithmetic_overflow_is_flagged(self):
        tables = self.tables(predictors={"xgboost": lambda x: np.where(x[HISTORICAL_SHOCK_NAMES[0]] > 0, 1e308, -1e308)})
        row = tables["sensitivity_summary"].iloc[0]
        self.assertFalse(row.numerically_valid)
        self.assertTrue(pd.isna(row.raw_response_range))
        self.assertIn("NONFINITE_RESPONSE_ARITHMETIC", json.loads(row.warnings_json))

    def test_nondeterminism_flagged(self):
        count = 0
        def predict(frame):
            nonlocal count
            count += 1
            return np.full(len(frame), float(count))
        summary = self.tables(predictors={"xgboost": predict})["sensitivity_summary"]
        self.assertFalse(summary.repeatable.any())
        self.assertTrue(summary.maximum_repeat_difference.eq(1.0).all())

    def test_deterministic_repeated_diagnostics(self):
        first, second = self.tables(), self.tables()
        for name in first:
            pd.testing.assert_frame_equal(first[name], second[name])

    def test_all_eight_flat_inputs_flagged_with_numerical_evidence(self):
        tables = self.tables(predictors={"xgboost": lambda x: np.zeros(len(x))})
        row = sensitivity.responsiveness_flags(tables["sensitivity_summary"]).iloc[0]
        self.assertEqual(row.uniformly_near_flat_shocks, 8)
        self.assertIn("ALL_EIGHT_SHOCKS_NEAR_FLAT", json.loads(row.warnings_json))
        self.assertEqual(len(json.loads(row.shock_evidence_json)), 8)

    def test_nonflat_sign_instability_across_states_and_folds(self):
        summary = self.tables(predictors={"ridge": lambda x: x[HISTORICAL_SHOCK_NAMES[0]].to_numpy() * np.where(x.vix_level_points > 25, -1.0, 1.0)})["sensitivity_summary"]
        second = summary.copy()
        second["fold"] = 2
        second["endpoint_change"] *= -1
        row = sensitivity.responsiveness_flags(pd.concat([summary, second], ignore_index=True)).iloc[0]
        evidence = json.loads(row.sign_instability_json)
        self.assertEqual({item["dimension"] for item in evidence}, {"across_states", "across_folds"})

    def test_reconstruction_uses_original_alpha_pipeline_and_boundaries(self):
        saved, _ = saved_ridge_fixture(self.training, self.validation, self.record)
        with patch.object(sensitivity, "make_ridge_pipeline", wraps=make_ridge_pipeline) as factory:
            pipeline, audit = sensitivity.reconstruct_verified_ridge(self.training, self.validation, FINAL_FEATURE_NAMES, target="sp500", fold=1, fit_record=self.record, saved_predictions=saved)
        factory.assert_called_once_with(10.0)
        self.assertIsNotNone(pipeline)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["comparison_count"], 20)
        self.assertEqual(audit["outer_training_rows"], 160)
        self.assertEqual(audit["maximum_absolute_difference"], 0.0)
        self.assertLess(audit["training_last_target_end"], audit["validation_first_baseline"])

    def test_saved_prediction_alignment_uses_dates_not_row_position(self):
        saved, _ = saved_ridge_fixture(self.training, self.validation, self.record)
        _, audit = sensitivity.reconstruct_verified_ridge(self.training, self.validation, FINAL_FEATURE_NAMES, target="sp500", fold=1, fit_record=self.record, saved_predictions=saved.iloc[::-1])
        self.assertTrue(audit["passed"])

    def test_reconstruction_accepts_prespecified_tolerance(self):
        saved, _ = saved_ridge_fixture(self.training, self.validation, self.record)
        saved["prediction"] += 0.5e-10
        _, audit = sensitivity.reconstruct_verified_ridge(self.training, self.validation, FINAL_FEATURE_NAMES, target="sp500", fold=1, fit_record=self.record, saved_predictions=saved)
        self.assertTrue(audit["passed"])

    def test_reconstruction_mismatch_excluded_and_blocks_promotion(self):
        saved, _ = saved_ridge_fixture(self.training, self.validation, self.record)
        saved.loc[saved.index[0], "prediction"] += 0.001
        model, audit = sensitivity.reconstruct_verified_ridge(self.training, self.validation, FINAL_FEATURE_NAMES, target="sp500", fold=1, fit_record=self.record, saved_predictions=saved)
        self.assertIsNone(model)
        self.assertFalse(audit["passed"])
        self.assertTrue(audit["promotion_approval_blocked"])

    def test_changed_training_boundary_rejected_before_fit(self):
        saved, _ = saved_ridge_fixture(self.training, self.validation, self.record)
        self.record["fit_last_sample"] = "wrong"
        with patch.object(sensitivity, "make_ridge_pipeline") as factory:
            with self.assertRaisesRegex(ValueError, "original fitting scope"):
                sensitivity.reconstruct_verified_ridge(self.training, self.validation, FINAL_FEATURE_NAMES, target="sp500", fold=1, fit_record=self.record, saved_predictions=saved)
            factory.assert_not_called()

    def test_unpurged_training_rejected(self):
        saved, _ = saved_ridge_fixture(self.training, self.validation, self.record)
        self.training.loc[0, "target_end_date"] = self.validation.target_baseline_date.iloc[0]
        with self.assertRaisesRegex(ValueError, "purge"):
            sensitivity.reconstruct_verified_ridge(self.training, self.validation, FINAL_FEATURE_NAMES, target="sp500", fold=1, fit_record=self.record, saved_predictions=saved)

    def test_missing_saved_comparison_rows_rejected(self):
        saved, _ = saved_ridge_fixture(self.training, self.validation, self.record)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            sensitivity.reconstruct_verified_ridge(self.training, self.validation, FINAL_FEATURE_NAMES, target="sp500", fold=1, fit_record=self.record, saved_predictions=saved.iloc[:-1])

    def test_reuses_passed_xgboost_and_excludes_failed_ridge(self):
        saved, _ = saved_ridge_fixture(self.training, self.validation, self.record)
        saved["prediction"] += 0.001
        predict = Mock(side_effect=lambda x: np.zeros(len(x)))
        xgb = SimpleNamespace(target="sp500", unit="percent", outer_fold=1, feature_names=FINAL_FEATURE_NAMES, predict=predict)
        with patch.object(workflow, "fit_target_xgboost", side_effect=AssertionError("A second XGBoost fit is forbidden")):
            result = sensitivity.diagnose_target_fold(self.training, self.validation, FINAL_FEATURE_NAMES, target="sp500", fold=1, xgboost_fit=xgb, fit_record=self.record, saved_predictions=saved)
        self.assertEqual(set(result["sensitivity_detail"].model_family), {"xgboost"})
        self.assertEqual(predict.call_count, 48)
        self.assertFalse(result["ridge_reconstruction_audit"].passed.iloc[0])

    def test_module_has_no_loaders_or_final_test_access_path(self):
        source = inspect.getsource(sensitivity)
        self.assertNotIn("read_csv", source)
        self.assertNotIn("load_all_pretest_rows", source)
        self.assertNotIn("MATRIX_PATH", source)
        self.assertNotIn("fit_target_xgboost", source)


class OptionalSensitivityIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="insightpulse-task4-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        cls.frame = synthetic_pretest()
        cls.manifest, cls.folds = workflow.build_walk_forward_folds(cls.frame)
        predictions, audit = synthetic_evidence(cls.frame, cls.folds)
        write_synthetic_evidence(cls.directory, predictions, audit, cls.manifest)
        cls.split = cls.directory / "split.csv"
        cls.split.write_text("synthetic_only\n", encoding="utf-8")
        cls.fitted, cls.reconstructed = [], []
        def fitter(training, features, **arguments):
            model = SyntheticFit(training, features, arguments["target"], arguments["outer_fold"], arguments["outer_validation_baseline"])
            cls.fitted.append(model)
            return model
        def reconstruct(training, validation, features, *, target, fold, fit_record, saved_predictions):
            # Fitting is tested separately above. This double exercises all 60
            # runner integrations and one explicit reconstruction failure.
            cls.reconstructed.append((training.copy(), validation.copy(), target, fold, fit_record))
            passed = not (fold == 2 and target == "sp500")
            pipeline = SimpleNamespace(predict=lambda x: 0.1 * x[HISTORICAL_SHOCK_NAMES[0]].to_numpy())
            record = {"fold": fold, "target": target, "unit": comparison._target_unit(target),
                      "selected_alpha": fit_record["selected_alpha"], "outer_training_rows": len(training),
                      "comparison_count": len(validation), "maximum_absolute_difference": 0.0 if passed else 0.1,
                      "passed": passed, "status": "PASS" if passed else "FAIL", "promotion_approval_blocked": not passed}
            return (pipeline if passed else None), record
        guard = {"matrix_rows_declared": 584, "locked_final_test_rows_declared": 114, "locked_final_test_target_rows_parsed": 0}
        with (
            patch.object(comparison, "load_all_pretest_rows", return_value=(cls.frame.copy(), FINAL_FEATURE_NAMES, comparison.TARGET_NAMES, guard)),
            patch.object(comparison, "build_walk_forward_folds", side_effect=workflow.build_walk_forward_folds),
            patch.object(comparison, "fit_target_xgboost", side_effect=fitter) as fits,
            patch.object(comparison, "APPROVED_PRETEST_SHA256", comparison.authorized_pretest_fingerprint(cls.frame)),
            patch.object(comparison, "SPLIT_MANIFEST_PATH", cls.split),
            patch.object(comparison, "_source_identity", return_value={"commit": "synthetic", "branch": "synthetic", "source_files_sha256": {}}),
            patch.object(sensitivity, "reconstruct_verified_ridge", side_effect=reconstruct),
        ):
            cls.artifacts = comparison.build_all_target_comparison_artifacts(evidence_dir=cls.directory, include_sensitivity=True)
            cls.fit_count = fits.call_count

    def test_optional_runner_reuses_exact_sixty_task2_fits(self):
        self.assertEqual(self.fit_count, 60)
        self.assertEqual(len(self.reconstructed), 60)
        self.assertTrue(all(model.predict.call_count > 2 for model in self.fitted))

    def test_diagnostic_validation_has_no_target_labels(self):
        for training, validation, target, fold, record in self.reconstructed:
            self.assertTrue(set(validation.columns).isdisjoint(comparison.TARGET_NAMES))
            self.assertEqual(set(training.columns).intersection(comparison.TARGET_NAMES), {target})
            self.assertEqual(tuple(validation.columns[3:]), FINAL_FEATURE_NAMES)
            self.assertEqual(record["selected_alpha"], 1000.0)

    def test_optional_outputs_have_all_models_except_failed_ridge(self):
        self.assertEqual(len(self.artifacts["ridge_reconstruction_audit"]), 60)
        self.assertEqual(len(self.artifacts["sensitivity_summary"]), (120 - 1) * 24)
        summary = self.artifacts["sensitivity_summary"]
        failed = summary.query("target == 'sp500' and fold == 2")
        self.assertEqual(set(failed.model_family), {"xgboost"})
        self.assertTrue(self.artifacts["summary"]["sensitivity_metadata"]["promotion_approval_blocked_by_ridge_mismatch"])

    def test_metadata_truthful_and_original_evaluation_preserved(self):
        summary = self.artifacts["summary"]
        self.assertTrue(summary["ridge_refitted"])
        self.assertTrue(summary["ridge_evaluation_predictions_reused"])
        self.assertTrue(summary["ridge_reconstruction_for_sensitivity_only"])
        self.assertEqual(len(self.artifacts["authoritative_predictions"]), 14400)
        self.assertEqual(summary["matrix_guard"]["locked_final_test_target_rows_parsed"], 0)

    def test_optional_persistence_writes_nineteen_checksummed_files(self):
        destination = self.directory / "comparison"
        paths = comparison.persist_comparison_artifacts(self.artifacts, output_dir=destination)
        self.assertEqual(len(paths), 19)
        checksums = json.loads((destination / "comparison_checksums.json").read_text())["files"]
        self.assertEqual(len(checksums), 18)
        for name, expected in checksums.items():
            self.assertEqual(hashlib.sha256((destination / name).read_bytes()).hexdigest(), expected)
        summary = json.loads((destination / "comparison_summary.json").read_text())
        self.assertEqual(summary["table_rows"]["ridge_reconstruction_audit"], 60)

    def test_validator_rejects_unmatched_model_inputs(self):
        changed = dict(self.artifacts)
        changed["sensitivity_detail"] = changed["sensitivity_detail"].copy()
        changed["sensitivity_detail"].loc[0, "input_sha256"] = "mismatch"
        with self.assertRaisesRegex(ValueError, "identical"):
            sensitivity.validate_sensitivity_artifacts(changed)

    def test_validator_rejects_missing_optional_table(self):
        changed = dict(self.artifacts)
        del changed["sensitivity_flags"]
        with self.assertRaisesRegex(ValueError, "artifact names"):
            comparison.validate_comparison_artifacts(changed)

    def test_validator_rejects_nonzero_neutral(self):
        changed = dict(self.artifacts)
        changed["sensitivity_summary"] = changed["sensitivity_summary"].copy()
        changed["sensitivity_summary"].loc[0, "neutral_incremental_effect"] = 1.0
        with self.assertRaisesRegex(ValueError, "neutral"):
            sensitivity.validate_sensitivity_artifacts(changed)
