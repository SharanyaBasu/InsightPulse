"""Synthetic comparison infrastructure tests; historical inputs stay unopened."""

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
from scenario_modeling.src import walk_forward_validation as workflow
from scenario_modeling.src.feature_sources import frame_to_csv_text
from scenario_modeling.src.model_matrix import FINAL_FEATURE_NAMES
from scenario_modeling.src.xgboost_prototype import ARCHITECTURES, SEED, _xgboost_parameters


def synthetic_pretest() -> pd.DataFrame:
    """Create independent observations with the original fold-size geometry."""
    rng = np.random.default_rng(73)
    frame = pd.DataFrame(rng.normal(size=(464, 58)), columns=FINAL_FEATURE_NAMES)
    for name in FINAL_FEATURE_NAMES:
        if name.endswith("_release_flag"):
            frame[name] = rng.integers(0, 2, len(frame))
    steps = np.full(464, 7)
    for start, purge in zip((164, 224, 284, 344, 404), (4, 2, 4, 2, 3), strict=True):
        steps[start - 5:start + 1] = 21 // purge
    offsets = np.cumsum(steps) - steps[0]
    origin = pd.Timestamp("2000-01-03")
    baselines = pd.DatetimeIndex([origin + pd.offsets.BDay(int(value)) for value in offsets])
    frame["sample_date"] = (baselines + pd.offsets.BDay(1)).strftime("%Y-%m-%d")
    frame["target_baseline_date"] = baselines.strftime("%Y-%m-%d")
    frame["target_end_date"] = (baselines + pd.offsets.BDay(21)).strftime("%Y-%m-%d")
    frame["vix_level_points"] = np.resize([12.0, 22.0, 35.0], len(frame))
    for number, target in enumerate(comparison.TARGET_NAMES, 1):
        frame[target] = number * (0.5 * frame[FINAL_FEATURE_NAMES[0]] + 0.2 * frame[FINAL_FEATURE_NAMES[1]])
    return frame


def synthetic_evidence(frame, folds):
    rows, audits = [], []
    for fold, (training, validation) in enumerate(folds, 1):
        inner_training, inner_validation, purge = workflow._inner_split(training)
        for target in comparison.TARGET_NAMES:
            audits.append({
                "fold": fold, "target": target, "unit": comparison._target_unit(target),
                "selected_alpha": 1000.0, "outer_training_rows": len(training),
                "fit_first_sample": training["sample_date"].iloc[0],
                "fit_last_sample": training["sample_date"].iloc[-1],
                "inner_training_rows": len(inner_training), "inner_validation_rows": len(inner_validation),
                "inner_purge_rows": purge, "inner_validation_mae": 0.25,
                "training_mae": 0.25, "training_rmse": 0.25,
            })
            for observation in validation.itertuples(index=False):
                actual = float(getattr(observation, target))
                values = {
                    "naive_zero": 0.0,
                    "naive_training_mean": float(training[target].mean()),
                    "naive_recent_trailing_21_observation": actual + 0.5,
                    comparison.RIDGE_STRATEGY: actual + 0.25,
                }
                for strategy, value in values.items():
                    rows.append({
                        "fold": fold, "sample_date": observation.sample_date,
                        "target_baseline_date": observation.target_baseline_date,
                        "target_end_date": observation.target_end_date,
                        "target": target, "unit": comparison._target_unit(target), "strategy": strategy,
                        "actual": actual, "prediction": value, "residual": actual - value,
                        "absolute_error": abs(actual - value), "vix_level_points": observation.vix_level_points,
                    })
    return pd.DataFrame(rows, columns=comparison.SAVED_COLUMNS), pd.DataFrame(audits)


def write_synthetic_evidence(directory, predictions, audit, manifest):
    outputs = {}
    for name, frame in (("fold_predictions", predictions), ("fit_audit", audit), ("fold_manifest", manifest)):
        path = directory / f"all_targets_{name}.csv"
        content = frame_to_csv_text(frame).encode("utf-8")
        path.write_bytes(content)
        outputs[name] = {"path": str(path), "sha256": hashlib.sha256(content).hexdigest()}
    (directory / "all_targets_summary.json").write_text(json.dumps({"output_files": outputs}), encoding="utf-8")


class SyntheticFit:
    """A deterministic prediction double with the complete Task 2 interface."""

    def __init__(self, training, features, target, fold, boundary):
        self.target, self.outer_fold = target, fold
        self.unit = comparison._target_unit(target)
        self.feature_names = tuple(features)
        inner_train, inner_validation, purge = workflow._inner_split(training)
        self.inner_training_rows, self.inner_validation_rows = len(inner_train), len(inner_validation)
        self.inner_purge_rows, self.outer_training_rows = purge, len(training)
        self.seed = SEED
        self.candidates = tuple(
            workflow.XGBoostCandidateAudit(
                architecture=item, inner_mae=0.1 * (index + 1), best_iteration=7,
                boosting_rounds=8, parameters=_xgboost_parameters(item, base_score=float(inner_train[target].mean())),
            )
            for index, item in enumerate(ARCHITECTURES)
        )
        self.selected_candidate = self.candidates[0]
        self.practical_tie_mae_limit = 0.101
        self.refit_parameters = _xgboost_parameters(ARCHITECTURES[0], base_score=float(training[target].mean()))
        self.boundaries = {
            "outer_training_first_sample": training["sample_date"].iloc[0],
            "outer_training_last_sample": training["sample_date"].iloc[-1],
            "outer_training_last_target_end": training["target_end_date"].max(),
            "outer_validation_first_baseline": boundary,
            "inner_training_first_sample": inner_train["sample_date"].iloc[0],
            "inner_training_last_sample": inner_train["sample_date"].iloc[-1],
            "inner_training_last_target_end": inner_train["target_end_date"].max(),
            "inner_validation_first_sample": inner_validation["sample_date"].iloc[0],
            "inner_validation_last_sample": inner_validation["sample_date"].iloc[-1],
            "inner_validation_first_baseline": inner_validation["target_baseline_date"].iloc[0],
        }
        self.predict = Mock(side_effect=self._predict)

    def _predict(self, frame):
        if tuple(frame.columns) != self.feature_names:
            raise AssertionError("Predictor received labels or an incorrect feature order")
        scale = comparison.TARGET_NAMES.index(self.target) + 1
        return scale * (0.5 * frame[FINAL_FEATURE_NAMES[0]].to_numpy() + 0.2 * frame[FINAL_FEATURE_NAMES[1]].to_numpy()) + 0.03125 * 2 ** (self.outer_fold - 1)


class AllTargetComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="insightpulse-task3-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        cls.frame = synthetic_pretest()
        cls.manifest, cls.folds = workflow.build_walk_forward_folds(cls.frame)
        cls.predictions, cls.audit = synthetic_evidence(cls.frame, cls.folds)
        write_synthetic_evidence(cls.directory, cls.predictions, cls.audit, cls.manifest)
        cls.saved = comparison.load_authoritative_evidence(cls.directory)
        cls.split_path = cls.directory / "synthetic_split.csv"
        cls.split_path.write_text("sample_date,role\n2000-01-04,synthetic\n", encoding="utf-8")
        cls.models, cls.fit_calls = [], []

        def fitter(training, features, *, target, outer_validation_baseline, outer_fold):
            cls.fit_calls.append((training.copy(), tuple(features), target, outer_validation_baseline, outer_fold))
            fitted = SyntheticFit(training, features, target, outer_fold, outer_validation_baseline)
            cls.models.append(fitted)
            return fitted

        cls.artifacts = cls.build(fitter=fitter)

    @classmethod
    def build(cls, *, fitter=None, approved_hash=None, **kwargs):
        if fitter is None:
            fitter = lambda training, features, **arguments: SyntheticFit(
                training, features, arguments["target"], arguments["outer_fold"], arguments["outer_validation_baseline"]
            )
        guard = {"matrix_rows_declared": 584, "locked_final_test_rows_declared": 114, "locked_final_test_target_rows_parsed": 0}
        with (
            patch.object(comparison, "load_all_pretest_rows", return_value=(cls.frame.copy(), FINAL_FEATURE_NAMES, comparison.TARGET_NAMES, guard)),
            patch.object(comparison, "build_walk_forward_folds", side_effect=workflow.build_walk_forward_folds),
            patch.object(comparison, "fit_target_xgboost", side_effect=fitter),
            patch.object(comparison, "APPROVED_PRETEST_SHA256", approved_hash or comparison.authorized_pretest_fingerprint(cls.frame)),
            patch.object(comparison, "SPLIT_MANIFEST_PATH", cls.split_path),
            patch.object(comparison, "_source_identity", return_value={"commit": "synthetic-test", "branch": "synthetic-test", "source_files_sha256": {"fixture": "test-only"}}),
        ):
            return comparison.build_all_target_comparison_artifacts(evidence_dir=cls.directory, **kwargs)

    def test_twelve_targets_five_folds_delegate_to_task2_fitter(self):
        self.assertEqual(len(self.fit_calls), 60)
        self.assertEqual({(call[4], call[2]) for call in self.fit_calls}, {(f, t) for f in range(1, 6) for t in comparison.TARGET_NAMES})
        for training, features, target, boundary, fold in self.fit_calls:
            expected_train, validation = self.folds[fold - 1]
            self.assertEqual(features, FINAL_FEATURE_NAMES)
            self.assertEqual(training["sample_date"].tolist(), expected_train["sample_date"].tolist())
            self.assertEqual(set(training.columns).intersection(comparison.TARGET_NAMES), {target})
            self.assertTrue(set(training["sample_date"]).isdisjoint(validation["sample_date"]))
            self.assertEqual(boundary, validation["target_baseline_date"].iloc[0])
            self.assertTrue((pd.to_datetime(training["target_end_date"]) < pd.Timestamp(boundary)).all())

    def test_prediction_calls_receive_features_only_after_fitting(self):
        for model in self.models:
            self.assertEqual(model.predict.call_count, 2)
            first, second = model.predict.call_args_list
            self.assertEqual(len(first.args[0]), 60)
            self.assertEqual(len(second.args[0]), model.outer_training_rows)
            self.assertEqual(tuple(first.args[0].columns), FINAL_FEATURE_NAMES)

    def test_expected_shapes_and_candidate_audits(self):
        for name, count in comparison.TABLE_ROWS.items():
            self.assertEqual(len(self.artifacts[name]), count, name)
        candidates = self.artifacts["candidate_audit"]
        self.assertTrue(candidates.groupby(["target", "fold"]).size().eq(3).all())
        self.assertTrue(candidates.groupby(["target", "fold"])["selected"].sum().eq(1).all())
        self.assertEqual(set(candidates["architecture"]), {item.name for item in ARCHITECTURES})
        self.assertTrue(candidates["seed"].eq(SEED).all())
        self.assertEqual(candidates.groupby("fold")["outer_purge_rows"].first().tolist(), [4, 2, 4, 2, 3])
        self.assertTrue(candidates["outer_validation_rows"].eq(60).all())

    def test_authoritative_records_and_configurations_are_preserved(self):
        pd.testing.assert_frame_equal(self.artifacts["authoritative_predictions"], self.saved.predictions)
        pd.testing.assert_frame_equal(self.artifacts["ridge_fit_audit"], self.saved.ridge_fit_audit)
        table = self.artifacts["predictions"]
        preserved = table.loc[table["strategy"].isin(comparison.SAVED_STRATEGIES), list(comparison.SAVED_COLUMNS)].sort_values(list(comparison.KEY_COLUMNS)).reset_index(drop=True)
        original = self.saved.predictions.sort_values(list(comparison.KEY_COLUMNS)).reset_index(drop=True)
        pd.testing.assert_frame_equal(preserved, original, check_dtype=False, check_exact=True)
        self.assertFalse(table.duplicated(list(comparison.KEY_COLUMNS)).any())
        self.assertEqual(len(table), 18000)
        self.assertTrue(table.groupby(["target", "strategy"]).size().eq(300).all())

    def test_duplicate_missing_and_unexpected_keys_are_rejected(self):
        original = self.saved.predictions
        unexpected = original.copy()
        unexpected.loc[0, "sample_date"] = "1900-01-01"
        for invalid in (original.iloc[:-1], pd.concat([original, original.iloc[:1]]), unexpected):
            with self.assertRaisesRegex(ValueError, "keys"):
                comparison.validate_authoritative_predictions(self.frame, self.folds, invalid)

    def test_actual_unit_and_date_mismatches_are_rejected(self):
        for column, value in (("actual", 10000.0), ("unit", "wrong"), ("target_end_date", "2099-01-01"), ("target_baseline_date", "1900-01-01")):
            invalid = self.saved.predictions.copy()
            invalid.loc[0, column] = value
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, column):
                comparison.validate_authoritative_predictions(self.frame, self.folds, invalid)

    def test_task1_tolerance_allows_only_small_numeric_roundoff(self):
        near = self.saved.predictions.copy()
        near.loc[0, "actual"] += 1e-12
        comparison.validate_authoritative_predictions(self.frame, self.folds, near)
        near.loc[0, "actual"] += 1e-4
        with self.assertRaisesRegex(ValueError, "actual"):
            comparison.validate_authoritative_predictions(self.frame, self.folds, near)

    def test_pooled_metrics_and_fold_metrics_match_shared_conventions(self):
        table = self.artifacts["predictions"]
        for target in comparison.TARGET_NAMES:
            group = table.loc[table["target"].eq(target) & table["strategy"].eq(comparison.XGBOOST_STRATEGY)]
            pooled = self.artifacts["pooled_metrics"].loc[lambda df: df["target"].eq(target) & df["strategy"].eq(comparison.XGBOOST_STRATEGY)].iloc[0]
            self.assertAlmostEqual(pooled["mae"], float(np.abs(group["actual"] - group["prediction"]).mean()))
            self.assertAlmostEqual(pooled["rmse"], float(np.sqrt(np.square(group["actual"] - group["prediction"]).mean())))
            fold_metrics = self.artifacts["fold_metrics"].loc[lambda df: df["target"].eq(target) & df["strategy"].eq(comparison.XGBOOST_STRATEGY)]
            self.assertAlmostEqual(pooled["mae"], fold_metrics["mae"].mean())
            self.assertAlmostEqual(pooled["rmse"] ** 2, np.square(fold_metrics["rmse"]).mean())

    def test_fold_wins_stability_and_worst_fold_are_correct(self):
        for row in self.artifacts["family_comparison"].itertuples(index=False):
            metrics = self.artifacts["fold_metrics"].loc[lambda df: df["target"].eq(row.target)].pivot(index="fold", columns="strategy", values="mae")
            ridge, boosted = metrics[comparison.RIDGE_STRATEGY], metrics[comparison.XGBOOST_STRATEGY]
            self.assertEqual(row.xgboost_mae_fold_wins, int((boosted < ridge).sum()))
            self.assertEqual(row.ridge_mae_fold_wins, int((ridge < boosted).sum()))
            self.assertEqual(row.mae_fold_ties, int((ridge == boosted).sum()))
            self.assertEqual(row.xgboost_mae_fold_wins + row.ridge_mae_fold_wins + row.mae_fold_ties, 5)
            self.assertAlmostEqual(row.xgboost_worst_fold_mae, boosted.max())
            self.assertAlmostEqual(row.xgboost_fold_mae_std, boosted.std(ddof=1))

    def test_naive_comparisons_use_matching_target_units_and_errors(self):
        pooled = self.artifacts["pooled_metrics"].set_index(["target", "strategy"])
        for row in self.artifacts["naive_comparison"].itertuples(index=False):
            model = pooled.loc[(row.target, row.strategy)]
            baseline = pooled.loc[(row.target, row.benchmark)]
            self.assertEqual(row.unit, comparison._target_unit(row.target))
            self.assertEqual(row.model_mae, model["mae"])
            self.assertEqual(row.benchmark_mae, baseline["mae"])
            self.assertAlmostEqual(row.mae_improvement_pct, 100 * (baseline["mae"] - model["mae"]) / baseline["mae"])

    def test_undefined_correlations_and_constant_actuals_have_reasons(self):
        folds = self.artifacts["fold_metrics"]
        constant = folds.loc[folds["strategy"].eq("naive_zero")]
        self.assertTrue(constant["prediction_actual_correlation"].isna().all())
        self.assertTrue(constant["prediction_actual_correlation_undefined_reason"].eq("CONSTANT_PREDICTIONS").all())
        result = comparison._metrics(pd.DataFrame({"actual": [1.0, 1.0], "prediction": [2.0, 3.0]}))
        self.assertIsNone(result["oos_r_squared"])
        self.assertEqual(result["oos_r_squared_undefined_reason"], "CONSTANT_ACTUAL_VALUES")

    def test_regime_support_and_missing_ridge_training_metrics_are_explicit(self):
        table = self.artifacts["predictions"].copy()
        table["vix_level_points"] = 12.0
        ridge = self.saved.ridge_fit_audit.drop(columns=["training_mae", "training_rmse"])
        metrics = comparison.comparison_metric_tables(table, ridge, self.artifacts["xgboost_fit_audit"])
        unsupported = metrics["regime_metrics"].loc[lambda df: df["vix_regime"].ne("low_vix_lt_20")]
        self.assertTrue(unsupported["mae"].isna().all())
        self.assertTrue(unsupported["metric_status"].eq("INSUFFICIENT_SAMPLES_REQUIRES_20").all())
        missing = metrics["generalization"].loc[lambda df: df["strategy"].eq(comparison.RIDGE_STRATEGY)]
        self.assertTrue(missing["training_mae"].isna().all())
        self.assertTrue(missing["training_metric_status"].eq("UNAVAILABLE_IN_SAVED_EVIDENCE").all())

    def test_generalization_uses_saved_ridge_and_new_xgboost_training_metrics(self):
        result = self.artifacts["generalization"]
        self.assertTrue(result["training_metric_status"].eq("AVAILABLE").all())
        np.testing.assert_allclose(result["outer_minus_training_mae"], result["outer_validation_mae"] - result["training_mae"])

    def test_pretest_identity_mismatch_blocks_fitting(self):
        fitter = Mock(side_effect=AssertionError("Fitting should not begin"))
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self.build(fitter=fitter, approved_hash="0" * 64)
        fitter.assert_not_called()

    def test_fingerprint_preserves_row_column_order_and_values(self):
        expected = comparison.authorized_pretest_fingerprint(self.frame)
        self.assertEqual(len(expected), 64)
        self.assertNotEqual(expected, comparison.authorized_pretest_fingerprint(self.frame.iloc[::-1]))
        changed = self.frame.copy()
        changed.loc[0, FINAL_FEATURE_NAMES[0]] += 1e-8
        self.assertNotEqual(expected, comparison.authorized_pretest_fingerprint(changed))
        self.assertNotEqual(expected, comparison.authorized_pretest_fingerprint(self.frame, tuple(reversed(FINAL_FEATURE_NAMES))))

    def test_metadata_records_identity_versions_parameters_and_dependence(self):
        summary = self.artifacts["summary"]
        self.assertEqual(summary["observed_pretest_sha256"], comparison.authorized_pretest_fingerprint(self.frame))
        self.assertEqual(summary["source_identity"]["commit"], "synthetic-test")
        self.assertEqual(summary["pretest_serialization"]["columns"], ["sample_date", *FINAL_FEATURE_NAMES, *comparison.TARGET_NAMES])
        self.assertEqual(summary["pretest_serialization"]["format"], comparison.FINGERPRINT_FORMAT)
        self.assertEqual(summary["execution_settings"]["max_bin"], 128)
        self.assertEqual(summary["execution_settings"]["seed"], SEED)
        self.assertIn("overlap", summary["validation_dependence"])
        self.assertIn("xgboost", summary["software_versions"])
        self.assertEqual(len(summary["architectures"]), 3)
        self.assertFalse(summary["ridge_refitted"])

    def test_source_identity_includes_untracked_source_hashes(self):
        with patch.object(comparison.subprocess, "run", side_effect=[SimpleNamespace(stdout="commit\n"), SimpleNamespace(stdout="branch\n")]):
            identity = comparison._source_identity()
        self.assertEqual(identity["commit"], "commit")
        self.assertEqual(identity["branch"], "branch")
        self.assertEqual(len(identity["source_files_sha256"]), 8)
        self.assertTrue(all(len(value) == 64 for value in identity["source_files_sha256"].values()))

    def test_cached_checksum_mismatch_is_rejected(self):
        path = self.directory / "all_targets_fit_audit.csv"
        content = path.read_bytes()
        try:
            path.write_bytes(content + b"\n")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                comparison.load_authoritative_evidence(self.directory)
        finally:
            path.write_bytes(content)

    def test_evidence_path_cannot_escape_into_protected_matrix(self):
        path = self.directory / "all_targets_summary.json"
        content = path.read_bytes()
        try:
            metadata = json.loads(content)
            metadata["output_files"]["fit_audit"]["path"] = str(comparison.MATRIX_PATH)
            path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "inside"):
                comparison.load_authoritative_evidence(self.directory)
        finally:
            path.write_bytes(content)
        with self.assertRaisesRegex(ValueError, "hashing is forbidden"):
            comparison._sha256(comparison.MATRIX_PATH)

    def test_saved_fold_identity_mismatch_blocks_fitting(self):
        evidence = comparison.SavedComparisonEvidence(self.saved.predictions, self.saved.ridge_fit_audit, self.saved.fold_manifest.copy(), self.saved.hashes, self.saved.summary_sha256)
        evidence.fold_manifest.loc[0, "training_rows"] = 999
        with patch.object(comparison, "load_authoritative_evidence", return_value=evidence):
            fitter = Mock(side_effect=AssertionError("Fitting should not begin"))
            with self.assertRaisesRegex(ValueError, "cached folds"):
                self.build(fitter=fitter)
            fitter.assert_not_called()

    def test_nonfinite_model_predictions_are_rejected(self):
        def invalid_fit(training, features, **arguments):
            fitted = SyntheticFit(training, features, arguments["target"], arguments["outer_fold"], arguments["outer_validation_baseline"])
            fitted.predict = Mock(return_value=np.full(60, np.nan))
            return fitted
        with self.assertRaisesRegex(ValueError, "60 finite"):
            self.build(fitter=invalid_fit)

    def test_inconsistent_fitter_scope_is_rejected(self):
        def invalid_fit(training, features, **arguments):
            fitted = SyntheticFit(training, features, arguments["target"], arguments["outer_fold"], arguments["outer_validation_baseline"])
            fitted.outer_training_rows += 1
            return fitted
        with self.assertRaisesRegex(ValueError, "training count"):
            self.build(fitter=invalid_fit)

    def test_invalid_regime_support_setting_is_rejected_before_loading(self):
        with patch.object(comparison, "load_all_pretest_rows") as loader:
            with self.assertRaisesRegex(ValueError, "minimum"):
                comparison.build_all_target_comparison_artifacts(minimum_regime_samples=1)
            loader.assert_not_called()

    def test_persistence_writes_complete_checksummed_synthetic_output(self):
        output = self.directory / "synthetic-persist"
        paths = comparison.persist_comparison_artifacts(self.artifacts, output_dir=output)
        self.assertEqual(len(paths), 15)
        checksums = json.loads((output / "comparison_checksums.json").read_text(encoding="utf-8"))
        self.assertEqual(len(checksums["files"]), 14)
        for name, digest in checksums["files"].items():
            self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), digest)
        summary = json.loads((output / "comparison_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["table_rows"], comparison.TABLE_ROWS)
        self.assertNotIn("NaN", (output / "comparison_summary.json").read_text(encoding="utf-8"))

    def test_persistence_refuses_existing_output_and_protected_locations(self):
        existing = self.directory / "existing"
        existing.mkdir()
        marker = existing / "keep.txt"
        marker.write_text("preserve", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            comparison.persist_comparison_artifacts(self.artifacts, output_dir=existing)
        self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")
        for directory in (comparison.PROJECT_ROOT / "scenario_modeling" / "forbidden", comparison.RIDGE_EVIDENCE_ROOT, comparison.PROJECT_ROOT / "backend" / "forbidden"):
            with self.subTest(directory=directory), self.assertRaises(ValueError):
                comparison.persist_comparison_artifacts(self.artifacts, output_dir=directory)

    def test_persistence_validates_shapes_before_creating_output(self):
        invalid = dict(self.artifacts)
        invalid["predictions"] = self.artifacts["predictions"].iloc[:-1]
        output = self.directory / "bad-shape"
        with self.assertRaisesRegex(ValueError, "shape"):
            comparison.persist_comparison_artifacts(invalid, output_dir=output)
        self.assertFalse(output.exists())

    def test_failed_publication_cleans_only_its_owned_output(self):
        output = self.directory / "publication-failure"
        original_rename = Path.rename
        def fail_second(path, destination):
            if Path(destination).name == "comparison_xgboost_predictions.csv":
                raise OSError("Synthetic publication failure")
            return original_rename(path, destination)
        with patch.object(Path, "rename", new=fail_second):
            with self.assertRaisesRegex(OSError, "Synthetic publication"):
                comparison.persist_comparison_artifacts(self.artifacts, output_dir=output)
        self.assertFalse(output.exists())
        comparison.persist_comparison_artifacts(self.artifacts, output_dir=output)

    def test_task2_and_original_step12_interfaces_remain_available(self):
        self.assertIs(comparison.fit_target_xgboost, workflow.fit_target_xgboost)
        self.assertIn("outer_validation_baseline", inspect.signature(workflow.fit_target_xgboost).parameters)
        self.assertNotIn("outer_validation", inspect.signature(workflow.fit_target_xgboost).parameters)
        self.assertEqual(tuple(inspect.signature(workflow._fit_xgboost).parameters), ("training", "features", "strategy"))


if __name__ == "__main__":
    unittest.main()
