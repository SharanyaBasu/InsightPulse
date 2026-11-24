"""Task 2 tests using synthetic in-memory observations exclusively."""

from __future__ import annotations

import inspect
import unittest
from dataclasses import asdict, replace
from unittest.mock import patch

import numpy as np
import pandas as pd

from scenario_modeling.src import walk_forward_validation as workflow
from scenario_modeling.src.model_matrix import FINAL_FEATURE_NAMES
from scenario_modeling.src.sampling import TARGET_SPECS
from scenario_modeling.src.xgboost_prototype import ARCHITECTURES, SEED


def synthetic_training() -> pd.DataFrame:
    """Create 140 observations with synthetic 21-business-session windows."""
    rng = np.random.default_rng(20260725)
    values = rng.normal(size=(140, len(FINAL_FEATURE_NAMES)))
    frame = pd.DataFrame(values, columns=FINAL_FEATURE_NAMES)
    for flag in (
        "fomc_release_flag", "cpi_release_flag", "unemployment_release_flag",
        "gdp_release_flag", "pmi_release_flag",
    ):
        frame[flag] = rng.integers(0, 2, len(frame))
    baselines = pd.bdate_range("2000-01-03", periods=len(frame))
    frame["sample_date"] = (baselines + pd.offsets.BDay(1)).strftime("%Y-%m-%d")
    frame["target_baseline_date"] = baselines.strftime("%Y-%m-%d")
    frame["target_end_date"] = (baselines + pd.offsets.BDay(21)).strftime("%Y-%m-%d")
    signal = 0.8 * values[:, 0] - 0.5 * values[:, 1] + rng.normal(0, 0.05, len(frame))
    for number, spec in enumerate(TARGET_SPECS, 1):
        frame[spec.target] = signal * number
    return frame


def outer_boundary(frame: pd.DataFrame) -> str:
    """Provide only a synthetic next-validation baseline date."""
    return (pd.to_datetime(frame["target_end_date"]).max() + pd.offsets.BDay(1)).date().isoformat()


class FakeBooster:
    def __init__(self, score: float, best_iteration: int) -> None:
        self.best_score = str(score)
        self.best_iteration = best_iteration

    def predict(self, matrix, *, iteration_range):
        return np.full(matrix.num_row(), 0.25)


class TargetXGBoostFittingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = synthetic_training()

    def mocked_fit(self, *, frame=None, scores=None, iterations=None, **kwargs):
        data = self.frame if frame is None else frame
        candidate_scores = scores or {
            "regularized_stumps": 1.009,
            "regularized_depth2": 1.0,
            "strongly_regularized_depth2": 1.001,
        }
        candidate_iterations = iterations or {item.name: 6 for item in ARCHITECTURES}

        def train(parameters, matrix, **options):
            architecture = next(
                item for item in ARCHITECTURES
                if item.reg_alpha == parameters["alpha"]
            )
            return FakeBooster(
                candidate_scores[architecture.name],
                candidate_iterations[architecture.name],
            )

        arguments = {
            "target": "sp500",
            "outer_validation_baseline": outer_boundary(self.frame),
            "outer_fold": 2,
            **kwargs,
        }
        with patch.object(workflow.xgb, "train", side_effect=train) as trainer:
            fitted = workflow.fit_target_xgboost(
                data, FINAL_FEATURE_NAMES, **arguments
            )
        return fitted, trainer.call_args_list

    def test_all_twelve_targets_use_the_explicit_selected_labels(self) -> None:
        for spec in TARGET_SPECS:
            with self.subTest(target=spec.target):
                fitted, calls = self.mocked_fit(target=spec.target)
                inner_train, inner_validation, _ = workflow._inner_split(self.frame)
                np.testing.assert_allclose(
                    calls[0].args[1].get_label(), inner_train[spec.target], rtol=1e-6
                )
                np.testing.assert_allclose(
                    calls[0].kwargs["evals"][-1][0].get_label(),
                    inner_validation[spec.target], rtol=1e-6,
                )
                self.assertEqual(fitted.target, spec.target)
                expected_unit = "basis_points" if spec.target == "ten_year_yield_bps" else "percent"
                self.assertEqual(fitted.unit, expected_unit)
                self.assertEqual(len(calls), 4)

    def test_feature_order_is_enforced_for_training_and_prediction(self) -> None:
        shuffled = self.frame.loc[:, list(reversed(self.frame.columns))]
        fitted, calls = self.mocked_fit(frame=shuffled)
        self.assertEqual(fitted.feature_names, FINAL_FEATURE_NAMES)
        self.assertEqual(calls[0].args[1].feature_names, list(FINAL_FEATURE_NAMES))
        self.assertEqual(calls[-1].args[1].feature_names, list(FINAL_FEATURE_NAMES))
        with patch.object(fitted.booster, "predict", wraps=fitted.booster.predict) as predictor:
            fitted.predict(shuffled)
        self.assertEqual(predictor.call_args.args[0].feature_names, list(FINAL_FEATURE_NAMES))
        self.assertEqual(predictor.call_args.kwargs["iteration_range"], (0, 7))

    def test_wrong_feature_order_is_rejected_before_training(self) -> None:
        with patch.object(workflow.xgb, "train") as trainer:
            with self.assertRaisesRegex(ValueError, "58-feature order"):
                workflow.fit_target_xgboost(
                    self.frame, tuple(reversed(FINAL_FEATURE_NAMES)), target="sp500",
                    outer_validation_baseline=outer_boundary(self.frame),
                )
            trainer.assert_not_called()

    def test_inner_chronology_and_target_end_purge_are_preserved(self) -> None:
        fitted, calls = self.mocked_fit()
        inner_train, inner_validation, purge = workflow._inner_split(self.frame)
        self.assertEqual((len(inner_train), len(inner_validation), purge), (91, 28, 21))
        self.assertEqual(fitted.inner_training_rows, 91)
        self.assertEqual(fitted.inner_validation_rows, 28)
        self.assertEqual(fitted.inner_purge_rows, 21)
        self.assertTrue((
            pd.to_datetime(inner_train["target_end_date"])
            < pd.Timestamp(inner_validation["target_baseline_date"].iloc[0])
        ).all())
        self.assertLess(
            fitted.boundaries["inner_training_last_target_end"],
            fitted.boundaries["inner_validation_first_baseline"],
        )
        for call in calls[:3]:
            self.assertEqual(call.args[1].num_row(), 91)
            self.assertEqual(call.kwargs["evals"][-1][0].num_row(), 28)

    def test_outer_label_overlap_is_rejected_before_training(self) -> None:
        with patch.object(workflow.xgb, "train") as trainer:
            with self.assertRaisesRegex(ValueError, "overlaps outer validation"):
                self.mocked_fit(
                    outer_validation_baseline=self.frame["target_end_date"].max()
                )
            trainer.assert_not_called()

    def test_audit_preserves_all_three_frozen_architectures_and_settings(self) -> None:
        fitted, calls = self.mocked_fit()
        self.assertEqual(tuple(item.architecture for item in fitted.candidates), ARCHITECTURES)
        self.assertEqual([item.architecture.complexity_rank for item in fitted.candidates], [1, 2, 3])
        for audit, call in zip(fitted.candidates, calls[:3], strict=True):
            architecture = audit.architecture
            parameters = call.args[0]
            self.assertEqual(parameters, audit.parameters)
            self.assertEqual(parameters["max_depth"], architecture.max_depth)
            self.assertEqual(parameters["eta"], architecture.learning_rate)
            self.assertEqual(parameters["min_child_weight"], architecture.min_child_weight)
            self.assertEqual(parameters["subsample"], architecture.subsample)
            self.assertEqual(parameters["colsample_bytree"], architecture.colsample_bytree)
            self.assertEqual(parameters["alpha"], architecture.reg_alpha)
            self.assertEqual(parameters["lambda"], architecture.reg_lambda)
            self.assertEqual(parameters["gamma"], architecture.gamma)
            self.assertEqual(parameters["max_bin"], 128)
            self.assertEqual(parameters["seed"], SEED)
            self.assertEqual(parameters["nthread"], 1)
            self.assertEqual(parameters["device"], "cpu")
            self.assertEqual(parameters["tree_method"], "hist")
            self.assertEqual(parameters["objective"], "reg:squarederror")
            self.assertEqual(parameters["eval_metric"], "mae")
            self.assertEqual(call.kwargs["num_boost_round"], architecture.max_rounds)
            self.assertEqual(call.kwargs["early_stopping_rounds"], architecture.early_stopping_rounds)
        self.assertEqual(fitted.seed, SEED)
        self.assertEqual(fitted.outer_fold, 2)
        self.assertEqual(fitted.outer_training_rows, 140)
        self.assertEqual(fitted.boundaries["outer_training_first_sample"], self.frame["sample_date"].iloc[0])
        self.assertEqual(fitted.boundaries["outer_training_last_sample"], self.frame["sample_date"].iloc[-1])
        self.assertEqual(fitted.boundaries["inner_validation_last_sample"], self.frame["sample_date"].iloc[-1])

    def test_one_percent_practical_tie_prefers_simpler_architecture(self) -> None:
        fitted, _ = self.mocked_fit()
        self.assertEqual(fitted.selected_candidate.architecture.name, "regularized_stumps")
        self.assertAlmostEqual(fitted.practical_tie_mae_limit, 1.01)
        self.assertEqual([item.inner_mae for item in fitted.candidates], [1.009, 1.0, 1.001])

    def test_clear_non_tie_prefers_lower_inner_mae(self) -> None:
        fitted, _ = self.mocked_fit(scores={
            "regularized_stumps": 1.02,
            "regularized_depth2": 1.0,
            "strongly_regularized_depth2": 1.1,
        })
        self.assertEqual(fitted.selected_candidate.architecture.name, "regularized_depth2")

    def test_practical_tie_boundary_is_inclusive(self) -> None:
        fitted, _ = self.mocked_fit(scores={
            "regularized_stumps": 1.01,
            "regularized_depth2": 1.0,
            "strongly_regularized_depth2": 2.0,
        })
        self.assertEqual(fitted.selected_candidate.architecture.name, "regularized_stumps")

    def test_boosting_rounds_use_best_iteration_plus_one_during_refit(self) -> None:
        for best_iteration in (0, ARCHITECTURES[0].max_rounds - 1):
            with self.subTest(best_iteration=best_iteration):
                fitted, calls = self.mocked_fit(
                    iterations={item.name: best_iteration for item in ARCHITECTURES}
                )
                self.assertEqual(fitted.selected_candidate.best_iteration, best_iteration)
                self.assertEqual(fitted.selected_candidate.boosting_rounds, best_iteration + 1)
                self.assertEqual(calls[-1].kwargs["num_boost_round"], best_iteration + 1)
                self.assertEqual(calls[-1].args[1].num_row(), len(self.frame))
                self.assertNotIn("evals", calls[-1].kwargs)
                self.assertNotIn("early_stopping_rounds", calls[-1].kwargs)

    def test_invalid_best_iteration_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid best iteration"):
            self.mocked_fit(iterations={item.name: -1 for item in ARCHITECTURES})

    def test_base_scores_use_only_their_current_training_scope(self) -> None:
        fitted, calls = self.mocked_fit(target="gold")
        inner_train, _, _ = workflow._inner_split(self.frame)
        for call in calls[:3]:
            self.assertEqual(call.args[0]["base_score"], float(inner_train["gold"].mean()))
        self.assertEqual(fitted.refit_parameters["base_score"], float(self.frame["gold"].mean()))

    def test_outer_validation_frame_is_not_an_accepted_argument(self) -> None:
        parameters = inspect.signature(workflow.fit_target_xgboost).parameters
        self.assertNotIn("outer_validation", parameters)
        with patch.object(workflow.xgb, "train") as trainer:
            with self.assertRaises(TypeError):
                workflow.fit_target_xgboost(
                    self.frame, FINAL_FEATURE_NAMES, target="sp500",
                    outer_validation_baseline=outer_boundary(self.frame),
                    outer_validation=pd.DataFrame({"sp500": [1e20]}),
                )
            trainer.assert_not_called()

    def test_unselected_target_and_extra_columns_are_never_read(self) -> None:
        poisoned = self.frame.copy()
        poisoned["sp500"] = [object() for _ in range(len(poisoned))]
        poisoned["outer_validation_target"] = [object() for _ in range(len(poisoned))]
        with patch.object(workflow, "load_development_and_validation", side_effect=AssertionError("Historical loading forbidden")):
            original, _ = self.mocked_fit(target="nasdaq")
            changed, _ = self.mocked_fit(frame=poisoned, target="nasdaq")
        self.assertEqual(asdict(original.selected_candidate), asdict(changed.selected_candidate))
        np.testing.assert_array_equal(original.predict(self.frame), changed.predict(poisoned))

    def test_invalid_targets_and_invalid_training_values_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown Scenario Playground target"):
            self.mocked_fit(target="unknown")
        for column in (FINAL_FEATURE_NAMES[0], "sp500"):
            invalid = self.frame.copy()
            invalid.loc[0, column] = np.inf
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "finite"):
                self.mocked_fit(frame=invalid)
        missing_label = self.frame.copy()
        missing_label.loc[0, "sp500"] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            self.mocked_fit(frame=missing_label)

    def test_empty_duplicate_and_unordered_observations_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            self.mocked_fit(frame=self.frame.iloc[:0])
        duplicate = self.frame.copy()
        duplicate.loc[1, "sample_date"] = duplicate.loc[0, "sample_date"]
        for invalid in (duplicate, self.frame.iloc[::-1]):
            with self.assertRaisesRegex(ValueError, "unique and chronological"):
                self.mocked_fit(frame=invalid)

    def test_native_feature_missing_values_are_retained(self) -> None:
        missing = self.frame.copy()
        missing.loc[0, FINAL_FEATURE_NAMES[0]] = np.nan
        _, calls = self.mocked_fit(frame=missing)
        self.assertEqual(calls[0].args[1].num_nonmissing(), 91 * 58 - 1)

    def test_selection_rejects_nonfinite_or_negative_mae(self) -> None:
        fitted, _ = self.mocked_fit()
        for score in (float("nan"), float("inf"), -1.0):
            with self.subTest(score=score), self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                workflow.select_xgboost_candidate([
                    replace(fitted.candidates[0], inner_mae=score)
                ])

    def test_legacy_sp500_fit_retains_original_step12_parameters(self) -> None:
        with patch.object(workflow.xgb, "train", return_value=FakeBooster(1.0, 3)) as trainer:
            fitted = workflow._fit_xgboost(self.frame, FINAL_FEATURE_NAMES, "full_history")
        self.assertEqual(trainer.call_count, 2)
        initial, refit = trainer.call_args_list
        self.assertEqual(initial.kwargs["num_boost_round"], 500)
        self.assertEqual(initial.kwargs["early_stopping_rounds"], 40)
        self.assertNotIn("max_bin", initial.args[0])
        self.assertEqual(initial.args[0]["max_depth"], 1)
        self.assertEqual(refit.kwargs["num_boost_round"], 4)
        self.assertEqual(fitted.selected_parameter, 4)
        np.testing.assert_array_equal(fitted.predict(self.frame), np.full(len(self.frame), 0.25))

    def test_real_synthetic_fits_are_deterministic_and_refit_exact_rounds(self) -> None:
        data = self.frame.copy()
        data.loc[0, FINAL_FEATURE_NAMES[0]] = np.nan
        kwargs = {
            "target": "ten_year_yield_bps",
            "outer_validation_baseline": outer_boundary(data),
            "outer_fold": 1,
        }
        first = workflow.fit_target_xgboost(data, FINAL_FEATURE_NAMES, **kwargs)
        second = workflow.fit_target_xgboost(data, FINAL_FEATURE_NAMES, **kwargs)
        self.assertEqual(first.unit, "basis_points")
        self.assertEqual(asdict(first.selected_candidate), asdict(second.selected_candidate))
        self.assertEqual(first.booster.num_boosted_rounds(), first.selected_candidate.boosting_rounds)
        self.assertEqual(first.booster.feature_names, list(FINAL_FEATURE_NAMES))
        prediction = first.predict(data)
        self.assertTrue(np.isfinite(prediction).all())
        np.testing.assert_array_equal(prediction, second.predict(data))


if __name__ == "__main__":
    unittest.main()
