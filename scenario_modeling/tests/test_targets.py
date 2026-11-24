from __future__ import annotations

import math
import unittest

import pandas as pd

from scenario_modeling.src.targets import (
    TARGET_HORIZON_SESSIONS,
    TARGET_NAMES,
    TARGET_OUTPUT_COLUMNS,
    TARGET_SPECS,
    _exact_values,
    _frame_to_csv_text,
    build_target_artifacts,
    calculate_price_return_pct,
    calculate_yield_change_bps,
)


class TargetFormulaUnitTests(unittest.TestCase):
    def test_price_return_formula_uses_percent_units(self) -> None:
        baseline = pd.Series([100.0, 200.0])
        future = pd.Series([121.0, 180.0])
        result = calculate_price_return_pct(baseline, future)
        self.assertAlmostEqual(result.iloc[0], 21.0)
        self.assertAlmostEqual(result.iloc[1], -10.0)

    def test_yield_conversion_is_percentage_points_to_basis_points(self) -> None:
        baseline = pd.Series([3.90, 4.05])
        future = pd.Series([4.05, 3.90])
        result = calculate_yield_change_bps(baseline, future)
        self.assertAlmostEqual(result.iloc[0], 15.0)
        self.assertAlmostEqual(result.iloc[1], -15.0)

    def test_zero_price_baseline_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "baseline is zero"):
            calculate_price_return_pct(
                pd.Series([0.0]), pd.Series([10.0])
            )

    def test_exact_lookup_does_not_forward_fill(self) -> None:
        source = pd.Series(
            [100.0, 102.0],
            index=["2024-01-02", "2024-01-04"],
        )
        with self.assertRaisesRegex(ValueError, "exact baseline"):
            _exact_values(
                source,
                pd.Series(["2024-01-03"]),
                target_name="test",
                endpoint_name="baseline",
            )

    def test_adjustment_policies_match_contract(self) -> None:
        fields = {spec.target: spec.value_field for spec in TARGET_SPECS}
        for target in (
            "gold",
            "technology",
            "energy",
            "financials",
            "utilities",
            "healthcare",
            "consumer_discretionary",
        ):
            self.assertEqual(fields[target], "Adj Close")
        for target in ("sp500", "nasdaq", "dxy", "oil"):
            self.assertEqual(fields[target], "Close")
        self.assertEqual(fields["ten_year_yield_bps"], "value")


class CachedTargetAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifacts = build_target_artifacts()

    def test_all_selected_rows_and_contract_targets_are_present(self) -> None:
        targets = self.artifacts["targets"]
        self.assertEqual(len(targets), 585)
        self.assertEqual(tuple(targets.columns), TARGET_OUTPUT_COLUMNS)
        self.assertEqual(
            tuple(column for column in targets if column in TARGET_NAMES),
            TARGET_NAMES,
        )
        self.assertFalse(targets[list(TARGET_NAMES)].isna().any().any())
        self.assertTrue(
            targets[list(TARGET_NAMES)].map(math.isfinite).all().all()
        )

    def test_every_target_window_is_exactly_21_sessions(self) -> None:
        summary = self.artifacts["summary"]
        self.assertEqual(
            summary["horizon_validation"]["minimum_session_gap"],
            TARGET_HORIZON_SESSIONS,
        )
        self.assertEqual(
            summary["horizon_validation"]["maximum_session_gap"],
            TARGET_HORIZON_SESSIONS,
        )
        self.assertTrue(
            summary["validations"][
                "all_target_windows_are_exactly_21_sessions"
            ]
        )

    def test_independent_manual_checks_pass_and_known_rows_match(self) -> None:
        checks = self.artifacts["manual_checks"]
        self.assertEqual(len(checks), 10)
        self.assertTrue(checks["passed"].all())

        targets = self.artifacts["targets"]
        first = targets.iloc[0]
        pandemic = targets[
            targets["sample_date"] == "2020-02-21"
        ].iloc[0]
        first_yield = first["ten_year_yield_bps"]
        self.assertAlmostEqual(first["sp500"], -4.3295637409683795)
        self.assertAlmostEqual(
            pandemic["sp500"], -28.845740381710596
        )
        self.assertAlmostEqual(first_yield, 4.0)

    def test_extreme_values_are_retained_for_review_not_removed(self) -> None:
        extremes = self.artifacts["extremes"]
        self.assertEqual(len(extremes), 4)
        self.assertEqual(
            extremes.groupby("target").size().to_dict(),
            {"energy": 1, "oil": 2, "ten_year_yield_bps": 1},
        )
        self.assertTrue(
            (
                extremes["action"]
                == "retained_for_documented_review"
            ).all()
        )
        self.assertEqual(len(self.artifacts["targets"]), 585)

    def test_distributions_are_complete_and_finite(self) -> None:
        distributions = self.artifacts["summary"]["distributions"]
        self.assertEqual(set(distributions), set(TARGET_NAMES))
        for record in distributions.values():
            self.assertEqual(record["count"], 585)
            self.assertEqual(record["missing_count"], 0)
            self.assertEqual(record["non_finite_count"], 0)

    def test_output_order_and_serialization_are_repeatable(self) -> None:
        targets = self.artifacts["targets"]
        self.assertTrue(targets["sample_date"].is_monotonic_increasing)
        self.assertFalse(targets["sample_date"].duplicated().any())
        self.assertEqual(
            _frame_to_csv_text(targets),
            _frame_to_csv_text(targets.copy()),
        )


if __name__ == "__main__":
    unittest.main()
