from __future__ import annotations

import math
import unittest

import pandas as pd

from scenario_modeling.src.feature_sources import asof_lookup, frame_to_csv_text
from scenario_modeling.src.state_features import (
    MACRO_STATE_FEATURES,
    MARKET_DERIVED_FEATURES,
    STATE_FEATURE_NAMES,
    STATE_OUTPUT_COLUMNS,
    build_state_artifacts,
    calculate_market_state_history,
)


class StateFeatureUnitTests(unittest.TestCase):
    def test_asof_lookup_never_selects_a_future_value(self) -> None:
        values = pd.Series(
            [1.0, 2.0],
            index=pd.to_datetime(["2024-01-02", "2024-01-04"]),
        )
        result = asof_lookup(
            values, pd.Series(["2024-01-03", "2024-01-04"])
        )
        self.assertEqual(result.values.tolist(), [1.0, 2.0])
        self.assertEqual(
            result.source_dates.tolist(), ["2024-01-02", "2024-01-04"]
        )
        self.assertEqual(result.age_calendar_days.tolist(), [1, 0])

    def test_market_history_uses_trailing_observations_only(self) -> None:
        index = pd.bdate_range("2020-01-01", periods=300)
        values = pd.Series(
            [100.0 + position for position in range(300)],
            index=index,
        )
        history = calculate_market_state_history(values)
        last = history.iloc[-1]
        self.assertAlmostEqual(
            last["return_5d_pct"],
            100.0 * (399.0 / 394.0 - 1.0),
        )
        self.assertAlmostEqual(
            last["ma_distance_50d_pct"],
            100.0 * (399.0 / values.iloc[-50:].mean() - 1.0),
        )
        self.assertLessEqual(history.index.max(), index.max())


class CachedStateAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifacts = build_state_artifacts()

    def test_all_rows_and_disciplined_feature_count_are_present(self) -> None:
        features = self.artifacts["features"]
        self.assertEqual(len(features), 585)
        self.assertEqual(tuple(features.columns), STATE_OUTPUT_COLUMNS)
        self.assertEqual(len(STATE_FEATURE_NAMES), 46)
        self.assertEqual(len(MACRO_STATE_FEATURES), 10)
        self.assertEqual(len(MARKET_DERIVED_FEATURES), 35)
        self.assertGreaterEqual(len(STATE_FEATURE_NAMES), 25)
        self.assertLessEqual(len(STATE_FEATURE_NAMES), 60)

    def test_state_values_are_complete_and_finite(self) -> None:
        features = self.artifacts["features"]
        values = features[list(STATE_FEATURE_NAMES)]
        self.assertFalse(values.isna().any().any())
        self.assertTrue(values.map(math.isfinite).all().all())

    def test_all_availability_dates_are_on_or_before_cutoff(self) -> None:
        summary = self.artifacts["summary"]
        self.assertTrue(
            all(
                summary[
                    "point_in_time_by_availability_column"
                ].values()
            )
        )
        self.assertTrue(
            summary["validations"][
                "macro_release_values_not_backfilled_before_availability"
            ]
        )
        self.assertTrue(
            summary["validations"][
                "rolling_features_end_on_or_before_state_cutoff"
            ]
        )

    def test_derived_macro_formulas_match_components(self) -> None:
        features = self.artifacts["features"]
        slope_difference = (
            features["yield_curve_slope_10y_2y_pct_points"]
            - (
                features["ten_year_yield_pct"]
                - features["two_year_yield_pct"]
            )
        ).abs()
        real_rate_difference = (
            features["real_rate_proxy_pct_points"]
            - (
                features["fed_funds_target_midpoint_pct"]
                - features["cpi_yoy_pct"]
            )
        ).abs()
        self.assertTrue(slope_difference.lt(1e-12).all())
        self.assertTrue(real_rate_difference.lt(1e-12).all())

    def test_state_is_strictly_before_each_operational_sample(self) -> None:
        features = self.artifacts["features"]
        self.assertTrue(
            (
                pd.to_datetime(features["state_cutoff_date"])
                < pd.to_datetime(features["sample_date"])
            ).all()
        )

    def test_oil_nonpositive_history_is_disclosed_not_hidden(self) -> None:
        diagnostics = self.artifacts["summary"]["market_diagnostics"]
        self.assertGreaterEqual(
            diagnostics["oil"]["nonpositive_raw_value_count"], 1
        )

    def test_serialization_is_repeatable(self) -> None:
        features = self.artifacts["features"]
        self.assertEqual(
            frame_to_csv_text(features),
            frame_to_csv_text(features.copy()),
        )


if __name__ == "__main__":
    unittest.main()
