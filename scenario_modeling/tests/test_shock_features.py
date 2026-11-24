from __future__ import annotations

import math
import unittest

import pandas as pd

from scenario_modeling.src.feature_sources import AlfredVintagePanel
from scenario_modeling.src.feature_sources import frame_to_csv_text
from scenario_modeling.src.shock_features import (
    CONTRACT_TO_HISTORICAL_FEATURE,
    EVENT_FLAG_NAMES,
    HISTORICAL_SHOCK_NAMES,
    SHOCK_OUTPUT_COLUMNS,
    build_shock_artifacts,
    calculate_cpi_mom_actual,
    calculate_percent_change,
    calculate_point_change,
)


class ShockFormulaUnitTests(unittest.TestCase):
    def test_percent_and_point_units_are_distinct(self) -> None:
        self.assertAlmostEqual(calculate_percent_change(100.0, 105.0), 5.0)
        self.assertAlmostEqual(calculate_point_change(15.0, 20.0), 5.0)
        with self.assertRaisesRegex(ValueError, "start value is zero"):
            calculate_percent_change(0.0, 1.0)

    def test_cpi_actual_uses_same_vintage_current_and_prior_levels(self) -> None:
        panel = AlfredVintagePanel(
            "CPIAUCSL",
            {
                ("2024-01-01", "2024-03-01"): 100.0,
                ("2024-02-01", "2024-03-01"): 100.3,
            },
            {},
            [],
        )
        actual, current, previous = calculate_cpi_mom_actual(
            panel, "2024-02-01", "2024-03-01"
        )
        self.assertAlmostEqual(actual, 0.3)
        self.assertEqual(current, 100.3)
        self.assertEqual(previous, 100.0)

    def test_regional_pmi_proxy_name_is_not_silently_nationalized(self) -> None:
        self.assertEqual(
            CONTRACT_TO_HISTORICAL_FEATURE["pmi_change_points"],
            "kc_fed_manufacturing_pmi_proxy_change_points",
        )


class CachedShockAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifacts = build_shock_artifacts()

    def test_all_selected_rows_and_feature_columns_are_present(self) -> None:
        features = self.artifacts["features"]
        self.assertEqual(len(features), 585)
        self.assertEqual(tuple(features.columns), SHOCK_OUTPUT_COLUMNS)
        self.assertEqual(len(HISTORICAL_SHOCK_NAMES), 8)
        self.assertEqual(len(EVENT_FLAG_NAMES), 5)
        self.assertTrue(features["sample_date"].is_monotonic_increasing)
        self.assertFalse(features["sample_date"].duplicated().any())

    def test_only_documented_cpi_events_are_missing(self) -> None:
        features = self.artifacts["features"]
        missing = features[list(HISTORICAL_SHOCK_NAMES)].isna().sum()
        self.assertEqual(int(missing["cpi_surprise_pct"]), 1)
        self.assertEqual(int(missing.drop("cpi_surprise_pct").sum()), 0)
        nonmissing = (
            features[list(HISTORICAL_SHOCK_NAMES)]
            .stack(future_stack=True)
            .dropna()
        )
        self.assertTrue(nonmissing.map(math.isfinite).all())

    def test_event_counts_flags_and_audit_records_match(self) -> None:
        summary = self.artifacts["summary"]
        self.assertEqual(
            summary["event_counts"],
            {
                "cpi": 152,
                "fomc": 104,
                "gdp": 51,
                "pmi": 154,
                "unemployment": 154,
            },
        )
        self.assertEqual(summary["event_audit_row_count"], 615)
        self.assertEqual(
            summary["event_counts"], summary["release_flag_counts"]
        )
        self.assertTrue(
            self.artifacts["event_audit"]["point_in_time_pass"].all()
        )

    def test_true_surprises_and_proxy_disclosure_are_preserved(self) -> None:
        audit = self.artifacts["event_audit"]
        cpi = audit[audit["event_type"] == "cpi"]
        available_cpi = cpi[cpi["feature_available"]]
        self.assertEqual(len(available_cpi), 151)
        self.assertTrue(
            (
                pd.to_datetime(available_cpi["expectation_date"])
                < pd.to_datetime(available_cpi["original_release_date"])
            ).all()
        )
        gdp = audit[audit["event_type"] == "gdp"]
        self.assertTrue(
            (
                gdp["shock_value"]
                - (gdp["actual_value"] - gdp["comparator_value"])
            )
            .abs()
            .lt(1e-12)
            .all()
        )
        pmi = audit[audit["event_type"] == "pmi"]
        self.assertTrue(pmi["proxy_flag"].all())
        self.assertTrue(
            (
                pmi["shock_feature"]
                == "kc_fed_manufacturing_pmi_proxy_change_points"
            ).all()
        )

    def test_emergency_fomc_shocks_are_recreated(self) -> None:
        audit = self.artifacts["event_audit"]
        fomc = audit[audit["event_type"] == "fomc"].set_index(
            "original_release_date"
        )
        self.assertAlmostEqual(fomc.loc["2020-03-03", "shock_value"], -50.0)
        self.assertAlmostEqual(fomc.loc["2020-03-15", "shock_value"], -100.0)

    def test_market_windows_never_use_future_source_values(self) -> None:
        audit = self.artifacts["market_audit"]
        self.assertEqual(len(audit), 585 * 3)
        self.assertTrue(audit["point_in_time_pass"].all())
        self.assertTrue(
            (
                pd.to_datetime(audit["window_start_source_date"])
                <= pd.to_datetime(audit["window_start_boundary"])
            ).all()
        )
        self.assertTrue(
            (
                pd.to_datetime(audit["window_end_source_date"])
                <= pd.to_datetime(audit["window_end_boundary"])
            ).all()
        )

    def test_serialization_is_repeatable(self) -> None:
        features = self.artifacts["features"]
        self.assertEqual(
            frame_to_csv_text(features),
            frame_to_csv_text(features.copy()),
        )


if __name__ == "__main__":
    unittest.main()
