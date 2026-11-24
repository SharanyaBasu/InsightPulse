from __future__ import annotations

import unittest

from scenario_modeling.src.leakage_audit import (
    build_leakage_audit_artifacts,
)
from scenario_modeling.src.model_matrix import OIL_VOLATILITY_FEATURE


class LeakageAuditAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifacts = build_leakage_audit_artifacts()

    def test_complete_leakage_checklist_passes(self) -> None:
        checklist = self.artifacts["leakage_checklist"]
        self.assertEqual(len(checklist), 16)
        self.assertTrue(checklist["passed"].all())
        self.assertEqual(
            self.artifacts["summary"]["training_approval"],
            "APPROVED_FOR_STEP_10_RIDGE_TRAINING",
        )

    def test_audit_corrects_and_rechecks_the_oil_feature(self) -> None:
        sequence = self.artifacts["summary"]["audit_sequence"]
        final_matrix = self.artifacts["final_matrix_artifacts"]["matrix"]
        review = self.artifacts["oil_volatility_review"]
        self.assertEqual(sequence["initial_feature_count"], 59)
        self.assertEqual(sequence["final_feature_count"], 58)
        self.assertTrue(sequence["re_audit_passed"])
        self.assertNotIn(OIL_VOLATILITY_FEATURE, final_matrix.columns)
        self.assertEqual(len(review), 5)
        self.assertEqual(
            set(review["nonpositive_source_date"]), {"2020-04-20"}
        )
        self.assertAlmostEqual(
            float(review["nonpositive_close"].iloc[0]),
            -37.630001068115234,
        )
        self.assertGreater(float(review["feature_value"].max()), 1196.0)

    def test_final_test_is_not_in_value_based_audit_scopes(self) -> None:
        summary = self.artifacts["summary"]["audit_scope"]
        self.assertEqual(summary["pre_final_test_feature_rows"], 470)
        self.assertEqual(summary["purged_pre_final_test_target_rows"], 464)
        self.assertEqual(summary["final_test_structural_rows"], 114)
        self.assertFalse(
            summary["final_test_value_based_analysis_performed"]
        )
        for name in (
            "feature_distributions",
            "target_distributions",
            "feature_correlations",
            "target_correlations",
            "feature_target_correlations",
            "scenario_shock_ranges",
            "extreme_observations",
        ):
            values = set(self.artifacts[name]["scope"].astype(str))
            self.assertTrue(
                values
                <= {
                    "PRE_FINAL_TEST_FEATURES_ONLY",
                    "PURGED_PRE_FINAL_TEST_TARGET_AUDIT",
                }
            )

    def test_all_eight_user_facing_shock_ranges_are_present(self) -> None:
        ranges = self.artifacts["scenario_shock_ranges"]
        self.assertEqual(len(ranges), 8)
        self.assertEqual(
            set(ranges["contract_input"]),
            {
                "fed_funds_rate_change_bps",
                "cpi_surprise_pct",
                "oil_price_change_pct",
                "gdp_growth_surprise_pct",
                "unemployment_change_pct",
                "pmi_change_points",
                "dxy_change_pct",
                "vix_change_points",
            },
        )
        self.assertTrue((ranges["nonzero_count"] >= 20).all())
        pmi = ranges[ranges["contract_input"] == "pmi_change_points"].iloc[0]
        self.assertTrue(bool(pmi["proxy"]))
        self.assertEqual(
            pmi["historical_feature"],
            "kc_fed_manufacturing_pmi_proxy_change_points",
        )

    def test_eda_tables_cover_every_final_feature_and_target(self) -> None:
        self.assertEqual(len(self.artifacts["feature_distributions"]), 58)
        self.assertEqual(len(self.artifacts["target_distributions"]), 12)
        self.assertEqual(len(self.artifacts["feature_correlations"]), 1653)
        self.assertEqual(len(self.artifacts["target_correlations"]), 66)
        self.assertEqual(
            len(self.artifacts["feature_target_correlations"]), 696
        )
        self.assertEqual(len(self.artifacts["extreme_observations"]), 140)
        self.assertEqual(len(self.artifacts["regime_coverage"]), 10)

    def test_boundary_purges_pass_target_overlap_checks(self) -> None:
        validations = self.artifacts["summary"]["split"]["validations"]
        self.assertTrue(
            validations[
                "development_labels_end_before_validation_baseline"
            ]
        )
        self.assertTrue(
            validations[
                "validation_labels_end_before_final_test_baseline"
            ]
        )
        self.assertTrue(
            validations[
                "purge_uses_target_end_dates_not_event_row_counts"
            ]
        )


if __name__ == "__main__":
    unittest.main()
