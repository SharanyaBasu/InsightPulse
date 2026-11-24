from __future__ import annotations

import math
import unittest

import pandas as pd

from scenario_modeling.src.feature_sources import frame_to_csv_text
from scenario_modeling.src.model_matrix import (
    FINAL_FEATURE_NAMES,
    INITIAL_FEATURE_NAMES,
    OIL_VOLATILITY_FEATURE,
    PROVENANCE_COLUMNS,
    SOURCE_ROW_AUDIT_COLUMNS,
    build_model_matrix_artifacts,
)
from scenario_modeling.src.targets import TARGET_NAMES


class PointInTimeMatrixAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifacts = build_model_matrix_artifacts()

    def test_final_raw_matrix_shape_and_roles_are_exact(self) -> None:
        matrix = self.artifacts["matrix"]
        self.assertEqual(matrix.shape, (584, 71))
        self.assertEqual(len(FINAL_FEATURE_NAMES), 58)
        self.assertEqual(len(TARGET_NAMES), 12)
        self.assertEqual(
            tuple(matrix.columns),
            ("sample_date", *FINAL_FEATURE_NAMES, *TARGET_NAMES),
        )
        self.assertNotIn(OIL_VOLATILITY_FEATURE, matrix.columns)

    def test_missing_cpi_row_is_excluded_without_imputation(self) -> None:
        exclusions = self.artifacts["row_exclusions"]
        matrix = self.artifacts["matrix"]
        self.assertEqual(len(exclusions), 1)
        self.assertEqual(exclusions.iloc[0]["sample_date"], "2013-08-16")
        self.assertEqual(
            exclusions.iloc[0]["column_name"], "cpi_surprise_pct"
        )
        self.assertFalse(bool(exclusions.iloc[0]["imputation_performed"]))
        self.assertNotIn("2013-08-16", set(matrix["sample_date"]))

    def test_matrix_is_complete_finite_raw_and_unscaled(self) -> None:
        matrix = self.artifacts["matrix"]
        values = matrix[[*FINAL_FEATURE_NAMES, *TARGET_NAMES]]
        self.assertFalse(values.isna().any().any())
        self.assertTrue(values.map(math.isfinite).all().all())
        self.assertTrue(
            self.artifacts["summary"]["validations"][
                "model_matrix_values_are_raw_source_values"
            ]
        )
        self.assertTrue(
            self.artifacts["summary"]["validations"][
                "no_imputation_scaling_clipping_or_winsorization"
            ]
        )

    def test_provenance_and_audit_columns_are_separate(self) -> None:
        matrix = self.artifacts["matrix"]
        provenance = self.artifacts["provenance"]
        source_audit = self.artifacts["source_row_audit"]
        self.assertEqual(tuple(provenance.columns), PROVENANCE_COLUMNS)
        self.assertEqual(
            tuple(source_audit.columns), SOURCE_ROW_AUDIT_COLUMNS
        )
        self.assertFalse(
            (set(PROVENANCE_COLUMNS) - {"sample_date"})
            & set(matrix.columns)
        )
        self.assertFalse(
            (set(SOURCE_ROW_AUDIT_COLUMNS) - {"sample_date"})
            & set(matrix.columns)
        )

    def test_all_target_horizons_are_exactly_21_sessions(self) -> None:
        self.assertTrue(
            self.artifacts["summary"]["validations"][
                "all_targets_span_exactly_21_canonical_sessions"
            ]
        )
        self.assertTrue(
            (
                self.artifacts["provenance"]["target_horizon_sessions"]
                == 21
            ).all()
        )

    def test_chronological_split_and_target_end_purge_are_exact(self) -> None:
        manifest = self.artifacts["split_manifest"]
        counts = manifest["modeling_role"].value_counts().to_dict()
        self.assertEqual(counts["development_train"], 354)
        self.assertEqual(counts["development_boundary_purge"], 4)
        self.assertEqual(counts["validation"], 110)
        self.assertEqual(counts["validation_boundary_purge"], 2)
        self.assertEqual(counts["final_test_locked"], 114)

        development = manifest[
            manifest["modeling_role"] == "development_train"
        ]
        validation = manifest[manifest["modeling_role"] == "validation"]
        self.assertTrue(
            (
                pd.to_datetime(development["target_end_date"])
                < pd.Timestamp("2021-05-06")
            ).all()
        )
        self.assertTrue(
            (
                pd.to_datetime(validation["target_end_date"])
                < pd.Timestamp("2023-12-07")
            ).all()
        )
        self.assertEqual(
            int(manifest["purged_for_target_overlap"].sum()), 6
        )
        self.assertNotEqual(
            int(manifest["purged_for_target_overlap"].sum()), 21
        )

    def test_every_matrix_column_has_definition_and_provenance(self) -> None:
        catalog = self.artifacts["column_catalog"]
        matrix = self.artifacts["matrix"]
        cataloged = catalog[
            catalog["storage_artifact"]
            == "scenario_model_matrix_raw.csv"
        ]
        self.assertEqual(set(cataloged["column_name"]), set(matrix.columns))
        self.assertFalse(
            cataloged[
                [
                    "source",
                    "source_artifact",
                    "definition_or_formula",
                    "point_in_time_rule",
                ]
            ]
            .isna()
            .any()
            .any()
        )

    def test_pre_audit_matrix_differs_by_only_the_oil_feature(self) -> None:
        initial = build_model_matrix_artifacts(excluded_features=())
        self.assertEqual(len(INITIAL_FEATURE_NAMES), 59)
        self.assertIn(OIL_VOLATILITY_FEATURE, initial["matrix"].columns)
        self.assertTrue(
            initial["matrix"]
            .drop(columns=[OIL_VOLATILITY_FEATURE])
            .equals(self.artifacts["matrix"])
        )

    def test_serialization_is_repeatable(self) -> None:
        matrix = self.artifacts["matrix"]
        self.assertEqual(
            frame_to_csv_text(matrix),
            frame_to_csv_text(matrix.copy()),
        )


if __name__ == "__main__":
    unittest.main()
